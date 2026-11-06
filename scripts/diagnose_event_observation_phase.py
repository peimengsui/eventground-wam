"""One preselected replay: inspect sensor timestamps and six diagnostic images."""

import argparse
import functools
import json
import signal
import time
from pathlib import Path

import numpy as np
from diagnose_event_regeneration import compare, read_sealed, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    contract = read_sealed(args.base / "contract.json")
    assert sha(__file__) == contract["runner_sha256"]
    assert (
        sha(Path(__file__).with_name("diagnose_event_regeneration.py")) == contract["helper_sha256"]
    )
    for name, digest in contract["input_sha256"].items():
        assert sha(name) == digest, name
    out = args.base / "run"
    out.mkdir(exist_ok=False)
    start = time.monotonic()

    def timeout(signum, frame):
        raise TimeoutError("Sealed one-case deadline")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["wall_seconds"])
    env = None
    steps = 0
    result = {
        "contract_digest": contract["digest"],
        "labels_admitted": False,
        "scene_identity_verified": False,
        "status": "started",
    }
    try:
        import av
        from libero.libero.envs.env_wrapper import ControlEnv
        from PIL import Image
        from robosuite.utils.transform_utils import quat2axisangle

        # Only this selected training episode's two files are decoded, once each.
        target_images = {}
        for camera, path in contract["videos"].items():
            if "cached_target_frames" in contract:
                for index in contract["image_frames"]:
                    frame_path = Path(contract["cached_target_frames"]) / (
                        f"target-{camera}-{index:03d}.png"
                    )
                    with Image.open(frame_path) as image:
                        array = np.asarray(image.convert("RGB")).copy()
                    assert array.shape == (256, 256, 3)
                    target_images[(camera, index)] = array
                    Image.fromarray(array).save(out / frame_path.name)
                continue
            with av.open(path) as container:
                count = 0
                for index, frame in enumerate(container.decode(video=0)):
                    if index >= 88:
                        raise ValueError("Unexpected video length")
                    count += 1
                    if index in contract["image_frames"]:
                        array = frame.to_ndarray(format="rgb24")
                        assert array.shape == (256, 256, 3)
                        target_images[(camera, index)] = array
                        Image.fromarray(array).save(out / f"target-{camera}-{index:03d}.png")
                assert count == 88

        original = Path(contract["original_base"])
        recorded = np.load(original / "recorded-state-diagnostic/candidate-04/recorded-states.npy")
        with np.load(original / "run/candidate-04/target-numeric.npz") as data:
            target_states, target_actions = data["states"].copy(), data["actions"].copy()
        with np.load(original / "run/candidate-04/demo_17-numeric.npz") as data:
            actions = data["actions"].copy()
        np.random.seed(0)
        env = ControlEnv(
            bddl_file_name=contract["bddl"],
            robots=["Panda"],
            controller="OSC_POSE",
            initialization_noise=None,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
        )
        env.seed(0)
        env.env.reset()
        env.set_init_state(recorded[0])
        last_samples = {}
        sensor_names = [
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]
        for name in sensor_names:
            observable = env.env._observables[name]
            original_sensor = observable._sensor

            @functools.wraps(original_sensor)
            def sensor(cache, sensor_name=name, base_sensor=original_sensor):
                value = base_sensor(cache)
                # Read only: no forwarding, stepping or forced observable update.
                last_samples[sensor_name] = {
                    "time": float(env.sim.data.time),
                    "full_state": env.sim.get_state().flatten().copy(),
                }
                return value

            observable.set_sensor(sensor)
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            steps += 1
        observations, sample_states, sample_times, end_times, image_metrics = [], [], [], [], []
        for index, action in enumerate(actions):
            observations.append(
                np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
            )
            sample_states.append(last_samples["robot0_gripper_qpos"]["full_state"].copy())
            sample_times.append([last_samples[name]["time"] for name in sensor_names])
            end_times.append(float(env.sim.data.time))
            if index in contract["image_frames"]:
                for camera, obskey in [
                    ("front", "agentview_image"),
                    ("wrist", "robot0_eye_in_hand_image"),
                ]:
                    raw = obs[obskey].copy()
                    Image.fromarray(raw).save(out / f"replayed-{camera}-{index:03d}-raw.png")
                    target = target_images[(camera, index)].astype(np.float64)
                    for rotation in [0, 180]:
                        rotated = raw if rotation == 0 else raw[::-1, ::-1]
                        error = np.abs(target - rotated.astype(np.float64))
                        image_metrics.append(
                            {
                                "frame": index,
                                "camera": camera,
                                "rotation_degrees": rotation,
                                "rgb_mean_absolute_error_0_to_255": float(error.mean()),
                                "pixel_exact_fraction": float((error == 0).all(-1).mean()),
                            }
                        )
            assert steps < 100
            obs, _, _, _ = env.step(action.tolist())
            steps += 1
        states = np.asarray(observations)
        sample_states = np.asarray(sample_states)
        times = np.asarray(sample_times)
        qpos_columns = 1 + np.asarray(
            [
                env.sim.model.get_joint_qpos_addr(j)
                for j in ("gripper0_finger_joint1", "gripper0_finger_joint2")
            ]
        )
        np.savez_compressed(
            out / "timing.npz",
            states=states,
            sensor_full_states=sample_states,
            sensor_times=times,
            return_times=np.asarray(end_times),
            actions=actions,
        )
        (out / "constructed-model.xml").write_text(env.sim.model.get_xml())
        result.update(
            {
                "status": "completed",
                "numeric_metrics": compare(target_states, target_actions, states, actions),
                "sensor_order": sensor_names,
                "max_sensor_timestamp_spread_seconds": float(np.ptp(times, axis=1).max()),
                "sample_to_return_lag_seconds_min_max": [
                    float((np.asarray(end_times) - times[:, 0]).min()),
                    float((np.asarray(end_times) - times[:, 0]).max()),
                ],
                "max_observed_vs_sampled_state_finger_error": float(
                    np.abs(states[:, 6:] - sample_states[:, qpos_columns]).max()
                ),
                "image_metrics": image_metrics,
                "image_comparison_is_diagnostic_not_an_admission_gate": True,
            }
        )
    except Exception as exc:
        result.update({"status": "unresolved_runtime", "error": repr(exc)})
        raise
    finally:
        signal.alarm(0)
        result.update(
            {
                "control_steps": steps,
                "elapsed_seconds": time.monotonic() - start,
                "training_updates": 0,
                "heldout_payload_reads": 0,
                "evidence_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
            }
        )
        save(out / "result.json", result)
        print(json.dumps(result), flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
