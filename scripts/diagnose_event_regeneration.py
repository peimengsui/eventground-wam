"""Bounded, hypothesis-only replay. This script cannot admit physical labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import signal
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_native(value):
    if isinstance(value, dict):
        return {k: json_native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def seal(record):
    record = json_native(dict(record))
    record.pop("digest", None)
    payload = json.dumps(
        record, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return {**record, "digest": hashlib.sha256(payload).hexdigest()}


def save(path, record):
    with Path(path).open("x") as handle:
        json.dump(seal(record), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_sealed(path):
    record = json.loads(Path(path).read_text())
    assert seal(record)["digest"] == record["digest"], path
    return record


def compare(target_states, target_actions, states, actions):
    """Diagnostic metrics, deliberately separate from recorded-source admission."""
    if states.shape != target_states.shape or actions.shape != target_actions.shape:
        return {"complete_numeric_agreement": False, "reason": "full_sequence_shape_mismatch"}
    assert all(np.isfinite(x).all() for x in (target_states, target_actions, states, actions))
    error = np.abs(states.astype(np.float64) - target_states.astype(np.float64))
    action_error = np.abs(actions.astype(np.float64) - target_actions.astype(np.float64))
    exact_gripper = bool(np.array_equal(actions[:, -1], target_actions[:, -1]))
    return {
        "complete_numeric_agreement": bool(
            np.all(error <= 1e-5) and np.all(action_error <= 1e-6) and exact_gripper
        ),
        "max_state_absolute_error": float(error.max()),
        "per_dimension_state_max_error": error.max(axis=0).tolist(),
        "max_eef_position_distance_mm": float(np.linalg.norm(error[:, :3], axis=1).max() * 1000),
        "max_finger_error_mm": float(error[:, 6:].max() * 1000),
        "max_finger_error_after_frame_zero_mm": float(error[1:, 6:].max() * 1000),
        "frames_with_state_mismatch": int(np.any(error > 1e-5, axis=1).sum()),
        "frame_zero_state_error": error[0].tolist(),
        "max_action_absolute_error": float(action_error.max()),
        "gripper_commands_exact": exact_gripper,
    }


class Deadline(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    contract = read_sealed(args.base / "contract.json")
    assert sha(__file__) == contract["runner_sha256"]
    source = Path(contract["source_base"])
    for rel, digest in contract["input_sha256"].items():
        assert sha(source / rel) == digest, rel
    for path, digest in contract["runtime"]["files"].items():
        assert sha(path) == digest, path
    for name, version in contract["runtime"]["versions"].items():
        assert importlib.metadata.version(name) == version, name
    out = args.base / "run"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    steps = 0
    rows = []
    aborted = False

    def timeout(signum, frame):
        raise Deadline("Sealed global wall-clock deadline")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["wall_seconds"])
    save(
        out / "launch.json",
        {
            "contract_digest": contract["digest"],
            "started_unix": time.time(),
            "runner_sha256": sha(__file__),
        },
    )
    try:
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.transform_utils import quat2axisangle

        for item in contract["targets"]:
            rank = item["rank"]
            case = source / "run" / f"candidate-{rank:02d}"
            original = source / "recorded-state-diagnostic" / f"candidate-{rank:02d}"
            evidence = read_sealed(case / "result.json")
            assert evidence["candidate"]["split"] == "train"
            assert evidence["candidate"]["source_episode_index"] == item["episode"]
            assert rank not in contract["excluded_queue_ranks"]
            with np.load(case / "target-numeric.npz", allow_pickle=False) as data:
                target_states, target_actions = data["states"].copy(), data["actions"].copy()
            with np.load(case / f"{item['demo']}-numeric.npz", allow_pickle=False) as data:
                actions = data["actions"].copy()
            recorded = np.load(original / "recorded-states.npy", allow_pickle=False)
            assert actions.shape == target_actions.shape == (item["frames"], 7)
            assert np.all(np.abs(actions - target_actions) <= 1e-6)
            assert np.array_equal(actions[:, -1], target_actions[:, -1])
            source_joints = [
                node.attrib["name"]
                for node in ET.parse(original / "model.xml").getroot().find("worldbody").iter()
                if node.tag in {"joint", "freejoint"}
            ]

            for condition in contract["conditions"]:
                name = f"candidate-{rank:02d}-{condition['name']}"
                folder = out / name
                folder.mkdir()
                env = None
                row = {
                    "case": name,
                    "target_episode": item["episode"],
                    "settle_steps": condition["settle_steps"],
                    "status": "started",
                    "labels_admitted": False,
                    "scene_identity_verified": False,
                }
                before = steps
                observations, full_states, retained_actions, source_frames = [], [], [], []
                try:
                    np.random.seed(0)
                    env = ControlEnv(
                        bddl_file_name=item["bddl"],
                        robots=["Panda"],
                        controller="OSC_POSE",
                        initialization_noise=None,
                        use_camera_obs=False,
                        has_renderer=False,
                        has_offscreen_renderer=False,
                        control_freq=20,
                        horizon=1000,
                        ignore_done=False,
                        hard_reset=True,
                    )
                    env.seed(0)
                    # One attempt only; avoid ControlEnv.reset's unbounded error retry loop.
                    env.env.reset()
                    assert list(env.sim.model.joint_names) == source_joints
                    assert env.sim.get_state().flatten().shape == recorded[0].shape
                    xml = env.sim.model.get_xml()
                    (folder / "constructed-model.xml").write_text(xml)
                    assets = {}
                    for element in ET.fromstring(xml).iter():
                        if "file" in element.attrib:
                            path = Path(element.attrib["file"])
                            assets[str(path)] = sha(path)
                    save(
                        folder / "model-provenance.json",
                        {
                            "asset_sha256": assets,
                            "joint_names": source_joints,
                            "controller_config_excluding_sim_object": {
                                k: v
                                for k, v in env.env.robots[0].controller_config.items()
                                if k != "sim"
                            },
                            "model_timestep": float(env.sim.model.opt.timestep),
                            "model_sha256": sha(folder / "constructed-model.xml"),
                            "historical_model_identity_verified": False,
                        },
                    )
                    obs = env.set_init_state(recorded[0])

                    def step(action, current_env=env):
                        nonlocal steps
                        if steps >= contract["resources"]["control_steps"]:
                            raise RuntimeError("Sealed control-step budget exhausted")
                        steps += 1
                        return current_env.step(action)

                    for _ in range(condition["settle_steps"]):
                        obs, _, _, _ = step([0, 0, 0, 0, 0, 0, -1])
                    last = None
                    for frame, action in enumerate(actions):
                        if np.linalg.norm(action[:-1]) < 1e-4 and (
                            last is None or action[-1] == last[-1]
                        ):
                            continue
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        observations.append(state.copy())
                        full_states.append(env.sim.get_state().flatten().copy())
                        retained_actions.append(action.copy())
                        source_frames.append(frame)
                        last = action
                        obs, _, done, _ = step(action.tolist())
                    states = np.asarray(observations)
                    retained = np.asarray(retained_actions)
                    row.update(
                        {
                            "status": "completed",
                            "frames": len(states),
                            "final_done": bool(done),
                            "noops_dropped": len(actions) - len(states),
                            "metrics": compare(target_states, target_actions, states, retained),
                        }
                    )
                    finger_addresses = [
                        env.sim.model.get_joint_qpos_addr(j)
                        for j in ("gripper0_finger_joint1", "gripper0_finger_joint2")
                    ]
                    fingers = np.asarray(full_states)[:, np.asarray(finger_addresses) + 1]
                    row["max_observation_vs_current_full_state_finger_error"] = float(
                        np.abs(states[:, 6:] - fingers).max()
                    )
                except Exception as exc:  # noqa: BLE001 -- preserve each bounded runtime failure
                    row.update(
                        {
                            "status": "unresolved_runtime",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                except Deadline as exc:
                    row.update({"status": "resource_deadline", "error": str(exc)})
                    raise
                finally:
                    if observations:
                        np.savez_compressed(
                            folder / "replayed.npz",
                            states=np.asarray(observations),
                            full_states=np.asarray(full_states),
                            actions=np.asarray(retained_actions),
                            original_frame=np.asarray(source_frames, dtype=np.int64),
                        )
                    row["control_steps"] = steps - before
                    row["evidence_sha256"] = {
                        p.name: sha(p) for p in folder.iterdir() if p.is_file()
                    }
                    save(folder / "result.json", row)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if env is not None:
                        env.close()
                if (
                    sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                    > contract["resources"]["output_bytes"]
                ):
                    raise RuntimeError("Sealed output ceiling exceeded")
    except Deadline:
        aborted = True
    finally:
        signal.alarm(0)
        save(
            out / "summary.json",
            {
                "contract_digest": contract["digest"],
                "results": rows,
                "aborted": aborted,
                "elapsed_seconds": time.monotonic() - start,
                "control_steps": steps,
                "labels_admitted": False,
                "training_updates": 0,
                "new_episode_payload_downloads": 0,
                "heldout_payload_reads": 0,
                "numeric_match_does_not_establish_historical_scene_or_frame_identity": True,
            },
        )


if __name__ == "__main__":
    main()
