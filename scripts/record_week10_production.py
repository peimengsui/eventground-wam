"""Record the approved initial 200 v3 sources; requires separate exact launch approval."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import signal
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_snapshot import verify_snapshot_forward
from eventground_wam.production_capture import initial_capture_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, record):
    with Path(path).open("x") as handle:
        json.dump(seal_contract(native(record)), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load(path):
    record = json.loads(Path(path).read_text())
    assert contract_digest(record) == record["digest"], path
    return record


def descendants(model, root):
    result = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def geometry_map(env):
    """Exact model ancestry; retain excluded geometry with an explicit class."""
    model = env.sim.model._model
    robot = env.env.robots[0]
    gripper = robot.gripper
    gripper_root = env.sim.model.body_name2id(gripper.root_body)
    gripper_bodies = descendants(model, gripper_root)
    sides = {
        side: [env.sim.model.geom_name2id(name) for name in gripper.important_geoms[side]]
        for side in ("left_finger", "right_finger")
    }
    objects = []
    ownership = {}
    for name, obj in sorted(env.env.objects_dict.items()):
        root = env.sim.model.body_name2id(obj.root_body)
        bodies = descendants(model, root)
        free = any(
            int(model.jnt_bodyid[j]) == root and int(model.jnt_type[j]) == 0
            for j in range(model.njnt)
        )
        geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]
        for geom in geoms:
            if geom in ownership:
                raise ValueError("Ambiguous object geometry ownership")
            ownership[geom] = name
        objects.append(
            {
                "object_id": name,
                "body_id": root,
                "root_body": obj.root_body,
                "movable_free_object": free,
                "geom_ids": geoms,
            }
        )
    assert objects and any(o["movable_free_object"] for o in objects)
    geoms = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        owner = ownership.get(geom)
        classification = "excluded_fixture_robot_or_background"
        if owner is not None:
            obj = next(o for o in objects if o["object_id"] == owner)
            classification = (
                "movable_object" if obj["movable_free_object"] else "excluded_fixed_object"
            )
        if body in gripper_bodies:
            assert owner is None
            classification = "gripper"
        geoms.append(
            {
                "geom_id": geom,
                "name": env.sim.model.geom_id2name(geom),
                "body_id": body,
                "object_id": owner,
                "class": classification,
                "collision_enabled": bool(model.geom_contype[geom] or model.geom_conaffinity[geom]),
                "finger_sides": [side for side, values in sides.items() if geom in values],
            }
        )
    return {
        "objects": objects,
        "geoms": geoms,
        "finger_geom_ids": sides,
        "eef_site_id": int(robot.eef_site_id),
        "eef_body_id": env.sim.model.body_name2id(robot.robot_model.eef_name),
        "finger_qpos_addresses": list(robot._ref_gripper_joint_pos_indexes),
        "gripper_root_body": gripper.root_body,
    }


def contact_rows(data):
    return [
        {
            "geom1": int(c.geom1),
            "geom2": int(c.geom2),
            "distance_m": float(c.dist),
            "position": c.pos.copy().tolist(),
            "frame": c.frame.copy().tolist(),
            "efc_address": int(c.efc_address),
        }
        for c in data.contact[: data.ncon]
    ]


def contact_geometry(data):
    return np.array(
        [[c.geom1, c.geom2, c.dist, *c.pos, *c.frame] for c in data.contact[: data.ncon]],
        dtype=np.float64,
    ).reshape(-1, 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    contract = load(base / "recording-contract.json")
    authorization = load(args.authorization)
    targets = initial_capture_targets(contract, load(base / "partitions.json"), authorization)
    for rel, expected in contract["bundle_files_sha256"].items():
        assert sha(base / rel) == expected, rel
    for name, expected in contract["input_sha256"].items():
        assert sha(name) == expected, name
    for name, expected in contract["runtime"]["versions"].items():
        assert importlib.metadata.version(name) == expected, name
    assert shutil.disk_usage(base).free >= contract["resources"]["minimum_free_bytes"]
    out = base / "run"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    steps = 0
    rows = []

    def deadline(signum, frame):
        raise TimeoutError("Frozen production wall-clock limit")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(contract["resources"]["wall_seconds"])
    save(out / "authorization.json", {k: v for k, v in authorization.items() if k != "digest"})
    save(
        out / "launch.json",
        {
            "contract_digest": contract["digest"],
            "authorization_digest": authorization["digest"],
            "started_unix": time.time(),
        },
    )
    failure = None
    try:
        import mujoco
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
        from robosuite.utils.transform_utils import convert_quat, quat2axisangle

        integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        physics_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

        def get_state(model, data, spec=integration_spec):
            value = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
            mujoco.mj_getState(model, data, value, spec)
            return value

        for item in targets:
            folder = out / item["recording_id"]
            folder.mkdir()
            env = snapshot = None
            before = steps
            result = {
                "recording_id": item["recording_id"],
                "source_id": item["source_id"],
                "status": "started",
                "labels_admitted": False,
            }
            try:
                with np.load(item["actions_file"], allow_pickle=False) as payload:
                    actions = payload["actions"].copy()
                original = np.load(item["initial_state_file"], allow_pickle=False)
                n = item["frames"]
                assert (
                    actions.shape == (n, 7) and original.ndim == 1 and np.isfinite(original).all()
                )
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
                env.env.reset()  # one attempt; avoid wrapper's unbounded reset retries
                source_joints = [
                    node.attrib["name"]
                    for node in ET.parse(item["source_model_file"])
                    .getroot()
                    .find("worldbody")
                    .iter()
                    if node.tag in {"joint", "freejoint"}
                ]
                assert list(env.sim.model.joint_names) == source_joints
                assert env.sim.get_state().flatten().shape == original.shape
                env.set_init_state(original)
                mapping = geometry_map(env)
                save(folder / "geometry-map.json", mapping)
                model = env.sim.model._model
                live = env.sim.data._data
                snapshot = MjSim(copy.copy(model))
                assert not np.shares_memory(model.body_pos, snapshot.model._model.body_pos)
                MjRenderContextOffscreen(snapshot, device_id=0, max_width=256, max_height=256)
                sm, sd = snapshot.model._model, snapshot.data._data
                # A second independent mjData verifies position/contact reconstruction.
                audit = mujoco.MjData(sm)
                xml = env.sim.model.get_xml()
                assert hashlib.sha256(xml.encode()).hexdigest() == item["canonical_model_sha256"]
                (folder / "model.xml").write_text(xml)
                mujoco.mj_saveModel(sm, str(folder / "capture-model.mjb"), None)
                assets = {
                    element.attrib["file"]: sha(element.attrib["file"])
                    for element in ET.fromstring(xml).iter()
                    if "file" in element.attrib
                }
                save(
                    folder / "scene.json",
                    {
                        "object_mapping_digest": load(folder / "geometry-map.json")["digest"],
                        "bddl_sha256": sha(item["bddl"]),
                        "asset_sha256": assets,
                        "controller": {
                            k: v
                            for k, v in env.env.robots[0].controller_config.items()
                            if k != "sim"
                        },
                        "camera_names": ["agentview", "robot0_eye_in_hand"],
                        "camera_transform": "rotate raw pixels 180 degrees; lossless RGB uint8",
                        "corruption": "none",
                        "control_hz": 20,
                        "physics_timestep": float(sm.opt.timestep),
                        "integration_state_spec": int(integration_spec),
                        "compiled_capture_model_sha256": sha(folder / "capture-model.mjb"),
                        "source_historical_scene_identity_claimed": False,
                    },
                )

                def step(action, current_env=env):
                    nonlocal steps
                    if steps >= contract["resources"]["control_steps"]:
                        raise RuntimeError("Frozen control-step ceiling")
                    steps += 1
                    return current_env.step(action)

                for _ in range(10):
                    step([0, 0, 0, 0, 0, 0, -1])
                states, times, integration, poses, rotations = [], [], [], [], []
                gripper_positions, gripper_rotations, checks = [], [], []
                capture_qpos, normalization_adjustments = [], []
                quaternion_indices = []
                for joint in range(sm.njnt):
                    kind, address = int(sm.jnt_type[joint]), int(sm.jnt_qposadr[joint])
                    if kind in (0, 1):
                        start_address = address + (3 if kind == 0 else 0)
                        quaternion_indices.extend(range(start_address, start_address + 4))
                objects = mapping["objects"]
                object_body_ids = [o["body_id"] for o in objects]
                site = mapping["eef_site_id"]
                eef_body = mapping["eef_body_id"]
                fingers = mapping["finger_qpos_addresses"]
                with (
                    h5py.File(folder / "recording.h5", "x") as handle,
                    (folder / "contacts.jsonl").open("x") as contacts,
                ):
                    handle.attrs["recording_contract_digest"] = contract["digest"]
                    handle.attrs["source_id"] = item["source_id"]
                    handle.attrs["observation_phase"] = "fresh_snapshot_before_source_action"
                    image_arrays = {
                        name: handle.create_dataset(
                            f"observation/{name}",
                            (n, 256, 256, 3),
                            dtype="u1",
                            chunks=(1, 256, 256, 3),
                            compression="gzip",
                            compression_opts=1,
                        )
                        for name in ("front", "wrist")
                    }
                    for frame, action in enumerate(actions):
                        live_before = get_state(model, live)
                        # No preexisting production trace: exact live-state nonmutation is checked below.
                        mujoco.mj_setState(sm, sd, live_before, integration_spec)
                        physics_before = get_state(sm, sd, physics_spec)
                        canonical_qpos = sd.qpos.copy()
                        mujoco.mj_normalizeQuat(sm, canonical_qpos)
                        snapshot.forward()  # only the independent capture simulator
                        physics_after_forward = get_state(sm, sd, physics_spec)
                        normalization_adjustment = verify_snapshot_forward(
                            physics_before,
                            physics_after_forward,
                            canonical_qpos,
                            quaternion_indices,
                        )
                        mujoco.mj_setState(sm, audit, live_before, integration_spec)
                        mujoco.mj_fwdPosition(sm, audit)
                        geom_error = float(np.max(np.abs(sd.geom_xpos - audit.geom_xpos)))
                        rotation_error = float(np.max(np.abs(sd.geom_xmat - audit.geom_xmat)))
                        first, second = contact_geometry(sd), contact_geometry(audit)
                        assert first.shape == second.shape
                        contact_error = float(np.max(np.abs(first - second))) if first.size else 0.0
                        assert max(geom_error, rotation_error, contact_error) <= 1e-10
                        data_before_render = (
                            sd.geom_xpos.copy(),
                            sd.geom_xmat.copy(),
                            first.copy(),
                        )
                        state = np.concatenate(
                            (
                                sd.site_xpos[site],
                                quat2axisangle(convert_quat(sd.xquat[eef_body], to="xyzw")),
                                sd.qpos[fingers],
                            )
                        )
                        for name, camera in [
                            ("front", "agentview"),
                            ("wrist", "robot0_eye_in_hand"),
                        ]:
                            rgb = snapshot.render(256, 256, camera_name=camera, device_id=0)
                            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
                            image_arrays[name][frame] = rgb[::-1, ::-1]
                        assert np.array_equal(data_before_render[0], sd.geom_xpos)
                        assert np.array_equal(data_before_render[1], sd.geom_xmat)
                        assert np.array_equal(data_before_render[2], contact_geometry(sd))
                        assert np.array_equal(
                            physics_after_forward, get_state(sm, sd, physics_spec)
                        )
                        assert np.array_equal(live_before, get_state(model, live))
                        contacts.write(
                            json.dumps(
                                {
                                    "frame_index": frame,
                                    "time": float(sd.time),
                                    "contacts": contact_rows(sd),
                                }
                            )
                            + "\n"
                        )
                        capture_qpos.append(sd.qpos.copy())
                        normalization_adjustments.append(normalization_adjustment)
                        states.append(state.copy())
                        times.append(float(sd.time))
                        integration.append(live_before.copy())
                        poses.append(sd.xpos[object_body_ids].copy())
                        rotations.append(sd.xmat[object_body_ids].copy())
                        gripper_positions.append(sd.site_xpos[site].copy())
                        gripper_rotations.append(sd.site_xmat[site].copy())
                        checks.append([geom_error, rotation_error, contact_error])
                        _, _, done, _ = step(action.tolist())
                        if frame % 25 == 0:
                            assert (
                                shutil.disk_usage(base).free
                                >= contract["resources"]["minimum_free_bytes"]
                            )
                            assert (
                                sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                                <= contract["resources"]["output_bytes"]
                            )
                    for key, value in {
                        "observation/state": states,
                        "time": times,
                        "integration_state": integration,
                        "capture_qpos": capture_qpos,
                        "quaternion_normalization_adjustment": normalization_adjustments,
                        "action": actions,
                        "frame_index": np.arange(n),
                        "source_action_index": np.arange(n),
                        "object_position": poses,
                        "object_rotation_matrix": rotations,
                        "gripper_position": gripper_positions,
                        "gripper_rotation_matrix": gripper_rotations,
                        "consistency_errors": checks,
                    }.items():
                        array = np.asarray(value)
                        assert np.isfinite(array).all()
                        handle.create_dataset(key, data=array, compression="gzip")
                    handle.attrs["complete"] = True
                assert np.allclose(np.diff(times), 0.05, atol=1e-9, rtol=0)
                assert (
                    sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                    <= contract["resources"]["output_bytes"]
                )
                result.update(
                    {
                        "status": "completed",
                        "frames": n,
                        "final_environment_done": bool(done),
                        "preexisting_baseline_available": False,
                        "live_integration_state_unchanged_by_every_capture": True,
                        "geometry_contact_reconstruction_passed_all_frames": True,
                        "rendering_preserved_capture_state_all_frames": True,
                        "maximum_consistency_errors": np.max(checks, axis=0).tolist(),
                        "maximum_quaternion_normalization_adjustment": max(
                            normalization_adjustments
                        ),
                        "forward_state_equals_exact_MuJoCo_canonicalization": True,
                        "observation_period_seconds": 0.05,
                        "source_observations_replaced_by_new_recording": True,
                        "label_extraction_performed": False,
                    }
                )
            except Exception as exc:  # noqa: BLE001 -- preserve failure; never retry the trajectory
                result.update(
                    {
                        "status": "failed_integrity_or_runtime",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                failure = result["error"]
            finally:
                result["control_steps"] = steps - before
                result["evidence_sha256"] = {
                    p.name: sha(p) for p in folder.iterdir() if p.is_file()
                }
                save(folder / "result.json", result)
                rows.append(result)
                print(json.dumps(result), flush=True)
                if snapshot is not None:
                    snapshot.free()
                if env is not None:
                    env.close()
            if failure:
                break
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        signal.alarm(0)
        save(
            out / "completion.json",
            {
                "contract_digest": contract["digest"],
                "results": rows,
                "status": "complete"
                if len(rows) == len(targets) and failure is None
                else "incomplete",
                "requested_sources": len(targets),
                "reserve_sources_recorded": 0,
                "error": failure,
                "control_steps": steps,
                "elapsed_seconds": time.monotonic() - start,
                "heldout_payload_reads": 0,
                "external_downloads": 0,
                "training_updates": 0,
                "label_extraction_performed": False,
                "labels_admitted": False,
            },
        )

    if failure:
        raise RuntimeError(f"Production capture stopped: {failure}")


if __name__ == "__main__":
    main()
