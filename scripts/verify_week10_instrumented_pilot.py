"""Independently reload completed pilot evidence; no stepping or event extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    assert contract_digest(value) == value["digest"], path
    return value


def assert_near(first, second, tolerance=1e-10):
    first, second = np.asarray(first), np.asarray(second)
    assert first.shape == second.shape
    assert np.isfinite(first).all() and np.isfinite(second).all()
    error = float(np.max(np.abs(first - second))) if first.size else 0.0
    assert error <= tolerance, error
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    audit = load(base / "verification-contract.json")
    assert sha(__file__) == audit["verifier_sha256"]
    assert not (base / "verification.json").exists()
    start = time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("Verification wall-time ceiling")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(audit["wall_seconds"])
    import mujoco
    from PIL import Image, ImageDraw
    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

    contract = load(base / "recording-contract.json")
    completion = load(base / "run/completion.json")
    assert completion["status"] == "complete"
    assert completion["contract_digest"] == contract["digest"] == audit["recording_digest"]
    for rel, expected in contract["bundle_files_sha256"].items():
        assert sha(base / rel) == expected, rel
    for path, expected in contract["input_sha256"].items():
        assert sha(path) == expected, path
    previous = load(Path(audit["previous_pilot"]) / "startup-recovery/run/completion.json")
    assert previous["control_steps"] == 110
    assert previous["control_steps"] + completion["control_steps"] <= 700
    source_catalog = load(base / "source-catalog.json")
    sources = {r["source_id"]: r for r in source_catalog["sources"]}
    rows = []
    image_count = 0
    preview = Image.new("RGB", (6 * 256, 3 * 286), "white")
    draw = ImageDraw.Draw(preview)
    for case_index, (item, outcome) in enumerate(
        zip(contract["targets"], completion["results"], strict=True)
    ):
        folder = base / "run" / item["recording_id"]
        assert outcome["status"] == "completed" and outcome["recording_id"] == item["recording_id"]
        assert outcome["source_id"] == item["source_id"]
        assert sources[item["source_id"]]["known_alias_partition"]["partition"] == "train"
        for rel, expected in outcome["evidence_sha256"].items():
            assert sha(folder / rel) == expected, rel
        mapping, scene = load(folder / "geometry-map.json"), load(folder / "scene.json")
        for path, expected in scene["asset_sha256"].items():
            assert sha(path) == expected, path
        assert scene["compiled_capture_model_sha256"] == sha(folder / "capture-model.mjb")
        model = mujoco.MjModel.from_binary_path(str(folder / "capture-model.mjb"))
        sim = MjSim(model)
        MjRenderContextOffscreen(sim, device_id=0, max_width=256, max_height=256)
        data = sim.data._data
        # Check geometry ownership independently by walking each geom's ancestors.
        object_ids = [o["body_id"] for o in mapping["objects"]]
        gripper_root = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, mapping["gripper_root_body"]
        )
        for geom in mapping["geoms"]:
            gi = geom["geom_id"]
            body = int(model.geom_bodyid[gi])
            assert geom["body_id"] == body
            assert geom["name"] == mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            ancestors = {body}
            while body:
                body = int(model.body_parentid[body])
                ancestors.add(body)
            owners = [o["object_id"] for o in mapping["objects"] if o["body_id"] in ancestors]
            assert len(owners) <= 1
            assert geom["object_id"] == (owners[0] if owners else None)
            assert (geom["class"] == "gripper") == (gripper_root in ancestors)
        for obj in mapping["objects"]:
            free = any(
                int(model.jnt_bodyid[j]) == obj["body_id"] and int(model.jnt_type[j]) == 0
                for j in range(model.njnt)
            )
            assert obj["movable_free_object"] == free
        contacts = [
            json.loads(line) for line in (folder / "contacts.jsonl").read_text().splitlines()
        ]
        with np.load(item["baseline_file"]) as saved:
            baseline = saved["full_states"].copy()
        with np.load(item["actions_file"]) as saved:
            actions = saved["actions"].copy()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        n = item["frames"]
        selected = [0, n // 2, n - 1]
        state_error = pose_error = contact_error = 0.0
        contact_count = 0
        with h5py.File(folder / "recording.h5", "r") as h:
            assert h.attrs["complete"] and h.attrs["source_id"] == item["source_id"]
            assert h.attrs["recording_contract_digest"] == contract["digest"]
            assert h.attrs["observation_phase"] == "fresh_snapshot_before_source_action"
            assert len(contacts) == n == len(h["time"]) == outcome["frames"]
            assert np.array_equal(h["frame_index"][:], np.arange(n))
            assert np.array_equal(h["source_action_index"][:], np.arange(n))
            assert np.array_equal(h["action"][:], actions)
            assert np.allclose(np.diff(h["time"][:]), 0.05, atol=1e-9, rtol=0)
            assert h["integration_state"].shape == (n, mujoco.mj_stateSize(model, spec))
            assert np.max(h["consistency_errors"][:]) <= 1e-10
            for camera in ("front", "wrist"):
                assert h[f"observation/{camera}"].shape == (n, 256, 256, 3)
            for frame in range(n):
                raw = h["integration_state"][frame]
                assert np.isfinite(raw).all()
                mujoco.mj_setState(model, data, raw, spec)
                flat = np.concatenate(([data.time], data.qpos, data.qvel))
                assert np.array_equal(flat, baseline[frame])
                canonical = data.qpos.copy()
                mujoco.mj_normalizeQuat(model, canonical)
                time_before, velocity_before = data.time, data.qvel.copy()
                mujoco.mj_forward(model, data)
                assert np.array_equal(data.qpos, canonical)
                assert np.array_equal(data.qvel, velocity_before) and data.time == time_before
                assert np.array_equal(h["capture_qpos"][frame], data.qpos)
                assert float(h["time"][frame]) == data.time == contacts[frame]["time"]
                quaternion = data.xquat[mapping["eef_body_id"]].copy()
                w = float(np.clip(quaternion[0], -1.0, 1.0))
                den = math.sqrt(1.0 - w * w)
                axis = (
                    np.zeros(3)
                    if math.isclose(den, 0.0)
                    else quaternion[1:] * (2 * math.acos(w)) / den
                )
                state = np.concatenate(
                    (
                        data.site_xpos[mapping["eef_site_id"]],
                        axis,
                        data.qpos[mapping["finger_qpos_addresses"]],
                    )
                )
                state_error = max(state_error, assert_near(state, h["observation/state"][frame]))
                for expected, saved in (
                    (data.xpos[object_ids], h["object_position"][frame]),
                    (data.xmat[object_ids], h["object_rotation_matrix"][frame]),
                    (data.site_xpos[mapping["eef_site_id"]], h["gripper_position"][frame]),
                    (data.site_xmat[mapping["eef_site_id"]], h["gripper_rotation_matrix"][frame]),
                ):
                    pose_error = max(pose_error, assert_near(expected, saved))
                contact_record = contacts[frame]
                assert (
                    contact_record["frame_index"] == frame
                    and len(contact_record["contacts"]) == data.ncon
                )
                contact_count += data.ncon
                for saved, actual in zip(
                    contact_record["contacts"], data.contact[: data.ncon], strict=True
                ):
                    assert (saved["geom1"], saved["geom2"], saved["efc_address"]) == (
                        int(actual.geom1),
                        int(actual.geom2),
                        int(actual.efc_address),
                    )
                    for a, b in (
                        (saved["distance_m"], actual.dist),
                        (saved["position"], actual.pos),
                        (saved["frame"], actual.frame),
                    ):
                        contact_error = max(contact_error, assert_near(a, b))
                for camera in ("front", "wrist"):
                    assert np.ptp(h[f"observation/{camera}"][frame]) > 0
                if frame in selected:
                    for camera_index, (name, camera) in enumerate(
                        (("front", "agentview"), ("wrist", "robot0_eye_in_hand"))
                    ):
                        image_count += 1
                        assert image_count <= audit["maximum_rerendered_images"]
                        pixels = sim.render(256, 256, camera_name=camera, device_id=0)[::-1, ::-1]
                        saved = h[f"observation/{name}"][frame]
                        assert np.array_equal(pixels, saved), (item["recording_id"], frame, name)
                        column = selected.index(frame) * 2 + camera_index
                        preview.paste(Image.fromarray(saved), (column * 256, case_index * 286 + 30))
                        draw.text(
                            (column * 256 + 4, case_index * 286 + 7),
                            f"ep {item['episode']}  frame {frame}  {name}",
                            fill="black",
                        )
        sim.free()
        rows.append(
            {
                "recording_id": item["recording_id"],
                "old_episode_reference": item["episode"],
                "frames": n,
                "baseline_mapping_exact": True,
                "all_contact_records_reconstructed": contact_count,
                "maximum_proprio_error": state_error,
                "maximum_object_gripper_pose_error": pose_error,
                "maximum_contact_error": contact_error,
                "rerendered_frame_indices": selected,
                "both_cameras_bitwise_equal_on_selected_frames": True,
            }
        )
    preview.save(base / "camera-preview.png")
    previous_bytes = sum(
        p.stat().st_size for p in Path(audit["previous_pilot"]).rglob("*") if p.is_file()
    )
    output_bytes = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    assert previous_bytes + output_bytes <= contract["resources"]["cumulative_output_bytes_ceiling"]
    elapsed = time.monotonic() - start
    cumulative_seconds = previous["elapsed_seconds"] + completion["elapsed_seconds"] + elapsed
    assert cumulative_seconds <= contract["resources"]["cumulative_runner_wall_seconds_ceiling"]
    signal.alarm(0)
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.instrumented_pilot_independent_verification",
            "status": "passed_recording_consistency_not_event_label_QA",
            "recording_contract_digest": contract["digest"],
            "completion_digest": completion["digest"],
            "verification_contract_digest": audit["digest"],
            "cases": rows,
            "frames_verified": sum(r["frames"] for r in rows),
            "images_rerendered": image_count,
            "control_steps_added_by_verification": 0,
            "cumulative_pilot_control_steps": previous["control_steps"]
            + completion["control_steps"],
            "verification_seconds": elapsed,
            "cumulative_recording_and_verification_seconds": cumulative_seconds,
            "previous_evidence_bytes": previous_bytes,
            "new_evidence_bytes": output_bytes,
            "cumulative_evidence_bytes": previous_bytes + output_bytes,
            "camera_preview_sha256": sha(base / "camera-preview.png"),
            "heldout_payload_reads": 0,
            "label_extraction_performed": False,
            "labels_admitted": False,
            "production_dataset_admitted": False,
            "training_updates": 0,
        }
    )
    with (base / "verification.json").open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
