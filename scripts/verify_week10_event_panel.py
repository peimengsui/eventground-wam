"""Independently verify exact held-out panel recordings with the unchanged physical/visual reconstruction core."""

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
from eventground_wam.eventground_weak_evaluation import validate_panel_contract
from eventground_wam.visual_scene import static_scene_digest


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
    specs = load(base / "visual-scene-specs.json")["tasks"]
    for key, expected in contract["verification"].items():
        assert audit[key] == expected, key
    completion = load(base / "run/completion.json")
    assert completion["status"] == "complete"
    assert completion["contract_digest"] == contract["digest"] == audit["recording_digest"]
    for rel, expected in contract["bundle_files_sha256"].items():
        assert sha(base / rel) == expected, rel
    for path, expected in contract["input_sha256"].items():
        assert sha(path) == expected, path
    assert completion["control_steps"] == contract["resources"]["remaining_control_steps"]
    assert completion["cumulative_control_steps"] == contract["resources"]["control_steps"]
    budget_seconds = contract["resources"]["wall_seconds"]
    assert completion["elapsed_seconds"] < budget_seconds
    remaining = int(budget_seconds - completion["elapsed_seconds"] - (time.monotonic() - start))
    assert remaining > 0, "Cumulative recording and verification deadline exhausted"
    signal.alarm(min(audit["wall_seconds"], remaining))
    split = load(base / "partitions.json")
    panel=load(base / "panel-contract.json")
    sources=validate_panel_contract(panel,split,load(base / "source-metadata.json"))
    targets=contract["targets"]
    assert len(targets)==len(sources)==len(completion["results"])
    assert [r["source_id"] for r in targets]==[r["source_id"] for r in sources]
    previews = base / "verification-previews"
    previews.mkdir(exist_ok=False)
    rows = []
    image_count = 0
    pixel_comparisons = []
    for case_index, (item, outcome) in enumerate(zip(targets, completion["results"], strict=True)):
        folder = base / "run" / item["recording_id"]
        assert outcome["status"] == "completed" and outcome["recording_id"] == item["recording_id"]
        assert outcome["source_id"] == item["source_id"]
        assert sources[case_index]["source_id"] == item["source_id"]
        preview = Image.new("RGB", (6 * 256, 286), "white")
        draw = ImageDraw.Draw(preview)
        for rel, expected in outcome["evidence_sha256"].items():
            assert sha(folder / rel) == expected, rel
        mapping, scene = load(folder / "geometry-map.json"), load(folder / "scene.json")
        visual_spec = specs[item["task_key"]]
        assert scene["visual_state_spec"] == visual_spec
        assert (
            sha(folder / "canonical-reset.xml")
            == item["canonical_model_sha256"]
            == visual_spec["canonical_reset_sha256"]
        )
        assert (
            static_scene_digest(
                (folder / "model.xml").read_text(),
                [f["site_name"] for f in visual_spec["dynamic_sites"]],
            )
            == visual_spec["static_scene_digest"]
        )
        for path, expected in scene["asset_sha256"].items():
            assert sha(path) == expected, path
        assert scene["compiled_capture_model_sha256"] == sha(folder / "capture-model.mjb")
        model = mujoco.MjModel.from_binary_path(str(folder / "capture-model.mjb"))
        sim = MjSim(model)
        MjRenderContextOffscreen(sim, device_id=0, max_width=256, max_height=256)
        data = sim.data._data
        # Resolve the pinned rule against this independently loaded model, not
        # recorder array indices alone. FlatStove visibility is qpos >= 0.5.
        for field in visual_spec["dynamic_sites"]:
            assert field["predicate"] == "flat_stove_turn_on_qpos_ge_0.5"
            assert (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, field["site_name"])
                == field["site_id"]
            )
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, field["joint_name"])
            assert int(model.jnt_qposadr[joint]) == field["joint_qpos_address"]
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
            assert h.attrs["visual_state_format"] == "qualified_site_rgba_v1"
            assert h["visual/site_rgba"].shape == (n, model.nsite, 4)
            for camera in ("front", "wrist"):
                assert h[f"observation/{camera}"].shape == (n, 256, 256, 3)
            for frame in range(n):
                raw = h["integration_state"][frame]
                assert np.isfinite(raw).all()
                mujoco.mj_setState(model, data, raw, spec)
                expected_rgba = np.asarray(visual_spec["reset_site_rgba"], dtype=np.float32).copy()
                for field in visual_spec["dynamic_sites"]:
                    site_id = field["site_id"]
                    visible = float(data.qpos[field["joint_qpos_address"]]) >= 0.5
                    alpha = expected_rgba[site_id, 3]
                    if (alpha > 0) != visible:
                        expected_rgba[site_id, 3] = 1 - alpha
                assert np.array_equal(h["visual/site_rgba"][frame], expected_rgba)
                model.site_rgba[:] = expected_rgba
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
                        assert np.array_equal(model.site_rgba, expected_rgba)
                        saved = h[f"observation/{name}"][frame]
                        delta = pixels.astype(np.int16) - saved.astype(np.int16)
                        pixel_comparisons.append(
                            {
                                "recording_id": item["recording_id"],
                                "frame": frame,
                                "camera": name,
                                "bitwise_equal": bool(np.array_equal(pixels, saved)),
                                "maximum_channel_error": int(np.abs(delta).max()),
                                "mean_absolute_channel_error": float(np.abs(delta).mean()),
                                "changed_pixels": int(
                                    np.count_nonzero(np.any(delta != 0, axis=-1))
                                ),
                                "total_pixels": 256 * 256,
                                "saved_sha256": hashlib.sha256(saved.tobytes()).hexdigest(),
                                "rerender_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                            }
                        )
                        if not np.array_equal(pixels, saved):
                            comparison_name = f"{item['recording_id']}-frame-{frame}-{name}"
                            Image.fromarray(pixels).save(
                                previews / f"{comparison_name}-rerender.png"
                            )
                            Image.fromarray(saved).save(previews / f"{comparison_name}-saved.png")
                            Image.fromarray(np.minimum(np.abs(delta) * 50, 255).astype("u1")).save(
                                previews / f"{comparison_name}-difference-x50.png"
                            )
                        column = selected.index(frame) * 2 + camera_index
                        preview.paste(Image.fromarray(saved), (column * 256, 30))
                        draw.text(
                            (column * 256 + 4, 7),
                            f"{item['demo_id']}  frame {frame}  {name}",
                            fill="black",
                        )
        preview.save(previews / f"{item['recording_id']}.png")
        sim.free()
        rows.append(
            {
                "recording_id": item["recording_id"],
                "source_id": item["source_id"],
                "frames": n,
                "preexisting_baseline_available": False,
                "all_contact_records_reconstructed": contact_count,
                "maximum_proprio_error": state_error,
                "maximum_object_gripper_pose_error": pose_error,
                "maximum_contact_error": contact_error,
                "all_visual_states_independently_reconstructed": True,
                "rerendered_frame_indices": selected,
                "both_cameras_bitwise_equal_on_selected_frames": all(
                    p["bitwise_equal"]
                    for p in pixel_comparisons
                    if p["recording_id"] == item["recording_id"]
                ),
            }
        )
        with (base / "verification-progress.jsonl").open("a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
    output_bytes = sum(p.stat().st_size for p in (base / "run").rglob("*") if p.is_file()) + sum(
        p.stat().st_size for p in previews.rglob("*") if p.is_file()
    )
    assert output_bytes <= contract["resources"]["output_bytes"]
    elapsed = time.monotonic() - start
    cumulative_seconds = completion["elapsed_seconds"] + elapsed
    assert cumulative_seconds <= budget_seconds
    signal.alarm(0)
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.production_capture_independent_verification",
            "status": "passed_state_contact_checks_with_pixel_equality_qualification"
            if not all(p["bitwise_equal"] for p in pixel_comparisons)
            else "passed_recording_consistency_not_event_label_QA",
            "pixel_comparisons": pixel_comparisons,
            "strict_pixel_equality_claimed": all(p["bitwise_equal"] for p in pixel_comparisons),
            "new_camera_disagreement_requires_qualification_review": not all(
                p["bitwise_equal"] for p in pixel_comparisons
            ),
            "total_verification_images_rendered": image_count,
            "recording_contract_digest": contract["digest"],
            "completion_digest": completion["digest"],
            "verification_contract_digest": audit["digest"],
            "cases": rows,
            "frames_verified": sum(r["frames"] for r in rows),
            "images_rerendered": image_count,
            "control_steps_added_by_verification": 0,
            "cumulative_production_control_steps": completion["control_steps"],
            "preserved_plus_new_production_control_steps": completion["cumulative_control_steps"],
            "all_panel_visual_states_reconstructed": True,
            "retained_original_recordings": 0,
            "verification_seconds": elapsed,
            "cumulative_recording_and_verification_seconds": cumulative_seconds,
            "cumulative_including_prior_and_qualification_seconds": cumulative_seconds
            + contract["resources"]["prior_seconds_charged"],
            "new_evidence_bytes": output_bytes,
            "preview_sha256": {p.name: sha(p) for p in previews.iterdir()},
            "heldout_sources_verified": len(rows),
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
