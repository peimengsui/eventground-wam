"""Re-render preserved states into a new artifact without repeating source actions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.visual_scene import expected_site_rgba, static_scene_digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    d = json.loads(Path(path).read_text())
    assert d["digest"] == contract_digest(d)
    return d


def repair_saved_recording(item, folder, contract, spec, prior_outcome, visual_audit):
    import mujoco
    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

    old = Path(contract["continuation"]["prior_run"]) / "run" / item["recording_id"]
    assert load(old / "result.json") == seal_contract(prior_outcome)
    for rel, h in prior_outcome["evidence_sha256"].items():
        assert sha(old / rel) == h
    assert sha(old / "recording.h5") == visual_audit["original_recording_sha256"]
    assert sha(visual_audit["vendor_site_rgba_file"]) == visual_audit["vendor_site_rgba_sha256"]
    assert sha(old / "model.xml") == spec["canonical_reset_sha256"]
    for name in (
        "geometry-map.json",
        "contacts.jsonl",
        "model.xml",
        "capture-model.mjb",
        "recording.h5",
    ):
        shutil.copy2(old / name, folder / name)
    shutil.copy2(old / "model.xml", folder / "canonical-reset.xml")
    with np.load(visual_audit["vendor_site_rgba_file"], allow_pickle=False) as saved:
        vendor_rgba = saved["site_rgba"].copy()
    model = mujoco.MjModel.from_binary_path(str(folder / "capture-model.mjb"))
    sim = MjSim(model)
    changed_frames = changed_pixels = maximum_error = 0
    physical = mujoco.mjtState.mjSTATE_FULLPHYSICS
    try:
        MjRenderContextOffscreen(sim, device_id=0, max_width=256, max_height=256)
        with h5py.File(folder / "recording.h5", "r+") as h:
            h.attrs["recording_contract_digest"] = contract["digest"]
            h.attrs["derived_from_recording_sha256"] = sha(old / "recording.h5")
            h.attrs["visuals_reconstructed_without_source_replay"] = True
            h.attrs["visual_state_format"] = "qualified_site_rgba_v1"
            rgba_dataset = h.create_dataset(
                "visual/site_rgba", data=vendor_rgba, compression="gzip"
            )
            for frame in range(item["frames"]):
                raw = h["integration_state"][frame]
                mujoco.mj_setState(model, sim.data._data, raw, mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(model, sim.data._data)
                expected = expected_site_rgba(spec, sim.data.qpos)
                assert np.array_equal(expected, rgba_dataset[frame])
                model.site_rgba[:] = expected
                before = np.empty(mujoco.mj_stateSize(model, physical))
                mujoco.mj_getState(model, sim.data._data, before, physical)
                different = False
                for name, camera in (("front", "agentview"), ("wrist", "robot0_eye_in_hand")):
                    rgb = sim.render(256, 256, camera_name=camera, device_id=0)[::-1, ::-1]
                    previous = h[f"observation/{name}"][frame]
                    delta = np.abs(rgb.astype(np.int16) - previous.astype(np.int16))
                    maximum_error = max(maximum_error, int(delta.max()))
                    changed_pixels += int(np.count_nonzero(np.any(delta != 0, axis=-1)))
                    different |= bool(np.any(delta))
                    h[f"observation/{name}"][frame] = rgb
                after = np.empty_like(before)
                mujoco.mj_getState(model, sim.data._data, after, physical)
                assert np.array_equal(before, after) and np.array_equal(model.site_rgba, expected)
                changed_frames += int(different)
                if frame % 25 == 0:
                    assert (
                        shutil.disk_usage(folder).free
                        >= contract["resources"]["minimum_free_bytes"]
                    )
                    assert (
                        sum(p.stat().st_size for p in folder.parent.rglob("*") if p.is_file())
                        <= contract["resources"]["output_bytes"]
                    )
        # Every non-image original payload is copied bit-for-bit, including outcome,
        # actions, integration states, frame indices and physical evidence.
        with (
            h5py.File(old / "recording.h5", "r") as before,
            h5py.File(folder / "recording.h5", "r") as after,
        ):

            def compare(name, obj):
                if isinstance(obj, h5py.Dataset) and name not in (
                    "observation/front",
                    "observation/wrist",
                ):
                    assert np.array_equal(obj[:], after[name][:]), name

            before.visititems(compare)
    finally:
        sim.free()
    scene = load(old / "scene.json")
    scene.pop("digest")
    scene.update(
        {
            "visual_state_spec": spec,
            "canonical_reset_model_sha256": spec["canonical_reset_sha256"],
            "static_scene_digest": static_scene_digest(
                (old / "model.xml").read_text(), [r["site_name"] for r in spec["dynamic_sites"]]
            ),
            "retained_original_scene_digest": load(old / "scene.json")["digest"],
        }
    )
    with (folder / "scene.json").open("x") as f:
        json.dump(seal_contract(scene), f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "status": "completed",
        "frames": item["frames"],
        "final_environment_done": prior_outcome["final_environment_done"],
        "retained_original_result_digest": load(old / "result.json")["digest"],
        "retained_original_recording_sha256": sha(old / "recording.h5"),
        "original_control_steps_preserved": prior_outcome["control_steps"],
        "source_actions_replayed": 0,
        "physical_payloads_bitwise_preserved": True,
        "images_reconstructed_from_saved_states": True,
        "frames_with_changed_pixels": changed_frames,
        "changed_pixels_across_both_views": changed_pixels,
        "maximum_channel_change_from_preserved_original": maximum_error,
        "vendor_visual_state_matches_all_frames": True,
        "maximum_consistency_errors": prior_outcome["maximum_consistency_errors"],
        "labels_admitted": False,
    }
