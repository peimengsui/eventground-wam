"""Qualify all canonical visual rules and audit saved states without integration or rendering."""

from __future__ import annotations
import argparse, gc, hashlib, json, signal, time, traceback
from pathlib import Path
import h5py
import numpy as np
from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.visual_scene import describe_visual_scene, verify_visual_scene


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    d = json.loads(Path(p).read_text())
    assert contract_digest(d) == d["digest"]
    return d


def save(p, d):
    with Path(p).open("x") as f:
        json.dump(seal_contract(d), f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    a = p.parse_args()
    base = a.base.resolve()
    c = load(base / "contract.json")
    for rel, h in c["bundle_sha256"].items():
        assert sha(base / rel) == h, rel
    cap = load(c["parent_capture_contract"])
    assert cap["digest"] == c["parent_capture_digest"]
    for path, h in cap["input_sha256"].items():
        assert sha(path) == h, path
    parent = Path(c["prior_run"])
    done = load(parent / "run/completion.json")
    assert done["digest"] == c["prior_completion_digest"]
    completed = {r["recording_id"]: r for r in done["results"] if r["status"] == "completed"}
    assert len(completed) == 15
    targets = [t for t in cap["targets"] if t["queue_role"] == "initial_review"]
    assert len(targets) == 200
    out = base / "run"
    out.mkdir(exist_ok=False)
    (out / "retained-visual-audit").mkdir()
    start = time.monotonic()
    stage_calls = 0
    env = None
    specs = {}
    audits = []
    sources = 0
    failure = None

    def forbidden(*a, **k):
        raise RuntimeError("Qualification forbids integration and rendering")

    def deadline(*a):
        raise TimeoutError("Visual qualification ceiling")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(c["wall_seconds"])
    try:
        import mujoco

        mujoco.mj_step = forbidden
        mujoco.mj_step2 = forbidden
        step1 = mujoco.mj_step1

        def checked(m, d):
            nonlocal stage_calls
            t = d.time
            v = d.qvel.copy()
            act = d.act.copy()
            q = d.qpos.copy()
            mujoco.mj_normalizeQuat(m, q)
            step1(m, d)
            stage_calls += 1
            assert (
                stage_calls <= 80
                and d.time == t
                and np.array_equal(d.qvel, v)
                and np.array_equal(d.act, act)
                and np.array_equal(d.qpos, q)
            )

        mujoco.mj_step1 = checked
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.binding_utils import MjSim

        MjSim.step = forbidden
        MjSim.render = forbidden
        for task in sorted({t["task_key"] for t in targets}):
            group = [t for t in targets if t["task_key"] == task]
            t = group[0]
            np.random.seed(0)
            env = ControlEnv(
                bddl_file_name=t["bddl"],
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
            env.env.reset()
            spec = describe_visual_scene(env)
            assert spec["canonical_reset_sha256"] == t["canonical_model_sha256"]
            specs[task] = spec
            for t in group:
                env.set_init_state(np.load(t["initial_state_file"], allow_pickle=False))
                verify_visual_scene(
                    env.sim.model.get_xml(), env.sim.model.site_rgba, env.sim.data.qpos, spec
                )
                sources += 1
                if t["recording_id"] not in completed:
                    continue
                old = parent / "run" / t["recording_id"]
                r = completed[t["recording_id"]]
                for rel, h in r["evidence_sha256"].items():
                    assert sha(old / rel) == h, rel
                assert sha(old / "model.xml") == spec["canonical_reset_sha256"]
                arrays = []
                mismatched = 0
                with h5py.File(old / "recording.h5", "r") as h:
                    assert h.attrs["complete"] and h["integration_state"].shape[0] == t["frames"]
                    for raw in h["integration_state"]:
                        mujoco.mj_setState(
                            env.sim.model._model,
                            env.sim.data._data,
                            raw,
                            mujoco.mjtState.mjSTATE_INTEGRATION,
                        )
                        env.sim.forward()
                        env._post_process()
                        rgba = verify_visual_scene(
                            env.sim.model.get_xml(),
                            env.sim.model.site_rgba,
                            env.sim.data.qpos,
                            spec,
                        )
                        arrays.append(rgba.copy())
                        mismatched += int(
                            not np.array_equal(
                                rgba, np.asarray(spec["reset_site_rgba"], dtype=np.float32)
                            )
                        )
                dest = out / "retained-visual-audit" / f"{t['recording_id']}.npz"
                np.savez_compressed(dest, site_rgba=np.stack(arrays))
                audits.append(
                    {
                        "recording_id": t["recording_id"],
                        "source_id": t["source_id"],
                        "frames": t["frames"],
                        "frames_with_visual_model_different_from_original_snapshot": mismatched,
                        "vendor_reconstruction_equals_independent_predicate_all_frames": True,
                        "vendor_site_rgba_file": str(dest),
                        "vendor_site_rgba_sha256": sha(dest),
                        "original_recording_sha256": r["evidence_sha256"]["recording.h5"],
                        "original_outcome_preserved": r["final_environment_done"],
                    }
                )
            env.close()
            env = None
            gc.collect()
            print(
                json.dumps(
                    {
                        "task": task,
                        "status": "passed",
                        "sources_checked": sources,
                        "retained_recordings_audited": len(audits),
                    }
                ),
                flush=True,
            )
    except Exception as exc:
        failure = {"error": repr(exc), "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            env.close()
        signal.alarm(0)
        save(
            out / "visual-scene-specs.json",
            {
                "artifact_type": "eventground_wam.qualified_visual_scene_specs",
                "qualification_contract_digest": c["digest"],
                "tasks": specs,
            },
        )
        result = {
            "status": "complete"
            if failure is None and len(specs) == 40 and sources == 200 and len(audits) == 15
            else "incomplete",
            "contract_digest": c["digest"],
            "failure": failure,
            "tasks": len(specs),
            "initial_sources_checked": sources,
            "retained_recordings_audited": len(audits),
            "retained_frames_audited": sum(r["frames"] for r in audits),
            "retained_audits": audits,
            "specs_sha256": sha(out / "visual-scene-specs.json"),
            "elapsed_seconds": time.monotonic() - start,
            "position_stage_calls": stage_calls,
            "integration_steps": 0,
            "rendered_images": 0,
            "source_actions": 0,
            "training_updates": 0,
        }
        save(out / "completion.json", result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "retained_audits"}, indent=2),
            flush=True,
        )
    if failure:
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
