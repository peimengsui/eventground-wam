"""Compile canonical schemas with physics stepping and rendering disabled."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import signal
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as f:
        json.dump(seal_contract(value), f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=Path)
    a = p.parse_args()
    base = a.base.resolve()
    c = json.loads((base / "contract.json").read_text())
    assert contract_digest(c) == c["digest"]
    for rel, h in c["bundle_sha256"].items():
        assert sha(base / rel) == h, rel
    for path, h in c["runtime_files_sha256"].items():
        assert sha(path) == h, path
    for n, v in c["versions"].items():
        assert importlib.metadata.version(n) == v, n
    out = base / "run"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    rows = []
    failure = None

    def forbidden(*args, **kwargs):
        raise RuntimeError("Schema preflight forbids physics steps and rendering")

    def deadline(signum, frame):
        raise TimeoutError("Schema preflight time ceiling")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    save(out / "launch.json", {"contract_digest": c["digest"], "started_unix": time.time()})
    try:
        import mujoco

        for name in ("mj_step", "mj_step2"):
            setattr(mujoco, name, forbidden)
        original_step1 = mujoco.mj_step1
        position_stage_calls = 0

        def checked_position_stage(model, data):
            nonlocal position_stage_calls
            before_time, before_velocity, before_act = data.time, data.qvel.copy(), data.act.copy()
            canonical = data.qpos.copy()
            mujoco.mj_normalizeQuat(model, canonical)
            original_step1(model, data)
            assert data.time == before_time and np.array_equal(data.qvel, before_velocity)
            assert np.array_equal(data.act, before_act) and np.array_equal(data.qpos, canonical)
            position_stage_calls += 1
            assert position_stage_calls <= 160

        mujoco.mj_step1 = checked_position_stage
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.utils.binding_utils import MjSim

        MjSim.step = forbidden
        MjSim.render = forbidden
        for task in c["tasks"]:
            env = None
            try:
                assert sha(task["bddl"]) == task["bddl_sha256"]
                np.random.seed(0)
                env = ControlEnv(
                    bddl_file_name=task["bddl"],
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
                xml = env.sim.model.get_xml()
                assets = {
                    e.attrib["file"]: sha(e.attrib["file"])
                    for e in ET.fromstring(xml).iter()
                    if "file" in e.attrib
                }
                dest = out / task["task_key"].replace("/", "-")
                dest.mkdir()
                (dest / "canonical-model.xml").write_text(xml)
                result = {
                    "task_key": task["task_key"],
                    "bddl": task["bddl"],
                    "bddl_sha256": task["bddl_sha256"],
                    "joint_names": list(env.sim.model.joint_names),
                    "flattened_state_shape": list(env.sim.get_state().flatten().shape),
                    "nq": env.sim.model.nq,
                    "nv": env.sim.model.nv,
                    "asset_sha256": assets,
                    "canonical_model_sha256": sha(dest / "canonical-model.xml"),
                    "language_instruction": env.language_instruction,
                    "rendered_images": 0,
                    "physics_steps": 0,
                    "status": "passed",
                }
                save(dest / "result.json", result)
                rows.append(json.loads((dest / "result.json").read_text()))
                print(task["task_key"] + " passed", flush=True)
            finally:
                if env is not None:
                    env.close()
                gc.collect()
    except Exception as exc:  # noqa: BLE001 -- preserve all bounded qualification failures
        failure = {"error": repr(exc), "traceback": traceback.format_exc()}
    finally:
        signal.alarm(0)
        save(
            out / "completion.json",
            {
                "contract_digest": c["digest"],
                "status": "complete" if len(rows) == 40 and failure is None else "incomplete",
                "tasks": rows,
                "failure": failure,
                "elapsed_seconds": time.monotonic() - started,
                "position_velocity_stage_calls_without_integration": position_stage_calls,
                "new_control_steps": 0,
                "rendered_images": 0,
                "recorded_frames": 0,
                "heldout_payload_reads": 0,
                "training_updates": 0,
            },
        )
    if failure:
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
