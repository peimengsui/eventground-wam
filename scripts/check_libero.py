"""Record a short zero-action policy rollout in one LIBERO-Plus environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation" / "splits" / "libero-plus-v1" / "full-10K.json"
)


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=expanded_path,
        default=expanded_path("~/data/eventground-wam/libero-plus/smoke"),
    )
    parser.add_argument(
        "--manifest",
        type=expanded_path,
        default=DEFAULT_MANIFEST_PATH,
        help="Full locked task manifest used to attach perturbation labels.",
    )
    parser.add_argument("--checkpoint-id", default="builtin/zero-action-smoke-policy")
    parser.add_argument("--checkpoint-revision", default="zero-action-smoke-v1")
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def task_record_from_manifest(
    manifest_path: Path,
    protocol: dict[str, Any],
    suite: str,
    runtime_task_index: int,
) -> dict[str, Any]:
    from eventground_wam.evaluation_protocol import protocol_sha256

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_id") != "full-10K":
        raise SystemExit(f"Expected the full-10K manifest: {manifest_path}")
    if manifest.get("protocol_sha256") != protocol_sha256(protocol):
        raise SystemExit("Task manifest does not match the locked evaluation protocol")
    matches = [
        task
        for task in manifest.get("tasks", [])
        if task.get("suite") == suite
        and task.get("runtime_task_index") == runtime_task_index
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one manifest task for {suite}[{runtime_task_index}], found {len(matches)}"
        )
    return matches[0]


def transformed_rgb(observation: dict[str, Any], key: str, resolution: list[int]) -> Any:
    import numpy as np

    image = np.asarray(observation[key])
    expected_shape = (*resolution, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise RuntimeError(
            f"Observation {key!r} must have shape {expected_shape} and dtype uint8; "
            f"found shape={image.shape}, dtype={image.dtype}"
        )
    return np.ascontiguousarray(image[::-1, ::-1])


def proprioception_from_observation(observation: dict[str, Any]) -> Any:
    import numpy as np
    from robosuite.utils import transform_utils

    eef_position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    eef_quaternion = np.asarray(
        observation["robot0_eef_quat"], dtype=np.float64
    ).copy()
    eef_axis_angle = transform_utils.quat2axisangle(eef_quaternion)
    gripper_qpos = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
    proprioception = np.concatenate((eef_position, eef_axis_angle, gripper_qpos))
    if proprioception.shape != (8,):
        raise RuntimeError(f"Unexpected proprioception shape: {proprioception.shape}")
    return proprioception


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise SystemExit("--steps cannot be negative")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    import cv2
    import numpy as np
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    from eventground_wam.episode_recorder import (
        EpisodeMetadata,
        EpisodeRecorder,
        StepRecord,
        sha256_array,
        sha256_file,
        validate_episode_file,
    )
    from eventground_wam.evaluation_protocol import (
        load_evaluation_protocol,
        protocol_sha256,
    )
    from eventground_wam.libero_bddl import resolve_libero_bddl_paths
    from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

    install_codec_independent_motion_blur()
    protocol = load_evaluation_protocol()
    benchmark_class = benchmark.get_benchmark_dict().get(args.suite.lower())
    if benchmark_class is None:
        available = ", ".join(sorted(benchmark.get_benchmark_dict()))
        raise SystemExit(f"Unknown suite {args.suite!r}; available suites: {available}")
    suite_specification = protocol["benchmark"]["suites"].get(args.suite)
    if suite_specification is None:
        raise SystemExit(f"Suite {args.suite!r} is not in the locked protocol")
    if not 0 <= args.task_id < suite_specification["task_count"]:
        raise SystemExit(f"--task-id is outside suite {args.suite!r}")
    if args.steps > suite_specification["max_policy_steps"]:
        raise SystemExit(
            f"--steps exceeds the {suite_specification['max_policy_steps']}-step suite budget"
        )

    suite = benchmark_class()
    task = suite.get_task(args.task_id)
    bddl_paths = resolve_libero_bddl_paths(suite.get_task_bddl_file_path(args.task_id))
    init_states = suite.get_task_init_states(args.task_id)
    initial_state = np.asarray(
        init_states[protocol["benchmark"]["initial_state"]["state_index"]]
    )
    manifest_task = task_record_from_manifest(
        args.manifest,
        protocol,
        args.suite,
        args.task_id,
    )
    if manifest_task["task_name"] != task.name:
        raise SystemExit(
            "Runtime task name does not match the locked manifest: "
            f"{task.name!r} != {manifest_task['task_name']!r}"
        )

    episode_seed = manifest_task["episode_seed"]
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)

    metadata = EpisodeMetadata(
        protocol_id=protocol["protocol_id"],
        protocol_sha256=protocol_sha256(protocol),
        project_git_revision=git_revision(PROJECT_ROOT),
        benchmark_revision=protocol["benchmark"]["revision"],
        checkpoint_id=args.checkpoint_id,
        checkpoint_revision=args.checkpoint_revision,
        suite=args.suite,
        runtime_task_index=args.task_id,
        classification_task_id=manifest_task["classification_task_id"],
        perturbation_category=manifest_task["perturbation_category"],
        difficulty_level=manifest_task["difficulty_level"],
        instruction=task.language,
        bddl_path=str(bddl_paths.source),
        bddl_sha256=sha256_file(bddl_paths.source),
        initial_state_sha256=sha256_array(initial_state),
        episode_seed=episode_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"{args.suite}-task-{args.task_id}.hdf5"
    image_path = args.output_dir / f"{args.suite}-task-{args.task_id}.png"
    report_path = args.output_dir / f"{args.suite}-task-{args.task_id}.json"
    resolution = protocol["observation"]["capture_resolution"]
    cameras = protocol["observation"]["cameras"]
    settling_action = np.asarray(
        protocol["simulation"]["settling_action"], dtype=np.float64
    )
    control_frequency = protocol["simulation"]["control_frequency_hz"]
    action_chunk_length = protocol["simulation"]["action"]["chunk_length"]

    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_paths.runtime),
        camera_heights=resolution[0],
        camera_widths=resolution[1],
        control_freq=control_frequency,
    )
    observation: dict[str, Any]
    done = False
    try:
        environment.seed(protocol["seeding"]["environment_seed"])
        environment.reset()
        observation = environment.set_init_state(initial_state)
        with EpisodeRecorder(artifact_path, protocol, metadata) as recorder:
            for _ in range(protocol["simulation"]["settling_steps"]):
                step_index = recorder.step_count
                state = environment.get_sim_state()
                next_observation, reward, done, _ = environment.step(settling_action)
                recorder.append(
                    StepRecord(
                        step_index=step_index,
                        phase="settling",
                        sim_time_seconds=step_index / control_frequency,
                        agent_rgb=transformed_rgb(
                            observation, cameras["agent"], resolution
                        ),
                        wrist_rgb=transformed_rgb(
                            observation, cameras["wrist"], resolution
                        ),
                        proprioception=proprioception_from_observation(observation),
                        predicted_action_chunk=None,
                        executed_action=settling_action,
                        action_chunk_index=-1,
                        mujoco_state=state,
                        reward=reward,
                        done=done,
                        policy_latency_ms=None,
                    )
                )
                observation = next_observation
                if done:
                    break

            action_chunk = np.zeros((action_chunk_length, 7), dtype=np.float64)
            for policy_step_index in range(args.steps if not done else 0):
                action_chunk_index = policy_step_index % action_chunk_length
                latency_ms = None
                if action_chunk_index == 0:
                    query_started = time.perf_counter()
                    action_chunk = np.zeros(
                        (action_chunk_length, 7), dtype=np.float64
                    )
                    latency_ms = (time.perf_counter() - query_started) * 1_000
                action = action_chunk[action_chunk_index]
                step_index = recorder.step_count
                state = environment.get_sim_state()
                next_observation, reward, done, _ = environment.step(action)
                recorder.append(
                    StepRecord(
                        step_index=step_index,
                        phase="policy",
                        sim_time_seconds=step_index / control_frequency,
                        agent_rgb=transformed_rgb(
                            observation, cameras["agent"], resolution
                        ),
                        wrist_rgb=transformed_rgb(
                            observation, cameras["wrist"], resolution
                        ),
                        proprioception=proprioception_from_observation(observation),
                        predicted_action_chunk=action_chunk,
                        executed_action=action,
                        action_chunk_index=action_chunk_index,
                        mujoco_state=state,
                        reward=reward,
                        done=done,
                        policy_latency_ms=latency_ms,
                    )
                )
                observation = next_observation
                if done:
                    break

            termination_reason = "environment_done" if done else "smoke_policy_step_limit"
            recorder.finalize(success=bool(done), termination_reason=termination_reason)
    finally:
        environment.close()

    agent_rgb = transformed_rgb(observation, cameras["agent"], resolution)
    if not cv2.imwrite(str(image_path), agent_rgb[:, :, ::-1]):
        raise SystemExit(f"Failed to write smoke-test image: {image_path}")

    artifact_summary = validate_episode_file(artifact_path, protocol)
    report = {
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_summary": artifact_summary,
        "bddl_runtime_path": str(bddl_paths.runtime),
        "bddl_source_file": str(bddl_paths.source),
        "cuda_available": torch.cuda.is_available(),
        "image_shape": list(agent_rgb.shape),
        "instruction": task.language,
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "policy_steps_requested": args.steps,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"episode: {artifact_path}")
    print(f"render: {image_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
