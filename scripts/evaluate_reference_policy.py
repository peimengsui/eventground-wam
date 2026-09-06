#!/usr/bin/env python3
"""Evaluate the pinned OpenVLA-OFT checkpoint on a committed task manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from check_reference_policy import (
    EVALUATOR_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    verify_checkpoint,
    verify_evaluator,
    verify_libero,
)

from eventground_wam.episode_recorder import (
    EpisodeMetadata,
    EpisodeRecorder,
    StepRecord,
    read_episode_metadata,
    sha256_array,
    sha256_file,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import (
    load_evaluation_protocol,
    protocol_sha256,
)
from eventground_wam.evaluation_splits import validate_manifest
from eventground_wam.libero_bddl import resolve_libero_bddl_paths
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "splits" / "libero-plus-v1" / "sanity-35.json"


class InvalidPolicyAction(RuntimeError):
    """Raised when the reference policy does not return a valid locked action chunk."""


@dataclass(frozen=True)
class EpisodeResult:
    suite: str
    runtime_task_index: int
    perturbation_category: str
    difficulty_level: int | None
    success: bool
    termination_reason: str
    artifact: str | None
    artifact_sha256: str | None
    policy_step_count: int
    episode_wall_time_seconds: float | None
    policy_latencies_ms: list[float]
    retry_count: int
    exception: str | None


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    data_root = expanded_path(
        os.environ.get(
            "EVENTGROUND_REFERENCE_POLICY_DATA_ROOT",
            "~/data/eventground-wam/reference-policy",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=expanded_path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument(
        "--task-limit",
        "--limit",
        dest="task_limit",
        type=int,
        default=None,
        help="Evaluate this many manifest tasks after --task-offset (default: all).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=expanded_path,
        default=expanded_path(
            os.environ.get(
                "EVENTGROUND_REFERENCE_POLICY_CHECKPOINT",
                str(data_root / "checkpoints" / "openvla-7b-oft-libero-plus-a85655ec"),
            )
        ),
    )
    parser.add_argument(
        "--evaluator-root",
        type=expanded_path,
        default=expanded_path(
            os.environ.get("EVENTGROUND_OPENVLA_OFT_ROOT", "~/code/vendor/openvla-oft")
        ),
    )
    parser.add_argument(
        "--libero-root",
        type=expanded_path,
        default=expanded_path(
            os.environ.get("EVENTGROUND_LIBERO_ROOT", "~/code/vendor/LIBERO-plus")
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=expanded_path,
        default=data_root / "evaluations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the manifest selection without loading model or simulator.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load the pinned model and heads, report CUDA memory, then exit.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Reuse completed output episodes after validating their provenance (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Fail instead of reusing already validated output episodes.",
    )
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError(
            f"Refusing to record a dirty project checkout as a Git revision:\n{status}"
        )
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_manifest(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load task manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError("Task manifest must contain a JSON object")
    validate_manifest(manifest, protocol)
    return manifest


def select_tasks(
    manifest: dict[str, Any],
    *,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("--task-offset cannot be negative")
    if limit is not None and limit < 0:
        raise ValueError("--task-limit cannot be negative")
    tasks = manifest["tasks"]
    selected = tasks[offset:] if limit is None else tasks[offset : offset + limit]
    if not selected and not (limit == 0):
        raise ValueError("Manifest selection contains no tasks")
    return selected


def transform_rgb(
    observation: dict[str, Any],
    key: str,
    resolution: list[int],
) -> np.ndarray:
    image = np.asarray(observation[key])
    expected_shape = (*resolution, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise RuntimeError(
            f"Observation {key!r} must be uint8 {expected_shape}; found {image.dtype} {image.shape}"
        )
    return np.ascontiguousarray(image[::-1, ::-1])


def proprioception_from_observation(observation: dict[str, Any]) -> np.ndarray:
    from robosuite.utils import transform_utils

    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    axis_angle = transform_utils.quat2axisangle(quaternion)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
    proprioception = np.concatenate((position, axis_angle, gripper))
    if proprioception.shape != (8,) or not np.all(np.isfinite(proprioception)):
        raise RuntimeError("LIBERO observation did not produce finite 8-D proprioception")
    return proprioception


def process_openvla_action_chunk(actions: Any) -> np.ndarray:
    """Apply the official LIBERO gripper conversion to a complete action chunk."""

    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.shape != (8, 7):
        raise InvalidPolicyAction(f"Expected an 8x7 action chunk, found {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise InvalidPolicyAction("Policy action chunk has non-finite values")
    processed = chunk.copy()
    # Official OpenVLA-OFT: [0, 1] -> [-1, 1], binarize, then flip the
    # gripper sign back to LIBERO's convention.
    processed[:, -1] = -np.sign(2.0 * processed[:, -1] - 1.0)
    if np.any(processed < -1.0) or np.any(processed > 1.0):
        raise InvalidPolicyAction("Processed action chunk is outside the locked [-1, 1] range")
    return processed


def _load_state_dict(torch: Any, path: Path) -> dict[str, Any]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


class OpenVLAOFTPolicy:
    """Read-only wrapper around the pinned official OpenVLA-OFT inference logic."""

    def __init__(self, checkpoint_dir: Path) -> None:
        import torch
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA-OFT evaluation requires CUDA")
        self._torch = torch
        self._checkpoint_dir = checkpoint_dir
        self._device = torch.device("cuda:0")
        self._config = SimpleNamespace(
            model_family="openvla",
            num_images_in_input=2,
            use_proprio=True,
            center_crop=True,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            unnorm_key="",
        )

        self._model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.vision_backbone.set_num_images_in_input(2)
        self._model.eval().to(self._device)
        statistics_text = (checkpoint_dir / "dataset_statistics.json").read_text(encoding="utf-8")
        self._model.norm_stats = json.loads(statistics_text)

        action_head = L1RegressionActionHead(
            input_dim=self._model.llm_dim,
            hidden_dim=self._model.llm_dim,
            action_dim=7,
        )
        action_state = _load_state_dict(
            torch,
            checkpoint_dir / "action_head--150000_checkpoint.pt",
        )
        action_head.load_state_dict(action_state)
        self._action_head = action_head.to(torch.bfloat16).to(self._device).eval()

        projector = ProprioProjector(llm_dim=self._model.llm_dim, proprio_dim=8)
        projector_state = _load_state_dict(
            torch,
            checkpoint_dir / "proprio_projector--150000_checkpoint.pt",
        )
        projector.load_state_dict(projector_state)
        self._proprio_projector = projector.to(torch.bfloat16).to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(
            str(checkpoint_dir),
            trust_remote_code=True,
            local_files_only=True,
        )

    def predict_chunk(
        self,
        *,
        agent_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        proprioception: np.ndarray,
        instruction: str,
        suite: str,
    ) -> np.ndarray:
        from experiments.robot.openvla_utils import get_vla_action

        if suite not in self._model.norm_stats:
            raise RuntimeError(f"Checkpoint has no normalization statistics for {suite}")
        self._config.unnorm_key = suite
        observation = {
            "full_image": agent_rgb,
            "wrist_image": wrist_rgb,
            "state": proprioception.copy(),
        }
        raw_actions = get_vla_action(
            cfg=self._config,
            vla=self._model,
            processor=self._processor,
            obs=observation,
            task_label=instruction,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            noisy_action_projector=None,
            use_film=False,
        )
        return process_openvla_action_chunk(raw_actions)

    def memory_summary(self) -> dict[str, Any]:
        allocated = self._torch.cuda.memory_allocated(self._device)
        reserved = self._torch.cuda.memory_reserved(self._device)
        return {
            "device": self._torch.cuda.get_device_name(self._device),
            "capability": list(self._torch.cuda.get_device_capability(self._device)),
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
        }


def make_episode_metadata(
    *,
    protocol: dict[str, Any],
    manifest_task: dict[str, Any],
    task: Any,
    bddl_source_path: Path,
    initial_state: np.ndarray,
    project_revision: str,
) -> EpisodeMetadata:
    return EpisodeMetadata(
        protocol_id=protocol["protocol_id"],
        protocol_sha256=protocol_sha256(protocol),
        project_git_revision=project_revision,
        benchmark_revision=protocol["benchmark"]["revision"],
        checkpoint_id=MODEL_ID,
        checkpoint_revision=MODEL_REVISION,
        suite=manifest_task["suite"],
        runtime_task_index=manifest_task["runtime_task_index"],
        classification_task_id=manifest_task["classification_task_id"],
        perturbation_category=manifest_task["perturbation_category"],
        difficulty_level=manifest_task["difficulty_level"],
        instruction=task.language,
        bddl_path=str(bddl_source_path),
        bddl_sha256=sha256_file(bddl_source_path),
        initial_state_sha256=sha256_array(initial_state),
        episode_seed=manifest_task["episode_seed"],
    )


def _append_transition(
    recorder: EpisodeRecorder,
    *,
    protocol: dict[str, Any],
    observation: dict[str, Any],
    state: np.ndarray,
    phase: str,
    action_chunk: np.ndarray | None,
    action: np.ndarray,
    action_chunk_index: int,
    reward: float,
    done: bool,
    policy_latency_ms: float | None,
) -> None:
    cameras = protocol["observation"]["cameras"]
    resolution = protocol["observation"]["capture_resolution"]
    frequency = protocol["simulation"]["control_frequency_hz"]
    recorder.append(
        StepRecord(
            step_index=recorder.step_count,
            phase=phase,
            sim_time_seconds=recorder.step_count / frequency,
            agent_rgb=transform_rgb(observation, cameras["agent"], resolution),
            wrist_rgb=transform_rgb(observation, cameras["wrist"], resolution),
            proprioception=proprioception_from_observation(observation),
            predicted_action_chunk=action_chunk,
            executed_action=action,
            action_chunk_index=action_chunk_index,
            mujoco_state=state,
            reward=reward,
            done=done,
            policy_latency_ms=policy_latency_ms,
        )
    )


def run_episode(
    *,
    environment: Any,
    policy: OpenVLAOFTPolicy,
    protocol: dict[str, Any],
    metadata: EpisodeMetadata,
    initial_state: np.ndarray,
    artifact_path: Path,
) -> EpisodeResult:
    started = time.perf_counter()
    environment.seed(protocol["seeding"]["environment_seed"])
    environment.reset()
    observation = environment.set_init_state(initial_state)
    done = False
    policy_steps = 0
    latencies: list[float] = []
    policy_exception: str | None = None

    with EpisodeRecorder(artifact_path, protocol, metadata) as recorder:
        settling_action = np.asarray(
            protocol["simulation"]["settling_action"],
            dtype=np.float64,
        )
        for _ in range(protocol["simulation"]["settling_steps"]):
            state = np.asarray(environment.get_sim_state(), dtype=np.float64)
            next_observation, reward, done, _ = environment.step(settling_action)
            _append_transition(
                recorder,
                protocol=protocol,
                observation=observation,
                state=state,
                phase="settling",
                action_chunk=None,
                action=settling_action,
                action_chunk_index=-1,
                reward=reward,
                done=done,
                policy_latency_ms=None,
            )
            observation = next_observation
            if done:
                break

        suite = metadata.suite
        max_policy_steps = protocol["benchmark"]["suites"][suite]["max_policy_steps"]
        action_chunk = np.empty((0, 7), dtype=np.float64)
        for chunk_start in range(0, max_policy_steps, 8):
            if done:
                break
            agent_rgb = transform_rgb(
                observation,
                protocol["observation"]["cameras"]["agent"],
                protocol["observation"]["capture_resolution"],
            )
            wrist_rgb = transform_rgb(
                observation,
                protocol["observation"]["cameras"]["wrist"],
                protocol["observation"]["capture_resolution"],
            )
            proprioception = proprioception_from_observation(observation)
            query_started = time.perf_counter()
            try:
                action_chunk = policy.predict_chunk(
                    agent_rgb=agent_rgb,
                    wrist_rgb=wrist_rgb,
                    proprioception=proprioception,
                    instruction=metadata.instruction,
                    suite=suite,
                )
            except Exception as error:  # noqa: BLE001 - policy errors are task failures
                policy_exception = f"{type(error).__name__}: {error}"
                break
            latency_ms = (time.perf_counter() - query_started) * 1_000
            latencies.append(latency_ms)

            remaining = min(8, max_policy_steps - chunk_start)
            for action_chunk_index, action in enumerate(action_chunk[:remaining]):
                state = np.asarray(environment.get_sim_state(), dtype=np.float64)
                next_observation, reward, done, _ = environment.step(action)
                _append_transition(
                    recorder,
                    protocol=protocol,
                    observation=observation,
                    state=state,
                    phase="policy",
                    action_chunk=action_chunk,
                    action=action,
                    action_chunk_index=action_chunk_index,
                    reward=reward,
                    done=done,
                    policy_latency_ms=latency_ms if action_chunk_index == 0 else None,
                )
                observation = next_observation
                policy_steps += 1
                if done:
                    break

        if done:
            termination_reason = "environment_done"
        elif policy_exception is not None:
            termination_reason = f"policy_exception:{policy_exception.split(':', 1)[0]}"
        else:
            termination_reason = "policy_step_limit"
        recorder.finalize(success=bool(done), termination_reason=termination_reason)

    summary = validate_episode_file(artifact_path, protocol)
    return EpisodeResult(
        suite=metadata.suite,
        runtime_task_index=metadata.runtime_task_index,
        perturbation_category=metadata.perturbation_category,
        difficulty_level=metadata.difficulty_level,
        success=bool(done),
        termination_reason=termination_reason,
        artifact=str(artifact_path),
        artifact_sha256=sha256_file(artifact_path),
        policy_step_count=summary["policy_step_count"],
        episode_wall_time_seconds=time.perf_counter() - started,
        policy_latencies_ms=latencies,
        retry_count=0,
        exception=policy_exception,
    )


def episode_stem(index: int, manifest_task: dict[str, Any]) -> str:
    return f"{index:04d}-{manifest_task['suite']}-task-{manifest_task['runtime_task_index']:04d}"


def recovered_result(
    path: Path,
    protocol: dict[str, Any],
    *,
    retry_count: int,
    expected_project_revision: str,
    manifest_task: dict[str, Any],
    expected_instruction: str,
) -> EpisodeResult:
    import h5py

    summary = validate_episode_file(path, protocol)
    metadata = read_episode_metadata(path)
    expected_provenance = {
        "project_git_revision": expected_project_revision,
        "checkpoint_id": MODEL_ID,
        "checkpoint_revision": MODEL_REVISION,
        "suite": manifest_task["suite"],
        "runtime_task_index": manifest_task["runtime_task_index"],
        "classification_task_id": manifest_task["classification_task_id"],
        "perturbation_category": manifest_task["perturbation_category"],
        "difficulty_level": manifest_task["difficulty_level"],
        "episode_seed": manifest_task["episode_seed"],
        "instruction": expected_instruction,
    }
    for field, expected in expected_provenance.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"Refusing to resume {path}: expected {field}={expected!r}, "
                f"found {metadata.get(field)!r}"
            )
    with h5py.File(path, "r") as artifact:
        latencies = artifact["steps/policy_latency_ms"][:]
    finite_latencies = [float(value) for value in latencies if np.isfinite(value)]
    return EpisodeResult(
        suite=summary["suite"],
        runtime_task_index=summary["runtime_task_index"],
        perturbation_category=metadata["perturbation_category"],
        difficulty_level=metadata["difficulty_level"],
        success=bool(summary["success"]),
        termination_reason=summary["termination_reason"],
        artifact=str(path),
        artifact_sha256=sha256_file(path),
        policy_step_count=summary["policy_step_count"],
        episode_wall_time_seconds=None,
        policy_latencies_ms=finite_latencies,
        retry_count=retry_count,
        exception=("recovered_after_infrastructure_failure" if retry_count else None),
    )


def attempt_path(base_path: Path, retry_count: int) -> Path:
    if retry_count == 0:
        return base_path
    return base_path.with_name(f"{base_path.stem}-retry-{retry_count}{base_path.suffix}")


def find_resumable_episode(
    base_path: Path,
    protocol: dict[str, Any],
    *,
    retry_limit: int,
) -> tuple[Path, int] | None:
    """Return the latest valid completed attempt, including a retry artifact."""

    for retry_count in range(retry_limit, -1, -1):
        candidate = attempt_path(base_path, retry_count)
        if candidate.exists():
            validate_episode_file(candidate, protocol)
            return candidate, retry_count
    return None


def evaluate_task(
    *,
    ordinal: int,
    manifest_task: dict[str, Any],
    suite: Any,
    policy: OpenVLAOFTPolicy,
    protocol: dict[str, Any],
    output_dir: Path,
    project_revision: str,
    resume: bool,
) -> EpisodeResult:
    task_index = manifest_task["runtime_task_index"]
    task = suite.get_task(task_index)
    if task.name != manifest_task["task_name"]:
        raise RuntimeError(
            f"Runtime task does not match manifest: {task.name!r} != {manifest_task['task_name']!r}"
        )

    artifact_path = output_dir / f"{episode_stem(ordinal, manifest_task)}.hdf5"
    retry_limit = protocol["episode_outcomes"]["infrastructure_failure_retry_limit"]
    completed_attempt = find_resumable_episode(
        artifact_path,
        protocol,
        retry_limit=retry_limit,
    )
    if completed_attempt is not None:
        if not resume:
            raise FileExistsError(f"Episode already exists: {completed_attempt[0]}")
        return recovered_result(
            completed_attempt[0],
            protocol,
            retry_count=completed_attempt[1],
            expected_project_revision=project_revision,
            manifest_task=manifest_task,
            expected_instruction=task.language,
        )

    incomplete_attempts = [
        retry_count
        for retry_count in range(retry_limit + 1)
        if Path(str(attempt_path(artifact_path, retry_count)) + ".partial").exists()
    ]
    next_attempt = max(incomplete_attempts, default=-1) + 1
    if next_attempt > retry_limit:
        raise RuntimeError(f"All infrastructure attempts are incomplete for {artifact_path}")

    from libero.libero.envs import OffScreenRenderEnv

    bddl_paths = resolve_libero_bddl_paths(suite.get_task_bddl_file_path(task_index))
    initial_states = suite.get_task_init_states(task_index)
    state_index = protocol["benchmark"]["initial_state"]["state_index"]
    initial_state = np.asarray(initial_states[state_index])
    metadata = make_episode_metadata(
        protocol=protocol,
        manifest_task=manifest_task,
        task=task,
        bddl_source_path=bddl_paths.source,
        initial_state=initial_state,
        project_revision=project_revision,
    )

    resolution = protocol["observation"]["capture_resolution"]
    frequency = protocol["simulation"]["control_frequency_hz"]
    last_error: Exception | None = None
    for retry_count in range(next_attempt, retry_limit + 1):
        active_attempt_path = attempt_path(artifact_path, retry_count)
        environment = None
        try:
            environment = OffScreenRenderEnv(
                bddl_file_name=str(bddl_paths.runtime),
                camera_heights=resolution[0],
                camera_widths=resolution[1],
                control_freq=frequency,
            )
            result = run_episode(
                environment=environment,
                policy=policy,
                protocol=protocol,
                metadata=metadata,
                initial_state=initial_state,
                artifact_path=active_attempt_path,
            )
            if retry_count:
                result = EpisodeResult(**{**asdict(result), "retry_count": retry_count})
            return result
        except Exception as error:  # noqa: BLE001 - retry infrastructure boundary
            last_error = error
        finally:
            if environment is not None:
                environment.close()

    return EpisodeResult(
        suite=metadata.suite,
        runtime_task_index=metadata.runtime_task_index,
        perturbation_category=metadata.perturbation_category,
        difficulty_level=metadata.difficulty_level,
        success=False,
        termination_reason="infrastructure_failure",
        artifact=None,
        artifact_sha256=None,
        policy_step_count=0,
        episode_wall_time_seconds=0.0,
        policy_latencies_ms=[],
        retry_count=retry_limit,
        exception=f"{type(last_error).__name__}: {last_error}",
    )


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return [center - radius, center + radius]


def collect_runtime_metadata() -> dict[str, Any]:
    import tensorflow
    import torch
    import torchvision
    import transformers

    device = torch.device("cuda:0")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tensorflow.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
    }


def summary_filename(
    manifest: dict[str, Any],
    *,
    selected_task_offset: int,
    selected_task_count: int,
) -> str:
    if selected_task_offset == 0 and selected_task_count == manifest["task_count"]:
        return "summary.json"
    return f"summary-offset-{selected_task_offset:04d}-count-{selected_task_count:04d}.json"


def summarize_results(
    *,
    manifest: dict[str, Any],
    results: list[EpisodeResult],
    project_revision: str,
    runtime_metadata: dict[str, Any],
    environment_lock_sha256: str,
    selected_task_offset: int = 0,
    selected_task_count: int | None = None,
) -> dict[str, Any]:
    if selected_task_count is None:
        selected_task_count = len(results)
    successes = sum(result.success for result in results)
    latencies = np.asarray(
        [value for result in results for value in result.policy_latencies_ms],
        dtype=np.float64,
    )
    wall_times = [
        result.episode_wall_time_seconds
        for result in results
        if result.episode_wall_time_seconds is not None
    ]

    def breakdown(attribute_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[EpisodeResult]] = {}
        for result in results:
            labels = []
            for attribute_name in attribute_names:
                value = getattr(result, attribute_name)
                labels.append("unlabeled" if value is None else str(value))
            grouped.setdefault("|".join(labels), []).append(result)
        return {
            label: {
                "episode_count": len(group),
                "success_count": sum(result.success for result in group),
                "success_rate": sum(result.success for result in group) / len(group),
            }
            for label, group in sorted(grouped.items())
        }

    return {
        "schema_version": 1,
        "split_id": manifest["split_id"],
        "manifest_task_count": manifest["task_count"],
        "selected_task_offset": selected_task_offset,
        "selected_task_count": selected_task_count,
        "selection_is_complete_manifest": (
            selected_task_offset == 0 and selected_task_count == manifest["task_count"]
        ),
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "project_git_revision": project_revision,
        "checkpoint_id": MODEL_ID,
        "checkpoint_revision": MODEL_REVISION,
        "evaluator_revision": EVALUATOR_REVISION,
        "environment_lock": {
            "path": "environments/reference-policy/uv.lock",
            "sha256": environment_lock_sha256,
        },
        "runtime": runtime_metadata,
        "episode_count": len(results),
        "success_count": successes,
        "micro_success_rate": successes / len(results) if results else None,
        "wilson_95_percent": wilson_interval(successes, len(results)),
        "policy_latency_median_ms": float(np.median(latencies)) if latencies.size else None,
        "policy_latency_p95_ms": (float(np.percentile(latencies, 95)) if latencies.size else None),
        "episode_wall_time_seconds": (float(sum(wall_times)) if wall_times else None),
        "timeout_count": sum(
            result.termination_reason == "policy_step_limit" for result in results
        ),
        "exception_count": sum(result.exception is not None for result in results),
        "retry_count": sum(result.retry_count for result in results),
        "breakdowns": {
            "suite": breakdown(("suite",)),
            "perturbation_category": breakdown(("perturbation_category",)),
            "difficulty_level": breakdown(("difficulty_level",)),
            "perturbation_category_x_difficulty_level": breakdown(
                ("perturbation_category", "difficulty_level")
            ),
        },
        "episodes": [asdict(result) for result in results],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    protocol = load_evaluation_protocol()
    manifest = load_manifest(args.manifest, protocol)
    selected_tasks = select_tasks(
        manifest,
        offset=args.task_offset,
        limit=args.task_limit,
    )
    selection = {
        "manifest": str(args.manifest),
        "split_id": manifest["split_id"],
        "task_offset": args.task_offset,
        "task_count": len(selected_tasks),
        "checkpoint_id": MODEL_ID,
        "checkpoint_revision": MODEL_REVISION,
        "evaluator_revision": EVALUATOR_REVISION,
        "tasks": selected_tasks,
    }
    if args.dry_run:
        print(json.dumps(selection, indent=2))
        return

    project_revision = git_revision(PROJECT_ROOT)
    verify_evaluator(args.evaluator_root)
    verify_libero(args.libero_root)
    verify_checkpoint(args.checkpoint_dir, full_hash=False)
    policy = OpenVLAOFTPolicy(args.checkpoint_dir)
    runtime_metadata = collect_runtime_metadata()
    environment_lock_sha256 = sha256_file(
        PROJECT_ROOT / "environments" / "reference-policy" / "uv.lock"
    )
    if args.load_only:
        print(
            json.dumps(
                {
                    "selection": selection,
                    "project_git_revision": project_revision,
                    "environment_lock_sha256": environment_lock_sha256,
                    "runtime": runtime_metadata,
                    "model": policy.memory_summary(),
                },
                indent=2,
            )
        )
        return

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    import torch
    from libero.libero import benchmark

    install_codec_independent_motion_blur()
    output_dir = args.output_dir / manifest["split_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / summary_filename(
        manifest,
        selected_task_offset=args.task_offset,
        selected_task_count=len(selected_tasks),
    )
    suite_instances: dict[str, Any] = {}
    results: list[EpisodeResult] = []
    for ordinal, manifest_task in enumerate(selected_tasks, start=args.task_offset):
        seed = manifest_task["episode_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        suite_name = manifest_task["suite"]
        if suite_name not in suite_instances:
            suite_class = benchmark.get_benchmark_dict().get(suite_name)
            if suite_class is None:
                raise RuntimeError(f"LIBERO runtime does not provide suite {suite_name}")
            suite_instances[suite_name] = suite_class()
        result = evaluate_task(
            ordinal=ordinal,
            manifest_task=manifest_task,
            suite=suite_instances[suite_name],
            policy=policy,
            protocol=protocol,
            output_dir=output_dir,
            project_revision=project_revision,
            resume=args.resume,
        )
        results.append(result)
        report = summarize_results(
            manifest=manifest,
            results=results,
            project_revision=project_revision,
            runtime_metadata=runtime_metadata,
            environment_lock_sha256=environment_lock_sha256,
            selected_task_offset=args.task_offset,
            selected_task_count=len(selected_tasks),
        )
        write_json(report_path, report)
        print(
            f"[{len(results)}/{len(selected_tasks)}] {suite_name} "
            f"task={manifest_task['runtime_task_index']} success={result.success} "
            f"reason={result.termination_reason}"
        )

    report = summarize_results(
        manifest=manifest,
        results=results,
        project_revision=project_revision,
        runtime_metadata=runtime_metadata,
        environment_lock_sha256=environment_lock_sha256,
        selected_task_offset=args.task_offset,
        selected_task_count=len(selected_tasks),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"summary: {report_path}")


if __name__ == "__main__":
    main()
