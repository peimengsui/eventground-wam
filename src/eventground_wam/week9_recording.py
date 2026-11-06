"""Pinned recorder loop with synchronized full-query timing for every Week 9 arm.

The loop is copied from evaluate_reference_policy.run_episode; only the injectable
GPU synchronization boundaries are added. No policy action, image, timeout or
outcome logic is changed.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import numpy as np
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from evaluate_reference_policy import (
    ActionChunkPolicy, EpisodeResult, EpisodeMetadata, EpisodeRecorder,
    _append_transition, transform_rgb, proprioception_from_observation,
    validate_episode_file, sha256_file,
)

def run_episode(
    *,
    environment: Any,
    policy: ActionChunkPolicy,
    protocol: dict[str, Any],
    metadata: EpisodeMetadata,
    initial_state: np.ndarray,
    artifact_path: Path,
    timing_synchronize=None,
) -> EpisodeResult:
    if timing_synchronize is None:
        import torch
        timing_synchronize = torch.cuda.synchronize
    started = time.perf_counter()
    environment.seed(protocol["seeding"]["environment_seed"])
    environment.reset()
    observation = environment.set_init_state(initial_state)
    done = False
    policy_steps = 0
    latencies: list[float] = []
    queries = []
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
            timing_synchronize()
            query_started = time.perf_counter()
            query_error = None
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
                query_error = policy_exception
            timing_synchronize()
            latency_ms = (time.perf_counter() - query_started) * 1_000
            queries.append({'latency_ms': latency_ms, 'exception': query_error})
            if query_error is not None:
                break
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
        atomic_json(artifact_path.with_suffix('.timing.json'), seal_contract({
            'artifact_type': 'eventground_wam.week9_episode_timing',
            'checkpoint_revision': metadata.checkpoint_revision,
            'cohort_digest': metadata.cohort_digest,
            'episode_wall_time_seconds': time.perf_counter() - started,
            'queries': queries, 'gpu_sync_at_both_boundaries': True,
            'no_queries_excluded': True}))
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
