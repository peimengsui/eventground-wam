"""Shared recorder with durable attempt accounting, including pre-render crashes."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import atomic_json, read_sealed
from eventground_wam.demonstration_contracts import seal_contract


def claim_attempt(path, identity, limit):
    """A persisted claim consumes an attempt even if the process dies before HDF5 opens."""
    if path.exists():
        journal = read_sealed(path)
        require(journal['identity'] == identity, 'attempt identity changed')
        attempts = journal['attempts']
        require(attempts == list(range(len(attempts))) and len(attempts) <= limit + 1,
                'attempt journal changed')
    else:
        attempts = []
    if len(attempts) > limit:
        return None
    attempt = len(attempts)
    atomic_json(path, seal_contract({'identity': identity, 'attempts': attempts + [attempt]}))
    return attempt


def evaluate_task(*, ordinal, manifest_task, suite, policy, protocol, output_dir,
                  project_revision, resume, checkpoint_id, checkpoint_revision):
    """Same environment and run_episode as B0/M1; no fresh retry allowance on restart."""
    import numpy as np
    import evaluate_reference_policy as ref
    from evaluate_clean_comparison import seed_episode

    task = suite.get_task(manifest_task['runtime_task_index'])
    require(task.name == manifest_task['task_name'], 'runtime task differs')
    require(isinstance(task.language, str) and bool(task.language), 'missing task instruction')
    path = output_dir / f'{ref.episode_stem(ordinal, manifest_task)}.hdf5'
    limit = protocol['episode_outcomes']['infrastructure_failure_retry_limit']
    journal_path = path.with_suffix('.attempts.json')
    identity = {'task': manifest_task, 'ordinal': ordinal, 'checkpoint_id': checkpoint_id,
                'checkpoint_revision': checkpoint_revision, 'project_revision': project_revision,
                'protocol_sha256': ref.protocol_sha256(protocol)}
    if journal_path.exists():
        require(read_sealed(journal_path)['identity'] == identity, 'attempt identity changed')
    completed = ref.find_resumable_episode(path, protocol, retry_limit=limit)
    if completed is not None:
        require(resume, 'episode already exists')
        return ref.recovered_result(completed[0], protocol, retry_count=completed[1],
            expected_project_revision=project_revision, manifest_task=manifest_task,
            expected_instruction=task.language, expected_checkpoint_id=checkpoint_id,
            expected_checkpoint_revision=checkpoint_revision)
    require(journal_path.exists() or not any(
        Path(str(ref.attempt_path(path, i)) + '.partial').exists() for i in range(limit + 1)),
        'partial artifacts without attempt journal')
    last_error = 'attempts exhausted before completion; retain partial artifacts'
    while (attempt := claim_attempt(journal_path, identity, limit)) is not None:
        environment = None
        try:
            # The journal is durable before imports, scene loading, rendering or policy calls.
            from libero.libero.envs import OffScreenRenderEnv
            seed_episode(manifest_task['episode_seed'])
            paths = ref.resolve_libero_bddl_paths(suite.get_task_bddl_file_path(manifest_task['runtime_task_index']))
            initial = np.asarray(suite.get_task_init_states(manifest_task['runtime_task_index'])[
                protocol['benchmark']['initial_state']['state_index']])
            metadata = ref.make_episode_metadata(protocol=protocol, manifest_task=manifest_task,
                task=task, bddl_source_path=paths.source, initial_state=initial,
                project_revision=project_revision, checkpoint_id=checkpoint_id,
                checkpoint_revision=checkpoint_revision, policy_instruction=task.language)
            resolution = protocol['observation']['capture_resolution']
            environment = OffScreenRenderEnv(bddl_file_name=str(paths.runtime),
                camera_heights=resolution[0], camera_widths=resolution[1],
                control_freq=protocol['simulation']['control_frequency_hz'])
            result = ref.run_episode(environment=environment, policy=policy, protocol=protocol,
                metadata=metadata, initial_state=initial, artifact_path=ref.attempt_path(path, attempt))
            return ref.EpisodeResult(**{**asdict(result), 'retry_count': attempt})
        except Exception as error:  # infrastructure boundary; run_episode scores policy failures itself
            last_error = f'{type(error).__name__}: {error}'
            atomic_json(path.with_suffix(f'.attempt-{attempt}-error.json'),
                        seal_contract({'identity': identity, 'attempt': attempt, 'error': last_error}))
        finally:
            if environment is not None:
                environment.close()
    return ref.EpisodeResult(**{key: manifest_task[key] for key in
        ('suite', 'runtime_task_index', 'perturbation_category', 'difficulty_level')},
        success=False, termination_reason='infrastructure_failure', artifact=None,
        artifact_sha256=None, policy_step_count=0, episode_wall_time_seconds=0.,
        policy_latencies_ms=[], retry_count=limit, exception=last_error)
