"""Week 11 durable execution with classified retries and event-log validation."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import ComparisonMetadata, atomic_json, inspect_artifact
from eventground_wam.coupled_rollouts import claim_attempt
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import EpisodeMetadata
from eventground_wam.week11_recording import run_episode


@dataclass(frozen=True, kw_only=True)
class Week9Metadata(EpisodeMetadata):
    cohort_digest: str
    week9_protocol_digest: str
    policy_objective: str
    admission_digest: str
    legacy_profile_and_acceptance_fields_are_format_only: bool = True

@dataclass(frozen=True, kw_only=True)
class Week9CleanMetadata(ComparisonMetadata):
    cohort_digest: str
    week9_protocol_digest: str
    policy_objective: str
    admission_digest: str
    legacy_profile_and_acceptance_fields_are_format_only: bool = True


def bind_metadata(metadata, identity):
    cls = Week9CleanMetadata if isinstance(metadata, ComparisonMetadata) else Week9Metadata
    return cls(**asdict(metadata), **{k: identity[k] for k in (
        'cohort_digest', 'week9_protocol_digest', 'policy_objective', 'admission_digest')})


def immutable(path, value):
    if path.exists():
        require(read_sealed(path) == value, f'immutable artifact changed: {path}')
    else:
        atomic_json(path, value)


def inspect_timed(path, protocol, metadata):
    row = inspect_artifact(path, protocol, metadata)
    timing_path = path.with_suffix('.timing.json')
    timing = read_sealed(timing_path)
    require(timing['checkpoint_revision'] == metadata.checkpoint_revision
            and timing['cohort_digest'] == metadata.cohort_digest
            and timing['gpu_sync_at_both_boundaries'] and timing['no_queries_excluded'], 'timing binding changed')
    successful = [q['latency_ms'] for q in timing['queries'] if q['exception'] is None]
    require(successful == row['policy_latencies_ms'], 'query timer disagrees with recorded actions')
    row['recorded_action_query_latencies_ms'] = row['policy_latencies_ms']
    row['policy_latencies_ms'] = [q['latency_ms'] for q in timing['queries']]
    require(all(np.isfinite(x) and x >= 0 for x in row['policy_latencies_ms']), 'invalid timing values')
    row.update(timing_artifact=str(timing_path), timing_sha256=sha256_file(timing_path),
               query_exceptions=[q['exception'] for q in timing['queries']],
               episode_wall_time_seconds=timing['episode_wall_time_seconds'])
    return row


def execute_episode(*, root, ordinal, task, identity, protocol, metadata, initial_state,
                    policy, environment_factory, seed_episode, episode_runner=run_episode):
    """Persist attempts before environment creation; no extra retries on resume."""
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f'result-{ordinal:04d}.json'
    base = root / f'episode-{ordinal:04d}'
    journal = root / f'episode-{ordinal:04d}.attempts.json'
    key = {'identity_digest': identity['digest'], 'ordinal': ordinal, 'task': task}
    def inspect(path, attempt):
        row = inspect_timed(path, protocol, metadata)
        row.update({k: task[k] for k in ('suite', 'runtime_task_index', 'perturbation_category', 'difficulty_level')})
        row['retry_count'] = attempt
        return row
    if receipt_path.exists():
        receipt = read_sealed(receipt_path)
        require(all(receipt[k] == v for k, v in key.items()), 'receipt identity changed')
        require(receipt['attempt_journal_sha256'] == sha256_file(journal), 'attempt journal changed')
        row = receipt['result']
        if row['artifact']:
            require(inspect(Path(row['artifact']), row['retry_count']) == row, 'saved episode changed')
        else:
            require(row['success'] is False and row['termination_reason'] == 'infrastructure_failure'
                    and read_sealed(journal)['attempts'] == [0, 1], 'invalid exhausted failure')
        return row
    if journal.exists():
        attempts = read_sealed(journal)
        require(attempts['identity'] == key, 'attempt identity changed')
        for attempt in attempts['attempts']:
            path = Path(f'{base}-attempt-{attempt}.hdf5')
            if path.exists():
                row = inspect(path, attempt)
                break
        else:
            row = None
    else:
        require(not list(root.glob(f'{base.name}-attempt-*')), 'unaccounted attempt files')
        row = None
    while row is None and (attempt := claim_attempt(journal, key, 1)) is not None:
        path = Path(f'{base}-attempt-{attempt}.hdf5')
        environment = None
        started = time.monotonic()
        try:
            seed_episode(task['episode_seed'])
            environment = environment_factory()
            episode_runner(environment=environment, policy=policy, protocol=protocol,
                metadata=metadata, initial_state=initial_state, artifact_path=path)
        except Exception as exc:  # policy errors are already scored inside the recorder
            # Only recognized infrastructure faults can consume the one retry.
            # Assertion, contract and artifact-integrity failures stop admission.
            infrastructure = isinstance(exc, (OSError, EOFError)) or type(exc).__name__ in (
                'FatalError', 'MjError', 'GLFWError', 'GLError')
            atomic_json(Path(f'{base}-attempt-{attempt}-error.json'), seal_contract({
                **key, 'attempt': attempt, 'exception': f'{type(exc).__name__}: {exc}',
                'wall_seconds': time.monotonic()-started, 'classified_infrastructure': infrastructure}))
            if not infrastructure:
                raise
        finally:
            if environment is not None:
                try:
                    environment.close()
                except Exception as exc:  # noqa: BLE001 - preserve close failure evidence
                    atomic_json(Path(f'{base}-attempt-{attempt}-close-error.json'),
                        seal_contract({**key, 'exception': str(exc)}))
        if path.exists():
            row = inspect(path, attempt)  # corrupted complete artifacts halt, never rerun
    if row is None:
        row = {**{k: task[k] for k in ('suite', 'runtime_task_index', 'perturbation_category', 'difficulty_level')},
               'success': False, 'termination_reason': 'infrastructure_failure', 'artifact': None,
               'artifact_sha256': None, 'policy_step_count': 0, 'policy_latencies_ms': [],
               'episode_wall_time_seconds': None, 'retry_count': 1}
    immutable(receipt_path, seal_contract({**key, 'result': row,
                                          'attempt_journal_sha256': sha256_file(journal)}))
    return row


def validate_query_artifact(row, arm, admission):
    """Check durable event logs against the independently sealed timing sidecar."""
    import json
    from dataclasses import asdict

    from eventground_wam.eventground_horizon import select_horizon
    if not row['artifact']:
        return False
    timing = read_sealed(Path(row['timing_artifact']))
    path = Path(row['artifact']).with_suffix('.queries.jsonl')
    require(sha256_file(path) == timing['query_artifact_sha256'], 'Query artifact changed')
    queries = [json.loads(line) for line in path.read_text().splitlines()]
    require(queries == timing['queries'], 'Durable query log differs from timing sidecar')
    for index, query in enumerate(queries):
        require(query['query_index'] == index and query['chunk_start'] == index*8,
                'Query order or execution cadence changed')
        p = query['policy']
        require(p['arm'] == arm and p['branch_identities'] == admission, 'Wrong query branch identities')
        require(query['exception'] == p['exception'], 'Query exception binding differs')
        if query['exception'] is not None:
            continue
        require(p['future_calls'] == 1 and p['proposal_actions'] == 8 and p['zero_padded_actions'] == 24,
                'Proposal/future-core contract differs')
        if arm == 'M4-fixed':
            require(p['prediction'] is p['decision'] is p['event'] is None and p['used_horizon'] == 8,
                    'Baseline unexpectedly used event branch')
        else:
            decision = select_horizon(p['prediction'])
            require(asdict(decision) == p['decision'], 'Event mapping record changed')
            require(p['used_horizon'] == (8 if arm == 'M4-EG-fixed8' else decision.selected_horizon),
                    'Wrong deployed horizon')
            for name, values in p['event']['categorical_probabilities'].items():
                require(len(values) == (4 if name == 'event' else 32)
                        and np.isfinite(values).all() and min(values) >= 0
                        and abs(sum(values)-1) < 1e-5, 'Invalid categorical probabilities')
    return True
