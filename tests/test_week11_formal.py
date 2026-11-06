"""Admission, crash accounting, runtime identity and failed-attempt timing."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_training import read_sealed
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.week11_formal import (
    FormalResourceGuard,
    all_attempt_latencies,
    projection_admission,
    relocated_runtime,
    schedule,
)
from eventground_wam.week11_resources import QualificationStopped

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen():
    return (read_sealed(ROOT/'configs/evaluation/week11_integration_v1.json'),
            read_sealed(ROOT/'docs/evaluation/week11-integration-v1/analysis.json'))


def test_exact_queue_and_scope(frozen):
    cfg, _ = frozen
    queue = schedule(cfg)
    assert len(queue) == 6 and sum(cfg['panels'][p]['count'] for _, p in queue) == 2001
    assert queue[0] == ('M4-fixed', 'targeted-467')
    assert queue[-1] == ('M4-EG-adaptive', 'clean-200')
    cfg['execution']['arm_order'].reverse()
    with pytest.raises(ValueError):
        schedule(cfg)


def test_projection_uses_full_suite_bound_and_reserve(frozen):
    cfg, analysis = frozen
    free = 240763498496
    result = projection_admission(cfg, analysis, free)
    assert result['admitted']
    assert result['additional_bytes_projected_1p2'] == pytest.approx(174833347477.92468)
    assert result['total_active_seconds_projected_1p2'] == pytest.approx(36625.431590868204)
    with pytest.raises(ValueError, match='disk reserve'):
        projection_admission(cfg, analysis, 200*1024**3)
    with pytest.raises(ValueError, match='aggregate active'):
        projection_admission(cfg, analysis, free, active_seconds=48*3600)
    with pytest.raises(ValueError, match='aggregate output'):
        projection_admission(cfg, analysis, free, output_bytes=100*1024**3)


def test_resume_subtracts_only_sealed_panels(frozen):
    cfg, analysis = frozen
    result = projection_admission(cfg, analysis, 210*1024**3,
        completed=[('M4-fixed', 'targeted-467')], active_seconds=2000, output_bytes=30*1024**3)
    assert result['remaining_panel_count'] == 5
    assert result['additional_bytes_projected_1p2'] == pytest.approx(
        analysis['formal_additional_bytes_projected_1p2']
        - analysis['formal_step_limit_projection']['M4-fixed']['targeted-467']['hdf5_bytes_1p2'])


def test_runtime_only_relocates_source():
    parent = seal_contract({'environment': {'PYTHONPATH': 'old', 'MUJOCO_GL': 'egl'},
                            'versions': {'torch': 'pinned'}, 'gpu': 'pinned'})
    moved = relocated_runtime(parent, Path('/child'), Path('/libero'))
    assert moved['environment'] == {'PYTHONPATH': '/child/src:/child/scripts:/libero', 'MUJOCO_GL': 'egl'}
    assert moved['versions'] == parent['versions'] and moved['gpu'] == parent['gpu']
    assert parent['environment']['PYTHONPATH'] == 'old' and moved['digest'] != parent['digest']


def test_both_time_caps_include_prior_costs(frozen):
    cfg, _ = frozen
    guard = object.__new__(FormalResourceGuard)
    guard.reason = None
    guard.formal_limits = cfg['resources']
    guard.panel_lock = threading.RLock()
    guard.qualification_seconds = 262
    guard.budget = SimpleNamespace(seconds=172600)
    guard.panel_budget = SimpleNamespace(seconds=10)
    guard.panel_key = 'M4-fixed/targeted-467'
    with pytest.raises(QualificationStopped, match='Aggregate'):
        guard.check()
    guard.budget.seconds = 10
    guard.panel_budget.seconds = 43200
    with pytest.raises(QualificationStopped, match='arm/panel'):
        guard.check()
    guard.panel_budget.seconds = 43199
    guard.check()


def test_panel_ledgers_resume_and_reject_changed_identity(tmp_path):
    guard = object.__new__(FormalResourceGuard)
    guard.panel_lock = threading.RLock()
    guard.output, guard.identity = tmp_path, 'source'
    guard.panel_key = guard.panel_budget = None
    guard.check = lambda: None
    guard.set_panel('M4-fixed', 'targeted-467')
    guard.panel_budget.prior = 123.
    guard.set_panel('M4-fixed', 'clean-200')
    guard.set_panel('M4-fixed', 'targeted-467')
    assert guard.panel_budget.seconds >= 123
    guard.panel_budget.close()
    with pytest.raises(ValueError):
        ActiveBudget(tmp_path/'panel-budgets/M4-fixed/targeted-467.json', 'other-source')


def test_failed_and_retry_query_times_are_kept(tmp_path):
    identity = {'cohort_digest': 'cohort', 'admission_digest': 'admission'}
    for attempt, latencies in enumerate([[100., 900.], [40., 60.]]):
        rows = [{**identity, 'query_index': i, 'latency_ms': v,
                 'exception': 'failed' if attempt == 0 else None} for i, v in enumerate(latencies)]
        (tmp_path/f'episode-0000-attempt-{attempt}.queries.jsonl').write_text(
            ''.join(json.dumps(row)+'\n' for row in rows))
    result = all_attempt_latencies(tmp_path, identity)
    assert result['latency']['all_queries']['count'] == 4
    assert result['latency']['all_queries']['p95_ms'] == pytest.approx(780.)
    assert result['latency']['first_queries']['count'] == 2
    with pytest.raises(ValueError, match='identity'):
        all_attempt_latencies(tmp_path, {**identity, 'admission_digest': 'wrong'})
