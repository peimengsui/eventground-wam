from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import train_coupled_lambda as launch
from coupled_runtime import execution_contract, project_hours
from eventground_wam.b0_training import read_sealed
from eventground_wam.coupled_policy import FULL_BATCH_EXECUTION, LOSS_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    amendment = read_sealed(ROOT/'configs/training/week8_m4_lambda_budget_v1.json')
    contract = execution_contract('M4')
    runtime = {'digest': amendment['runtime_digest'], 'cuda_total_bytes': 40*1024**3}
    mean, p95, hours = project_hours([10.]*128, 18.)
    report = {'digest': amendment['resource_report_digest'], 'passed': False, 'arm': 'M4',
        'rows': [{'global_step': i+1, 'seconds': 10.} for i in range(128)],
        'backward_execution': FULL_BATCH_EXECUTION, 'loss_weight': LOSS_WEIGHTS['M4'],
        'checkpoint_seconds': 18., 'mean_step_seconds': mean, 'p95_step_seconds': p95,
        'projected_training_hours': hours, 'host_peak_rss_bytes': 12*1024**3,
        'gpu_peak_reserved_bytes': 14*1024**3, 'gpu_peak_allocated_bytes': 12*1024**3,
        'teacher_unchanged': True, 'pilot_weights_excluded_from_production': True,
        'edge_probes': [{'finite': True, 'sequence_length': 198}]}
    return contract, amendment, runtime, report


def test_time_amendment_changes_only_limits_and_provenance():
    contract, amendment, runtime, report = inputs()
    before = copy.deepcopy(contract)
    effective = launch.amended_contract(contract, amendment, runtime, report)
    assert contract == before
    assert effective['resources'] == {**contract['resources'],
        'maximum_active_hours': 480, 'maximum_projected_hours': 480}
    assert {k:v for k,v in effective.items() if k not in ('digest','resources',
        'unamended_contract_digest','time_budget_amendment_digest')} == {
        k:v for k,v in contract.items() if k not in ('digest','resources')}
    launch.validate_resources(report, effective, runtime)


@pytest.mark.parametrize('change', ['budget', 'runtime', 'evidence', 'arm', 'failed_flag'])
def test_reject_unapproved_budget_or_wrong_scope(change):
    contract, amendment, runtime, report = inputs()
    if change == 'budget': amendment['replacement_limits']['maximum_active_hours'] = 481
    if change == 'runtime': runtime['digest'] = 'f'*64
    if change == 'evidence': report['digest'] = 'f'*64
    if change == 'arm': contract['arm'] = 'M3'
    if change == 'failed_flag': report['passed'] = True
    with pytest.raises(ValueError):
        launch.amended_contract(contract, amendment, runtime, report)


@pytest.mark.parametrize('change', ['ram', 'gpu', 'teacher', 'edge', 'sequence', 'exposure', 'projection'])
def test_time_exception_cannot_override_other_admission_failures(change):
    contract, amendment, runtime, report = inputs()
    effective = launch.amended_contract(contract, amendment, runtime, report)
    if change == 'ram': report['host_peak_rss_bytes'] = 29*1024**3
    if change == 'gpu': report['gpu_peak_reserved_bytes'] = 37*1024**3
    if change == 'teacher': report['teacher_unchanged'] = False
    if change == 'edge': report['edge_probes'][0]['finite'] = False
    if change == 'sequence': report['edge_probes'][0]['sequence_length'] = 513
    if change == 'exposure': report['rows'].pop()
    if change == 'projection': report['projected_training_hours'] -= 1
    with pytest.raises(ValueError):
        launch.validate_resources(report, effective, runtime)


def test_even_approved_budget_still_has_a_finite_time_limit():
    contract, amendment, runtime, report = inputs()
    effective = launch.amended_contract(contract, amendment, runtime, report)
    for row in report['rows']: row['seconds'] = 20.
    mean,p95,hours = project_hours([20.]*128, report['checkpoint_seconds'])
    report.update(mean_step_seconds=mean,p95_step_seconds=p95,projected_training_hours=hours)
    with pytest.raises(ValueError, match='time limit'):
        launch.validate_resources(report, effective, runtime)
