from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from auxiliary_lambda_runtime import project_hours
from train_auxiliary_lambda import validate_resources

from eventground_wam.b0_training import read_sealed


def evidence():
    mean, p95, hours = project_hours([2.]*128, 1., 117132)
    recipe = {'loss': {'outer_future_weight': 1.5140534185764003}, 'total_updates': 117132,
              'resources': {'maximum_projected_hours': 168}}
    report = {'passed': True, 'loss_weight': recipe['loss']['outer_future_weight'],
              'rows': [{'global_step': i+1, 'seconds': 2.} for i in range(128)],
              'checkpoint_seconds': 1., 'mean_step_seconds': mean, 'p95_step_seconds': p95,
              'projected_training_hours': hours, 'gpu_peak_reserved_bytes': 12*1024**3,
              'host_peak_rss_bytes': 15*1024**3, 'teacher_unchanged': True,
              'pilot_weights_excluded_from_production': True,
              'edge_probes': [{'finite': True, 'sequence_length': 400}]}
    return report, recipe, {'cuda_total_bytes': 24*1024**3}


def test_valid_qualification_can_launch_without_changing_recipe():
    report, recipe, runtime = evidence()
    original = copy.deepcopy(recipe)
    validate_resources(report, recipe, runtime)
    assert recipe == original


def test_passed_flag_cannot_override_slow_measured_steps():
    report, recipe, runtime = evidence()
    report['rows'] = [{'global_step': i+1, 'seconds': 8.} for i in range(128)]
    mean, p95, hours = project_hours([8.]*128, 1., 117132)
    report.update(mean_step_seconds=mean, p95_step_seconds=p95, projected_training_hours=hours)
    with pytest.raises(ValueError, match='time limit'):
        validate_resources(report, recipe, runtime)


@pytest.mark.parametrize(('field', 'value'), [
    ('loss_weight', 1.), ('projected_training_hours', 1.),
    ('pilot_weights_excluded_from_production', False), ('teacher_unchanged', False),
    ('gpu_peak_reserved_bytes', 23*1024**3), ('host_peak_rss_bytes', 29*1024**3),
    ('edge_probes', []),
])
def test_stale_or_failed_resource_evidence_is_rejected(field, value):
    report, recipe, runtime = evidence()
    report[field] = value
    with pytest.raises(ValueError):
        validate_resources(report, recipe, runtime)


def amended_evidence():
    root = Path(__file__).resolve().parents[1]
    pilot = root/'docs/evaluation/week7-m2-lambda-pilot-v1'
    return (read_sealed(pilot/'run/resources.json'),
            read_sealed(root/'configs/training/week7_m2_auxiliary_v1.json'),
            read_sealed(pilot/'runtime.json'),
            read_sealed(root/'configs/training/week7_m2_lambda_budget_v1.json'))


def test_real_a10_failure_requires_explicit_amendment():
    report, recipe, runtime, amendment = amended_evidence()
    original = copy.deepcopy((report, recipe))
    with pytest.raises(ValueError, match='resource qualification failed'):
        validate_resources(report, recipe, runtime)
    validate_resources(report, recipe, runtime, amendment)
    assert (report, recipe) == original
    assert not report['passed']
    assert recipe['resources']['maximum_active_hours'] == 192


def test_amendment_cannot_be_used_for_another_runtime():
    report, recipe, runtime, amendment = amended_evidence()
    runtime['digest'] = 'different-gpu'
    with pytest.raises(ValueError, match='runtime mismatch'):
        validate_resources(report, recipe, runtime, amendment)


def test_amendment_cannot_raise_budget_again():
    report, recipe, runtime, amendment = amended_evidence()
    amendment['replacement_limits']['maximum_active_hours'] = 999
    with pytest.raises(ValueError, match='unapproved budget'):
        validate_resources(report, recipe, runtime, amendment)


@pytest.mark.parametrize(('field', 'value'), [
    ('loss_weight', 1.), ('gpu_peak_reserved_bytes', 24*1024**3),
    ('host_peak_rss_bytes', 29*1024**3), ('teacher_unchanged', False),
    ('edge_probes', []), ('pilot_weights_excluded_from_production', False),
])
def test_time_amendment_does_not_relax_other_requirements(field, value):
    report, recipe, runtime, amendment = amended_evidence()
    report[field] = value
    with pytest.raises(ValueError):
        validate_resources(report, recipe, runtime, amendment)


def test_amended_projection_still_enforces_480_hours():
    report, recipe, runtime, amendment = amended_evidence()
    seconds = [20.]*128
    for row in report['rows']:
        row['seconds'] = 20.
    mean, p95, hours = project_hours(seconds, report['checkpoint_seconds'], recipe['total_updates'])
    report.update(mean_step_seconds=mean, p95_step_seconds=p95, projected_training_hours=hours)
    with pytest.raises(ValueError, match='time limit'):
        validate_resources(report, recipe, runtime, amendment)
