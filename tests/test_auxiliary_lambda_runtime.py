from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from auxiliary_lambda_runtime import project_hours, validate_runtime


def runtime():
    return {'python': '3.10.21', 'packages': {'torch': '2.13.0'}, 'cuda_build': '13.0',
            'architecture': 'x86_64', 'gpu_count': 1, 'gpu_name': 'NVIDIA A10',
            'bf16_supported': True, 'matmul_allow_tf32': False}


@pytest.mark.parametrize(('key', 'value'), [
    ('python', '3.12.0'), ('packages', {'torch': '2.12.0'}), ('cuda_build', '12.8'),
    ('architecture', 'aarch64'), ('gpu_count', 2), ('gpu_name', 'NVIDIA A100'),
    ('bf16_supported', False), ('matmul_allow_tf32', True),
])
def test_only_declared_hardware_changes_are_allowed(key, value):
    original = runtime()
    actual = copy.deepcopy(original)
    actual[key] = value
    with pytest.raises(ValueError):
        validate_runtime(actual, original)


def test_a10_does_not_need_the_feifei_host_or_device_identity():
    original = {**runtime(), 'host': 'feifei', 'gpu_name': 'NVIDIA GeForce RTX 5090'}
    validate_runtime({**runtime(), 'host': 'lambda'}, original)


def test_slow_tail_can_fail_feasibility_despite_fast_mean():
    mean, p95, hours = project_hours([2.]*120+[10.]*8, 1., 117132)
    assert mean == 2.5 and p95 == 10.
    assert hours > 168
    assert mean*117132/3600 < 168


@pytest.mark.parametrize('seconds', [[2.]*127, [float('nan')]*128, [0.]*128])
def test_incomplete_or_invalid_pilot_cannot_produce_a_projection(seconds):
    with pytest.raises(ValueError):
        project_hours(seconds, 1., 117132)
