"""Explicit Lambda host boundary around the unchanged Week 7 implementation."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import torch

from eventground_wam.b0_training import read_sealed
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
M2_DIGEST = '2c2fd3c3fe67a06f236727de2016d38b306f3121175b679f899034cc34a9cf27'


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_runtime(actual, original):
    """Permit the declared host/GPU change, preserving software and precision."""
    for key in ('python', 'packages', 'cuda_build'):
        require(actual[key] == original[key], f'Lambda software mismatch: {key}')
    require(actual['architecture'] == 'x86_64', 'x86_64 runtime required')
    require(actual['gpu_count'] == 1, 'exactly one GPU required')
    require(actual['gpu_name'] == 'NVIDIA A10', 'this profile is for the selected A10')
    require(actual['bf16_supported'], 'BF16 support required')
    require(actual['matmul_allow_tf32'] is False, 'policy precision changed')


def capture_runtime():
    import json
    original = json.loads((ROOT/'launch-evidence/runtime.json').read_text())
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    actual = {
        'host': platform.node(), 'architecture': platform.machine(),
        'python': platform.python_version(),
        'packages': {key: importlib.metadata.version(key) for key in original['packages']},
        'cuda_build': torch.version.cuda,
        'gpu_count': torch.cuda.device_count(), 'gpu_name': torch.cuda.get_device_name(0),
        'cuda_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'bf16_supported': torch.cuda.is_bf16_supported(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'gpu': subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],
            text=True).strip(),
    }
    validate_runtime(actual, original)
    return seal_contract({'artifact_type': 'eventground_wam.lambda_runtime', **actual})


def verify_idle_gpu():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    require(all(int(pid.strip()) == os.getpid() for pid in rows if pid.strip()),
            'another GPU process is running')


def verify_runtime(profile_path):
    actual = capture_runtime()
    require(actual == read_sealed(profile_path), 'Lambda runtime changed since inspection')
    verify_idle_gpu()
    return actual


def verify_recipe(recipe):
    require(recipe['digest'] == M2_DIGEST and recipe['arm'] == 'M2', 'frozen M2 recipe required')


def project_hours(seconds, checkpoint_seconds, total_updates):
    import math
    import statistics
    require(len(seconds) == 128, 'resource pilot must contain 128 updates')
    require(all(math.isfinite(x) and x > 0 for x in seconds), 'invalid measured duration')
    require(math.isfinite(checkpoint_seconds) and checkpoint_seconds >= 0, 'invalid save duration')
    ordered = sorted(seconds)
    position = .95*(len(ordered)-1)
    lower = int(position)
    p95 = ordered[lower]+(position-lower)*(ordered[lower+1]-ordered[lower])
    mean = statistics.mean(seconds)
    hours = 1.2*max(mean, p95)*total_updates/3600
    hours += checkpoint_seconds*math.ceil(total_updates/2000)/3600
    return mean, p95, hours
