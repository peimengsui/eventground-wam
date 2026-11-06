"""Pinned hardware/source boundary and shared setup for coupled training."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import check_auxiliary_training as pilot
import torch

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_model import b0_trainable_state_dict
from eventground_wam.b0_training import read_sealed, seed_training, sha256_file
from eventground_wam.coupled_policy import (
    CANDIDATE_MICROBATCH, COUPLING_PARAMETERS, FULL_BATCH_EXECUTION, METHODS, CoupledPolicy,
)
from eventground_wam.coupled_setup import load_coupled_design
from eventground_wam.coupled_training import CoupledTrainingEngine
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_predictor import tensor_state_digest

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def execution_contract(arm):
    design = load_coupled_design(ROOT)
    parent = read_sealed(ROOT/f'configs/training/week7_m{1 if arm == "M3" else 2}_auxiliary_v1.json')
    keys = ('epochs', 'total_updates', 'updates_per_epoch', 'action_presentations', 'future_pairs',
            'mixed_plan_digests', 'batching', 'loss', 'training_seed', 'pilot_rule_digest',
            'base_recipe_digest', 'optimizer', 'checkpoint_selection', 'initialization')
    return seal_contract({'artifact_type': 'eventground_wam.coupled_execution_contract',
        **{key: parent[key] for key in keys}, 'design_digest': design['digest'],
        'arm': arm, 'method': METHODS[arm], 'parent_recipe_digest': parent['digest'],
        'backward_execution': FULL_BATCH_EXECUTION,
        'batching': {**parent['batching'], 'predictor_microbatch': 'whole logical future batch',
                     'candidate_microbatch': CANDIDATE_MICROBATCH,
                     'maximum_future_pairs_per_logical_batch': 160},
        'execution_revision_reason': 'Microbatch-10 joint BF16 equivalence failed on A100; '
            'use the original complete objective with fresh correctness, recovery and resource admission.',
        'optimizer': {**parent['optimizer'], 'groups': ['head_and_state', 'lora', 'future', 'coupling'],
                      'coupling_learning_rate': 1e-4},
        'coupling_parameters': COUPLING_PARAMETERS,
        'total_trainable_parameters': parent['shared_trainable_parameters']
            +parent['future_trainable_parameters']+COUPLING_PARAMETERS,
        'recovery': {'checkpoint_interval_seconds': 300, 'checkpoint_interval_steps': 2000,
                     'checkpoint_keep_last': 3},
        'resources': {'maximum_projected_hours': 168, 'maximum_active_hours': 192,
                      'minimum_free_disk_bytes': 50*1024**3, 'output_limit_bytes': 100*1024**3},
        'status': 'requires_separate_hardware_qualification',
        'pilot_weights_may_be_loaded': False})


def verify_source():
    source = read_sealed(ROOT/'coupled-source-manifest.json')
    require(source['design_digest'] == load_coupled_design(ROOT)['digest'], 'source design differs')
    for name, digest in source['files'].items():
        require(sha256_file(ROOT/name) == digest, f'coupled source changed: {name}')
    return source


def capture_runtime():
    expected = json.loads((ROOT/'launch-evidence/runtime.json').read_text())
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    actual = {'host': platform.node(), 'architecture': platform.machine(),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in expected['packages']},
        'cuda_build': torch.version.cuda, 'gpu_count': torch.cuda.device_count(),
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'allocator_environment': {name: os.environ.get(name) for name in
            ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')},
        'bf16_supported': torch.cuda.is_bf16_supported(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'gpu': subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()}
    for key in ('python', 'packages', 'cuda_build'):
        require(actual[key] == expected[key], f'pinned software differs: {key}')
    require(actual['architecture'] == 'x86_64' and actual['gpu_count'] == 1, 'single x86_64 GPU required')
    require(actual['bf16_supported'] and not actual['matmul_allow_tf32'], 'policy precision differs')
    require(actual['cudnn_allow_tf32'] == expected['baseline_cudnn_allow_tf32'], 'base cuDNN precision differs')
    require(actual['gpu_name'] in ('NVIDIA A100-SXM4-40GB', 'NVIDIA GeForce RTX 5090'),
            'host GPU needs a separately declared qualification profile')
    return seal_contract({'artifact_type': 'eventground_wam.coupled_runtime', **actual})


def verify_idle_gpu():
    if torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5090':
        # Reuse the accepted Feifei guard: its driver reports the two small
        # desktop applications as compute jobs. They remain in the memory trace.
        from train_auxiliary_policy import other_gpu_jobs
        rows = subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True)
        require(not other_gpu_jobs(rows, os.getpid()), 'another substantial GPU job exists')
        return
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    require(all(int(pid.strip()) == os.getpid() for pid in rows if pid.strip()), 'another GPU process exists')


def verify_runtime(path):
    actual = capture_runtime()
    require(actual == read_sealed(path), 'runtime changed since inspection')
    verify_idle_gpu()
    return actual


def initialized_models(args, base, contract):
    values = pilot.models(args, base)
    policy, names, predictor, teacher, teacher_hash, verified, header = values
    state = b0_trainable_state_dict(policy, allowed_exact_names=names)
    require(tensor_state_digest(state) == contract['initialization']['shared_tensor_digest'],
            'shared initialization differs')
    require(tensor_tree_digest(state) == contract['initialization']['shared']['b0_initial_task_state_digest'],
            'B0-matched initialization differs')
    require(sha256_file(Path(contract['initialization']['future_module_path']))
            == contract['initialization']['future_module_sha256'], 'original CPT differs')
    normalizer = load_b0_action_normalizer(ROOT/'docs/model/week4-fresh-v2/action-statistics.json')
    model = CoupledPolicy(policy, predictor, normalizer, arm=args.arm).cuda().train()
    if teacher is not None:
        teacher.eval()
        require(not teacher.visual.training and not any(p.requires_grad for p in teacher.parameters()), 'teacher unfrozen')
    return model, predictor, names, teacher, teacher_hash, verified, header


def make_engine(model, predictor, names, base, identity):
    config = {**base, 'auxiliary_learning_rate': 1e-4}
    engine = CoupledTrainingEngine(model, names, predictor, config, identity_digest=identity['digest'],
                                  steps_per_epoch=58566, device=torch.device('cuda'))
    contract = execution_contract(model.arm)
    require([g['name'] for g in engine.optimizer.param_groups] == contract['optimizer']['groups'],
            'optimizer group inventory differs')
    require(sum(p.numel() for p in engine.parameters) == contract['total_trainable_parameters'],
            'parameter inventory differs')
    seed_training(17)
    return engine


def project_hours(seconds, save_seconds):
    import math
    import statistics
    require(len(seconds) == 128 and all(math.isfinite(x) and x > 0 for x in seconds), 'invalid timing stream')
    require(math.isfinite(save_seconds) and save_seconds >= 0, 'invalid checkpoint timing')
    ordered = sorted(seconds)
    position = .95*(len(ordered)-1)
    low = int(position)
    p95 = ordered[low]+(position-low)*(ordered[low+1]-ordered[low])
    mean = statistics.mean(seconds)
    hours = 1.2*max(mean, p95)*117132/3600+save_seconds*math.ceil(117132/2000)/3600
    return mean, p95, hours
