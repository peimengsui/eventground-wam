#!/usr/bin/env python3
"""Requalify frozen M2 on Lambda A10; never recalibrate or launch production."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import check_auxiliary_training as pilot
import torch
import train_auxiliary_policy as production
from auxiliary_lambda_runtime import (
    ROOT,
    capture_runtime,
    project_hours,
    require,
    verify_idle_gpu,
    verify_recipe,
    verify_runtime,
)

from eventground_wam.auxiliary_data import AuxiliaryBatches, AuxiliaryTargetCache
from eventground_wam.auxiliary_training import AuxiliaryTrainingEngine
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_model import b0_trainable_state_dict
from eventground_wam.b0_training import (
    CheckpointStore,
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_predictor import tensor_state_digest


def initialized_models(args, base, recipe):
    values = pilot.models(args, base)
    policy, names, predictor, teacher, _, _, _ = values
    state = b0_trainable_state_dict(policy, allowed_exact_names=names)
    require(tensor_state_digest(state) == recipe['initialization']['shared_tensor_digest'],
            'shared initialization changed')
    require(tensor_tree_digest(state) == recipe['initialization']['shared']['b0_initial_task_state_digest'],
            'initial task state differs from B0')
    require(sum(p.numel() for p in policy.parameters() if p.requires_grad)
            == recipe['shared_trainable_parameters'], 'shared parameter count changed')
    require(sum(p.numel() for p in predictor.parameters()) == recipe['future_trainable_parameters'],
            'future parameter count changed')
    require(teacher is not None and not any(p.requires_grad for p in teacher.parameters()),
            'teacher must remain frozen')
    return values


def correctness(args, rule, base, batches, calibration, edges, recipe, identity):
    parity = pilot.parity(batches, sorted({calibration[0], *edges}))
    policy, names, predictor, teacher, teacher_hash, verified, _ = initialized_models(args, base, recipe)
    small = AuxiliaryBatches(batches.action_dataset, batches.pairs,
                             [batches.plan[calibration[0]][:4]], args.cache)
    batch = small[0]
    truth, _ = AuxiliaryTargetCache(teacher).encode(batch, 'cuda')
    repeat, _ = AuxiliaryTargetCache(teacher).encode(batch, 'cuda')
    require(torch.equal(truth, repeat), 'teacher target repeat mismatch')
    del repeat
    checks = pilot.correctness(args, rule, policy, names, predictor, batch, truth,
                               recipe['loss']['outer_future_weight'])
    require(tensor_state_digest(teacher.visual.state_dict()) == teacher_hash, 'teacher changed')
    report = seal_contract({'artifact_type': 'eventground_wam.lambda_correctness',
        'identity_digest': identity['digest'], 'parity': parity, 'correctness': checks,
        'target_repeat_bitwise_equal': True, 'teacher_unchanged': True,
        'verified_model_files': verified, 'optimizer_updates': 0,
        'loss_weight': recipe['loss']['outer_future_weight'], 'recalibrated': False,
        'passed': True})
    production.immutable_json(args.output/'correctness.json', report)


def resources(args, rule, base, batches, edges, recipe, identity):
    policy, names, predictor, teacher, teacher_hash, _, _ = initialized_models(args, base, recipe)
    config = {**base, 'auxiliary_learning_rate': recipe['optimizer']['auxiliary_learning_rate']}
    engine = AuxiliaryTrainingEngine(policy, names, predictor, 'jepa', config,
        identity_digest=identity['digest'], steps_per_epoch=recipe['updates_per_epoch'],
        device=torch.device('cuda'), loss_weight=recipe['loss']['outer_future_weight'],
        microbatch=recipe['batching']['predictor_microbatch'])
    seed_training(recipe['training_seed'])
    cache = AuxiliaryTargetCache(teacher, maximum_bytes=recipe['batching']['target_cache_maximum_bytes'])
    batches.pairs._cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i in range(rule['resource']['updates']):
        production.check_resources(args.output, recipe)
        torch.cuda.synchronize()
        started = time.monotonic()
        batch = batches[i]
        truth, extra = pilot.get_targets(batch, 'jepa', cache)
        loaded = time.monotonic()
        row = engine.step(batch, truth)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        rows.append({**row, **extra, 'seconds': elapsed, 'data_and_teacher_seconds': loaded-started})
        if (i+1) % 8 == 0:
            pilot.event(args, 'resources', step=i+1, seconds=elapsed)
        del truth, batch
    started = time.monotonic()
    store = CheckpointStore(args.output/'pilot-recovery', identity['digest'], keep=2)
    record = store.save(engine.snapshot())
    checkpoint_seconds = time.monotonic()-started
    edge_rows = []
    for index in edges:
        batch = batches[index]
        truth, _ = pilot.get_targets(batch, 'jepa', cache)
        pilot.clear_grad(policy, predictor)
        output, features = pilot.policy_with_features(policy, *pilot.move_batch(batch, torch.device('cuda')))
        _, derivative = pilot.auxiliary_backward(predictor, 'jepa', features,
            batch['future_inputs'].to('cuda'), truth,
            microbatch=recipe['batching']['predictor_microbatch'],
            loss_weight=recipe['loss']['outer_future_weight'])
        torch.autograd.backward((output.loss, features.values), (None, derivative.to(features.values.dtype)))
        finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in engine.parameters)
        edge_rows.append({'batch': index, 'finite': finite, 'sequence_length': features.values.shape[1]})
        require(finite, 'nonfinite edge gradient')
        del output, features, derivative, truth, batch
    require(tensor_state_digest(teacher.visual.state_dict()) == teacher_hash, 'teacher changed')
    mean, p95, projected = project_hours([r['seconds'] for r in rows], checkpoint_seconds,
                                       recipe['total_updates'])
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    reserved = torch.cuda.max_memory_reserved()
    passed = (projected <= recipe['resources']['maximum_projected_hours']
        and reserved <= .9*torch.cuda.get_device_properties(0).total_memory
        and host <= 28*1024**3 and all(r['sequence_length'] <= 512 for r in edge_rows))
    report = seal_contract({'artifact_type': 'eventground_wam.lambda_resources',
        'identity_digest': identity['digest'], 'rows': rows, 'edge_probes': edge_rows,
        'mean_step_seconds': mean, 'p95_step_seconds': p95,
        'projected_training_hours': projected, 'checkpoint_seconds': checkpoint_seconds,
        'recovery_record': record, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'gpu_peak_reserved_bytes': reserved, 'host_peak_rss_bytes': host,
        'teacher_unchanged': True, 'loss_weight': recipe['loss']['outer_future_weight'],
        'pilot_weights_excluded_from_production': True, 'passed': passed})
    production.immutable_json(args.output/'resources.json', report)
    pilot.event(args, 'resources_complete', passed=passed, projected_hours=projected)
    require(passed, 'Lambda resource threshold failed; full production must not start')


def orchestrate(args, recipe, identity):
    started = time.monotonic()
    for phase in ('correctness', 'resource', 'uninterrupted', 'interrupt', 'resume'):
        command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase,
            '--recipe', str(args.recipe), '--profile', str(args.profile), '--data', str(args.data),
            '--cache', str(args.cache), '--output', str(args.output)]
        with (args.output/f'{phase}.log').open('x') as log, (args.output/f'{phase}-device.csv').open('x') as trace:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                '--format=csv,noheader,nounits', '-lms', '100'], stdout=trace)
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1, 3*3600-(time.monotonic()-started)), check=False)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        expected = -signal.SIGKILL if phase == 'interrupt' else 0
        require(result.returncode == expected, f'{phase}: exit {result.returncode}; inspect {phase}.log')
        readings = list(csv.reader((args.output/f'{phase}-device.csv').read_text().splitlines()))
        require(len(readings) > 10 and all(len(row) == 2 for row in readings), 'invalid device trace')
        require(all(float(used) <= .9*float(total) for total, used in readings), 'device memory ceiling exceeded')
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) <= 24*1024**3,
                'pilot output cap exceeded')
    resumed = read_sealed(args.output/'jepa-resume.json')
    uninterrupted = read_sealed(args.output/'jepa-uninterrupted.json')
    require(resumed['state_digest'] == uninterrupted['state_digest'], 'process resume mismatch')
    require(resumed['process_resume_bitwise_equal'] and resumed['damaged_latest_fallback'], 'recovery failed')
    reports = [read_sealed(args.output/name) for name in ('correctness.json', 'resources.json')]
    require(all(r['passed'] and r['identity_digest'] == identity['digest'] for r in reports), 'pilot failed')
    production.immutable_json(args.output/'qualification.json', seal_contract({
        'artifact_type': 'eventground_wam.lambda_qualification', 'identity_digest': identity['digest'],
        'recipe_digest': recipe['digest'], 'runtime_digest': identity['runtime_digest'],
        'production_source_digest': identity['production_source_digest'],
        'correctness_digest': reports[0]['digest'], 'resources_digest': reports[1]['digest'],
        'recovery_digest': resumed['digest'], 'passed': True, 'production_started': False,
        'scope': 'A10 hardware qualification with frozen M2 weight; no cross-GPU bitwise claim'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['inspect', 'all', 'correctness', 'resource',
                                          'uninterrupted', 'interrupt', 'resume'], required=True)
    parser.add_argument('--recipe', type=Path, default=ROOT/'configs/training/week7_m2_auxiliary_v1.json')
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/home/ubuntu/data/eventground-wam/libero-plus'))
    parser.add_argument('--cache', type=Path, default=Path('/home/ubuntu/data/eventground-wam/huggingface-cache'))
    args = parser.parse_args()
    args.method = 'jepa'
    args.output.mkdir(parents=True, exist_ok=True)
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    recipe = read_sealed(args.recipe)
    verify_recipe(recipe)
    source = production.verify_package(recipe)
    if args.phase == 'inspect':
        profile = capture_runtime()
        verify_idle_gpu()
        production.immutable_json(args.profile, profile)
        print(json.dumps(profile, indent=2), flush=True)
        return
    # The orchestrator must not create a CUDA context: children exclusively own
    # the device and each independently verifies the pinned runtime profile.
    runtime = read_sealed(args.profile) if args.phase == 'all' else verify_runtime(args.profile)
    if args.phase == 'all':
        verify_idle_gpu()
    production.check_resources(args.output, recipe, gpu=args.phase != 'all')
    identity = seal_contract({'artifact_type': 'eventground_wam.lambda_pilot_identity',
        'recipe_digest': recipe['digest'], 'runtime_digest': runtime['digest'],
        'production_source_digest': source['digest'], 'loss_weight': recipe['loss']['outer_future_weight'],
        'data_path': str(args.data.resolve()), 'cache_path': str(args.cache.resolve()),
        'phase_scope': 'correctness, 128 original updates and same-device exact process recovery',
        'pilot_weights_excluded_from_production': True})
    production.immutable_json(args.output/'pilot-identity.json', identity)
    if args.phase == 'all':
        with (args.output/'orchestrator.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            orchestrate(args, recipe, identity)
        return
    rule = read_sealed(pilot.RULE)
    require(rule['digest'] == recipe['pilot_rule_digest'], 'pilot rule changed')
    base, batches, calibration, edges, data = pilot.preflight(args, rule)
    require(data['action_plan_digests'] == recipe['mixed_plan_digests'], 'action plans changed')
    require(data['future_pairs_per_epoch']*recipe['epochs'] == recipe['future_pairs'], 'pair count changed')
    if args.phase == 'correctness':
        correctness(args, rule, base, batches, calibration, edges, recipe, identity)
    elif args.phase == 'resource':
        resources(args, rule, base, batches, edges, recipe, identity)
    else:
        # The original recovery implementation reads this filename; this record
        # references the already frozen weight and performs no new calibration.
        production.immutable_json(args.output/'jepa-calibration.json', seal_contract({
            'artifact_type': 'eventground_wam.lambda_frozen_weight_reference',
            'rule_digest': rule['digest'], 'source_digest': data['source_digest'],
            'recipe_digest': recipe['digest'], 'loss_weight': recipe['loss']['outer_future_weight'],
            'recalibrated': False, 'identity_digest': identity['digest']}))
        pilot.recovery(args, rule, base, batches, data)


if __name__ == '__main__':
    main()
