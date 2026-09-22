#!/usr/bin/env python3
"""Fresh two-epoch Qwen/OFT all-L1 training with exact mixed-cursor recovery."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import torch
import train_b0 as common
from train_b0_continuation import exact_replay

from eventground_wam.b0_continuation import ActiveBudget, mixed_loader, validation_metrics
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_fresh import (
    MANIFEST_TYPE,
    fresh_plan,
    grouped_validation,
    load_fresh_recipe,
)
from eventground_wam.b0_model import (
    QWEN_REVISION,
    b0_trainable_state_dict,
    load_b0_trainable_state_dict,
)
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    CheckpointStore,
    TrainingContractError,
    TrainingEngine,
    atomic_json,
    atomic_torch_save,
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'configs/training/week4_b0_fresh_all_l1_v2.json'
RULE = ROOT / 'configs/evaluation/week4_fresh_v2_acceptance.json'


def check_space(work_root, recipe):
    if shutil.disk_usage(work_root).free < recipe['minimum_free_bytes']:
        raise TrainingContractError('free-space reserve reached')
    if sum(p.stat().st_size for p in work_root.rglob('*') if p.is_file()) > recipe['output_limit_bytes']:
        raise TrainingContractError('experiment output cap reached')


def export_bundle(root, engine, identity, validation, epoch):
    root.mkdir(parents=True, exist_ok=True)
    state = b0_trainable_state_dict(engine.policy, allowed_exact_names=engine.exact_lora_names)
    path = root / 'B0.ckpt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['identity_digest'] != identity['digest']
                or tensor_tree_digest(saved['state_dict']) != tensor_tree_digest(state)):
            raise TrainingContractError('immutable epoch checkpoint changed')
    else:
        atomic_torch_save(path, {'format': POLICY_FORMAT, 'identity_digest': identity['digest'],
                                'qwen_revision': QWEN_REVISION, 'state_dict': state,
                                'objective': engine.recipe['objective']})
    saved = torch.load(path, map_location='cpu', weights_only=True)
    load_b0_trainable_state_dict(engine.policy, saved['state_dict'],
                                 allowed_exact_names=engine.exact_lora_names)
    if tensor_tree_digest(state) != tensor_tree_digest(b0_trainable_state_dict(
            engine.policy, allowed_exact_names=engine.exact_lora_names)):
        raise TrainingContractError('epoch export reload changed weights')
    report = seal_contract({
        'artifact_type': 'eventground_wam.b0_fresh_v2_epoch_report', 'status': 'complete',
        'identity_digest': identity['digest'], 'selected_epoch': epoch,
        'selected_step': engine.global_step, 'global_step': engine.global_step,
        'samples_seen': engine.samples_seen, 'checkpoint_sha256': sha256_file(path),
        'validation': validation, 'reload_validation_passed': True,
        'selection': 'fixed-epoch-2-endpoint; epoch-1-development-only',
    })
    atomic_json(root / 'training-manifest.json', identity)
    atomic_json(root / 'training-report.json', report)
    return report


def run(args, recipe, budget):
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 5090':
        raise TrainingContractError('expected RTX 5090 unavailable')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    train, validation, (_, val_plan), source_identity = common.inspect_data(args, recipe)
    if recipe['state_normalization']['source'] != {k: source_identity['source'][k]
                                                 for k in recipe['state_normalization']['source']}:
        raise TrainingContractError('action/state split identities differ')
    rule = read_sealed(RULE)
    plans_digests = [batches_digest(fresh_plan(train, recipe, epoch)) for epoch in range(2)]
    identity = seal_contract({**{k: v for k, v in source_identity.items() if k != 'digest'},
                              'artifact_type': MANIFEST_TYPE, 'mixed_plan_digests': plans_digests,
                              'acceptance_rule_digest': rule['digest'],
                              'parent_checkpoint': None, 'parent_optimizer': None})
    if (identity['steps_per_epoch'] != recipe['steps_per_epoch']
            or validation.sample_count != 230250):
        raise TrainingContractError('v2 epoch cardinality changed')
    output = args.work_root / ('pilot' if args.phase == 'pilot' else 'training')
    output.mkdir(parents=True, exist_ok=True)
    saved_manifest = output / 'training-manifest.json'
    if saved_manifest.exists() and read_sealed(saved_manifest) != identity:
        raise TrainingContractError('fresh training resume identity changed')
    atomic_json(saved_manifest, identity)
    if args.preflight_only:
        common.write_event(output, {'event': 'preflight_complete', 'identity_digest': identity['digest']})
        return 0
    if args.phase == 'train':
        pilot = read_sealed(args.work_root / 'pilot/pilot-report.json')
        smoke = read_sealed(args.work_root / 'pilot/rollouts-smoke/evaluation-report.json')
        if (pilot['status'] != 'passed' or pilot['recipe_digest'] != recipe['digest']
                or pilot['project_git_revision'] != identity['project_git_revision']
                or pilot['runtime_lock_sha256'] != identity['b0_runtime_lock_sha256']
                or pilot['process_resume_observed'] is not True
                or smoke['all_pairs_valid'] is not True or smoke['completed'] != 2
                or smoke['checkpoint_sha256'] != pilot['checkpoint_sha256']):
            raise TrainingContractError('production requires passed matching resume and rollout pilots')
    seed_training(recipe['seed'])
    policy, names = common.build_policy(recipe, str(args.cache_dir), device)
    engine = TrainingEngine(policy, names, recipe, identity_digest=identity['digest'],
                            steps_per_epoch=recipe['steps_per_epoch'], device=device)
    store = CheckpointStore(output / 'recovery', identity['digest'], keep=3)
    saved = store.load()
    if saved is None:
        atomic_json(output / 'fresh-initialization.json', seal_contract({
            'identity_digest': identity['digest'], 'qwen_revision': QWEN_REVISION,
            'task_state_digest': tensor_tree_digest(engine.snapshot()['state_dict']),
            'optimizer_state_entries': len(engine.optimizer.state), 'global_step': engine.global_step,
            'samples_seen': engine.samples_seen, 'parent_checkpoint_loaded': False,
        }))
    else:
        engine.restore(saved)
        atomic_json(output / 'process-resume.json', seal_contract({
            'identity_digest': identity['digest'], 'global_step': engine.global_step,
            'epoch': engine.epoch, 'next_batch': engine.next_batch, 'samples_seen': engine.samples_seen,
            'state_digest': tensor_tree_digest(engine.snapshot()), 'pid': os.getpid(),
            'recovery_notes': store.recovery_notes,
        }))
        del saved
    if args.phase == 'train' and engine.epoch >= args.until_epoch:
        return 0
    common.write_event(output, {'event': 'started', 'phase': args.phase,
                                'global_step': engine.global_step, 'epoch': engine.epoch,
                                'samples_seen': engine.samples_seen, 'identity_digest': identity['digest']})
    store.save(engine.snapshot())
    profile_path = output / 'profiling.json'
    profile = read_sealed(profile_path) if profile_path.exists() else {}
    measured_time = profile.get('measured_seconds', 0.)
    measured_samples = profile.get('measured_samples', 0)
    measured_steps = profile.get('measured_steps', 0)
    groups = [json.loads(x) for x in (ROOT / 'configs/datasets/libero-plus-trajectory-v2/groups.jsonl').read_text().splitlines()]
    memberships = {e: (g['task_key'], g['group_digest']) for g in groups for e in g['episode_indices']}

    def save_profile():
        atomic_json(profile_path, seal_contract({'measured_seconds': measured_time,
                                                 'measured_samples': measured_samples,
                                                 'measured_steps': measured_steps}))

    while engine.epoch < args.until_epoch:
        epoch = engine.epoch
        plan = fresh_plan(train, recipe, epoch)
        if batches_digest(plan) != identity['mixed_plan_digests'][epoch]:
            raise TrainingContractError('mixed epoch plan changed')
        loader = mixed_loader(train, plan, args.cache_dir, recipe, start=engine.next_batch,
                              expected_epoch=epoch)
        iterator = iter(loader)
        target = 128 if args.phase == 'pilot' else (epoch + 1) * recipe['steps_per_epoch']
        last_save = last_log = previous = time.monotonic()
        start_step = engine.global_step
        torch.cuda.reset_peak_memory_stats()
        while engine.global_step < target:
            if budget.seconds >= recipe['training_active_hours'] * 3600:
                store.save(engine.snapshot())
                raise TrainingContractError('training budget reached before fixed endpoint')
            batch = next(iterator)
            record = engine.step(batch)
            now = time.monotonic()
            if engine.global_step - start_step > 20:
                measured_time += now - previous
                measured_samples += batch['sample_count']
                measured_steps += 1
            previous = now
            if engine.global_step % 100 == 0 or now - last_log >= 60 or engine.global_step == 1:
                event = {'event': 'progress', **record, 'unique_task_count': batch['unique_task_count'],
                         'epoch_fraction': engine.samples_seen / 1706022,
                         'steady_samples_per_second': measured_samples / max(measured_time, 1e-9),
                         'budget_active_seconds': budget.seconds,
                         'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
                common.write_event(output, event)
                atomic_json(output / 'status.json', {'status': 'running', **event})
                save_profile()
                check_space(args.work_root, recipe)
                last_log = now
            if (engine.global_step == 1 or now - last_save >= 300
                    or engine.global_step % 2000 == 0 or common.STOP_REQUESTED):
                store.save(engine.snapshot())
                last_save = time.monotonic()
            if common.STOP_REQUESTED:
                save_profile()
                common.write_event(output, {'event': 'interrupted_checkpoint_saved',
                                            'global_step': engine.global_step})
                return 75
        store.save(engine.snapshot())
        save_profile()
        if args.phase == 'pilot':
            replay = exact_replay(engine, store, next(iterator))
            del iterator, loader
            probe = common.make_loader(validation, val_plan[:32], args, recipe)
            started = time.monotonic()
            metrics = validation_metrics(policy, probe, device)
            validation_seconds = time.monotonic() - started
            projected = 3412044 / (measured_samples / measured_time) * 1.25
            projected += 2 * 230250 * validation_seconds / metrics['sample_count']
            process = read_sealed(output / 'process-resume.json') if (output / 'process-resume.json').exists() else {}
            resumed = 0 < process.get('global_step', 0) < 128
            report = export_bundle(output, engine, identity, metrics, 0)
            passed = resumed and projected < recipe['maximum_projected_training_hours'] * 3600
            pilot = seal_contract({
                'artifact_type': 'eventground_wam.b0_fresh_v2_pilot', 'status': 'passed' if passed else 'failed',
                'identity_digest': identity['digest'], 'recipe_digest': recipe['digest'],
                'project_git_revision': identity['project_git_revision'],
                'runtime_lock_sha256': identity['b0_runtime_lock_sha256'],
                'process_resume_observed': resumed, 'reload_validation': replay,
                'samples_per_second': measured_samples / measured_time,
                'projected_training_and_validation_hours_with_margin': projected / 3600,
                'checkpoint_sha256': report['checkpoint_sha256'], 'validation_probe': metrics,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            })
            atomic_json(output / 'pilot-report.json', pilot)
            common.write_event(output, {'event': 'pilot_complete', **pilot})
            return 0 if passed else 78
        del iterator, loader
        val_path = output / f'validation-epoch-{epoch + 1:02d}.json'
        if val_path.exists():
            val = read_sealed(val_path)
            if val['identity_digest'] != identity['digest'] or val['global_step'] != engine.global_step:
                raise TrainingContractError('validation identity changed')
            metrics = val['metrics']
        else:
            common.write_event(output, {'event': 'validation_started', 'epoch': epoch + 1})
            metrics = grouped_validation(policy, common.make_loader(validation, val_plan, args, recipe),
                                          val_plan, memberships, device)
            if (metrics['sample_count'], metrics['episode_count'], metrics['group_count'], metrics['task_count']) != (230250, 1528, 180, 40):
                raise TrainingContractError('grouped validation did not cover the complete split')
            atomic_json(val_path, seal_contract({'identity_digest': identity['digest'],
                                                'global_step': engine.global_step, 'metrics': metrics}))
        endpoint = args.work_root / 'epochs' / f'epoch-{epoch + 1:02d}'
        report = export_bundle(endpoint, engine, identity, metrics, epoch + 1)
        if engine.samples_seen != (epoch + 1) * 1706022:
            raise TrainingContractError('full epoch exposure mismatch')
        engine.finish_epoch(metrics)
        store.save(engine.snapshot())
        common.write_event(output, {'event': 'epoch_complete', 'epoch': engine.epoch,
                                    'global_step': engine.global_step, 'samples_seen': engine.samples_seen,
                                    'checkpoint_sha256': report['checkpoint_sha256'],
                                    'validation_micro': metrics['micro'], 'validation_macro': metrics['task_macro']})
        atomic_json(output / 'status.json', {'status': 'epoch_complete' if engine.epoch == 1 else 'complete',
                                           'epoch': engine.epoch, 'global_step': engine.global_step,
                                           'samples_seen': engine.samples_seen})
    return 0


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'train'), required=True)
    parser.add_argument('--until-epoch', type=int, choices=(1, 2), default=2)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--data-root', type=Path, default=home / 'data/eventground-wam/libero-plus')
    parser.add_argument('--cache-dir', type=Path, default=home / 'data/eventground-wam/huggingface-cache')
    args = parser.parse_args()
    args.pilot_steps = 128 if args.phase == 'pilot' else 0
    args.frozen_root = args.data_root / 'derived/frozen/week2-dataset-v1'
    args.normalization = ROOT / 'docs/model/week4-fresh-v2/action-statistics.json'
    args.repository_root = args.data_root / 'datasets/libero-plus-lerobot/snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/repo'
    args.work_root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, common.request_stop)
    signal.signal(signal.SIGINT, common.request_stop)
    try:
        with (args.work_root / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recipe = load_fresh_recipe(RECIPE)
            budget = ActiveBudget(args.work_root / 'training-budget.json', recipe['digest'])
            try:
                code = run(args, recipe, budget)
            finally:
                budget.close()
        raise SystemExit(code)
    except (TrainingContractError, ValueError, BlockingIOError) as error:
        print(json.dumps({'integrity_stop': str(error)}), file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
