#!/usr/bin/env python3
"""Run a sealed Week 7 policy recipe using the unchanged, GPU-verified trainer."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import time
from pathlib import Path

import check_auxiliary_training as pilot
import torch

from eventground_wam.auxiliary_data import AuxiliaryBatches, AuxiliaryTargetCache
from eventground_wam.auxiliary_training import AuxiliaryTrainingEngine
from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_fresh import fresh_plan
from eventground_wam.b0_model import QWEN_REVISION, b0_trainable_state_dict
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    CheckpointStore,
    TrainingContractError,
    atomic_json,
    atomic_torch_save,
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_predictor import tensor_state_digest

ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def require(condition, message):
    if not condition:
        raise TrainingContractError(message)


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def emit(output, status, **values):
    row = {'status': status, 'time_unix': time.time(), **values}
    atomic_json(output/'status.json', row)
    print(json.dumps(row, allow_nan=False), flush=True)


def immutable_json(path, value):
    if path.exists():
        require(read_sealed(path) == value, f'immutable identity changed: {path}')
    else:
        atomic_json(path, value)


def verify_package(recipe):
    source = read_sealed(ROOT/'production-source-manifest.json')
    for name, digest in source['files'].items():
        require(sha256_file(ROOT/name) == digest, f'production source changed: {name}')
    original = read_sealed(ROOT/'source-manifest.json')
    require(original['digest'] == recipe['verified_source_digest'], 'pilot source mismatch')
    require(source['recipe_digest'] == recipe['digest'], 'packaged recipe mismatch')
    for name, digest in original['files'].items():
        require(source['files'].get(name) == digest, f'verified implementation changed: {name}')
    verification = read_sealed(ROOT/'launch-evidence/verification.json')
    require(verification['digest'] == recipe['verification_digest'], 'pilot audit mismatch')
    initialization = read_sealed(ROOT/'launch-evidence/initialization-audit.json')
    require(initialization['digest'] == recipe['initialization_audit_digest'], 'initial audit mismatch')
    return source


def verify_runtime(recipe):
    expected = json.loads((ROOT/'launch-evidence/runtime.json').read_text())
    actual = {'host': platform.node(), 'python': platform.python_version(),
              'packages': {name: importlib.metadata.version(name) for name in expected['packages']},
              'cuda_build': torch.version.cuda}
    for key, value in actual.items():
        require(value == expected[key], f'runtime changed: {key}')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    require(gpu == expected['gpu'], 'GPU or driver changed')
    require(torch.cuda.get_device_name(0) == recipe['resources']['gpu'], 'GPU changed')
    require(torch.cuda.get_device_properties(0).total_memory == expected['cuda_total_bytes'],
            'GPU capacity changed')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], text=True)
    require(not other_gpu_jobs(processes, os.getpid()),
            'another substantial GPU job is running')
    require(torch.backends.cuda.matmul.allow_tf32 is False, 'policy precision changed')
    actual['gpu'] = gpu
    return actual


def other_gpu_jobs(processes, own_pid):
    """Feifei's driver lists its idle graphics applications as compute processes."""
    blocked = []
    for row in csv.reader(processes.splitlines()):
        pid, name, memory = (value.strip() for value in row)
        if int(pid) == own_pid:
            continue
        desktop = (name == '/usr/bin/gnome-control-center'
                   or name.startswith('/opt/google/chrome/chrome --type=gpu-process '))
        if not (desktop and memory.isdigit() and int(memory) <= 256):
            blocked.append(int(pid))
    return blocked


def check_resources(output, recipe, *, reserve_bytes=0, gpu=True):
    limits = recipe['resources']
    require(shutil.disk_usage(output).free >= limits['minimum_free_disk_bytes']+reserve_bytes,
            'free-space reserve reached')
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    require(size+reserve_bytes <= limits['output_limit_bytes'], 'output cap reached')
    if gpu:
        require(torch.cuda.memory_reserved() <= .9*torch.cuda.get_device_properties(0).total_memory,
                'GPU reservation exceeds verified resource ceiling')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 28*1024**3,
                'host memory exceeds verified resource ceiling')
    return size


def check_exposure(engine, recipe, batches):
    """Audit resumed counters against the exact prefix before consuming another batch."""
    require(engine.global_step == engine.epoch*recipe['updates_per_epoch']+engine.next_batch,
            'update cursor mismatch')
    per_epoch = recipe['action_presentations']//recipe['epochs']
    pairs_per_epoch = recipe['future_pairs']//recipe['epochs']
    samples = pairs = 0
    if engine.epoch < recipe['epochs']:
        for tokens in batches.plan[:engine.next_batch]:
            samples += len(tokens)
            for _, episode, anchor in tokens:
                length = batches.pairs.sources[episode]['episode_length']
                pairs += sum(anchor+h < length for h in (2, 4, 8, 16, 32))
    require(engine.samples_seen == engine.epoch*per_epoch+samples, 'action exposure mismatch')
    require(engine.future_pairs_seen == engine.epoch*pairs_per_epoch+pairs, 'future exposure mismatch')
    require(engine.epoch_sample_count == samples, 'current epoch exposure mismatch')


def export_epoch(output, engine, recipe, identity):
    epoch = engine.epoch+1
    require(engine.global_step == epoch*recipe['updates_per_epoch'], 'incomplete endpoint')
    require(engine.samples_seen == epoch*recipe['action_presentations']//recipe['epochs'],
            'endpoint action exposure mismatch')
    require(engine.future_pairs_seen == epoch*recipe['future_pairs']//recipe['epochs'],
            'endpoint future exposure mismatch')
    root = output/'epochs'/f'epoch-{epoch:02d}'
    root.mkdir(parents=True, exist_ok=True)
    path = root/f"{recipe['arm']}.ckpt"
    state = b0_trainable_state_dict(engine.policy, allowed_exact_names=engine.exact_lora_names)
    bundle = {'format': POLICY_FORMAT, 'qwen_revision': QWEN_REVISION,
              'identity_digest': identity['digest'], 'objective': engine.recipe['objective'],
              'state_dict': state}
    if not path.exists():
        atomic_torch_save(path, bundle)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(set(saved) == set(bundle) and saved['identity_digest'] == identity['digest']
            and saved['format'] == POLICY_FORMAT and saved['qwen_revision'] == QWEN_REVISION
            and saved['objective'] == bundle['objective']
            and tensor_state_digest(saved['state_dict']) == tensor_state_digest(state),
            'action-only export roundtrip mismatch')
    future_path = root/'auxiliary-predictor.pt'
    future_state = {'identity_digest': identity['digest'], 'method': recipe['method'],
                    'state_dict': engine.predictor.state_dict(), 'global_step': engine.global_step}
    if not future_path.exists():
        atomic_torch_save(future_path, future_state)
    future_saved = torch.load(future_path, map_location='cpu', weights_only=True)
    require(tensor_tree_digest(future_saved) == tensor_tree_digest(future_state),
            'future export mismatch')
    report = seal_contract({'artifact_type': 'eventground_wam.week7_auxiliary_epoch',
        'identity_digest': identity['digest'], 'epoch': epoch, 'global_step': engine.global_step,
        'samples_seen': engine.samples_seen, 'future_pairs_seen': engine.future_pairs_seen,
        'sample_chain': engine.sample_chain, 'future_chain': engine.future_chain,
        'checkpoint_sha256': sha256_file(path), 'predictor_sha256': sha256_file(future_path),
        'action_only_export_roundtrip_equal': True,
        'selection': recipe['checkpoint_selection'], 'evaluation_status': 'pending-step-6'})
    immutable_json(root/'training-manifest.json', identity)
    immutable_json(root/'training-report.json', report)
    return report


def train_loop(engine, recipe, identity, output, store, budget, batch_factory, target_fn,
               *, resource_check=check_resources, exporter=export_epoch):
    """Epoch boundary checkpoints are restartable before and after export/finish."""
    last_save = last_log = time.monotonic()
    measured_steps = 0
    measured_seconds = 0.

    def save():
        nonlocal last_save
        resource_check(output, recipe, reserve_bytes=2*1024**3)
        record = store.save(engine.snapshot())
        last_save = time.monotonic()
        return record

    while engine.epoch < recipe['epochs']:
        batches = batch_factory(engine.epoch)
        check_exposure(engine, recipe, batches)
        while engine.next_batch < recipe['updates_per_epoch']:
            if STOP_REQUESTED or budget.seconds >= recipe['resources']['maximum_active_hours']*3600:
                save()
                status = 'stopped' if STOP_REQUESTED else 'budget_exhausted'
                emit(output, status, global_step=engine.global_step,
                     active_seconds=budget.seconds, identity_digest=identity['digest'])
                return 75 if STOP_REQUESTED else 78
            started = time.monotonic()
            batch = batches[engine.next_batch]
            targets, extra = target_fn(batch)
            row = engine.step(batch, targets)
            del targets, batch
            measured_seconds += time.monotonic()-started
            measured_steps += 1
            now = time.monotonic()
            if (engine.global_step == 1 or engine.global_step % 100 == 0 or now-last_log >= 60):
                resource_check(output, recipe)
                mean = measured_seconds/measured_steps
                emit(output, 'running', **row, **extra, identity_digest=identity['digest'],
                     active_seconds=budget.seconds, process_mean_step_seconds=mean,
                     remaining_mean_training_hours=(recipe['total_updates']-engine.global_step)*mean/3600)
                last_log = now
            if (engine.global_step == 1 or now-last_save >= recipe['recovery']['checkpoint_interval_seconds']
                    or engine.global_step % recipe['recovery']['checkpoint_interval_steps'] == 0):
                save()
        # A crash at any point below resumes without repeating optimizer updates.
        save()
        report = exporter(output, engine, recipe, identity)
        engine.finish_epoch({'status': 'pending-step-6', 'checkpoint_selection': 'fixed-epoch-2-endpoint'})
        save()
        emit(output, 'epoch_complete', epoch=engine.epoch, global_step=engine.global_step,
             samples_seen=engine.samples_seen, future_pairs_seen=engine.future_pairs_seen,
             checkpoint_sha256=report['checkpoint_sha256'])
    check_exposure(engine, recipe, None)
    endpoint = read_sealed(output/'epochs'/f"epoch-{recipe['epochs']:02d}"/'training-report.json')
    completion = seal_contract({'artifact_type': 'eventground_wam.week7_auxiliary_training_complete',
        'status': 'training_complete_evaluation_pending', 'identity_digest': identity['digest'],
        'recipe_digest': recipe['digest'], 'global_step': engine.global_step,
        'samples_seen': engine.samples_seen, 'future_pairs_seen': engine.future_pairs_seen,
        'sample_chain': engine.sample_chain, 'future_chain': engine.future_chain,
        'endpoint_report_digest': endpoint['digest'], 'checkpoint_sha256': endpoint['checkpoint_sha256']})
    immutable_json(output/'training-report.json', completion)
    emit(output, 'training_complete_evaluation_pending', **{k: completion[k] for k in
         ('global_step', 'samples_seen', 'future_pairs_seen', 'checkpoint_sha256')})
    return 0


def run(args, recipe, source, budget):
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    runtime = verify_runtime(recipe)
    torch.cuda.set_device(0)
    check_resources(args.output, recipe)
    rule = read_sealed(ROOT/'configs/training/week7_auxiliary_pilot_v2.json')
    require(rule['digest'] == recipe['pilot_rule_digest'], 'pilot rule mismatch')
    base, initial_batches, _, _, data = pilot.preflight(args, rule)
    require(base['digest'] == recipe['base_recipe_digest'], 'base recipe mismatch')
    require(data['action_plan_digests'] == recipe['mixed_plan_digests'], 'action plans changed')
    require(data['future_pairs_per_epoch']*recipe['epochs'] == recipe['future_pairs'],
            'future pair count changed')
    config = {**base, 'auxiliary_learning_rate': recipe['optimizer']['auxiliary_learning_rate']}
    for name in ('adam_betas', 'adam_epsilon', 'decoder_projector_learning_rate',
                 'lora_learning_rate', 'maximum_gradient_norm', 'minimum_learning_rate_ratio',
                 'warmup_fraction', 'weight_decay'):
        require(config[name] == recipe['optimizer'][name], f'optimizer changed: {name}')
    identity = seal_contract({'artifact_type': 'eventground_wam.week7_auxiliary_production',
        'recipe_digest': recipe['digest'], 'arm': recipe['arm'], 'method': recipe['method'],
        'production_source_digest': source['digest'], 'verified_source_digest': data['source_digest'],
        'mixed_plan_digests': data['action_plan_digests'], 'runtime': runtime,
        'initialization': recipe['initialization'], 'fixed_endpoint_updates': recipe['total_updates'],
        'validation_and_rollouts': 'pending-step-6; no selection or early stopping'})
    immutable_json(args.output/'training-manifest.json', identity)
    emit(args.output, 'loading_original_models', identity_digest=identity['digest'])
    policy, names, predictor, teacher, _, verified, header = pilot.models(args, config)
    state = b0_trainable_state_dict(policy, allowed_exact_names=names)
    require(tensor_state_digest(state) == recipe['initialization']['shared_tensor_digest'],
            'fresh shared initialization changed')
    require(tensor_tree_digest(state) == recipe['initialization']['shared']['b0_initial_task_state_digest'],
            'initialization no longer matches B0')
    require(sha256_file(Path(recipe['initialization']['future_module_path']))
            == recipe['initialization']['future_module_sha256'], 'CPT module changed')
    require(sum(p.numel() for p in policy.parameters() if p.requires_grad)
            == recipe['shared_trainable_parameters'], 'shared parameter count changed')
    require(sum(p.numel() for p in predictor.parameters()) == recipe['future_trainable_parameters'],
            'future parameter count changed')
    del state
    engine = AuxiliaryTrainingEngine(policy, names, predictor, recipe['method'], config,
        identity_digest=identity['digest'], steps_per_epoch=recipe['updates_per_epoch'],
        device=torch.device('cuda'), loss_weight=recipe['loss']['outer_future_weight'],
        microbatch=recipe['batching']['predictor_microbatch'])
    # Match the accepted resource/recovery pilot's RNG reset after module loading.
    seed_training(recipe['training_seed'])
    initial = seal_contract({'artifact_type': 'eventground_wam.week7_production_initialization',
        'identity_digest': identity['digest'], 'shared_tensor_digest': recipe['initialization']['shared_tensor_digest'],
        'module_header_digest': header['digest'], 'base_files': verified,
        'global_step': engine.global_step, 'optimizer_entries': len(engine.optimizer.state),
        'pilot_or_trained_b0_weights_loaded': False})
    immutable_json(args.output/'fresh-initialization.json', initial)
    store = CheckpointStore(args.output/'recovery', identity['digest'],
                            keep=recipe['recovery']['checkpoint_keep_last'])
    recovered = store.load()
    if recovered is None:
        require(not (args.output/'recovery-created.json').exists(), 'all recovery records missing')
        check_resources(args.output, recipe, reserve_bytes=2*1024**3)
        store.save(engine.snapshot())
        immutable_json(args.output/'recovery-created.json', seal_contract({'identity_digest': identity['digest']}))
    else:
        engine.restore(recovered)
        atomic_json(args.output/'last-resume.json', seal_contract({'identity_digest': identity['digest'],
            'global_step': engine.global_step, 'epoch': engine.epoch, 'next_batch': engine.next_batch,
            'pid': os.getpid(), 'time_unix': time.time(), 'recovery_notes': store.recovery_notes}))
        del recovered
    cache = AuxiliaryTargetCache(teacher, maximum_bytes=recipe['batching']['target_cache_maximum_bytes']) if teacher else None
    def batches(epoch):
        if epoch == 0:
            return initial_batches
        plan = fresh_plan(initial_batches.action_dataset, config, epoch)
        require(batches_digest(plan) == recipe['mixed_plan_digests'][epoch], 'epoch plan changed')
        return AuxiliaryBatches(initial_batches.action_dataset, initial_batches.pairs,
            plan, args.cache, expected_epoch=epoch, builder=initial_batches.builder)
    emit(args.output, 'initialized', global_step=engine.global_step,
         identity_digest=identity['digest'], total_updates=recipe['total_updates'])
    return train_loop(engine, recipe, identity, args.output, store, budget, batches,
                      lambda batch: pilot.get_targets(batch, args.method, cache))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path.home()/'data/eventground-wam/libero-plus')
    parser.add_argument('--cache', type=Path, default=Path.home()/'data/eventground-wam/huggingface-cache')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        recipe = read_sealed(args.recipe)
        require(recipe['status'] == 'numerical_recipe_and_feasibility_verified', 'recipe not verified')
        args.method, args.phase = recipe['method'], 'production'
        source = verify_package(recipe)
        with (args.output/'owner.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = ActiveBudget(args.output/'active-budget.json',
                                  seal_contract({'recipe': recipe['digest'], 'source': source['digest']})['digest'])
            try:
                code = run(args, recipe, source, budget)
            finally:
                budget.close()
        raise SystemExit(code)
    except (ValueError, OSError, AssertionError, RuntimeError) as error:
        emit(args.output, 'integrity_or_runtime_stop', error=str(error), exception=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
