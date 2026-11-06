#!/usr/bin/env python3
"""Prospectively bounded Week 7 integration pilots; no production or rollouts."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files

from eventground_wam.auxiliary_data import (
    AuxiliaryBatches,
    AuxiliaryPairDataset,
    AuxiliaryTargetCache,
)
from eventground_wam.auxiliary_losses import auxiliary_backward, joint_backward
from eventground_wam.auxiliary_policy import ObservationFeatures, policy_with_features
from eventground_wam.auxiliary_training import AuxiliaryTrainingEngine
from eventground_wam.b0_continuation import MixedBatches
from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_trajectory_v2_split_index,
)
from eventground_wam.b0_fresh import fresh_plan, load_fresh_recipe
from eventground_wam.b0_model import (
    QWEN_REVISION,
    b0_trainable_state_dict,
    load_b0_trainable_state_dict,
    load_pinned_qwen_model,
    load_pinned_qwen_processor,
)
from eventground_wam.b0_oft import build_oft_policy
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    CheckpointStore,
    atomic_json,
    atomic_torch_save,
    configure_deterministic_cuda,
    cpu_tree,
    move_batch,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.future_pairs import HORIZONS
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import jepa_prediction_loss, load_jepa_checkpoint
from eventground_wam.jepa_training import LOSS_RULE
from eventground_wam.pixel_predictor import (
    load_pixel_checkpoint,
    pixel_reconstruction_loss,
    tensor_state_digest,
)
from eventground_wam.pixel_training import tree_digest

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT/'configs/training/week7_auxiliary_pilot_v2.json'
PLAN_DIGESTS = ['57ec71527b26e5cd21ce7a16c85976b7d45c0246795cd5a1619d5162fa58d747',
                'bb44e37e1f5d4fec25b3eb55d66e928975086691a8e88d2a38a5209af34c8a9c']


def save(path, kind, **values):
    result = seal_contract({'artifact_type': kind, **values})
    atomic_json(path, result)
    return result


def event(args, phase, **values):
    record = {'phase': phase, 'method': args.method, 'time_unix': time.time(), **values}
    atomic_json(args.output/'progress.json', record)
    print(json.dumps(record), flush=True)


def preflight(args, rule):
    receipt = read_sealed(ROOT/'source-manifest.json')
    for name, digest in receipt['files'].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f'pilot source changed: {name}')
    recipe = load_fresh_recipe(ROOT/'configs/training/week4_b0_fresh_all_l1_v2.json')
    normalization = ROOT/'docs/model/week4-fresh-v2/action-statistics.json'
    assert sha256_file(normalization) == recipe['normalization_sha256']
    index = load_trajectory_v2_split_index(args.data/'derived/frozen/week2-dataset-v1',
                                         ROOT/'configs/datasets/libero-plus-trajectory-v2')
    repository = args.data/f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo'
    normalizer = load_b0_action_normalizer(normalization)
    dataset = B0StreamingDataset(DemonstrationReader(repository), index, normalizer,
                                config=B0DataConfig(seed=17))
    assert dataset.sample_count == 1706022
    for epoch in range(2):
        candidate = fresh_plan(dataset, recipe, epoch)
        assert len(candidate) == 58566 and sum(map(len, candidate)) == 1706022
        assert batches_digest(candidate) == PLAN_DIGESTS[epoch]
        del candidate
    plan = fresh_plan(dataset, recipe, 0)
    pairs = AuxiliaryPairDataset(args.data/'derived/future-pairs/trajectory-v2-future-pairs-v1/primary',
        repository, split='train', expected_manifest_sha256=read_sealed(
            ROOT/'configs/training/week6_jepa_cpt_v1.json')['pair_manifest_sha256'],
        normalizer=normalizer, state_bounds=recipe['state_normalization'],
        maximum_cache_frames=rule['batching']['decoded_cache_maximum_frames'])
    batches = AuxiliaryBatches(dataset, pairs, plan, args.cache)
    calibration = [next(i for i in range(start, start+128) if len(plan[i]) == 32)
                   for start in rule['calibration']['windows']]
    longest = max(dataset.episodes, key=lambda e: (e.length, -e.episode_index)).episode_index
    instruction = max(dataset.episodes, key=lambda e: (len(e.instruction), -e.episode_index)).episode_index
    probes = sorted({min(range(len(plan)), key=lambda i: len(plan[i])),
                     next(i for i, b in enumerate(plan) if any(t[1] == longest for t in b)),
                     next(i for i, b in enumerate(plan) if any(t[1] == instruction for t in b))})
    coverage = {str(h): sum(max(0, min(e.length-7, e.length-h)) for e in dataset.episodes)
                for h in HORIZONS}
    source = save(args.output/f'{args.method}-{args.phase}-preflight.json',
        'eventground_wam.week7_pilot_preflight', rule_digest=rule['digest'],
        source_digest=receipt['digest'], action_plan_digests=PLAN_DIGESTS,
        action_samples_per_epoch=dataset.sample_count, future_pairs_per_epoch=sum(coverage.values()),
        future_pairs_by_horizon=coverage, calibration_batches=calibration, edge_batches=probes,
        split='train', validation_payload_reads=0, test_payload_reads=0,
        teacher_target_version='qwen35-vision-merged-ln-l2-single-image-v2')
    return recipe, batches, calibration, probes, source


def models(args, recipe):
    verified = verify_base_files(args.cache, read_sealed(ROOT/'docs/model/Qwen3.5-2B-manifest.json'))
    teacher = None
    teacher_hash = None
    if args.method == 'jepa':
        full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True)
        processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
        teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor).cuda()
        del full, processor
        gc.collect()
        accepted = read_sealed(ROOT/'docs/evaluation/week6-jepa-targets-v2/report.json')
        teacher_hash = tensor_state_digest(teacher.visual.state_dict())
        assert teacher_hash == accepted['vision_state_after']
        for name, obj in [('vision', type(teacher.visual)), ('processor', type(teacher.image_processor))]:
            assert sha256_file(Path(inspect.getfile(obj))) == accepted['implementation_files'][name]['sha256']
    seed_training(17)
    policy, names = build_oft_policy(recipe, str(args.cache), torch.device('cuda'))
    # from_pretrained leaves the backbone in eval mode; training mode is required
    # to activate the inherited non-reentrant gradient checkpointing in diagnostics.
    policy.train()
    registry = read_sealed(ROOT/'configs/training/phase2_future_modules_v1.json')
    module = registry['modules']['C_pixel-v1' if args.method == 'pixel' else 'C_jepa-study-v1']
    path = Path(module['remote_path'])
    assert sha256_file(path) == module['sha256']
    predictor, header = (load_pixel_checkpoint(path) if args.method == 'pixel' else load_jepa_checkpoint(path))
    predictor.cuda().train()
    return policy, names, predictor, teacher, teacher_hash, verified, header


@torch.backends.cudnn.flags(allow_tf32=False)
def objective(model, method, condition, truth, groups):
    out = model(condition)
    return (pixel_reconstruction_loss(out.images, truth, ssim_weight=.1) if method == 'pixel' else
            jepa_prediction_loss(out.features, truth, groups, **LOSS_RULE).total)


def get_targets(batch, method, cache):
    if method == 'pixel':
        return batch['pixel_targets'].cuda(), {}
    return cache.encode(batch, 'cuda')


def clear_grad(policy, predictor):
    policy.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)


def gradient_activity(policy, names):
    groups = {'backbone': [], 'state': [], 'head': []}
    for name, value in policy.named_parameters():
        if not value.requires_grad:
            assert value.grad is None
            continue
        key = 'backbone' if name in names else ('state' if 'proprioception_projector' in name else 'head')
        if value.grad is not None:
            assert torch.isfinite(value.grad).all()
            groups[key].append(float(value.grad.float().square().sum()))
    return {k: float(np.sqrt(sum(v))) for k, v in groups.items()}


def parity(batches, indices):
    old = MixedBatches(batches.action_dataset, batches.plan, batches.cache_dir, expected_epoch=0)
    results = []
    for index in indices:
        new = batches[index]
        original = old[index]
        assert new['identity_digest'] == original['identity_digest']
        for key in ('targets', 'proprioception'):
            assert torch.equal(new[key], original[key]), key
        assert set(new['inputs']) == set(original['inputs'])
        assert all(torch.equal(new['inputs'][k], v) for k, v in original['inputs'].items())
        results.append({'batch': index, 'anchors': new['sample_count'],
                        'future_pairs': len(new['future_identities']), 'bitwise_equal': True})
    del old
    gc.collect()
    return results


def compare_gradients(expected, actual):
    squared = norm = maximum = 0.
    for a, b in zip(expected, actual, strict=True):
        delta = a.float()-b.float()
        squared += float(delta.square().sum())
        norm += float(a.float().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {'relative_l2': (squared/max(norm, 1e-30))**.5, 'maximum_absolute': maximum}


@torch.backends.cudnn.flags(allow_tf32=False)
def correctness(args, rule, policy, names, predictor, batch, truth, weight):
    clear_grad(policy, predictor)
    inputs, state, action = move_batch(batch, torch.device('cuda'))
    output, features = policy_with_features(policy, inputs, state, action)
    # Full action forward stays identical with or without the temporary hook and loss targets.
    with torch.no_grad():
        plain = policy(inputs, state)
        changed = policy(inputs, state, normalized_action_targets=-action)
    assert torch.equal(output.raw_action_predictions, plain.raw_action_predictions)
    assert torch.equal(output.raw_action_predictions, changed.raw_action_predictions)
    future = batch['future_inputs'].to('cuda')
    leaf = features.values.detach().float().requires_grad_(True)
    separate = ObservationFeatures(leaf, features.padding)
    truth = truth.detach().requires_grad_(True)
    loss = objective(predictor, args.method, future.condition(separate), truth.detach(), future.anchor_indices)
    loss.backward()
    expected = [p.grad.detach().clone() for p in predictor.parameters()]+[leaf.grad.detach().clone()]
    predictor.zero_grad(set_to_none=True)
    metrics, derivative = auxiliary_backward(predictor, args.method, features, future, truth,
                                              microbatch=rule['batching']['predictor_microbatch'])
    comparison = compare_gradients(expected, [p.grad for p in predictor.parameters()]+[derivative])
    limits = rule['correctness']
    assert abs(float(loss)-metrics['future_loss']) <= limits['full_vs_accumulated_future_loss_atol']
    assert comparison['relative_l2'] <= limits[f'{args.method}_fp32_gradient_relative_l2'], comparison
    assert comparison['maximum_absolute'] <= limits['fp32_gradient_max_absolute'], comparison
    assert truth.grad is None
    features.values.backward(derivative.to(features.values.dtype))
    activity = gradient_activity(policy, names)
    assert activity['backbone'] > 0 and activity['state'] > 0 and activity['head'] == 0
    assert not policy.conditioning_adapter.backbone._forward_hooks
    clear_grad(policy, predictor)
    output, features = policy_with_features(policy, inputs, state, action)
    complete_loss = objective(predictor, args.method, future.condition(
        ObservationFeatures(features.values.float(), features.padding)), truth.detach(), future.anchor_indices)
    (output.loss+weight*complete_loss).backward()
    trainable = [p for p in policy.parameters() if p.requires_grad]+list(predictor.parameters())
    expected_joint = [p.grad.detach().cpu().clone() for p in trainable]
    clear_grad(policy, predictor)
    output, features = policy_with_features(policy, inputs, state, action)
    joint_backward(output, features, predictor, args.method, future, truth,
                   microbatch=rule['batching']['predictor_microbatch'], loss_weight=weight)
    joint_comparison = compare_gradients(expected_joint, [p.grad.detach().cpu() for p in trainable])
    assert joint_comparison['relative_l2'] <= limits['joint_bf16_gradient_relative_l2'], joint_comparison
    # Auxiliary action/pixel/feature mutations cannot be arguments to the deployed policy.
    future.actions.fill_(100)
    with torch.no_grad():
        removed = policy(inputs, state)
    assert torch.equal(plain.raw_action_predictions, removed.raw_action_predictions)
    clear_grad(policy, predictor)
    return {'action_forward_bitwise_equal': True, 'future_target_isolation': True,
            'auxiliary_gradient_activity': activity, 'future_gradient_comparison': comparison,
            'joint_gradient_comparison': joint_comparison,
            'loss_difference': abs(float(loss)-metrics['future_loss']),
            'predictor_removal_bitwise_equal': True, 'target_gradients': False}


def evaluate_training(policy, predictor, batch, truth, method):
    inputs, state, target = move_batch(batch, torch.device('cuda'))
    with torch.no_grad():
        output, features = policy_with_features(policy, inputs, state, target)
        future = batch['future_inputs'].to('cuda')
        loss = objective(predictor, method, future.condition(
            ObservationFeatures(features.values.float(), features.padding)), truth, future.anchor_indices)
    return {'action_loss': float(output.loss), 'future_loss': float(loss)}


def probe(args, rule, recipe, batches, calibration, edges, source):
    parity_report = parity(batches, sorted({calibration[0], *edges}))
    event(args, 'data_parity_passed', checked=parity_report)
    policy, names, predictor, teacher, teacher_hash, verified, header = models(args, recipe)
    inventory = save(args.output/f'{args.method}-parameters.json',
        'eventground_wam.week7_trainable_inventory', method=args.method,
        shared=[{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype),
                 'group': 'lora' if n in names else 'head_and_state'}
                for n, p in policy.named_parameters() if p.requires_grad],
        future=[{'name': n, 'shape': list(p.shape), 'dtype': str(p.dtype)}
                for n, p in predictor.named_parameters()],
        shared_count=sum(p.numel() for p in policy.parameters() if p.requires_grad),
        future_count=sum(p.numel() for p in predictor.parameters()),
        frozen_count=sum(p.numel() for p in policy.parameters() if not p.requires_grad),
        teacher_trainable_count=0 if teacher is None else sum(
            p.numel() for p in teacher.parameters() if p.requires_grad))
    initial_policy = cpu_tree(b0_trainable_state_dict(policy, allowed_exact_names=names))
    initial_predictor = cpu_tree(predictor.state_dict())
    shared_digest = tensor_state_digest(initial_policy)
    cache = AuxiliaryTargetCache(teacher) if teacher else None
    rows = []
    for index in calibration:
        clear_grad(policy, predictor)
        batch = batches[index]
        truth, cache_info = get_targets(batch, args.method, cache)
        inputs, state, targets = move_batch(batch, torch.device('cuda'))
        output, features = policy_with_features(policy, inputs, state, targets)
        action_gradient, = torch.autograd.grad(output.loss, features.values, retain_graph=True)
        metrics, derivative = auxiliary_backward(predictor, args.method, features,
            batch['future_inputs'].to('cuda'), truth, microbatch=rule['batching']['predictor_microbatch'])
        a, f = float(action_gradient.float().norm()), float(derivative.norm())
        assert np.isfinite(a+f) and a > 0 and f > 0
        rows.append({'batch': index, 'action_hidden_gradient_norm': a,
                     'future_hidden_gradient_norm': f, 'ratio': .1*a/f,
                     'action_loss': float(output.loss), **metrics, **cache_info})
        event(args, 'calibration', batch=index, ratio=.1*a/f)
        del output, features, action_gradient, derivative, truth, batch
    weight = float(np.clip(np.median([r['ratio'] for r in rows]),
                           rule['calibration']['minimum'], rule['calibration']['maximum']))
    calibration_report = save(args.output/f'{args.method}-calibration.json',
        'eventground_wam.week7_auxiliary_calibration', rule_digest=rule['digest'],
        source_digest=source['source_digest'], method=args.method, shared_initialization=shared_digest,
        module_header_digest=header['digest'], loss_weight=weight, rows=rows,
        parameter_inventory_digest=inventory['digest'],
        optimizer_updates=0, validation_or_rollout_selection=False)
    # Four original anchors, with every valid horizon, for exactness and memorization.
    small = AuxiliaryBatches(batches.action_dataset, batches.pairs,
                             [batches.plan[calibration[0]][:4]], args.cache)
    batch = small[0]
    truth, _ = get_targets(batch, args.method, cache)
    target_repeat = True
    if teacher:
        again, _ = AuxiliaryTargetCache(teacher).encode(batch, 'cuda')
        target_repeat = torch.equal(truth, again)
        assert target_repeat
        del again
    checks = correctness(args, rule, policy, names, predictor, batch, truth, weight)
    event(args, 'correctness_passed', **checks)
    # Re-fetch after the intentional conditioning mutation in the isolation test.
    batch = small[0]
    clear_grad(policy, predictor)

    def engine(*, overfit=False):
        load_b0_trainable_state_dict(policy, initial_policy, allowed_exact_names=names)
        predictor.load_state_dict(initial_predictor)
        seed_training(17)
        config = {**recipe, 'auxiliary_learning_rate': rule['optimizer']['auxiliary_learning_rate']}
        if overfit:
            config.update(warmup_fraction=0., minimum_learning_rate_ratio=1.)
        identity = seal_contract({'rule': rule['digest'], 'source': source['source_digest'],
                                  'method': args.method, 'weight': weight, 'overfit': overfit})['digest']
        return AuxiliaryTrainingEngine(policy, names, predictor, args.method, config,
            identity_digest=identity, steps_per_epoch=58566, device=torch.device('cuda'),
            loss_weight=weight, microbatch=rule['batching']['predictor_microbatch'])

    current = engine(overfit=True)
    before = evaluate_training(policy, predictor, batch, truth, args.method)
    overfit_rows = []
    for i in range(rule['overfit']['updates']):
        row = current.step({**batch, 'index': i}, truth)
        overfit_rows.append(row)
        if (i+1) % 8 == 0:
            event(args, 'overfit', step=i+1, action_loss=row['action_loss'], future_loss=row['future_loss'])
    after = evaluate_training(policy, predictor, batch, truth, args.method)
    reductions = {k: 1-after[k]/before[k] for k in before}
    overfit = save(args.output/f'{args.method}-overfit.json', 'eventground_wam.week7_overfit',
        rule_digest=rule['digest'], before=before, after=after, reductions=reductions, rows=overfit_rows,
        passed=(reductions['action_loss'] >= rule['overfit']['minimum_action_loss_relative_reduction']
                and reductions['future_loss'] >= rule['overfit']['minimum_future_loss_relative_reduction']))
    assert overfit['passed'], overfit
    del current, truth, batch, small
    clear_grad(policy, predictor)
    gc.collect()
    torch.cuda.empty_cache()
    current = engine()
    # Release CPU initialization copies before measuring steady training resources.
    initial_policy = initial_predictor = None
    batches.pairs._cache.clear()
    if cache:
        cache.values.clear()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i in range(rule['resource']['updates']):
        torch.cuda.synchronize()
        started = time.monotonic()
        batch = batches[i]
        truth, cache_info = get_targets(batch, args.method, cache)
        loaded = time.monotonic()
        row = current.step(batch, truth)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        rows.append({**row, **cache_info, 'seconds': elapsed, 'data_and_teacher_seconds': loaded-started})
        if (i+1) % 8 == 0:
            event(args, 'resources', step=i+1, seconds=elapsed,
                  gpu_peak_bytes=torch.cuda.max_memory_allocated())
        del truth, batch
    # Checkpoint overhead and a standard action-only export; predictor is not in deployment bytes.
    started = time.monotonic()
    store = CheckpointStore(args.output/f'{args.method}-resource-recovery', current.identity_digest, keep=2)
    record = store.save(current.snapshot())
    checkpoint_seconds = time.monotonic()-started
    export = args.output/f'{args.method}-pilot-action-only.pt'
    atomic_torch_save(export, {'format': POLICY_FORMAT, 'qwen_revision': QWEN_REVISION,
        'identity_digest': current.identity_digest, 'objective': recipe['objective'],
        'state_dict': b0_trainable_state_dict(policy, allowed_exact_names=names)})
    saved = torch.load(export, map_location='cpu', weights_only=True)
    assert set(saved) == {'format', 'qwen_revision', 'identity_digest', 'objective', 'state_dict'}
    assert tensor_state_digest(saved['state_dict']) == tensor_state_digest(
        b0_trainable_state_dict(policy, allowed_exact_names=names))
    edge_rows = []
    for i in edges:
        batch = batches[i]
        truth, _ = get_targets(batch, args.method, cache)
        clear_grad(policy, predictor)
        output, features = policy_with_features(policy, *move_batch(batch, torch.device('cuda')))
        metrics, derivative = auxiliary_backward(predictor, args.method, features,
            batch['future_inputs'].to('cuda'), truth, microbatch=rule['batching']['predictor_microbatch'],
            loss_weight=weight)
        torch.autograd.backward((output.loss, features.values), (None, derivative.to(features.values.dtype)))
        edge_rows.append({'batch': i, 'anchors': batch['sample_count'], 'sequence_length': features.values.shape[1],
                          'future_pairs': metrics['future_pairs'], 'finite': all(
                              p.grad is None or torch.isfinite(p.grad).all().item() for p in current.parameters)})
        del output, features, derivative, truth, batch
    teacher_unchanged = teacher is None or tensor_state_digest(teacher.visual.state_dict()) == teacher_hash
    assert teacher_unchanged
    seconds = [r['seconds'] for r in rows]
    projected = 1.2*max(float(np.mean(seconds)), float(np.percentile(seconds, 95)))*117132/3600
    projected += checkpoint_seconds*math.ceil(117132/2000)/3600
    gpu = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    resource_passed = (projected <= rule['resource']['maximum_projected_hours_per_arm']
        and reserved <= torch.cuda.get_device_properties(0).total_memory*rule['resource']['maximum_gpu_fraction']
        and host <= rule['resource']['maximum_host_rss_gib']*1024**3 and all(r['finite'] for r in edge_rows))
    report = save(args.output/f'{args.method}-probe-report.json', 'eventground_wam.week7_probe',
        rule_digest=rule['digest'], source_digest=source['source_digest'],
        shared_initialization=shared_digest, calibrated_weight=weight,
        calibration_digest=calibration_report['digest'], parity=parity_report, correctness=checks,
        overfit_digest=overfit['digest'], overfit_passed=overfit['passed'],
        resource_passed=resource_passed, rows=rows, edge_probes=edge_rows,
        target_repeat_bitwise_equal=target_repeat, teacher_unchanged=teacher_unchanged,
        verified_model_files=verified, gpu_peak_allocated_bytes=gpu, gpu_peak_reserved_bytes=reserved,
        host_peak_rss_bytes=host, mean_step_seconds=float(np.mean(seconds)),
        p95_step_seconds=float(np.percentile(seconds, 95)), projected_training_hours=projected,
        checkpoint_seconds=checkpoint_seconds, recovery_record=record,
        action_only_export_sha256=sha256_file(export), action_only_export_roundtrip_equal=True,
        production_ready=False, reason='process recovery and final matched-arm review pending')
    event(args, 'probe_complete', report_digest=report['digest'], resource_passed=resource_passed,
          projected_hours=projected)
    assert resource_passed, 'resource threshold failed; retain report and revise prospectively if needed'


def recovery(args, rule, recipe, batches, source):
    cal = read_sealed(args.output/f'{args.method}-calibration.json')
    assert cal['rule_digest'] == rule['digest'] and cal['source_digest'] == source['source_digest']
    policy, names, predictor, teacher, _teacher_hash, _, _ = models(args, recipe)
    config = {**recipe, 'auxiliary_learning_rate': rule['optimizer']['auxiliary_learning_rate']}
    identity = seal_contract({'rule': rule['digest'], 'source': source['source_digest'],
                              'method': args.method, 'weight': cal['loss_weight'], 'recovery': True})['digest']
    current = AuxiliaryTrainingEngine(policy, names, predictor, args.method, config,
        identity_digest=identity, steps_per_epoch=58566, device=torch.device('cuda'),
        loss_weight=cal['loss_weight'], microbatch=rule['batching']['predictor_microbatch'])
    cache = AuxiliaryTargetCache(teacher) if teacher else None
    store = CheckpointStore(args.output/f'{args.method}-process-recovery', identity, keep=3)
    if args.phase == 'resume':
        current.restore(store.load())
        assert current.global_step == rule['recovery']['checkpoint_after']
    else:
        seed_training(17)
    for i in range(current.next_batch, rule['recovery']['updates']):
        batch = batches[i]
        truth, _ = get_targets(batch, args.method, cache)
        def kill(a, b, update=i+1):
            if args.phase == 'interrupt' and update == rule['recovery']['sigkill_during_update']:
                atomic_json(args.output/f'{args.method}-kill-marker.json',
                    {'pid': os.getpid(), 'global_step': current.global_step, 'microbatch_stop': b})
                os.kill(os.getpid(), signal.SIGKILL)
        current.step(batch, truth, hook=kill)
        if args.phase == 'interrupt' and current.global_step == rule['recovery']['checkpoint_after']:
            store.save(current.snapshot())
    snapshot = current.snapshot()
    digest = tree_digest(snapshot)
    summary = {'state_digest': digest, 'global_step': current.global_step,
               'samples_seen': current.samples_seen, 'future_pairs_seen': current.future_pairs_seen,
               'sample_chain': current.sample_chain, 'future_chain': current.future_chain,
               'rule_digest': rule['digest'], 'source_digest': source['source_digest']}
    if args.phase == 'resume':
        expected = read_sealed(args.output/f'{args.method}-uninterrupted.json')
        assert all(expected[k] == v for k, v in summary.items()), 'process recovery mismatch'
        damaged = store.save(snapshot)
        (store.root/damaged['checkpoint']).write_bytes(b'simulated torn checkpoint')
        intact = store.load()
        assert intact['global_step'] == rule['recovery']['checkpoint_after'] and store.recovery_notes
        summary['damaged_latest_fallback'] = True
        summary['recovery_notes'] = store.recovery_notes
        summary['process_resume_bitwise_equal'] = True
    save(args.output/f'{args.method}-{args.phase}.json', 'eventground_wam.week7_process_recovery', **summary)
    event(args, f'{args.phase}_complete', state_digest=digest)


def orchestrate(args):
    for method in ('pixel', 'jepa'):
        for phase in ('probe', 'uninterrupted', 'interrupt', 'resume'):
            command = [sys.executable, str(Path(__file__).resolve()), '--method', method, '--phase', phase,
                       '--data', str(args.data), '--cache', str(args.cache), '--output', str(args.output)]
            with (args.output/f'{method}-{phase}.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            expected = -signal.SIGKILL if phase == 'interrupt' else 0
            if result.returncode != expected:
                raise RuntimeError(f'{method}/{phase} returned {result.returncode}; expected {expected}; see log')
    reports = [read_sealed(args.output/f'{m}-probe-report.json') for m in ('pixel', 'jepa')]
    assert reports[0]['shared_initialization'] == reports[1]['shared_initialization']
    resumed = [read_sealed(args.output/f'{m}-resume.json') for m in ('pixel', 'jepa')]
    assert resumed[0]['sample_chain'] == resumed[1]['sample_chain']
    assert resumed[0]['future_chain'] == resumed[1]['future_chain']
    save(args.output/'completion.json', 'eventground_wam.week7_integration_completion',
        rule_digest=read_sealed(RULE)['digest'], probe_digests=[r['digest'] for r in reports],
        recovery_digests=[r['digest'] for r in resumed], matched_shared_initialization=True,
        matched_action_and_future_streams=True, steps_2_3_4_passed=True,
        production_training_started=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['all', 'probe', 'uninterrupted', 'interrupt', 'resume'], default='all')
    p.add_argument('--method', choices=['pixel', 'jepa'], default='pixel')
    p.add_argument('--data', type=Path, default=Path('/home/peimengsui/data/eventground-wam/libero-plus'))
    p.add_argument('--cache', type=Path, default=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'all':
        orchestrate(args)
        return
    torch.set_num_threads(1)
    configure_deterministic_cuda()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5090'
    rule = read_sealed(RULE)
    event(args, 'preflight')
    recipe, batches, calibration, edges, source = preflight(args, rule)
    if args.phase == 'probe':
        probe(args, rule, recipe, batches, calibration, edges, source)
    else:
        recovery(args, rule, recipe, batches, source)


if __name__ == '__main__':
    main()
