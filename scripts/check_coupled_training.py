#!/usr/bin/env python3
"""Train-only coupled correctness, exact recovery and actual-device resource admission."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import check_auxiliary_training as pilot
import torch
import train_auxiliary_policy as production
from coupled_runtime import (
    ROOT,
    capture_runtime,
    execution_contract,
    initialized_models,
    make_engine,
    project_hours,
    require,
    verify_idle_gpu,
    verify_runtime,
    verify_source,
)

from eventground_wam.auxiliary_data import AuxiliaryTargetCache
from eventground_wam.auxiliary_losses import auxiliary_backward
from eventground_wam.auxiliary_policy import ObservationFeatures
from eventground_wam.b0_training import (
    CheckpointStore,
    atomic_json,
    configure_deterministic_cuda,
    move_batch,
    read_sealed,
)
from eventground_wam.coupled_checkpoint import export_coupled_policy, load_coupled_policy
from eventground_wam.coupled_policy import (
    LOSS_WEIGHTS,
    METHODS,
    FULL_BATCH_EXECUTION,
    coupled_full_batch_backward,
    coupled_parameter_groups,
)
from eventground_wam.coupled_setup import build_fresh_coupled_policy
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_predictor import PixelCondition, tensor_state_digest
from eventground_wam.pixel_training import tree_digest


def clear(model, predictor):
    model.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)


def forward(model, batch):
    inputs, state, truth = move_batch(batch, torch.device('cuda'))
    return model.forward_with_features(inputs, state, batch['future_inputs'].current_images.cuda(),
                                       normalized_action_targets=truth)


def target_cache(teacher, contract):
    return AuxiliaryTargetCache(teacher, maximum_bytes=contract['batching']['target_cache_maximum_bytes']) if teacher else None


def check_teacher(teacher, original):
    if teacher is not None:
        require(not any(p.requires_grad or p.grad is not None for p in teacher.parameters()), 'teacher gradient exists')
        require(tensor_state_digest(teacher.visual.state_dict()) == original, 'teacher changed')


def correctness(args, rule, base, batches, calibration, edges, contract, identity):
    parity = pilot.parity(batches, sorted({calibration[0], *edges}))
    model, predictor, names, teacher, teacher_hash, verified, _ = initialized_models(args, base, contract)
    batch = batches[calibration[0]]
    truth, _ = pilot.get_targets(batch, args.method, target_cache(teacher, contract))
    if teacher is not None:
        again, _ = pilot.get_targets(batch, args.method, target_cache(teacher, contract))
        require(torch.equal(truth, again), 'teacher targets not repeatable')
        del again
    future = batch['future_inputs'].to('cuda')
    inputs, state, actions = move_batch(batch, torch.device('cuda'))
    with torch.no_grad():
        baseline = model.policy(inputs, state)
        result = forward(model, batch)
        require(torch.equal(result.action_output.raw_action_predictions, baseline.raw_action_predictions),
                'initial coupled policy differs from uncoupled policy')
        condition = PixelCondition(result.features.values, result.features.padding, future.current_images,
                                  future.normalized_state, result.candidate_actions,
                                  torch.full((len(state),), 8, device='cuda', dtype=torch.int64))
        with torch.backends.cudnn.flags(allow_tf32=False):
            original = torch.cat([predictor(PixelCondition(**{k: v[a:a+10]
                for k, v in vars(condition).items()})).future_tokens for a in range(0, len(state), 10)])
        require(torch.equal(original, result.future_tokens), 'token-only core differs')
        require(not result.candidate_actions.requires_grad, 'proposal not detached')
        features = result.features
    del result, original, baseline, condition
    # Compare the entire original logical batch, including cross-anchor JEPA variance.
    leaf = features.values.detach().float().requires_grad_(True)
    truth = truth.detach().requires_grad_(True)
    complete_loss = pilot.objective(predictor, args.method,
        future.condition(ObservationFeatures(leaf, features.padding)), truth.detach(), future.anchor_indices)
    with torch.backends.cudnn.flags(allow_tf32=False):
        complete_loss.backward()
    expected = [p.grad.detach().cpu().clone() for p in predictor.parameters()]+[leaf.grad.detach().cpu().clone()]
    loss_value = float(complete_loss.detach())
    clear(model, predictor)
    metrics, derivative = auxiliary_backward(predictor, args.method, features, future, truth, microbatch=10)
    comparison = pilot.compare_gradients(expected, [p.grad.cpu() for p in predictor.parameters()]+[derivative.cpu()])
    limits = rule['correctness']
    require(abs(loss_value-metrics['future_loss']) <= limits['full_vs_accumulated_future_loss_atol'], 'whole-batch loss differs')
    require(comparison['relative_l2'] <= limits[f'{args.method}_fp32_gradient_relative_l2']
            and comparison['maximum_absolute'] <= limits['fp32_gradient_max_absolute'], f'future gradients differ: {comparison}')
    require(truth.grad is None, 'target gradient exists')
    del expected, derivative, complete_loss, leaf, features
    clear(model, predictor)
    # Activate the coupling only for diagnostics; all qualification models are discarded.
    with torch.no_grad():
        model.coupling.output.weight.normal_(std=.01)
    result = forward(model, batch)
    result.proposal_raw.retain_grad()
    with torch.backends.cudnn.flags(allow_tf32=False):
        result.action_output.loss.backward()
    require(result.proposal_raw.grad is None, 'gradient enters detached proposal')
    activity = pilot.gradient_activity(model.policy, names)
    require(all(activity[k] > 0 for k in ('head', 'state', 'backbone')), 'action gradient missing')
    require(model.coupling.key.weight.grad.abs().sum() > 0
            and model.coupling.value.weight.grad.abs().sum() > 0
            and predictor.action_projection[0].weight.grad.abs().sum() > 0, 'coupled future gradient missing')
    require(all(p.grad is None for p in model.policy.parameters() if not p.requires_grad), 'frozen base gradient exists')
    del result
    clear(model, predictor)
    parameters = [p for g in coupled_parameter_groups(model, predictor, names) for p in g['params']]
    result = forward(model, batch)
    full = pilot.objective(predictor, args.method,
        future.condition(ObservationFeatures(result.features.values.float(), result.features.padding)),
        truth.detach(), future.anchor_indices)
    # Independent full-objective reference. Stop before the base, so its checkpoint
    # recomputation retains the same precision context as the original forward.
    above = (list(model.policy.action_decoder.parameters()) + list(model.coupling.parameters())
             + list(predictor.parameters()))
    with torch.backends.cudnn.flags(allow_tf32=False):
        gradients = torch.autograd.grad(result.action_output.loss+LOSS_WEIGHTS[args.arm]*full,
                                        [result.features.values, *above])
        for parameter, gradient in zip(above, gradients[1:], strict=True):
            parameter.grad = gradient.detach().clone()
    result.features.values.backward(gradients[0])
    expected = [p.grad.detach().cpu().clone() for p in parameters]
    del above, gradients
    del result, full
    clear(model, predictor)
    result = forward(model, batch)
    coupled_full_batch_backward(model, result, predictor, future, truth)
    joint_comparison = pilot.compare_gradients(expected, [p.grad.detach().cpu() for p in parameters])
    require(joint_comparison['relative_l2'] <= limits['joint_bf16_gradient_relative_l2'],
            f'joint gradients differ: {joint_comparison}')
    require(truth.grad is None, 'joint target gradient exists')
    del result, expected
    clear(model, predictor)
    model.eval()
    with torch.no_grad():
        reference = forward(model, batch)
        raw = reference.action_output.raw_action_predictions.detach().cpu()
        tokens = reference.future_tokens.detach().cpu()
        changed = model(inputs, state, future.current_images, normalized_action_targets=-actions)
        require(torch.equal(changed.raw_action_predictions.cpu(), raw), 'action target leakage')
        require(not torch.equal(model.coupling(reference.action_output.conditioning_memory.tokens, reference.future_tokens),
                                model.coupling(reference.action_output.conditioning_memory.tokens, reference.future_tokens.flip(2))),
                'coupling ignores predicted futures')
        alternate = replace(future, actions=-future.actions, horizons=torch.full_like(future.horizons, 2))
        predictor(alternate.condition(reference.features, 0, min(10, len(future.horizons))))
        # All training-only values may change; none are parameters of the action path.
        batch['future_mask'] = ~batch['future_mask']
        truth.neg_()
        require(torch.equal(forward(model, batch).action_output.raw_action_predictions.cpu(), raw), 'auxiliary target leakage')
    del reference, changed
    export_path = args.output/f'{args.arm}-diagnostic-export.pt'
    export_coupled_policy(export_path, model, names, metadata={'pilot_identity': identity['digest'], 'diagnostic_only': True})
    del model, predictor, parameters
    gc.collect()
    torch.cuda.empty_cache()
    restored, _restored_predictor, restored_names, _ = build_fresh_coupled_policy(ROOT, args.cache, torch.device('cuda'), arm=args.arm)
    load_coupled_policy(export_path, restored, restored_names)
    restored.eval()
    with torch.no_grad():
        after = forward(restored, batch)
        require(torch.equal(after.action_output.raw_action_predictions.cpu(), raw)
                and torch.equal(after.future_tokens.cpu(), tokens), 'real-Qwen export/reload differs')
    check_teacher(teacher, teacher_hash)
    report = seal_contract({'artifact_type': 'eventground_wam.coupled_correctness',
        'identity_digest': identity['digest'], 'arm': args.arm, 'passed': True,
        'backward_execution': FULL_BATCH_EXECUTION,
        'original_logical_batch_size': batch['sample_count'], 'future_pairs': len(future.horizons),
        'parity': parity, 'future_gradient_comparison': comparison, 'joint_gradient_comparison': joint_comparison,
        'future_loss_difference': abs(loss_value-metrics['future_loss']),
        'zero_coupling_action_bitwise_equal': True, 'token_only_bitwise_equal': True,
        'target_isolation': True, 'detached_proposal': True, 'action_gradient_activity': activity,
        'teacher_unchanged': True, 'target_repeat_bitwise_equal': True,
        'real_qwen_export_reload_bitwise_equal': True, 'verified_model_files': verified,
        'optimizer_updates': 0, 'pilot_weights_excluded_from_production': True})
    production.immutable_json(args.output/f'{args.arm}-correctness.json', report)
    pilot.event(args, 'correctness_complete', report_digest=report['digest'])


def resources(args, rule, base, batches, edges, contract, identity):
    model, predictor, names, teacher, teacher_hash, _, _ = initialized_models(args, base, contract)
    engine = make_engine(model, predictor, names, base, identity)
    cache = target_cache(teacher, contract)
    batches.pairs._cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i in range(128):
        production.check_resources(args.output, contract)
        torch.cuda.synchronize()
        started = time.monotonic()
        batch = batches[i]
        truth, extra = pilot.get_targets(batch, args.method, cache)
        loaded = time.monotonic()
        row = engine.step(batch, truth)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        rows.append({**row, **extra, 'seconds': elapsed, 'data_and_teacher_seconds': loaded-started})
        if (i+1) % 8 == 0:
            pilot.event(args, 'resources', step=i+1, seconds=elapsed,
                        data_and_teacher_seconds=loaded-started)
        del truth, batch
    started = time.monotonic()
    store = CheckpointStore(args.output/f'{args.arm}-pilot-recovery', identity['digest'], keep=2)
    record = store.save(engine.snapshot())
    save_seconds = time.monotonic()-started
    edge_rows = []
    for index in edges:
        batch = batches[index]
        truth, _ = pilot.get_targets(batch, args.method, cache)
        clear(model, predictor)
        result = forward(model, batch)
        coupled_full_batch_backward(model, result, predictor, batch['future_inputs'].to('cuda'), truth)
        finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in engine.parameters)
        edge_rows.append({'batch': index, 'anchors': batch['sample_count'], 'finite': finite,
                          'sequence_length': result.features.values.shape[1]})
        require(finite, 'edge gradient missing/nonfinite')
        del result, truth, batch
    check_teacher(teacher, teacher_hash)
    mean, p95, projected = project_hours([r['seconds'] for r in rows], save_seconds)
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    reserved = torch.cuda.max_memory_reserved()
    passed = (projected <= 168 and reserved <= .9*torch.cuda.get_device_properties(0).total_memory
              and host <= 28*1024**3 and all(r['sequence_length'] <= 512 for r in edge_rows))
    report = seal_contract({'artifact_type': 'eventground_wam.coupled_resources',
        'identity_digest': identity['digest'], 'arm': args.arm, 'passed': passed,
        'backward_execution': FULL_BATCH_EXECUTION,
        'rows': rows, 'edge_probes': edge_rows, 'mean_step_seconds': mean, 'p95_step_seconds': p95,
        'projected_training_hours': projected, 'checkpoint_seconds': save_seconds, 'recovery_record': record,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gpu_peak_reserved_bytes': reserved,
        'host_peak_rss_bytes': host, 'teacher_unchanged': True,
        'pilot_weights_excluded_from_production': True, 'loss_weight': LOSS_WEIGHTS[args.arm]})
    production.immutable_json(args.output/f'{args.arm}-resources.json', report)
    pilot.event(args, 'resources_complete', projected_hours=projected, passed=passed)
    require(passed, 'frozen resource limit failed; production must not start')


def recovery(args, base, batches, contract, identity):
    model, predictor, names, teacher, teacher_hash, _, _ = initialized_models(args, base, contract)
    engine = make_engine(model, predictor, names, base, identity)
    cache = target_cache(teacher, contract)
    store = CheckpointStore(args.output/f'{args.arm}-process-recovery', identity['digest'], keep=3)
    if args.phase == 'resume':
        engine.restore(store.load())
        require(engine.global_step == 2, 'wrong recovery point')
    for i in range(engine.next_batch, 4):
        batch = batches[i]
        truth, _ = pilot.get_targets(batch, args.method, cache)
        def kill(a, b, update=i+1):
            if args.phase == 'interrupt' and update == 3:
                atomic_json(args.output/f'{args.arm}-kill-marker.json',
                            {'pid': os.getpid(), 'global_step': engine.global_step, 'microbatch_stop': b})
                os.kill(os.getpid(), signal.SIGKILL)
        engine.step(batch, truth, hook=kill)
        if args.phase == 'interrupt' and engine.global_step == 2:
            store.save(engine.snapshot())
        del truth, batch
    snapshot = engine.snapshot()
    report = {'identity_digest': identity['digest'], 'state_digest': tree_digest(snapshot),
              'global_step': engine.global_step, 'samples_seen': engine.samples_seen,
              'future_pairs_seen': engine.future_pairs_seen,
              'sample_chain': engine.sample_chain, 'future_chain': engine.future_chain}
    if args.phase == 'resume':
        expected = read_sealed(args.output/f'{args.arm}-uninterrupted.json')
        require(all(expected[k] == v for k, v in report.items()), 'same-device process recovery differs')
        damaged = store.save(snapshot)
        (store.root/damaged['checkpoint']).write_bytes(b'simulated torn checkpoint')
        intact = store.load()
        require(intact['global_step'] == 2 and store.recovery_notes, 'damaged latest fallback failed')
        # Replay from the fallback too, not just verify which record was selected.
        engine.restore(intact)
        cache = target_cache(teacher, contract)
        for i in range(engine.next_batch, 4):
            batch = batches[i]
            truth, _ = pilot.get_targets(batch, args.method, cache)
            engine.step(batch, truth)
        require(tree_digest(engine.snapshot()) == report['state_digest'], 'fallback replay differs')
        report.update(process_resume_bitwise_equal=True, damaged_latest_fallback=True,
                      fallback_replay_bitwise_equal=True, recovery_notes=store.recovery_notes)
    check_teacher(teacher, teacher_hash)
    production.immutable_json(args.output/f'{args.arm}-{args.phase}.json', seal_contract(report))
    pilot.event(args, args.phase+'_complete', state_digest=report['state_digest'])


def identity_for(args, contract, runtime, source):
    return seal_contract({'artifact_type': 'eventground_wam.coupled_pilot_identity',
        'contract_digest': contract['digest'], 'arm': contract['arm'], 'runtime_digest': runtime['digest'],
        'source_digest': source['digest'], 'data_path': str(args.data.resolve()), 'cache_path': str(args.cache.resolve()),
        'pilot_weights_excluded_from_production': True})


def orchestrate(args, source, runtime):
    started = time.monotonic()
    phases = [(arm, phase) for arm in ('M3', 'M4')
              for phase in ('correctness', 'uninterrupted', 'interrupt', 'resume')]+[(args.arm, 'resource')]
    records = []
    for arm, phase in phases:
        command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase, '--arm', arm,
                   '--profile', str(args.profile), '--output', str(args.output), '--data', str(args.data), '--cache', str(args.cache)]
        production.emit(args.output, 'qualification_running', arm=arm, phase=phase)
        with (args.output/f'{arm}-{phase}.log').open('x') as log, (args.output/f'{arm}-{phase}-device.csv').open('x') as trace:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                '--format=csv,noheader,nounits', '-lms', '100'], stdout=trace)
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                                        timeout=max(1, 3*3600-(time.monotonic()-started)))
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        expected = -signal.SIGKILL if phase == 'interrupt' else 0
        records.append({'arm': arm, 'phase': phase, 'exit_code': result.returncode})
        atomic_json(args.output/'phase-records.json', records)
        require(result.returncode == expected, f'{arm} {phase} exited {result.returncode}; inspect its log')
        readings = list(csv.reader((args.output/f'{arm}-{phase}-device.csv').read_text().splitlines()))
        require(len(readings) > 10 and all(len(r) == 2 and float(r[1]) <= .9*float(r[0]) for r in readings), 'device memory ceiling failed')
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) <= 24*1024**3, 'pilot output limit failed')
    references = {}
    for arm in ('M3', 'M4'):
        contract = execution_contract(arm)
        identity = identity_for(args, contract, runtime, source)
        for phase in ('correctness', 'resume'):
            report = read_sealed(args.output/f'{arm}-{phase}.json')
            require(report['identity_digest'] == identity['digest'], 'evidence identity mismatch')
            require(report.get('passed', report.get('process_resume_bitwise_equal')), 'correctness or recovery failed')
            references[f'{arm}-{phase}'] = report['digest']
    resource_report = read_sealed(args.output/f'{args.arm}-resources.json')
    require(resource_report['passed'], 'resource qualification failed')
    references[f'{args.arm}-resources'] = resource_report['digest']
    qualification = seal_contract({'artifact_type': 'eventground_wam.coupled_qualification',
        'arm': args.arm, 'contract_digest': execution_contract(args.arm)['digest'],
        'source_digest': source['digest'], 'runtime_digest': runtime['digest'], 'reports': references,
        'phases': records, 'elapsed_seconds': time.monotonic()-started,
        'passed': True, 'production_started': False, 'scientific_design_changed': False})
    production.immutable_json(args.output/'qualification.json', qualification)
    production.emit(args.output, 'qualification_passed', qualification_digest=qualification['digest'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['inspect', 'all', 'correctness', 'resource', 'uninterrupted', 'interrupt', 'resume'], required=True)
    parser.add_argument('--arm', choices=['M3', 'M4'], default='M4')
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/home/peimengsui/data/eventground-wam/libero-plus'))
    parser.add_argument('--cache', type=Path, default=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'))
    args = parser.parse_args()
    args.method = METHODS[args.arm]
    args.output.mkdir(parents=True, exist_ok=True)
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    source = verify_source()
    contract = execution_contract(args.arm)
    if args.phase == 'inspect':
        runtime = capture_runtime()
        verify_idle_gpu()
        production.immutable_json(args.profile, runtime)
        print(json.dumps(runtime, indent=2), flush=True)
        return
    runtime = read_sealed(args.profile) if args.phase == 'all' else verify_runtime(args.profile)
    if args.phase == 'all':
        verify_idle_gpu()
        with (args.output/'orchestrator.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            orchestrate(args, source, runtime)
        return
    production.check_resources(args.output, contract)
    identity = identity_for(args, contract, runtime, source)
    production.immutable_json(args.output/f'{args.arm}-pilot-identity.json', identity)
    production.immutable_json(args.output/f'{args.arm}-contract.json', contract)
    rule = read_sealed(pilot.RULE)
    require(rule['digest'] == contract['pilot_rule_digest'], 'pilot rule changed')
    base, batches, calibration, edges, data = pilot.preflight(args, rule)
    require(data['action_plan_digests'] == contract['mixed_plan_digests']
            and data['future_pairs_per_epoch']*2 == contract['future_pairs'], 'data exposure changed')
    if args.phase == 'correctness':
        correctness(args, rule, base, batches, calibration, edges, contract, identity)
    elif args.phase == 'resource':
        resources(args, rule, base, batches, edges, contract, identity)
    else:
        recovery(args, base, batches, contract, identity)


if __name__ == '__main__':
    main()
