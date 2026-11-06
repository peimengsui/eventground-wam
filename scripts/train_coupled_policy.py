#!/usr/bin/env python3
"""Launch a fresh coupled arm only after symmetric correctness and actual-host admission."""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import time
from pathlib import Path

import check_auxiliary_training as pilot
import torch
import train_auxiliary_policy as production
from coupled_runtime import (
    execution_contract,
    initialized_models,
    make_engine,
    project_hours,
    require,
    verify_runtime,
    verify_source,
)

from eventground_wam.auxiliary_data import AuxiliaryBatches, AuxiliaryTargetCache
from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_fresh import fresh_plan
from eventground_wam.b0_training import (
    CheckpointStore,
    atomic_json,
    atomic_torch_save,
    configure_deterministic_cuda,
    cpu_tree,
    read_sealed,
    sha256_file,
)
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.coupled_checkpoint import export_coupled_policy
from eventground_wam.coupled_policy import FULL_BATCH_EXECUTION, LOSS_WEIGHTS, METHODS
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_predictor import tensor_state_digest


def verify_qualification(evidence, contract, source, runtime, args):
    qualification = read_sealed(evidence/'qualification.json')
    require(qualification['passed'] and qualification['arm'] == args.arm
            and qualification['contract_digest'] == contract['digest']
            and qualification['source_digest'] == source['digest']
            and qualification['runtime_digest'] == runtime['digest'], 'qualification identity differs')
    expected_phases = [(a, p) for a in ('M3', 'M4') for p in ('correctness', 'uninterrupted', 'interrupt', 'resume')]+[(args.arm, 'resource')]
    require([(r['arm'], r['phase']) for r in qualification['phases']] == expected_phases, 'qualification phases missing')
    require(all(r['exit_code'] == (-signal.SIGKILL if r['phase'] == 'interrupt' else 0)
                for r in qualification['phases']), 'qualification process failed')
    require(qualification['elapsed_seconds'] <= 3*3600, 'pilot time limit failed')
    for arm in ('M3', 'M4'):
        pilot_identity = read_sealed(evidence/f'{arm}-pilot-identity.json')
        require(pilot_identity['source_digest'] == source['digest']
                and pilot_identity['runtime_digest'] == runtime['digest']
                and pilot_identity['contract_digest'] == execution_contract(arm)['digest'], 'pilot identity differs')
        for key in ('data', 'cache'):
            require(pilot_identity[f'{key}_path'] == str(getattr(args, key).resolve()), f'{key} location differs')
        reports = {}
        for phase in ('correctness', 'resume'):
            report = read_sealed(evidence/f'{arm}-{phase}.json')
            require(report['digest'] == qualification['reports'][f'{arm}-{phase}']
                    and report['identity_digest'] == pilot_identity['digest'], 'qualification evidence differs')
            reports[phase] = report
        require(reports['correctness']['passed'] and reports['correctness']['real_qwen_export_reload_bitwise_equal']
                and reports['correctness']['target_isolation'] and reports['correctness']['teacher_unchanged'], 'correctness failed')
        require(reports['correctness']['backward_execution'] == FULL_BATCH_EXECUTION,
                'backward execution differs')
        require(reports['resume']['process_resume_bitwise_equal'] and reports['resume']['damaged_latest_fallback']
                and reports['resume']['fallback_replay_bitwise_equal'], 'recovery failed')
        reference = read_sealed(evidence/f'{arm}-uninterrupted.json')
        require(reference['state_digest'] == reports['resume']['state_digest'], 'uninterrupted state differs')
    report = read_sealed(evidence/f'{args.arm}-resources.json')
    require(report['passed'] and report['digest'] == qualification['reports'][f'{args.arm}-resources'], 'resources failed')
    require(report['identity_digest'] == read_sealed(evidence/f'{args.arm}-pilot-identity.json')['digest'],
            'resource evidence identity differs')
    require(report['loss_weight'] == LOSS_WEIGHTS[args.arm], 'loss weight differs')
    require(report['backward_execution'] == FULL_BATCH_EXECUTION, 'resource execution differs')
    require([r['global_step'] for r in report['rows']] == list(range(1, 129)), 'timing update stream differs')
    mean, p95, hours = project_hours([r['seconds'] for r in report['rows']], report['checkpoint_seconds'])
    for key, value in [('mean_step_seconds', mean), ('p95_step_seconds', p95), ('projected_training_hours', hours)]:
        require(abs(report[key]-value) < 1e-8, 'resource projection differs')
    require(hours <= 168 and report['host_peak_rss_bytes'] <= 28*1024**3
            and report['gpu_peak_reserved_bytes'] <= .9*runtime['cuda_total_bytes']
            and report['gpu_peak_allocated_bytes'] <= .9*runtime['cuda_total_bytes'], 'resource limit failed')
    require(report['teacher_unchanged'] and report['pilot_weights_excluded_from_production'], 'pilot isolation failed')
    require(report['edge_probes'] and all(r['finite'] and r['sequence_length'] <= 512 for r in report['edge_probes']), 'edge checks failed')
    return qualification


def export_epoch(output, engine, recipe, identity):
    epoch = engine.epoch+1
    require(engine.global_step == epoch*recipe['updates_per_epoch']
            and engine.samples_seen == epoch*recipe['action_presentations']//recipe['epochs']
            and engine.future_pairs_seen == epoch*recipe['future_pairs']//recipe['epochs'],
            'endpoint exposure differs')
    root = output/'epochs'/f'epoch-{epoch:02d}'
    root.mkdir(parents=True, exist_ok=True)
    path = root/f"{recipe['arm']}.ckpt"
    metadata = {'identity_digest': identity['digest'], 'global_step': engine.global_step,
                'epoch': epoch, 'checkpoint_selection': 'fixed-epoch-2-endpoint'}
    if not path.exists():
        export_coupled_policy(path, engine.model, engine.exact_lora_names, metadata=metadata)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(saved['header']['metadata'] == metadata, 'export identity differs')
    expected = {
        'policy': production.b0_trainable_state_dict(engine.policy, allowed_exact_names=engine.exact_lora_names),
        'future_core': engine.model.future_core.state_dict(), 'coupling': engine.model.coupling.state_dict()}
    for key, state in expected.items():
        require(tensor_state_digest(saved['states'][key]) == tensor_state_digest(state)
                == saved['header']['state_sha256'][key], f'export tensor roundtrip differs: {key}')
    future_path = root/'full-training-predictor.pt'
    future_state = cpu_tree({'identity_digest': identity['digest'], 'state_dict': engine.predictor.state_dict(),
                            'global_step': engine.global_step, 'method': recipe['method']})
    if not future_path.exists():
        atomic_torch_save(future_path, future_state)
    require(tensor_tree_digest(torch.load(future_path, map_location='cpu', weights_only=True))
            == tensor_tree_digest(future_state), 'full predictor export differs')
    report = seal_contract({'artifact_type': 'eventground_wam.coupled_epoch',
        'identity_digest': identity['digest'], 'epoch': epoch, 'global_step': engine.global_step,
        'samples_seen': engine.samples_seen, 'future_pairs_seen': engine.future_pairs_seen,
        'sample_chain': engine.sample_chain, 'future_chain': engine.future_chain,
        'checkpoint_sha256': sha256_file(path), 'predictor_sha256': sha256_file(future_path),
        'decoder_free_coupled_export': True, 'selection': 'fixed-epoch-2-endpoint', 'evaluation_status': 'pending'})
    production.immutable_json(root/'training-manifest.json', identity)
    production.immutable_json(root/'training-report.json', report)
    return report


def run(args, contract, source, runtime, qualification, budget):
    rule = read_sealed(pilot.RULE)
    base, initial_batches, _, _, data = pilot.preflight(args, rule)
    require(base['digest'] == contract['base_recipe_digest']
            and data['action_plan_digests'] == contract['mixed_plan_digests']
            and data['future_pairs_per_epoch']*2 == contract['future_pairs'], 'data or base recipe differs')
    for key in ('adam_betas', 'adam_epsilon', 'decoder_projector_learning_rate', 'lora_learning_rate',
                'maximum_gradient_norm', 'minimum_learning_rate_ratio', 'warmup_fraction', 'weight_decay'):
        require(base[key] == contract['optimizer'][key], f'optimizer differs: {key}')
    identity = seal_contract({'artifact_type': 'eventground_wam.coupled_production',
        'arm': args.arm, 'contract_digest': contract['digest'], 'design_digest': contract['design_digest'],
        'source_digest': source['digest'], 'runtime': runtime, 'qualification_digest': qualification['digest'],
        'mixed_plan_digests': data['action_plan_digests'], 'initialization': contract['initialization'],
        'fixed_endpoint_updates': 117132, 'training_host': runtime['host'],
        'evaluation': 'fixed endpoint; paired dev210 on common Feifei runtime, pending'})
    production.immutable_json(args.output/'training-manifest.json', identity)
    admitted_recipe = seal_contract({**{k: v for k, v in contract.items() if k != 'digest'},
        'status': 'numerical_recipe_and_feasibility_verified', 'contract_digest': contract['digest'],
        'qualification_digest': qualification['digest'], 'production_source_digest': source['digest'],
        'runtime_digest': runtime['digest']})
    production.immutable_json(args.output/'execution-recipe.json', admitted_recipe)
    model, predictor, names, teacher, _, verified, header = initialized_models(args, base, contract)
    engine = make_engine(model, predictor, names, base, identity)
    production.immutable_json(args.output/'fresh-initialization.json', seal_contract({
        'identity_digest': identity['digest'], 'shared_tensor_digest': contract['initialization']['shared_tensor_digest'],
        'module_header_digest': header['digest'], 'base_files': verified,
        'coupling_tensor_digest': tensor_state_digest(model.coupling.state_dict()),
        'global_step': 0, 'optimizer_entries': len(engine.optimizer.state),
        'pilot_or_trained_policy_weights_loaded': False}))
    store = CheckpointStore(args.output/'recovery', identity['digest'], keep=3)
    recovered = store.load()
    if recovered is None:
        require(not (args.output/'recovery-created.json').exists(), 'all recovery records missing')
        production.check_resources(args.output, contract, reserve_bytes=2*1024**3)
        store.save(engine.snapshot())
        production.immutable_json(args.output/'recovery-created.json', seal_contract({'identity_digest': identity['digest']}))
    else:
        engine.restore(recovered)
        atomic_json(args.output/'last-resume.json', seal_contract({'identity_digest': identity['digest'],
            'global_step': engine.global_step, 'epoch': engine.epoch, 'next_batch': engine.next_batch,
            'pid': os.getpid(), 'time_unix': time.time(), 'recovery_notes': store.recovery_notes}))
        del recovered
    cache = AuxiliaryTargetCache(teacher, maximum_bytes=contract['batching']['target_cache_maximum_bytes']) if teacher else None
    def batches(epoch):
        if epoch == 0:
            return initial_batches
        plan = fresh_plan(initial_batches.action_dataset, base, epoch)
        require(batches_digest(plan) == contract['mixed_plan_digests'][epoch], 'epoch plan differs')
        return AuxiliaryBatches(initial_batches.action_dataset, initial_batches.pairs, plan, args.cache,
                                expected_epoch=epoch, builder=initial_batches.builder)
    production.emit(args.output, 'initialized', global_step=engine.global_step, identity_digest=identity['digest'], total_updates=117132)
    return production.train_loop(engine, contract, identity, args.output, store, budget, batches,
                                 lambda batch: pilot.get_targets(batch, args.method, cache), exporter=export_epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['M3', 'M4'], required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/home/peimengsui/data/eventground-wam/libero-plus'))
    parser.add_argument('--cache', type=Path, default=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'))
    args = parser.parse_args()
    args.method, args.phase = METHODS[args.arm], 'production'
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, production.request_stop)
    signal.signal(signal.SIGINT, production.request_stop)
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    try:
        source, contract = verify_source(), execution_contract(args.arm)
        runtime = verify_runtime(args.profile)
        qualification = verify_qualification(args.qualification, contract, source, runtime, args)
        production.check_resources(args.output, contract)
        with (args.output/'owner.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = ActiveBudget(args.output/'active-budget.json', seal_contract({
                'contract': contract['digest'], 'source': source['digest'], 'qualification': qualification['digest']})['digest'])
            try:
                code = run(args, contract, source, runtime, qualification, budget)
            finally:
                budget.close()
        raise SystemExit(code)
    except (ValueError, OSError, AssertionError, RuntimeError) as error:
        production.emit(args.output, 'integrity_or_runtime_stop', error=str(error), exception=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
