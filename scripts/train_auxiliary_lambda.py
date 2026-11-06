#!/usr/bin/env python3
"""Run unchanged M2 training only after qualification on the pinned Lambda A10."""
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
from auxiliary_lambda_runtime import ROOT, project_hours, require, verify_recipe, verify_runtime
from check_auxiliary_lambda import initialized_models

from eventground_wam.auxiliary_data import AuxiliaryBatches, AuxiliaryTargetCache
from eventground_wam.auxiliary_training import AuxiliaryTrainingEngine
from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_fresh import fresh_plan
from eventground_wam.b0_training import (
    CheckpointStore,
    atomic_json,
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import seal_contract

AMENDMENT_DIGEST = 'c92bf8daecd92e6b5d0b555714a84b56d7fd99e60622ddb6d9a32b151a98477e'


def amended_limits(amendment, recipe, runtime, report):
    require(seal_contract({k: v for k, v in amendment.items() if k != 'digest'})['digest']
            == amendment['digest'] == AMENDMENT_DIGEST, 'unapproved budget amendment')
    require(amendment['recipe_digest'] == recipe['digest'], 'amendment recipe mismatch')
    require(amendment['runtime_digest'] == runtime['digest'], 'amendment runtime mismatch')
    require(amendment['original_resources_digest'] == report['digest'], 'amendment evidence mismatch')
    require(amendment['original_limits'] == {k: recipe['resources'][k] for k in
            ('maximum_projected_hours', 'maximum_active_hours')}, 'original limits changed')
    require(amendment['replacement_limits'] == {'maximum_projected_hours': 480,
            'maximum_active_hours': 480}, 'amendment scope changed')
    require(amendment['original_timing_gate_remains_failed'] and report['passed'] is False,
            'original timing failure must be retained')
    return {**recipe['resources'], **amendment['replacement_limits']}


def validate_resources(report, recipe, runtime, amendment=None):
    limits = amended_limits(amendment, recipe, runtime, report) if amendment else recipe['resources']
    require(report['passed'] is True or amendment is not None, 'Lambda resource qualification failed')
    require(report['loss_weight'] == recipe['loss']['outer_future_weight'], 'pilot loss weight changed')
    require([r['global_step'] for r in report['rows']] == list(range(1, 129)), 'pilot update stream changed')
    mean, p95, hours = project_hours([r['seconds'] for r in report['rows']],
                                    report['checkpoint_seconds'], recipe['total_updates'])
    for key, value in [('mean_step_seconds', mean), ('p95_step_seconds', p95),
                       ('projected_training_hours', hours)]:
        require(abs(report[key]-value) < 1e-8, f'pilot projection mismatch: {key}')
    require(hours <= limits['maximum_projected_hours'], 'pilot exceeds declared time limit')
    require(report['gpu_peak_reserved_bytes'] <= .9*runtime['cuda_total_bytes'], 'pilot GPU limit failed')
    require(report['host_peak_rss_bytes'] <= 28*1024**3, 'pilot host memory limit failed')
    require(report['teacher_unchanged'] and report['pilot_weights_excluded_from_production'],
            'pilot isolation failed')
    require(report['edge_probes'] and all(r['finite'] and r['sequence_length'] <= 512
                                         for r in report['edge_probes']), 'edge checks failed')


def verify_qualification(args, recipe, source, runtime, amendment):
    evidence = ROOT/'lambda-qualification'
    qualification = read_sealed(evidence/'qualification.json')
    identity = read_sealed(evidence/'pilot-identity.json')
    pilot_source = read_sealed(evidence/'source-manifest.json')
    require(qualification['passed'] is True, 'Lambda is not qualified')
    require(qualification['budget_amendment_digest'] == amendment['digest'], 'qualification budget changed')
    require(qualification['original_resource_gate_passed'] is False, 'original timing gate was relabeled')
    require(qualification['recipe_digest'] == identity['recipe_digest'] == recipe['digest'],
            'qualification recipe changed')
    require(qualification['runtime_digest'] == identity['runtime_digest'] == runtime['digest'],
            'qualification hardware/runtime changed')
    require(qualification['identity_digest'] == identity['digest'], 'qualification identity changed')
    require(qualification['production_source_digest'] == identity['production_source_digest']
            == pilot_source['digest'], 'qualification source identity changed')
    require(amendment['pilot_source_digest'] == pilot_source['digest']
            and amendment['pilot_identity_digest'] == identity['digest'], 'amendment pilot mismatch')
    for name, digest in pilot_source['files'].items():
        require(source['files'].get(name) == digest, f'qualified implementation changed: {name}')
    for key in ('data', 'cache'):
        require(identity[f'{key}_path'] == str(getattr(args, key).resolve()), f'{key} location changed')
    reports = {}
    for name, field in [('correctness', 'correctness_digest'), ('resources', 'resources_digest'),
                        ('jepa-resume', 'recovery_digest')]:
        report = read_sealed(evidence/f'{name}.json')
        require(report['digest'] == qualification[field], f'{name} evidence changed')
        reports[name] = report
    for name in ('correctness', 'resources'):
        require(reports[name]['identity_digest'] == identity['digest'], 'pilot evidence identity mismatch')
    correct = reports['correctness']
    require(correct['passed'] and correct['target_repeat_bitwise_equal'] and correct['teacher_unchanged'],
            'Lambda correctness failed')
    require(correct['loss_weight'] == recipe['loss']['outer_future_weight'] and not correct['recalibrated'],
            'qualification recalibrated frozen loss')
    validate_resources(reports['resources'], recipe, runtime, amendment)
    resumed = reports['jepa-resume']
    require(resumed['process_resume_bitwise_equal'] and resumed['damaged_latest_fallback'], 'recovery failed')
    uninterrupted = read_sealed(evidence/'jepa-uninterrupted.json')
    supplement = read_sealed(evidence/'recovery-supplement.json')
    require(supplement['digest'] == qualification['recovery_supplement_digest']
            and supplement['passed'], 'recovery supplement mismatch')
    require(supplement['identity_digest'] == identity['digest']
            and supplement['budget_amendment_digest'] == amendment['digest'], 'recovery provenance mismatch')
    require(supplement['recovery_digest'] == resumed['digest']
            and supplement['uninterrupted_digest'] == uninterrupted['digest']
            and resumed['state_digest'] == uninterrupted['state_digest'], 'recovery state mismatch')
    require(supplement['orchestrator_sha256'] == sha256_file(ROOT/'scripts/qualify_auxiliary_lambda_budget.py'),
            'recovery orchestrator changed')
    require([(p['phase'], p['exit_code']) for p in supplement['phases']]
            == [('uninterrupted', 0), ('interrupt', -9), ('resume', 0)], 'recovery phases missing')
    return qualification


def run(args, recipe, source, runtime, qualification, budget, amendment):
    rule = read_sealed(pilot.RULE)
    require(rule['digest'] == recipe['pilot_rule_digest'], 'pilot rule mismatch')
    base, initial_batches, _, _, data = pilot.preflight(args, rule)
    require(base['digest'] == recipe['base_recipe_digest'], 'base recipe mismatch')
    require(data['action_plan_digests'] == recipe['mixed_plan_digests'], 'action plans changed')
    require(data['future_pairs_per_epoch']*recipe['epochs'] == recipe['future_pairs'], 'pair count changed')
    config = {**base, 'auxiliary_learning_rate': recipe['optimizer']['auxiliary_learning_rate']}
    for name in ('adam_betas', 'adam_epsilon', 'decoder_projector_learning_rate',
                 'lora_learning_rate', 'maximum_gradient_norm', 'minimum_learning_rate_ratio',
                 'warmup_fraction', 'weight_decay'):
        require(config[name] == recipe['optimizer'][name], f'optimizer changed: {name}')
    identity = seal_contract({'artifact_type': 'eventground_wam.week7_auxiliary_production',
        'recipe_digest': recipe['digest'], 'arm': recipe['arm'], 'method': recipe['method'],
        'production_source_digest': source['digest'], 'verified_source_digest': data['source_digest'],
        'mixed_plan_digests': data['action_plan_digests'], 'runtime': runtime,
        'lambda_qualification_digest': qualification['digest'],
        'lambda_budget_amendment_digest': amendment['digest'],
        'effective_resource_limits': {**recipe['resources'], **amendment['replacement_limits']},
        'initialization': recipe['initialization'], 'fixed_endpoint_updates': recipe['total_updates'],
        'environment_change': 'Lambda A10 qualified separately; no cross-GPU bitwise equivalence claim',
        'validation_and_rollouts': 'pending-step-6; no selection or early stopping'})
    production.immutable_json(args.output/'training-manifest.json', identity)
    production.emit(args.output, 'loading_original_models', identity_digest=identity['digest'])
    policy, names, predictor, teacher, _, verified, header = initialized_models(args, config, recipe)
    engine = AuxiliaryTrainingEngine(policy, names, predictor, recipe['method'], config,
        identity_digest=identity['digest'], steps_per_epoch=recipe['updates_per_epoch'],
        device=torch.device('cuda'), loss_weight=recipe['loss']['outer_future_weight'],
        microbatch=recipe['batching']['predictor_microbatch'])
    seed_training(recipe['training_seed'])
    production.immutable_json(args.output/'fresh-initialization.json', seal_contract({
        'artifact_type': 'eventground_wam.week7_production_initialization',
        'identity_digest': identity['digest'], 'shared_tensor_digest': recipe['initialization']['shared_tensor_digest'],
        'module_header_digest': header['digest'], 'base_files': verified,
        'global_step': engine.global_step, 'optimizer_entries': len(engine.optimizer.state),
        'pilot_or_trained_b0_weights_loaded': False}))
    store = CheckpointStore(args.output/'recovery', identity['digest'],
                            keep=recipe['recovery']['checkpoint_keep_last'])
    recovered = store.load()
    if recovered is None:
        require(not (args.output/'recovery-created.json').exists(), 'all recovery records missing')
        production.check_resources(args.output, recipe, reserve_bytes=2*1024**3)
        store.save(engine.snapshot())
        production.immutable_json(args.output/'recovery-created.json', seal_contract({'identity_digest': identity['digest']}))
    else:
        engine.restore(recovered)
        atomic_json(args.output/'last-resume.json', seal_contract({'identity_digest': identity['digest'],
            'global_step': engine.global_step, 'epoch': engine.epoch, 'next_batch': engine.next_batch,
            'pid': os.getpid(), 'time_unix': time.time(), 'recovery_notes': store.recovery_notes}))
        del recovered
    cache = AuxiliaryTargetCache(teacher, maximum_bytes=recipe['batching']['target_cache_maximum_bytes'])

    def batches(epoch):
        if epoch == 0:
            return initial_batches
        plan = fresh_plan(initial_batches.action_dataset, config, epoch)
        require(batches_digest(plan) == recipe['mixed_plan_digests'][epoch], 'epoch plan changed')
        return AuxiliaryBatches(initial_batches.action_dataset, initial_batches.pairs,
            plan, args.cache, expected_epoch=epoch, builder=initial_batches.builder)

    production.emit(args.output, 'initialized', global_step=engine.global_step,
                    identity_digest=identity['digest'], total_updates=recipe['total_updates'])
    # This execution view is not a replacement sealed scientific recipe. The
    # manifest binds both the unchanged recipe and the separate time amendment.
    execution_recipe = {**recipe, 'resources': identity['effective_resource_limits']}
    return production.train_loop(engine, execution_recipe, identity, args.output, store, budget, batches,
                                 lambda batch: pilot.get_targets(batch, 'jepa', cache))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=ROOT/'configs/training/week7_m2_auxiliary_v1.json')
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, default=ROOT/'configs/training/week7_m2_lambda_budget_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/home/ubuntu/data/eventground-wam/libero-plus'))
    parser.add_argument('--cache', type=Path, default=Path('/home/ubuntu/data/eventground-wam/huggingface-cache'))
    args = parser.parse_args()
    args.method, args.phase = 'jepa', 'production'
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, production.request_stop)
    signal.signal(signal.SIGINT, production.request_stop)
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    try:
        recipe = read_sealed(args.recipe)
        verify_recipe(recipe)
        source = production.verify_package(recipe)
        runtime = verify_runtime(args.profile)
        amendment = read_sealed(args.amendment)
        qualification = verify_qualification(args, recipe, source, runtime, amendment)
        production.check_resources(args.output, recipe)
        with (args.output/'owner.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = ActiveBudget(args.output/'active-budget.json',
                seal_contract({'recipe': recipe['digest'], 'source': source['digest'],
                               'amendment': amendment['digest']})['digest'])
            try:
                code = run(args, recipe, source, runtime, qualification, budget, amendment)
            finally:
                budget.close()
        raise SystemExit(code)
    except (ValueError, OSError, AssertionError, RuntimeError) as error:
        production.emit(args.output, 'integrity_or_runtime_stop', error=str(error), exception=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
