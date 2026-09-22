#!/usr/bin/env python3
"""Evaluate the fixed all-L1 endpoint on development-200 with verified cached controls."""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import evaluate_clean_comparison as clean
from evaluate_b0_full import FullCheckpointIdentity
from train_b0_continuation import ROOT, check_space, experiment

from eventground_wam.b0_continuation import OBJECTIVE, ActiveBudget
from eventground_wam.b0_model import QWEN_REVISION
from eventground_wam.b0_training import TrainingContractError, read_sealed, sha256_file
from eventground_wam.clean_comparison import (
    ComparisonIntegrityError,
    atomic_json,
    compare_pair_receipts,
    execute_pair,
    metadata_for_pair,
    paths_for_pair,
    utc_now,
)
from eventground_wam.clean_reuse import verify_cached_record
from eventground_wam.continuation_freeze import assess_screen, outcome_counts
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import protocol_sha256
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur


def verify_bundle(root, normalization, config, *, pilot=False):
    manifest = read_sealed(root / 'training-manifest.json')
    report = read_sealed(root / ('pilot-report.json' if pilot else 'training-report.json'))
    recipe = manifest['recipe']
    expected_type = ('eventground_wam.all_l1_mixed_pilot' if pilot
                     else 'eventground_wam.all_l1_continuation_report')
    if (manifest['artifact_type'] != 'eventground_wam.b0_all_l1_continuation_manifest'
            or manifest['mode'] != ('pilot' if pilot else 'train')
            or manifest['arm'] != 'B' or manifest['experiment_digest'] != config['digest']
            or recipe['objective'] != OBJECTIVE or recipe['qwen_revision'] != QWEN_REVISION
            or recipe['policy_family'] != 'qwen35-oft-b0-v1' or recipe['lora_rank'] != 32
            or report['artifact_type'] != expected_type
            or report['identity_digest'] != manifest['digest']
            or report['experiment_digest'] != config['digest']
            or report['status'] != ('passed' if pilot else 'complete')):
        raise ComparisonIntegrityError('invalid all-L1 checkpoint bundle')
    if not pilot and (report['selected_additional_step'] != 15000
                      or report['additional_updates'] != 15000
                      or report['additional_samples'] != 435945
                      or report['cumulative_updates'] != 87000
                      or report['reload_validation_passed'] is not True
                      or report['validation']['metrics']['sample_count'] != 2560):
        raise ComparisonIntegrityError('endpoint or reload coverage changed')
    digest = sha256_file(root / 'B0.ckpt')
    if (digest != report['checkpoint_sha256']
            or sha256_file(normalization) != manifest['source']['normalization_file_sha256']):
        raise ComparisonIntegrityError('checkpoint or normalizer bytes changed')
    checkpoint = FullCheckpointIdentity(
        checkpoint_id=f"eventground-wam/{recipe['recipe_id']}" + ('-pilot' if pilot else ''),
        checkpoint_revision=digest, training_manifest_digest=manifest['digest'],
        training_report_digest=report['digest'], selected_epoch=None,
        selected_step=72000 + (report['updates'] if pilot else 15000))
    return manifest, checkpoint


def cached_controls(args, config, full_panel, pairs):
    old = read_sealed(args.previous_root / 'comparison-report.json')
    parent = read_sealed(args.parent_root / 'comparison-report.json')
    if (old['digest'] != config['sources']['previous_comparison']['digest']
            or parent['digest'] != config['sources']['clean_completion']['digest']):
        raise ComparisonIntegrityError('cached completed comparisons changed')
    pinned = read_sealed(ROOT / config['sources']['cached_baselines']['path'])
    by_id = {p['pair_id']: (i, p) for i, p in enumerate(full_panel['pairs'])}
    expected = {p['pair_id']: p for p in pinned['pairs']}
    specs = {
        'new_b0': (args.parent_root / 'b0', parent['new_policy_identity'],
                   parent['comparisons']['new_vs_openvla']['pairs'], 'b0'),
        'previous_b0': (args.previous_root / 'b0', old['policy_identities']['b0'], old['pairs'], 'b0'),
        'openvla': (args.previous_root / 'reference', old['policy_identities']['reference'],
                    old['pairs'], 'reference'),
    }
    controls = {}
    for name, (root, identity, rows, policy) in specs.items():
        if read_sealed(root / 'identity.json') != identity:
            raise ComparisonIntegrityError('cached control identity changed')
        controls[name] = []
        for pair in pairs:
            ordinal, original = by_id[pair['pair_id']]
            if pair != original:
                raise ComparisonIntegrityError('development pair changed')
            record = verify_cached_record(root, ordinal, pair, rows[ordinal], policy, identity)
            e = expected[pair['pair_id']]
            if (record['digest'] != e[f'{name}_receipt_digest']
                    or record['success'] != e[f'{name}_success']):
                raise ComparisonIntegrityError('cached development outcome changed')
            controls[name].append(record)
        print(json.dumps({'stage': 'cached-control-verified', 'policy': name,
                          'artifacts': len(pairs)}), flush=True)
    return controls


def run(args, config, budget):
    rule, full_panel, protocol, suites, states = clean.inventory(args)
    panel = read_sealed(ROOT / config['sources']['development_panel']['path'])
    pairs = panel['pairs'][:2] if args.pilot_smoke else panel['pairs']
    bundle = verify_bundle(args.training_root, args.normalization, config, pilot=args.pilot_smoke)
    manifest, checkpoint = bundle
    revision = clean.git_revision(ROOT)
    if revision != manifest['project_git_revision']:
        raise ComparisonIntegrityError('training and evaluation source revisions differ')
    versions = clean.runtime_versions()
    if versions['transformers'] != '5.17.0':
        raise ComparisonIntegrityError('Transformers runtime changed')
    identity = seal_contract({
        'artifact_type': 'eventground_wam.all_l1_development_run', 'policy': 'b0',
        'mode': 'pilot' if args.pilot_smoke else 'development', 'pair_count': len(pairs),
        'project_git_revision': revision, 'acceptance_rule_digest': rule['digest'],
        'panel_digest': panel['digest'], 'protocol_sha256': protocol_sha256(protocol),
        'experiment_digest': config['digest'], 'checkpoint_id': checkpoint.checkpoint_id,
        'checkpoint_revision': checkpoint.checkpoint_revision,
        'runtime_lock_sha256': sha256_file(ROOT / 'environments/b0/uv.lock'),
        'runtime_versions': versions, 'checkpoint_evidence': checkpoint.to_dict(),
    })
    if identity['runtime_lock_sha256'] != manifest['b0_runtime_lock_sha256']:
        raise ComparisonIntegrityError('training runtime lock changed')
    root = args.training_root / 'rollouts'
    saved = root / 'identity.json'
    if saved.exists() and read_sealed(saved) != identity:
        raise ComparisonIntegrityError('development resume identity changed')
    atomic_json(saved, identity)
    controls = cached_controls(args, config, full_panel, pairs)
    install_codec_independent_motion_blur()
    from libero.libero.envs import OffScreenRenderEnv

    policy = None
    records = []
    for ordinal, pair in enumerate(pairs):
        if budget.seconds >= config['budget']['total_active_seconds']:
            raise ComparisonIntegrityError('shared active budget reached')
        check_space(args, config)
        paths = clean.bddl_for(pair, suites)
        candidates = [paths_for_pair(root, ordinal, pair, i) for i in range(2)]
        if policy is None and not (any(a.exists() for a, _, _ in candidates)
                                  or candidates[0][2].exists()
                                  or all(j.exists() for _, j, _ in candidates)):
            policy = clean.load_policy(args, bundle)
        resolution = protocol['observation']['capture_resolution']
        record = execute_pair(
            root=root, ordinal=ordinal, pair=pair, identity=identity, protocol=protocol,
            metadata=metadata_for_pair(pair, protocol, identity, paths.source),
            initial_state=states[(pair['suite'], pair['runtime_task_index'])][pair['initial_state_index']],
            policy=policy,
            environment_factory=lambda paths=paths, resolution=resolution: OffScreenRenderEnv(
                bddl_file_name=str(paths.runtime), camera_heights=resolution[0],
                camera_widths=resolution[1], control_freq=protocol['simulation']['control_frequency_hz']),
            episode_runner=clean.run_episode, seed_episode=clean.seed_episode,
            expected_input=controls['new_b0'][ordinal]['first_query_input_hashes'])
        records.append(record)
        progress = {'stage': 'development-rollouts', 'mode': identity['mode'],
                    'pair_count': len(pairs), 'updated_at_utc': utc_now(),
                    'last_pair': pair['pair_id'], **clean.summarize(records)}
        atomic_json(root / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    records = clean.checked_receipts(root, pairs, identity, protocol, suites)
    comparisons = {name: [compare_pair_receipts(a, b) for a, b in zip(records, cached, strict=True)]
                   for name, cached in controls.items()}
    valid = all(r['assessable'] for checks in comparisons.values() for r in checks)
    outcomes = {r['pair_id']: r['success'] for r in records}
    control_outcomes = {name: {r['pair_id']: r['success'] for r in cached}
                        for name, cached in controls.items()}
    report = seal_contract({
        'artifact_type': 'eventground_wam.all_l1_development_report', 'status': 'complete',
        'identity_digest': identity['digest'], 'completed_at_utc': utc_now(),
        'checkpoint_sha256': checkpoint.checkpoint_revision, 'all_pairs_valid': valid,
        'receipt_digests': [r['digest'] for r in records], **clean.summarize(records),
        'comparisons': comparisons,
        'counts': None if args.pilot_smoke else {
            'all_l1_b': outcome_counts(panel, outcomes),
            **{name: outcome_counts(panel, values) for name, values in control_outcomes.items()}},
        'engineering_screen': None if args.pilot_smoke else assess_screen(
            panel, control_outcomes['new_b0'], outcomes, valid=valid),
        'interpretation': 'Development evidence only. No A/B selection, auto-extension or Week4 PASS.',
        'budget_active_seconds': budget.seconds,
    })
    atomic_json(root / 'evaluation-report.json', report)
    if not valid:
        raise ComparisonIntegrityError('rollout artifacts or matched initial inputs failed validation')
    print(json.dumps({'stage': 'development-complete', 'completed': len(records),
                      'counts': report['counts'], 'screen': report['engineering_screen']}), flush=True)


def main():
    home = Path.home()
    data = home / 'data/eventground-wam'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--pilot-smoke', action='store_true')
    defaults = {
        'normalization': data / 'libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json',
        'cache-dir': data / 'huggingface-cache', 'libero-root': home / 'code/vendor/LIBERO-plus',
        'previous-root': data / 'comparisons/week4-clean-comparison-v1',
        'parent-root': data / 'comparisons/week4-oft-clean-comparison-v1',
    }
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    args = parser.parse_args()
    args.policy = 'b0'
    args.training_root = args.work_root / ('pilot' if args.pilot_smoke else 'training')
    try:
        with (args.work_root / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = experiment()
            budget = ActiveBudget(args.work_root / 'budget.json', config['digest'])
            try:
                run(args, config, budget)
            finally:
                budget.close()
    except (ComparisonIntegrityError, TrainingContractError, BlockingIOError, ValueError) as error:
        atomic_json(args.training_root / 'rollout-integrity-error.json', {
            'at_utc': utc_now(), 'exception': f'{type(error).__name__}: {error}'})
        print(f'Integrity stop: {error}', file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
