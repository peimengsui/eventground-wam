#!/usr/bin/env python3
"""Fresh v2 clean rollouts with resumable artifacts and audited cached controls."""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import evaluate_clean_comparison as clean
from evaluate_b0_continuation import cached_controls
from evaluate_b0_full import FullCheckpointIdentity
from train_b0_fresh import RECIPE, ROOT, RULE, check_space

from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_fresh import load_fresh_recipe, verify_fresh_bundle
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
from eventground_wam.clean_reuse import (
    audit_cached_comparison,
    comparison_statistics,
    verify_cached_record,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import protocol_sha256
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur


def run(args, recipe, budget):
    rule, full, protocol, suites, states = clean.inventory(args)
    freeze = read_sealed(RULE)
    development = read_sealed(ROOT / freeze['development_panel'])
    panel = full if args.mode == 'clean-2000' else development
    pairs = panel['pairs'][:2] if args.mode == 'smoke' else panel['pairs']
    manifest, report = verify_fresh_bundle(args.training_root, args.normalization,
                                          allow_pilot=args.mode == 'smoke')
    if args.mode == 'clean-2000' and report['selected_epoch'] != 2:
        raise ComparisonIntegrityError('final clean evaluation requires epoch 2')
    revision = clean.git_revision(ROOT)
    if manifest['project_git_revision'] != revision or manifest['acceptance_rule_digest'] != freeze['digest']:
        raise ComparisonIntegrityError('training/evaluation freeze changed')
    checkpoint = FullCheckpointIdentity(
        checkpoint_id=f"eventground-wam/{recipe['recipe_id']}",
        checkpoint_revision=report['checkpoint_sha256'], training_manifest_digest=manifest['digest'],
        training_report_digest=report['digest'], selected_epoch=report['selected_epoch'],
        selected_step=report['global_step'])
    versions = clean.runtime_versions()
    if versions['transformers'] != '5.17.0':
        raise ComparisonIntegrityError('fresh policy runtime changed')
    identity = seal_contract({
        'artifact_type': 'eventground_wam.fresh_v2_clean_run', 'policy': 'b0',
        'mode': args.mode, 'pair_count': len(pairs), 'project_git_revision': revision,
        'acceptance_rule_digest': rule['digest'], 'fresh_acceptance_rule_digest': freeze['digest'],
        'panel_digest': panel['digest'], 'protocol_sha256': protocol_sha256(protocol),
        'checkpoint_id': checkpoint.checkpoint_id, 'checkpoint_revision': checkpoint.checkpoint_revision,
        'runtime_lock_sha256': sha256_file(ROOT / 'environments/b0/uv.lock'),
        'runtime_versions': versions, 'checkpoint_evidence': checkpoint.to_dict(),
    })
    root = args.training_root / f'rollouts-{args.mode}'
    if (root / 'identity.json').exists() and read_sealed(root / 'identity.json') != identity:
        raise ComparisonIntegrityError('fresh clean resume identity changed')
    atomic_json(root / 'identity.json', identity)
    if args.mode == 'clean-2000':
        reuse = read_sealed(ROOT / 'configs/evaluation/week4_oft_clean_reuse_v1.json')
        _, cached = audit_cached_comparison(args.previous_root, reuse, full)
        controls = {'previous_b0': cached['b0'], 'openvla': cached['reference']}
        parent = read_sealed(args.parent_root / 'comparison-report.json')
        expected_parent = read_sealed(ROOT / 'configs/training/week4_b0_oft_all_l1_v2.json')
        if parent['digest'] != expected_parent['sources']['clean_completion']['digest']:
            raise ComparisonIntegrityError('parent clean comparison changed')
        controls['new_b0'] = [verify_cached_record(
            args.parent_root / 'b0', i, p, parent['comparisons']['new_vs_openvla']['pairs'][i],
            'b0', parent['new_policy_identity']) for i, p in enumerate(pairs)]
    else:
        config = read_sealed(ROOT / 'configs/training/week4_b0_oft_all_l1_v2.json')
        controls = cached_controls(args, config, full, pairs)
    install_codec_independent_motion_blur()
    from libero.libero.envs import OffScreenRenderEnv
    policy = None
    records = []
    for ordinal, pair in enumerate(pairs):
        if budget.seconds >= 48 * 3600:
            raise ComparisonIntegrityError('evaluation active budget reached')
        check_space(args.work_root, recipe)
        paths = clean.bddl_for(pair, suites)
        candidates = [paths_for_pair(root, ordinal, pair, i) for i in range(2)]
        if policy is None and not (any(a.exists() for a, _, _ in candidates)
                                  or candidates[0][2].exists()
                                  or all(j.exists() for _, j, _ in candidates)):
            policy = clean.load_policy(args, (manifest, checkpoint))
        resolution = protocol['observation']['capture_resolution']
        record = execute_pair(
            root=root, ordinal=ordinal, pair=pair, identity=identity, protocol=protocol,
            metadata=metadata_for_pair(pair, protocol, identity, paths.source),
            initial_state=states[(pair['suite'], pair['runtime_task_index'])][pair['initial_state_index']],
            policy=policy, environment_factory=lambda paths=paths, resolution=resolution: OffScreenRenderEnv(
                bddl_file_name=str(paths.runtime), camera_heights=resolution[0],
                camera_widths=resolution[1], control_freq=protocol['simulation']['control_frequency_hz']),
            episode_runner=clean.run_episode, seed_episode=clean.seed_episode,
            expected_input=controls['openvla'][ordinal]['first_query_input_hashes'])
        records.append(record)
        progress = {'mode': args.mode, 'pair_count': len(pairs), 'updated_at_utc': utc_now(),
                    **clean.summarize(records)}
        atomic_json(root / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    records = clean.checked_receipts(root, pairs, identity, protocol, suites)
    checks = {name: [compare_pair_receipts(a, b) for a, b in zip(records, old, strict=True)]
              for name, old in controls.items()}
    valid = all(r['assessable'] for rows in checks.values() for r in rows)
    counts = {name: sum(r['success'] for r in rows) for name, rows in controls.items()}
    counts['fresh_v2'] = sum(r['success'] for r in records)
    result = seal_contract({
        'artifact_type': 'eventground_wam.fresh_v2_clean_report', 'status': 'complete',
        'identity_digest': identity['digest'], 'checkpoint_sha256': report['checkpoint_sha256'],
        'all_pairs_valid': valid, 'receipt_digests': [r['digest'] for r in records],
        'counts': counts, 'comparisons': checks, **clean.summarize(records),
        'historical_clean_statistics': {name: comparison_statistics(records, old, gate=name == 'openvla')
                                        for name, old in controls.items()} if args.mode == 'clean-2000' and valid else None,
        'acceptance': 'Development only' if args.mode != 'clean-2000' else 'Await matched dev-210 and v2 readiness adjudication',
    })
    atomic_json(root / 'evaluation-report.json', result)
    if not valid:
        raise ComparisonIntegrityError('fresh paired artifact validation failed')


def main():
    home = Path.home()
    data = home / 'data/eventground-wam'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, choices=(1, 2), default=2)
    parser.add_argument('--mode', choices=('smoke', 'development', 'clean-2000'), required=True)
    args = parser.parse_args()
    args.policy = 'b0'
    args.training_root = args.work_root / ('pilot' if args.mode == 'smoke' else f'epochs/epoch-{args.epoch:02d}')
    args.normalization = ROOT / 'docs/model/week4-fresh-v2/action-statistics.json'
    args.cache_dir = data / 'huggingface-cache'
    args.libero_root = home / 'code/vendor/LIBERO-plus'
    args.previous_root = data / 'comparisons/week4-clean-comparison-v1'
    args.parent_root = data / 'comparisons/week4-oft-clean-comparison-v1'
    try:
        with (args.work_root / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recipe = load_fresh_recipe(RECIPE)
            budget = ActiveBudget(args.work_root / 'evaluation-budget.json', recipe['digest'])
            try:
                run(args, recipe, budget)
            finally:
                budget.close()
    except (TrainingContractError, ComparisonIntegrityError, ValueError, BlockingIOError) as error:
        print(f'Integrity stop: {error}', file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
