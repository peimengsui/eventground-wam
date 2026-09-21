#!/usr/bin/env python3
"""Run the fixed OFT B0 on clean-2000 and compare with audited cached policies."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from evaluate_clean_comparison import (
    ROOT,
    bddl_for,
    checked_receipts,
    inventory,
    load_policy,
    policy_identity,
    run_episode,
    seed_episode,
    summarize,
)

from eventground_wam.clean_acceptance import read_sealed
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
    bind_reuse_identity,
    comparison_statistics,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import protocol_sha256
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

CONFIG = ROOT / 'configs/evaluation/week4_oft_clean_reuse_v1.json'
CONFIG_DIGEST = '588a19bab7f01ea7e9be6eb0327f7930c484193c63a197276b019e6fb5f8184a'


def emit(value):
    print(json.dumps(value), flush=True)


def run(args):
    parent, panel, protocol, suites, states = inventory(args)
    config = read_sealed(CONFIG)
    if (config['digest'] != CONFIG_DIGEST or config['parent_rule_digest'] != parent['digest']
            or config['panel'] != parent['panel'] or config['analysis'] != parent['analysis']
            or config['decision'] != parent['decision']
            or config['protocol_sha256'] != protocol_sha256(protocol)):
        raise ComparisonIntegrityError('reuse freeze or inherited scientific rules changed')
    if args.reuse_root == args.output_root or args.reuse_root in args.output_root.parents:
        raise ComparisonIntegrityError('new outputs must be separate from original evidence')
    identity, bundle = policy_identity(args, config, panel, protocol)
    identity = bind_reuse_identity(identity, config)
    if bundle[1].selected_step != config['policies']['b0']['selected_step']:
        raise ComparisonIntegrityError('selected training update changed')
    # Also pin the full B0 runtime to the previously verified evaluator runtime.
    original = read_sealed(ROOT / 'docs/evaluation/week4-clean-comparison-final-v1.json')
    if identity['runtime_versions'] != original['policy_identities']['b0']['runtime_versions']:
        raise ComparisonIntegrityError('new B0 evaluator runtime differs from prior B0')
    root = args.output_root / 'b0'
    saved = root / 'identity.json'
    if saved.exists() and read_sealed(saved) != identity:
        raise ComparisonIntegrityError('resume source, runtime or checkpoint identity changed')
    if args.preflight_only:
        emit({'stage': 'preflight-passed', 'identity': identity,
              'cached_artifact_audit': 'required at execution before policy loading'})
        return
    atomic_json(saved, identity)
    emit({'stage': 'auditing-cached-artifacts', 'root': str(args.reuse_root)})
    original, cached = audit_cached_comparison(
        args.reuse_root, config, panel,
        on_progress=lambda policy, count: emit({'stage': 'cached-byte-audit',
                                                'policy': policy, 'verified': count}))
    atomic_json(args.output_root / 'reuse-verification.json', seal_contract({
        'artifact_type': 'eventground_wam.cached_clean_verification',
        'verified_at_utc': utc_now(), 'reuse_config_digest': config['digest'],
        'original_comparison_report_digest': original['digest'],
        'verified_artifacts': 4000, 'verified_initial_input_pairs': 2000,
        'original_policy_identities': original['policy_identities'],
        'method': 'All receipt, artifact and attempt hashes match the original verified comparison.'}))
    pairs = panel['pairs'][:args.pilot_pairs] if args.pilot_pairs else panel['pairs']
    install_codec_independent_motion_blur()
    from libero.libero.envs import OffScreenRenderEnv

    policy = None
    records = []
    for ordinal, pair in enumerate(pairs):
        paths = bddl_for(pair, suites)
        candidates = [paths_for_pair(root, ordinal, pair, attempt) for attempt in range(2)]
        has_artifact = any(a.exists() for a, _, _ in candidates)
        has_receipt = candidates[0][2].exists()
        exhausted = all(j.exists() for _, j, _ in candidates)
        if policy is None and not (has_artifact or has_receipt or exhausted):
            emit({'stage': 'loading-policy', 'policy': 'new-b0'})
            policy = load_policy(args, bundle)
        resolution = protocol['observation']['capture_resolution']
        record = execute_pair(
            root=root, ordinal=ordinal, pair=pair, identity=identity, protocol=protocol,
            metadata=metadata_for_pair(pair, protocol, identity, paths.source),
            initial_state=states[(pair['suite'], pair['runtime_task_index'])][
                pair['initial_state_index']], policy=policy,
            environment_factory=lambda paths=paths, resolution=resolution: OffScreenRenderEnv(
                bddl_file_name=str(paths.runtime), camera_heights=resolution[0],
                camera_widths=resolution[1],
                control_freq=protocol['simulation']['control_frequency_hz']),
            episode_runner=run_episode, seed_episode=seed_episode,
            expected_input=cached['reference'][ordinal]['first_query_input_hashes'])
        records.append(record)
        progress = {'policy': 'new-b0', 'mode': identity['mode'], 'pair_count': len(pairs),
                    'updated_at_utc': utc_now(), 'last_pair': pair['pair_id'], **summarize(records),
                    'cached_same_prefix_successes': {
                        p: sum(r['success'] for r in rows[:ordinal + 1])
                        for p, rows in cached.items()}}
        atomic_json(root / 'progress.json', progress)
        emit(progress)
        if not all(compare_pair_receipts(record, cached[p][ordinal])['assessable']
                   for p in ('b0', 'reference')):
            raise ComparisonIntegrityError('new pair is unassessable or initial inputs differ')
    policy_report = seal_contract({
        'artifact_type': 'eventground_wam.clean_comparison_policy_report', 'status': 'complete',
        'identity_digest': identity['digest'],
        'receipt_digests': [r['digest'] for r in records], **summarize(records)})
    report_path = root / 'evaluation-report.json'
    if report_path.exists() and read_sealed(report_path) != policy_report:
        raise ComparisonIntegrityError('completed new B0 policy report changed')
    atomic_json(report_path, policy_report)
    # Re-open and fully validate all new HDF5 recordings before granting acceptance.
    records = checked_receipts(root, pairs, identity, protocol, suites)
    comparisons = {}
    for name, old_policy in [('new_vs_openvla', 'reference'), ('new_vs_previous_b0', 'b0')]:
        checks = [compare_pair_receipts(a, b) for a, b in zip(
            records, cached[old_policy][:len(pairs)], strict=True)]
        comparisons[name] = {
            'all_pairs_valid': all(p['assessable'] for p in checks), 'pairs': checks,
            'statistics': None if args.pilot_pairs else comparison_statistics(
                records, cached[old_policy], gate=old_policy == 'reference'),
            'outcome_contingency': dict(Counter(
                f"new_{int(p['b0_success'])}_comparator_{int(p['reference_success'])}"
                for p in checks))}
    tasks = {}
    for ordinal, pair in enumerate(pairs):
        key = f"{pair['suite']}/task-{pair['runtime_task_index']:02d}"
        row = tasks.setdefault(key, {'count': 0, 'new_b0': 0, 'previous_b0': 0, 'openvla': 0})
        row['count'] += 1
        row['new_b0'] += records[ordinal]['success']
        row['previous_b0'] += cached['b0'][ordinal]['success']
        row['openvla'] += cached['reference'][ordinal]['success']
    valid = all(c['all_pairs_valid'] for c in comparisons.values())
    decision = 'NOT_ASSESSED' if args.pilot_pairs else (
        comparisons['new_vs_openvla']['statistics']['statistical_decision']
        if valid else 'NOT_ASSESSABLE')
    final = seal_contract({
        'artifact_type': 'eventground_wam.clean_reused_three_policy_comparison',
        'status': 'complete', 'mode': identity['mode'], 'completed_at_utc': utc_now(),
        'comparison_id': config['comparison_id'], 'reuse_config_digest': config['digest'],
        'parent_acceptance_rule_digest': parent['digest'], 'panel_digest': panel['digest'],
        'protocol_sha256': protocol_sha256(protocol), 'new_policy_identity': identity,
        'reused_comparison_report_digest': original['digest'],
        'reused_policy_identities': original['policy_identities'],
        'new_policy_report_digest': policy_report['digest'],
        'pair_count': len(pairs), 'all_pairs_valid': valid, 'acceptance_decision': decision,
        'policy_summaries': {'new_b0': summarize(records),
                            'previous_b0': summarize(cached['b0'][:len(pairs)]),
                            'openvla': summarize(cached['reference'][:len(pairs)])},
        'comparisons': comparisons, 'per_task': tasks, 'interpretation': config['interpretation']})
    atomic_json(args.output_root / 'comparison-report.json', final)
    emit({'stage': 'comparison-complete', 'decision': decision, 'all_pairs_valid': valid,
          'policy_summaries': final['policy_summaries']})
    if not valid:
        raise ComparisonIntegrityError('final comparison failed scientific validity')


def main():
    home = Path.home()
    data = home / 'data/eventground-wam'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pilot-pairs', type=int, choices=(2,))
    parser.add_argument('--preflight-only', action='store_true')
    for name, default in {
        'reuse-root': data / 'comparisons/week4-clean-comparison-v1',
        'training-root': data / 'libero-plus/derived/b0/week4-b0-oft-v1',
        'normalization': data / 'libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json',
        'cache-dir': data / 'huggingface-cache',
        'libero-root': home / 'code/vendor/LIBERO-plus',
    }.items():
        parser.add_argument(f'--{name}', type=Path, default=default)
    args = parser.parse_args()
    args.policy = 'b0'
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    try:
        run(args)
    except (ComparisonIntegrityError, ValueError) as error:
        atomic_json(args.output_root / 'integrity-error.json', {
            'at_utc': utc_now(), 'exception': f'{type(error).__name__}: {error}'})
        print(f'Integrity stop: {error}', file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
