#!/usr/bin/env python3
"""Run the frozen paired panel, with durable per-episode resume and final validation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from check_reference_policy import verify_checkpoint, verify_evaluator, verify_libero
from evaluate_reference_policy import git_revision, run_episode
from validate_clean_acceptance import validate_freeze

from eventground_wam.clean_acceptance import analyze_paired_outcomes, read_sealed
from eventground_wam.clean_comparison import (
    ComparisonIntegrityError,
    atomic_json,
    compare_pair_receipts,
    execute_pair,
    load_pair_receipt,
    metadata_for_pair,
    paths_for_pair,
    utc_now,
    verify_receipt,
)
from eventground_wam.clean_evaluation import clean_manifest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import sha256_array, sha256_file
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.libero_bddl import resolve_libero_bddl_paths
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

ROOT = Path(__file__).resolve().parents[1]


def seed_episode(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inventory(args: Any) -> tuple[Any, Any, Any, Any, Any]:
    validate_freeze(ROOT)
    rule = read_sealed(ROOT / 'configs/evaluation/week4_clean_acceptance_v1.json')
    panel = read_sealed(ROOT / rule['panel']['path'])
    protocol = load_evaluation_protocol(ROOT / 'configs/evaluation/week4_clean_comparison_v1.json')
    verify_libero(args.libero_root)
    old = load_evaluation_protocol(ROOT / rule['rollout_contract']['inherited_profile_path'])
    manifest, suites = clean_manifest(args.libero_root, old)
    if manifest != read_sealed(ROOT / 'evaluation/splits/libero-clean-v1/clean-40.json'):
        raise ComparisonIntegrityError('canonical clean tasks changed')
    states = {key: np.asarray(suite.get_task_init_states(key[1]))
              for key, suite in suites.items()}
    for pair in panel['pairs']:
        values = states[(pair['suite'], pair['runtime_task_index'])]
        if len(values) != 50 or sha256_array(values[pair['initial_state_index']]) != (
            pair['initial_state_sha256']
        ):
            raise ComparisonIntegrityError('frozen initial states changed')
    return rule, panel, protocol, suites, states


def runtime_versions() -> dict[str, str]:
    values = {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'numpy', 'mujoco', 'robosuite', 'h5py', 'Pillow')}
    values['python'] = platform.python_version()
    for key, expected in {'numpy': '1.26.4', 'mujoco': '2.3.7', 'robosuite': '1.4.0'}.items():
        if values[key] != expected:
            raise ComparisonIntegrityError(f'pinned runtime changed: {key}')
    return values


def policy_identity(args: Any, rule: Any, panel: Any, protocol: Any) -> tuple[Any, Any]:
    pinned = rule['policies'][args.policy]
    training_bundle = None
    if args.policy == 'b0':
        from evaluate_b0_full import verify_training_bundle
        training, report, checkpoint = verify_training_bundle(args.training_root, args.normalization)
        if (checkpoint.checkpoint_revision != pinned['checkpoint_revision']
                or training['digest'] != pinned['training_manifest_digest']
                or report['digest'] != pinned['training_report_digest']
                or checkpoint.selected_epoch != pinned['selected_epoch']):
            raise ComparisonIntegrityError('B0 selection changed')
        evidence = checkpoint.to_dict()
        training_bundle = (training, checkpoint)
    else:
        verify_evaluator(args.evaluator_root)
        evidence = verify_checkpoint(args.checkpoint_dir, full_hash=True)
        if evidence['revision'] != pinned['checkpoint_revision']:
            raise ComparisonIntegrityError('reference selection changed')
    versions = runtime_versions()
    expected_transformers = '5.17.0' if args.policy == 'b0' else '4.40.1'
    if versions['transformers'] != expected_transformers:
        raise ComparisonIntegrityError('policy Transformers runtime changed')
    identity = seal_contract({
        'artifact_type': 'eventground_wam.clean_comparison_run', 'schema_version': 1,
        'policy': args.policy, 'mode': 'pilot' if args.pilot_pairs else 'formal',
        'pair_count': args.pilot_pairs or panel['pair_count'],
        'project_git_revision': git_revision(ROOT), 'acceptance_rule_digest': rule['digest'],
        'panel_digest': panel['digest'], 'protocol_sha256': protocol_sha256(protocol),
        'checkpoint_id': pinned['checkpoint_id'], 'checkpoint_revision': pinned['checkpoint_revision'],
        'runtime_lock_sha256': sha256_file(ROOT / pinned['runtime_lock_path']),
        'runtime_versions': versions, 'checkpoint_evidence': evidence,
    })
    return identity, training_bundle


def load_policy(args: Any, bundle: Any) -> Any:
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 5090':
        raise ComparisonIntegrityError('frozen RTX 5090 runtime unavailable')
    torch.cuda.set_device(0)
    if args.policy == 'reference':
        from evaluate_reference_policy import OpenVLAOFTPolicy
        return OpenVLAOFTPolicy(args.checkpoint_dir)
    from train_b0 import build_policy

    from eventground_wam.b0_data import load_b0_action_normalizer
    from eventground_wam.b0_inference import B0InferencePolicy
    from eventground_wam.b0_model import (
        QWEN_REVISION,
        QwenB0InputBuilder,
        load_b0_trainable_state_dict,
        load_pinned_qwen_processor,
    )
    from eventground_wam.b0_training import POLICY_FORMAT
    training, checkpoint = bundle
    device = torch.device('cuda:0')
    model, names = build_policy(training['recipe'], str(args.cache_dir), device)
    state = torch.load(args.training_root / 'B0.ckpt', map_location='cpu', weights_only=True)
    if (state['format'] != POLICY_FORMAT or state['identity_digest'] != training['digest']
            or state['qwen_revision'] != QWEN_REVISION):
        raise ComparisonIntegrityError('B0 tensor envelope changed')
    load_b0_trainable_state_dict(model, state['state_dict'], allowed_exact_names=names)
    input_builder = QwenB0InputBuilder
    if training['recipe'].get('policy_family') == 'qwen35-oft-b0-v1':
        from eventground_wam.b0_oft import OFTInputBuilder

        input_builder = OFTInputBuilder
    return B0InferencePolicy(
        policy=model, input_builder=input_builder(load_pinned_qwen_processor(
            cache_dir=str(args.cache_dir), local_files_only=True)),
        normalizer=load_b0_action_normalizer(args.normalization), device=device, checkpoint=checkpoint)


def bddl_for(pair: Any, suites: Any) -> Any:
    suite = suites[(pair['suite'], pair['runtime_task_index'])]
    return resolve_libero_bddl_paths(suite.get_task_bddl_file_path(pair['runtime_task_index']))


def checked_receipts(root: Path, pairs: Any, identity: Any, protocol: Any, suites: Any) -> list[Any]:
    records = []
    for ordinal, pair in enumerate(pairs):
        receipt = load_pair_receipt(root, ordinal, pair)
        records.append(verify_receipt(
            receipt, root=root, ordinal=ordinal, pair=pair, identity=identity, protocol=protocol,
            metadata=metadata_for_pair(pair, protocol, identity, bddl_for(pair, suites).source)))
    return records


def verify_saved_identity(identity: Any, policy: str, args: Any, rule: Any,
                          panel: Any, protocol: Any) -> None:
    expected = {'policy': policy, 'mode': 'pilot' if args.pilot_pairs else 'formal',
                'pair_count': args.pilot_pairs or 2000,
                'project_git_revision': git_revision(ROOT), 'acceptance_rule_digest': rule['digest'],
                'panel_digest': panel['digest'], 'protocol_sha256': protocol_sha256(protocol)}
    for key in ('checkpoint_id', 'checkpoint_revision', 'runtime_lock_sha256'):
        expected[key] = rule['policies'][policy][key]
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ComparisonIntegrityError('saved run identity differs from frozen comparison')


def summarize(records: Any) -> dict[str, Any]:
    return {'completed': len(records), 'successes': sum(r['success'] for r in records),
            'unassessable': sum(not r['assessable'] for r in records),
            'infrastructure_retries': sum(r['attempt_index'] for r in records),
            'termination_reasons': dict(Counter(r['termination_reason'] for r in records))}


def analyze(args: Any, rule: Any, panel: Any, protocol: Any, suites: Any, pairs: Any) -> None:
    runtime_versions()
    all_records = {}
    identities = {}
    for policy in ('b0', 'reference'):
        root = args.output_root / policy
        identities[policy] = read_sealed(root / 'identity.json')
        verify_saved_identity(identities[policy], policy, args, rule, panel, protocol)
        all_records[policy] = checked_receipts(root, pairs, identities[policy], protocol, suites)
        report = read_sealed(root / 'evaluation-report.json')
        if (report['identity_digest'] != identities[policy]['digest'] or report['receipt_digests'] !=
                [r['digest'] for r in all_records[policy]]):
            raise ComparisonIntegrityError('policy final report changed')
    checks = [compare_pair_receipts(a, b) for a, b in zip(
        all_records['b0'], all_records['reference'], strict=True)]
    valid = all(p['assessable'] for p in checks)
    statistics = None
    decision = 'NOT_ASSESSED' if args.pilot_pairs else 'NOT_ASSESSABLE'
    if valid and not args.pilot_pairs:
        statistics = analyze_paired_outcomes(*[
            np.array([r['success'] for r in all_records[p]], dtype=bool).reshape(4, 10, 50)
            for p in ('b0', 'reference')])
        decision = statistics['statistical_decision']
    tasks = {}
    for pair, check in zip(pairs, checks, strict=True):
        key = f"{pair['suite']}/task-{pair['runtime_task_index']:02d}"
        counts = tasks.setdefault(key, {'count': 0, 'b0_success': 0, 'reference_success': 0})
        counts['count'] += 1
        counts['b0_success'] += check['b0_success']
        counts['reference_success'] += check['reference_success']
    report = seal_contract({
        'artifact_type': 'eventground_wam.clean_paired_comparison', 'status': 'complete',
        'mode': 'pilot' if args.pilot_pairs else 'formal', 'completed_at_utc': utc_now(),
        'acceptance_rule_digest': rule['digest'], 'panel_digest': panel['digest'],
        'project_git_revision': git_revision(ROOT), 'policy_identities': identities,
        'pair_count': len(checks), 'all_pairs_valid': valid, 'acceptance_decision': decision,
        'statistics': statistics, 'per_task': tasks, 'pairs': checks,
        'outcome_contingency': dict(Counter(
            f"b0_{int(p['b0_success'])}_reference_{int(p['reference_success'])}" for p in checks)),
        'policy_summaries': {p: summarize(records) for p, records in all_records.items()},
    })
    atomic_json(args.output_root / 'comparison-report.json', report)
    print(json.dumps({'stage': 'comparison-complete', 'decision': decision,
                      'all_pairs_valid': valid, 'statistics': statistics}), flush=True)
    if not valid:
        raise ComparisonIntegrityError('paired artifacts failed scientific validity checks')


def run(args: Any) -> None:
    rule, panel, protocol, suites, states = inventory(args)
    pairs = panel['pairs'][:args.pilot_pairs] if args.pilot_pairs else panel['pairs']
    if args.policy == 'analyze':
        analyze(args, rule, panel, protocol, suites, pairs)
        return
    identity, bundle = policy_identity(args, rule, panel, protocol)
    root = args.output_root / args.policy
    saved = root / 'identity.json'
    if saved.exists() and read_sealed(saved) != identity:
        raise ComparisonIntegrityError('resume source, runtime, or checkpoint identity changed')
    if args.preflight_only:
        print(json.dumps({'status': 'preflight-passed', 'identity': identity}), flush=True)
        return
    atomic_json(saved, identity)
    if (root / 'evaluation-report.json').exists():
        records = checked_receipts(root, pairs, identity, protocol, suites)
        report = read_sealed(root / 'evaluation-report.json')
        if (report['identity_digest'] != identity['digest'] or
                report['receipt_digests'] != [r['digest'] for r in records]):
            raise ComparisonIntegrityError('completed report changed')
        print(json.dumps({'policy': args.policy, 'status': 'already-complete', **summarize(records)}),
              flush=True)
        return
    expected = None
    if args.policy == 'reference':
        left = args.output_root / 'b0'
        left_identity = read_sealed(left / 'identity.json')
        verify_saved_identity(left_identity, 'b0', args, rule, panel, protocol)
        expected = checked_receipts(left, pairs, left_identity, protocol, suites)
        report = read_sealed(left / 'evaluation-report.json')
        if (report['identity_digest'] != left_identity['digest'] or
                report['receipt_digests'] != [r['digest'] for r in expected]):
            raise ComparisonIntegrityError('B0 phase has not completed correctly')
    install_codec_independent_motion_blur()
    from libero.libero.envs import OffScreenRenderEnv
    policy = None
    records = []
    for ordinal, pair in enumerate(pairs):
        paths = bddl_for(pair, suites)
        metadata = metadata_for_pair(pair, protocol, identity, paths.source)
        # Completed artifacts are recovered without loading a model if all work was published.
        candidates = [paths_for_pair(root, ordinal, pair, attempt) for attempt in range(2)]
        has_artifact = any(a.exists() for a, _, _ in candidates)
        has_receipt = candidates[0][2].exists()
        exhausted = all(j.exists() for _, j, _ in candidates)
        if policy is None and not (has_artifact or has_receipt or exhausted):
            print(json.dumps({'stage': 'loading-policy', 'policy': args.policy}), flush=True)
            policy = load_policy(args, bundle)
        resolution = protocol['observation']['capture_resolution']
        record = execute_pair(
            root=root, ordinal=ordinal, pair=pair, identity=identity, protocol=protocol,
            metadata=metadata, initial_state=states[(pair['suite'], pair['runtime_task_index'])][
                pair['initial_state_index']], policy=policy,
            environment_factory=lambda paths=paths, resolution=resolution: OffScreenRenderEnv(
                bddl_file_name=str(paths.runtime), camera_heights=resolution[0],
                camera_widths=resolution[1], control_freq=protocol['simulation']['control_frequency_hz']),
            episode_runner=run_episode, seed_episode=seed_episode,
            expected_input=expected[ordinal]['first_query_input_hashes'] if expected else None)
        records.append(record)
        progress = {'policy': args.policy, 'mode': identity['mode'], 'pair_count': len(pairs),
                    'updated_at_utc': utc_now(), 'last_pair': pair['pair_id'], **summarize(records)}
        atomic_json(root / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    atomic_json(root / 'evaluation-report.json', seal_contract({
        'artifact_type': 'eventground_wam.clean_comparison_policy_report', 'status': 'complete',
        'identity_digest': identity['digest'], 'completed_at_utc': utc_now(),
        'receipt_digests': [r['digest'] for r in records], **summarize(records)}))


def main() -> None:
    home = Path.home()
    data = home / 'data/eventground-wam'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('b0', 'reference', 'analyze'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pilot-pairs', type=int, choices=(2,))
    parser.add_argument('--preflight-only', action='store_true')
    defaults = {
        'training-root': data / 'libero-plus/derived/b0/week4-b0-5epoch-v2',
        'normalization': data / 'libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json',
        'cache-dir': data / 'huggingface-cache', 'libero-root': home / 'code/vendor/LIBERO-plus',
        'evaluator-root': home / 'code/vendor/openvla-oft',
        'checkpoint-dir': data / 'reference-policy/checkpoints/openvla-7b-oft-libero-plus-a85655ec'}
    for name, value in defaults.items():
        parser.add_argument(f'--{name}', type=Path, default=value)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    try:
        run(args)
    except (ComparisonIntegrityError, ValueError) as error:
        atomic_json(args.output_root / f'{args.policy}-integrity-error.json', {
            'at_utc': utc_now(), 'exception': f'{type(error).__name__}: {error}'})
        print(f'Integrity stop: {error}', file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
