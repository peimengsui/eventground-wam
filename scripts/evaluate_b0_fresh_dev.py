#!/usr/bin/env python3
"""Run either fresh B0 or the pinned OpenVLA on the identical frozen dev-210."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from check_reference_policy import (
    MODEL_ID,
    MODEL_REVISION,
    verify_checkpoint,
    verify_evaluator,
    verify_libero,
)
from evaluate_b0_full import verify_training_bundle
from evaluate_clean_comparison import load_policy, runtime_versions, seed_episode
from evaluate_reference_policy import OpenVLAOFTPolicy, evaluate_task, git_revision, load_manifest

from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_fresh import load_fresh_recipe
from eventground_wam.b0_training import TrainingContractError, atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

ROOT = Path(__file__).resolve().parents[1]


def run(args, budget):
    recipe = load_fresh_recipe(ROOT / 'configs/training/week4_b0_fresh_all_l1_v2.json')
    protocol = load_evaluation_protocol()
    manifest = load_manifest(ROOT / 'evaluation/splits/libero-plus-v1/dev-210.json', protocol)
    if (manifest['task_count'] != 210 or manifest['task_records_sha256'] != recipe['dev_task_records_sha256']
            or protocol_sha256(protocol) != recipe['evaluation_protocol_sha256']):
        raise TrainingContractError('dev-210 identity changed')
    verify_libero(args.libero_root)
    revision = git_revision(ROOT)
    versions = runtime_versions()
    if versions['transformers'] != ('5.17.0' if args.policy == 'b0' else '4.40.1'):
        raise TrainingContractError('policy runtime mismatch')
    bundle = None
    if args.policy == 'b0':
        training, _, checkpoint = verify_training_bundle(args.training_root, args.normalization)
        bundle = (training, checkpoint)
        checkpoint_id, checkpoint_revision = checkpoint.checkpoint_id, checkpoint.checkpoint_revision
    else:
        verify_evaluator(args.evaluator_root)
        verify_checkpoint(args.checkpoint_dir, full_hash=True)
        checkpoint_id, checkpoint_revision = MODEL_ID, MODEL_REVISION
    lock_path = 'environments/b0/uv.lock' if args.policy == 'b0' else 'environments/reference-policy/uv.lock'
    identity = seal_contract({
        'artifact_type': 'eventground_wam.fresh_v2_dev_run', 'policy': args.policy,
        'project_git_revision': revision, 'checkpoint_id': checkpoint_id,
        'checkpoint_revision': checkpoint_revision, 'protocol_sha256': protocol_sha256(protocol),
        'task_records_sha256': manifest['task_records_sha256'], 'runtime_versions': versions,
        'runtime_lock_sha256': sha256_file(ROOT / lock_path),
        'acceptance_rule_digest': read_sealed(ROOT / 'configs/evaluation/week4_fresh_v2_acceptance.json')['digest'],
    })
    root = args.work_root / 'dev-210' / args.policy
    if (root / 'identity.json').exists() and read_sealed(root / 'identity.json') != identity:
        raise TrainingContractError('dev resume identity changed')
    atomic_json(root / 'identity.json', identity)
    atomic_json(root / 'task-manifest.json', manifest)
    install_codec_independent_motion_blur()
    policy = load_policy(args, bundle) if args.policy == 'b0' else OpenVLAOFTPolicy(args.checkpoint_dir)
    from libero.libero import benchmark
    suites = {}
    records = []
    for ordinal, task in enumerate(manifest['tasks']):
        if budget.seconds >= 48 * 3600:
            raise TrainingContractError('evaluation active budget reached')
        if (shutil.disk_usage(args.work_root).free < recipe['minimum_free_bytes']
                or sum(p.stat().st_size for p in args.work_root.rglob('*') if p.is_file()) > recipe['output_limit_bytes']):
            raise TrainingContractError('evaluation disk budget reached')
        if task['suite'] not in suites:
            suites[task['suite']] = benchmark.get_benchmark_dict()[task['suite']]()
        seed_episode(task['episode_seed'])
        result = evaluate_task(
            ordinal=ordinal, manifest_task=task, suite=suites[task['suite']], policy=policy,
            protocol=protocol, output_dir=root, project_revision=revision, resume=True,
            checkpoint_id=checkpoint_id, checkpoint_revision=checkpoint_revision)
        records.append(asdict(result))
        progress = {'completed': len(records), 'task_count': 210,
                    'successes': sum(r['success'] for r in records)}
        atomic_json(root / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    breakdowns = {}
    for field in ('suite', 'perturbation_category', 'difficulty_level'):
        groups = defaultdict(list)
        for row in records:
            groups[str(row[field])].append(row)
        breakdowns[field] = {key: {'count': len(rows), 'successes': sum(r['success'] for r in rows)}
                             for key, rows in groups.items()}
    atomic_json(root / 'evaluation-report.json', seal_contract({
        'artifact_type': 'eventground_wam.fresh_v2_dev_report', 'status': 'complete',
        'identity_digest': identity['digest'], 'episode_count': len(records),
        'success_count': sum(r['success'] for r in records), 'episodes': records,
        'breakdowns': breakdowns, 'all_artifacts_present': all(r['artifact'] for r in records),
        'paired_input_verification': 'required in final analysis',
    }))


def main():
    home = Path.home()
    data = home / 'data/eventground-wam'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--policy', choices=('b0', 'reference'), required=True)
    args = parser.parse_args()
    args.training_root = args.work_root / 'epochs/epoch-02'
    args.normalization = ROOT / 'docs/model/week4-fresh-v2/action-statistics.json'
    args.cache_dir = data / 'huggingface-cache'
    args.libero_root = home / 'code/vendor/LIBERO-plus'
    args.evaluator_root = home / 'code/vendor/openvla-oft'
    args.checkpoint_dir = data / 'reference-policy/checkpoints/openvla-7b-oft-libero-plus-a85655ec'
    try:
        with (args.work_root / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recipe = load_fresh_recipe(ROOT / 'configs/training/week4_b0_fresh_all_l1_v2.json')
            budget = ActiveBudget(args.work_root / 'evaluation-budget.json', recipe['digest'])
            try:
                run(args, budget)
            finally:
                budget.close()
    except (TrainingContractError, ValueError, BlockingIOError) as error:
        print(f'Integrity stop: {error}', file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
