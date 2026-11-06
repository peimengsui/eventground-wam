#!/usr/bin/env python3
"""Prepare, run, or analyze the fixed M1 endpoint; preflight never loads GPU models."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from eventground_wam.auxiliary_evaluation import (
    artifact_inputs,
    load_contract,
    paired_summary,
    require,
    verify_endpoint,
    verify_envelope,
    verify_training_identity,
)
from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256

ROOT = Path(__file__).resolve().parents[1]


def immutable(path, value):
    if path.exists():
        require(read_sealed(path) == value, f'immutable artifact changed: {path}')
    else:
        atomic_json(path, value)


def package_check():
    package = read_sealed(ROOT / 'evaluation-source-manifest.json')
    for name, digest in package['files'].items():
        require(sha256_file(ROOT / name) == digest, f'evaluation source changed: {name}')
    return package


def preflight(args, contract):
    from check_reference_policy import verify_libero
    from evaluate_clean_comparison import runtime_versions
    from evaluate_reference_policy import load_manifest

    package = package_check()
    training = verify_training_identity(args.training_root, contract)
    source = read_sealed(args.training_source / 'production-source-manifest.json')
    require(source['digest'] == contract['production_source_digest'], 'training source manifest changed')
    for name, digest in source['files'].items():
        require(sha256_file(args.training_source / name) == digest, f'training source changed: {name}')
    versions = runtime_versions()
    require(versions == contract['runtime_versions'], 'evaluation runtime differs from frozen B0')
    libero = verify_libero(args.libero_root)
    protocol = load_evaluation_protocol(ROOT / 'configs/evaluation/libero_plus_v1.json')
    tasks = load_manifest(ROOT / contract['panel'], protocol)
    require(protocol_sha256(protocol) == contract['protocol_sha256'], 'protocol changed')
    require(tasks['task_count'] == 210 and tasks['task_records_sha256'] == contract['task_records_sha256'],
            'dev panel changed')
    baseline = contract['baseline']
    for name, key in [('identity.json', 'identity_sha256'), ('evaluation-report.json', 'report_sha256')]:
        require(sha256_file(args.baseline_root / name) == baseline[key], f'B0 {name} changed')
    report = read_sealed(args.baseline_root / 'evaluation-report.json')
    require(len(report['episodes']) == 210, 'B0 panel incomplete')
    require(all(Path(row['artifact']).is_file() for row in report['episodes']), 'B0 episode files unavailable')
    endpoint_files = [args.training_root / 'training-report.json',
                      args.training_root / 'epochs/epoch-02/training-report.json',
                      args.training_root / 'epochs/epoch-02/training-manifest.json',
                      args.training_root / 'epochs/epoch-02/M1.ckpt',
                      args.training_root / 'epochs/epoch-02/auxiliary-predictor.pt']
    missing = [str(path) for path in endpoint_files if not path.is_file()]
    endpoint = verify_endpoint(args.training_root, contract)[1] if not missing else None
    return seal_contract({'artifact_type': 'eventground_wam.week7_m1_evaluation_preflight',
                          'status': 'endpoint_verified_launch_pending' if endpoint else 'waiting_for_epoch_2',
                          'contract_digest': contract['digest'], 'source_digest': package['digest'],
                          'training_identity_digest': training['digest'], 'runtime_versions': versions,
                          'libero': libero, 'baseline_artifacts_present': 210,
                          'baseline_pair_input_audit': 'pending actual M1 rollout artifacts',
                          'endpoint_report_digest': endpoint['digest'] if endpoint else None,
                          'missing_endpoint_files': missing, 'gpu_model_loaded': False}), tasks, package


def load_policy(args, contract, endpoint):
    import torch
    from evaluate_b0_full import FullCheckpointIdentity
    from train_b0 import build_policy

    from eventground_wam.b0_data import load_b0_action_normalizer
    from eventground_wam.b0_inference import B0InferencePolicy
    from eventground_wam.b0_model import load_b0_trainable_state_dict, load_pinned_qwen_processor
    from eventground_wam.b0_oft import OFTInputBuilder

    require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == contract['gpu'], 'wrong evaluation GPU')
    torch.cuda.set_device(0)
    state = torch.load(args.training_root / 'epochs/epoch-02/M1.ckpt', map_location='cpu', weights_only=True)
    verify_envelope(state, contract)
    recipe = read_sealed(ROOT / 'configs/training/week4_b0_fresh_all_l1_v2.json')
    require(recipe['digest'] == contract['base_recipe_digest'], 'wrong OFT architecture recipe')
    device = torch.device('cuda:0')
    model, names = build_policy(recipe, str(args.cache_dir), device)
    load_b0_trainable_state_dict(model, state['state_dict'], allowed_exact_names=names)
    model.eval()
    checkpoint = FullCheckpointIdentity(
        checkpoint_id=contract['checkpoint_id'], checkpoint_revision=endpoint['checkpoint_sha256'],
        training_manifest_digest=contract['training_identity_digest'], training_report_digest=endpoint['digest'],
        selected_epoch=2, selected_step=contract['updates'])
    return B0InferencePolicy(
        policy=model, input_builder=OFTInputBuilder(load_pinned_qwen_processor(
            cache_dir=str(args.cache_dir), local_files_only=True)),
        normalizer=load_b0_action_normalizer(ROOT / 'docs/model/week4-fresh-v2/action-statistics.json'),
        device=device, checkpoint=checkpoint)


def require_idle_gpu():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        if not line.strip():
            continue
        _, process, memory = [part.strip() for part in line.split(',', 2)]
        require(Path(process).name in {'chrome', 'gnome-shell', 'Xorg'} and int(memory) <= 256,
                f'GPU occupied: {line}')


def run(args, contract, tasks, package, budget):
    from evaluate_clean_comparison import seed_episode
    from evaluate_reference_policy import evaluate_task

    from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

    _, endpoint, completion = verify_endpoint(args.training_root, contract)
    require_idle_gpu()
    root = args.output_root / 'dev-210/M1'
    identity = seal_contract({
        'artifact_type': 'eventground_wam.week7_m1_dev_run', 'policy': 'M1',
        'checkpoint_id': contract['checkpoint_id'], 'checkpoint_revision': endpoint['checkpoint_sha256'],
        'project_git_revision': package['base_git_revision'],
        'project_revision_scope': 'base commit only; evaluation_source_digest identifies executed package',
        'evaluation_source_digest': package['digest'], 'contract_digest': contract['digest'],
        'training_identity_digest': contract['training_identity_digest'],
        'training_completion_digest': completion['digest'], 'runtime_versions': contract['runtime_versions'],
        'protocol_sha256': contract['protocol_sha256'], 'task_records_sha256': contract['task_records_sha256'],
        'runtime_lock_sha256': sha256_file(ROOT / 'environments/b0/uv.lock')})
    immutable(root / 'identity.json', identity)
    atomic_json(root / 'task-manifest.json', tasks)
    install_codec_independent_motion_blur()
    policy = load_policy(args, contract, endpoint)
    from libero.libero import benchmark
    protocol = load_evaluation_protocol(ROOT / 'configs/evaluation/libero_plus_v1.json')
    suites, records = {}, []
    for ordinal, task in enumerate(tasks['tasks']):
        require(budget.seconds < contract['maximum_active_hours'] * 3600, 'evaluation time cap reached')
        require(shutil.disk_usage(args.output_root).free >= contract['minimum_free_bytes'], 'disk reserve reached')
        require(sum(p.stat().st_size for p in args.output_root.rglob('*') if p.is_file())
                < contract['output_limit_bytes'], 'evaluation output cap reached')
        receipt_path = root / f'result-{ordinal:03d}.json'
        if receipt_path.exists():
            receipt = read_sealed(receipt_path)
            require(receipt['identity_digest'] == identity['digest'] and receipt['ordinal'] == ordinal
                    and receipt['task'] == task, 'resume receipt changed')
            row = receipt['result']
            # Preserve exhausted infrastructure failures; never buy extra retries by restarting.
            if row['artifact']:
                require(sha256_file(Path(row['artifact'])) == row['artifact_sha256'], 'resume episode changed')
        else:
            if task['suite'] not in suites:
                suites[task['suite']] = benchmark.get_benchmark_dict()[task['suite']]()
            seed_episode(task['episode_seed'])
            result = evaluate_task(
                ordinal=ordinal, manifest_task=task, suite=suites[task['suite']], policy=policy,
                protocol=protocol, output_dir=root, project_revision=identity['project_git_revision'],
                resume=True, checkpoint_id=identity['checkpoint_id'], checkpoint_revision=identity['checkpoint_revision'])
            row = asdict(result)
            immutable(receipt_path, seal_contract({'identity_digest': identity['digest'], 'ordinal': ordinal,
                                                  'task': task, 'result': row}))
        records.append(row)
        progress = {'completed': len(records), 'task_count': 210, 'successes': sum(r['success'] for r in records)}
        atomic_json(root / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    immutable(root / 'evaluation-report.json', seal_contract({
        'artifact_type': 'eventground_wam.week7_m1_dev_report', 'status': 'rollouts_finished_analysis_pending',
        'identity_digest': identity['digest'], 'episode_count': len(records),
        'success_count': sum(r['success'] for r in records), 'episodes': records,
        'all_artifacts_present': all(r['artifact'] for r in records)}))


def analyze(args, contract):
    import numpy as np
    from evaluate_reference_policy import load_manifest, wilson_interval

    protocol = load_evaluation_protocol(ROOT / 'configs/evaluation/libero_plus_v1.json')
    tasks = load_manifest(ROOT / contract['panel'], protocol)['tasks']
    roots = {'M1': args.output_root / 'dev-210/M1', 'B0': args.baseline_root}
    reports = {arm: read_sealed(root / 'evaluation-report.json') for arm, root in roots.items()}
    identities = {arm: read_sealed(root / 'identity.json') for arm, root in roots.items()}
    require(identities['M1']['contract_digest'] == contract['digest'], 'wrong M1 contract')
    endpoint = verify_endpoint(args.training_root, contract)[1]
    require(identities['M1']['checkpoint_revision'] == endpoint['checkpoint_sha256'], 'wrong M1 checkpoint')
    require(identities['B0']['digest'] == contract['baseline']['identity_digest'], 'wrong B0 identity')
    require(sha256_file(args.baseline_root / 'evaluation-report.json') == contract['baseline']['report_sha256'],
            'B0 report changed')
    for arm in roots:
        require(reports[arm]['identity_digest'] == identities[arm]['digest'], 'report identity changed')
        require(reports[arm]['episode_count'] == len(reports[arm]['episodes']) == 210, 'incomplete panel')
        require(identities[arm]['protocol_sha256'] == contract['protocol_sha256']
                and identities[arm]['task_records_sha256'] == contract['task_records_sha256'], 'wrong panel')
        require(identities[arm]['runtime_versions'] == contract['runtime_versions'], 'runtime mismatch')
    pairs = []
    for i, task in enumerate(tasks):
        inputs, errors = {}, {}
        for arm in roots:
            row = reports[arm]['episodes'][i]
            require(all(row[k] == task[k] for k in ('suite', 'runtime_task_index',
                                                   'perturbation_category', 'difficulty_level')), 'row task changed')
            try:
                inputs[arm] = artifact_inputs(Path(row['artifact']) if row['artifact'] else Path('/missing'),
                                              row, identities[arm], task, protocol)
                if inputs[arm] is None:
                    errors[arm] = 'no first policy query; input pairing unverified'
            except (OSError, ValueError, KeyError) as error:
                errors[arm] = str(error)
        pairs.append({'ordinal': i, 'exact_match': not errors and inputs['M1'] == inputs['B0'],
                      'artifact_errors': errors, 'input_digests': {arm: seal_contract(value)['digest']
                          if value else None for arm, value in inputs.items()}})
    summaries = {}
    for arm, report in reports.items():
        rows = report['episodes']
        successes = sum(row['success'] for row in rows)
        require(successes == report['success_count'], 'success count changed')
        latencies = [v for row in rows for v in row['policy_latencies_ms'] if np.isfinite(v)]
        breakdowns = {}
        for field in ('suite', 'perturbation_category', 'difficulty_level'):
            groups = defaultdict(list)
            for row in rows:
                groups[str(row[field])].append(row)
            breakdowns[field] = {key: {'count': len(values), 'successes': sum(v['success'] for v in values)}
                                 for key, values in groups.items()}
        summaries[arm] = {'count': 210, 'successes': successes, 'success_rate': successes / 210,
                          'wilson_95': wilson_interval(successes, 210), 'breakdowns': breakdowns,
                          'query_latency_ms': {'count': len(latencies),
                              'median': float(np.median(latencies)) if latencies else None,
                              'p95': float(np.quantile(latencies, .95)) if latencies else None}}
    all_valid = all(row['exact_match'] for row in pairs)
    result = seal_contract({
        'artifact_type': 'eventground_wam.week7_m1_development_analysis',
        'status': 'analyzed' if not any(p['artifact_errors'] for p in pairs) else 'artifact_audit_incomplete',
        'contract_digest': contract['digest'], 'report_digests': {arm: r['digest'] for arm, r in reports.items()},
        'pairing': 'exact canonical inputs for all 210' if all_valid else 'whole-panel contrast descriptive',
        'paired_input_audit': pairs, 'summaries': summaries,
        'M1_minus_B0': paired_summary([r['success'] for r in reports['M1']['episodes']],
                                    [r['success'] for r in reports['B0']['episodes']]),
        'limitations': 'Single-seed development evidence. Preserve prior B0 qualification and original JEPA FAIL.',
        'results_package': 'pending cost evidence, checkpoint backup, inventory and narrative review'})
    immutable(args.output_root / 'analysis.json', result)
    return result


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'run', 'analyze'))
    parser.add_argument('--training-root', type=Path,
                        default=home / 'data/eventground-wam/libero-plus/training/week7-m1-auxiliary-v2')
    parser.add_argument('--training-source', type=Path, default=home / 'code/eventground-wam-m1-auxiliary-v2')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--libero-root', type=Path, default=home / 'code/vendor/LIBERO-plus')
    parser.add_argument('--cache-dir', type=Path, default=home / 'data/eventground-wam/huggingface-cache')
    args = parser.parse_args()
    contract = load_contract(ROOT)
    args.baseline_root = args.baseline_root or Path(contract['baseline']['remote_directory'])
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output_root / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.mode == 'analyze':
                package_check()
                print(json.dumps({'analysis_digest': analyze(args, contract)['digest']}))
                return
            report, tasks, package = preflight(args, contract)
            atomic_json(args.output_root / 'preflight.json', report)
            print(json.dumps(report), flush=True)
            if args.mode == 'preflight':
                return
            require(report['status'] == 'endpoint_verified_launch_pending', 'waiting for complete epoch 2')
            with (args.training_root / 'owner.lock').open('r') as training_lock:
                fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                budget = ActiveBudget(args.output_root / 'active-budget.json', contract['digest'])
                try:
                    run(args, contract, tasks, package, budget)
                finally:
                    budget.close()
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Integrity stop: {error}', flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()
