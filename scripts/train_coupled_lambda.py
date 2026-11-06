#!/usr/bin/env python3
"""Launch the unchanged A100 M4 implementation under its approved time amendment."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import signal
from pathlib import Path

import torch
import train_auxiliary_policy as shared
import train_coupled_policy as coupled
from coupled_runtime import (
    ROOT, execution_contract, project_hours, require, verify_runtime, verify_source,
)

from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_training import configure_deterministic_cuda, read_sealed, sha256_file
from eventground_wam.coupled_policy import FULL_BATCH_EXECUTION, LOSS_WEIGHTS
from eventground_wam.demonstration_contracts import seal_contract

APPROVED_AMENDMENT_DIGEST = '3630837e2b179ee5de417f558510eec891a9874fd185791de4b910cdbe2b5e59'


def amended_contract(contract, amendment, runtime, report):
    require(amendment['digest'] == APPROVED_AMENDMENT_DIGEST
            == seal_contract({k: v for k, v in amendment.items() if k != 'digest'})['digest'],
            'unapproved or changed time amendment')
    require(contract['arm'] == 'M4' and amendment['status'] == 'approved', 'amendment is M4 only')
    require(amendment['original_contract_digest'] == contract['digest'], 'amendment recipe differs')
    require(amendment['runtime_digest'] == runtime['digest'], 'amendment runtime differs')
    require(amendment['resource_report_digest'] == report['digest'], 'amendment evidence differs')
    require(amendment['original_limits'] == {k: contract['resources'][k] for k in
            ('maximum_projected_hours', 'maximum_active_hours')}, 'original time limits differ')
    require(amendment['replacement_limits'] == {'maximum_projected_hours': 480,
            'maximum_active_hours': 480}, 'amendment scope differs')
    require(amendment['original_timing_gate_remains_failed'] and report['passed'] is False,
            'original failed gate must remain recorded')
    require(not amendment['numerical_hyperparameters_effective_batches_and_exposure_changed']
            and not amendment['memory_storage_bounds_changed'], 'non-time amendment is forbidden')
    return seal_contract({**{k: v for k, v in contract.items() if k != 'digest'},
        'resources': {**contract['resources'], **amendment['replacement_limits']},
        'unamended_contract_digest': contract['digest'],
        'time_budget_amendment_digest': amendment['digest']})


def validate_resources(report, contract, runtime):
    require(report['arm'] == 'M4' and report['backward_execution'] == FULL_BATCH_EXECUTION,
            'resource execution differs')
    require(report['loss_weight'] == LOSS_WEIGHTS['M4'], 'loss weight differs')
    require([r['global_step'] for r in report['rows']] == list(range(1, 129)), 'timing stream differs')
    mean, p95, hours = project_hours([r['seconds'] for r in report['rows']], report['checkpoint_seconds'])
    for key, value in [('mean_step_seconds', mean), ('p95_step_seconds', p95), ('projected_training_hours', hours)]:
        require(abs(report[key]-value) < 1e-8, 'resource projection differs')
    require(168 < hours <= contract['resources']['maximum_projected_hours'], 'amended time limit failed')
    require(report['host_peak_rss_bytes'] <= 28*1024**3
            and report['gpu_peak_reserved_bytes'] <= .9*runtime['cuda_total_bytes']
            and report['gpu_peak_allocated_bytes'] <= .9*runtime['cuda_total_bytes'], 'memory limit failed')
    require(report['teacher_unchanged'] and report['pilot_weights_excluded_from_production'],
            'teacher or pilot isolation failed')
    require(report['edge_probes'] and all(r['finite'] and r['sequence_length'] <= 512
            for r in report['edge_probes']), 'edge checks failed')


def verify_launch_source(root, pilot_source, amendment):
    source = read_sealed(root/'coupled-launch-source-manifest.json')
    require(source['pilot_source_digest'] == pilot_source['digest'] == amendment['pilot_source_digest'],
            'pilot source identity differs')
    require(source['time_budget_amendment_digest'] == amendment['digest'], 'launch amendment differs')
    for name, digest in pilot_source['files'].items():
        require(source['files'].get(name) == digest, f'qualified implementation changed: {name}')
    for name, digest in source['files'].items():
        require(sha256_file(root/name) == digest, f'launch source changed: {name}')
    return source


def verify_amended_admission(args, contract, source, runtime, amendment):
    evidence = args.qualification
    for name, digest in amendment['pilot_evidence_files'].items():
        require(sha256_file(evidence/name) == digest, f'approved evidence changed: {name}')
    require(source['digest'] == amendment['pilot_source_digest'], 'pilot source differs')
    report = read_sealed(evidence/'M4-resources.json')
    effective = amended_contract(contract, amendment, runtime, report)
    validate_resources(report, effective, runtime)
    phases = json.loads((evidence/'phase-records.json').read_text())
    expected = [(a, p) for a in ('M3', 'M4') for p in ('correctness', 'uninterrupted', 'interrupt', 'resume')]+[('M4', 'resource')]
    require([(r['arm'], r['phase']) for r in phases] == expected, 'pilot phases missing')
    require(all(r['exit_code'] == (-signal.SIGKILL if r['phase'] == 'interrupt' else
            1 if r['phase'] == 'resource' else 0) for r in phases), 'unexpected pilot failure')
    reports, traces = {}, {}
    for arm in ('M3', 'M4'):
        identity = read_sealed(evidence/f'{arm}-pilot-identity.json')
        require(identity['source_digest'] == source['digest']
                and identity['runtime_digest'] == runtime['digest']
                and identity['contract_digest'] == execution_contract(arm)['digest'], 'pilot identity differs')
        for key in ('data', 'cache'):
            require(identity[f'{key}_path'] == str(getattr(args, key).resolve()), f'{key} location differs')
        if arm == 'M4':
            require(identity['digest'] == amendment['pilot_identity_digest']
                    == report['identity_digest'], 'M4 resource identity differs')
        values = {p: read_sealed(evidence/f'{arm}-{p}.json') for p in ('correctness', 'uninterrupted', 'resume')}
        require(all(v['identity_digest'] == identity['digest'] for v in values.values()), 'report identity differs')
        correct, resumed = values['correctness'], values['resume']
        require(all(correct[k] for k in ('passed', 'real_qwen_export_reload_bitwise_equal',
                'target_isolation', 'teacher_unchanged', 'target_repeat_bitwise_equal',
                'zero_coupling_action_bitwise_equal', 'token_only_bitwise_equal', 'detached_proposal',
                'pilot_weights_excluded_from_production')), 'correctness gate failed')
        require(correct['backward_execution'] == FULL_BATCH_EXECUTION
                and correct['joint_gradient_comparison']['relative_l2'] <= .01, 'joint gradients differ')
        require(all(resumed[k] for k in ('process_resume_bitwise_equal', 'damaged_latest_fallback',
                'fallback_replay_bitwise_equal')) and resumed['state_digest'] == values['uninterrupted']['state_digest'],
                'recovery gate failed')
        reports.update({f'{arm}-{p}': v['digest'] for p, v in values.items()})
    for arm, phase in expected:
        path = evidence/f'{arm}-{phase}-device.csv'
        readings = list(csv.reader(path.read_text().splitlines()))
        require(len(readings) > 10 and all(len(r) == 2 and 0 <= float(r[1]) <= .9*float(r[0])
                for r in readings), 'device memory trace failed')
        traces[path.name] = sha256_file(path)
    events = []
    for name in ('service.log', 'M4-resource.log'):
        for line in (evidence/name).read_text().splitlines():
            if line.startswith('{'):
                events.append(json.loads(line))
    start = min(e['time_unix'] for e in events if e.get('status') == 'qualification_running')
    end = max(e['time_unix'] for e in events if e.get('phase') == 'resources_complete')
    require(0 < end-start <= 3*3600, 'pilot time ceiling failed')
    require(sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file()) <= 24*1024**3,
            'pilot output ceiling failed')
    admission = seal_contract({'artifact_type': 'eventground_wam.coupled_time_amended_admission',
        'arm': 'M4', 'passed_under_approved_time_amendment': True,
        'original_resource_gate_passed': False, 'original_timing_gate_remains_failed': True,
        'pilot_source_digest': source['digest'], 'runtime_digest': runtime['digest'],
        'unamended_contract_digest': contract['digest'], 'effective_contract_digest': effective['digest'],
        'time_budget_amendment_digest': amendment['digest'], 'reports': reports,
        'resource_report_digest': report['digest'], 'phases': phases, 'device_traces': traces,
        'elapsed_seconds': end-start, 'pilot_weights_excluded_from_production': True})
    return effective, admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, default=ROOT/'configs/training/week8_m4_lambda_budget_v1.json')
    parser.add_argument('--data', type=Path, default=Path('/home/peimengsui/data/eventground-wam/libero-plus'))
    parser.add_argument('--cache', type=Path, default=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.arm, args.method, args.phase = 'M4', 'jepa', 'production'
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, shared.request_stop)
    signal.signal(signal.SIGINT, shared.request_stop)
    configure_deterministic_cuda()
    torch.set_num_threads(1)
    try:
        pilot_source, contract = verify_source(), execution_contract('M4')
        amendment, runtime = read_sealed(args.amendment), verify_runtime(args.profile)
        launch_source = verify_launch_source(ROOT, pilot_source, amendment)
        effective, admission = verify_amended_admission(args, contract, pilot_source, runtime, amendment)
        shared.check_resources(args.output, effective)
        shared.immutable_json(args.output/'time-amended-admission.json', admission)
        shared.immutable_json(args.output/'approved-time-budget.json', amendment)
        if args.check_only:
            print(json.dumps({'admission_digest': admission['digest'], 'effective_contract_digest': effective['digest'],
                              'check_only': True, 'training_started': False}), flush=True)
            return
        with (args.output/'owner.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = ActiveBudget(args.output/'active-budget.json', seal_contract({
                'contract': effective['digest'], 'source': launch_source['digest'],
                'admission': admission['digest'], 'amendment': amendment['digest']})['digest'])
            try:
                code = coupled.run(args, effective, launch_source, runtime, admission, budget)
            finally:
                budget.close()
        raise SystemExit(code)
    except (ValueError, OSError, AssertionError, RuntimeError) as error:
        shared.emit(args.output, 'integrity_or_runtime_stop', error=str(error), exception=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
