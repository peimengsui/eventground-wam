#!/usr/bin/env python3
"""Bounded GPU/recorder qualification against the already packaged M3 evaluator."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
import evaluate_coupled_dev as evaluation
from eventground_wam.auxiliary_evaluation import artifact_inputs, require
from eventground_wam.b0_training import atomic_json, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args0 = parser.parse_args()
    root = args0.output
    root.mkdir(parents=True, exist_ok=True)
    require(not (root/'qualification.json').exists(), 'qualification already complete')
    args = SimpleNamespace(arm='M3',
        training_root=Path('/home/peimengsui/data/eventground-wam/libero-plus/training/week8-m3-coupled-v1'),
        training_source=Path('/home/peimengsui/code/eventground-wam-m3-coupled-v1'),
        libero_root=Path('/home/peimengsui/code/vendor/LIBERO-plus'),
        cache_dir=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'), output_root=root)
    contract = evaluation.load_contract(evaluation.ROOT, 'M3')
    args.baseline_root = Path(contract['baseline']['remote_directory'])
    preflight, tasks, package = evaluation.preflight(args, contract)
    atomic_json(root/'preflight.json', preflight)
    require(preflight['status'] == 'endpoint_verified_launch_pending', 'endpoint incomplete')
    evaluation.require_idle_gpu()
    endpoint = evaluation.verify_endpoint(args.training_root, contract)[1]
    policy = evaluation.load_policy(args, contract, endpoint)
    install_codec_independent_motion_blur()
    protocol = load_evaluation_protocol(evaluation.ROOT/'configs/evaluation/libero_plus_v1.json')
    from libero.libero import benchmark
    selected = {}
    for i, task in enumerate(tasks['tasks']):
        selected.setdefault(task['perturbation_category'], i)
    ordinals = sorted(set(selected.values()) | {100, 203, 204})
    identity = {'checkpoint_id': contract['checkpoint_id'], 'checkpoint_revision': endpoint['checkpoint_sha256'],
                'project_git_revision': package['base_git_revision']}
    audits, suites = [], {}
    for ordinal in ordinals:
        task = tasks['tasks'][ordinal]
        if task['suite'] not in suites:
            suites[task['suite']] = benchmark.get_benchmark_dict()[task['suite']]()
        results, inputs = [], []
        for repeat in range(2):
            out = root/f'repeat-{repeat}'
            out.mkdir(exist_ok=True)
            result = evaluation.evaluate_task(ordinal=ordinal, manifest_task=task,
                suite=suites[task['suite']], policy=policy, protocol=protocol, output_dir=out,
                project_revision=identity['project_git_revision'], resume=True,
                checkpoint_id=identity['checkpoint_id'], checkpoint_revision=identity['checkpoint_revision'])
            row = asdict(result)
            require(row['artifact'] and row['termination_reason'] in ('environment_done', 'policy_step_limit'),
                    f'model/recorder qualification failed: {row}')
            inputs.append(artifact_inputs(Path(row['artifact']), row, identity, task, protocol))
            require(inputs[-1] is not None and row['policy_latencies_ms'], 'no real policy query')
            results.append(row)
        audit = {'ordinal': ordinal, 'task': task, 'exact_repeat_inputs': inputs[0] == inputs[1],
                 'results': results, 'input_digests': [seal_contract(i)['digest'] for i in inputs]}
        audits.append(audit)
        atomic_json(root/f'task-{ordinal:03d}.json', seal_contract(audit))
        print(json.dumps({'ordinal': ordinal, 'exact_repeat_inputs': audit['exact_repeat_inputs']}), flush=True)
    atomic_json(root/'qualification.json', seal_contract({
        'artifact_type': 'eventground_wam.m3_dev_inference_qualification',
        'passed_model_and_recorder': True, 'arm': 'M3', 'contract_digest': contract['digest'],
        'evaluation_source_digest': package['digest'], 'qualifier_sha256': sha256_file(Path(__file__)),
        'checkpoint_sha256': endpoint['checkpoint_sha256'],
        'selection': 'First task of each category plus previously known M1 mismatch ordinals 100,203,204; two repeats each.',
        'ordinals': ordinals, 'repeat_input_audits': audits,
        'all_repeat_inputs_exact': all(a['exact_repeat_inputs'] for a in audits),
        'mismatch_rule': 'retain exact audit; whole affected contrast descriptive; no task dropping or extra repeats',
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'smoke_outcomes_used_in_formal_panel': False, 'formal_rollouts_started': False}))

if __name__ == '__main__':
    main()
