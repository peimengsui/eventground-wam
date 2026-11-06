#!/usr/bin/env python3
"""Seal M1/M2 numerical recipes only after the complete Week 7 evidence audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from verify_auxiliary_integration import verify

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    audit = verify(attempt)
    initialization = read_sealed(attempt.parent/'initialization-audit.json')
    device_root = attempt.parent/'device-memory-v1'
    device = read_sealed(device_root/'report.json')
    device_plan = read_sealed(device_root/'plan.json')
    assert device['plan_digest'] == device_plan['digest']
    assert device['source_digest'] == device_plan['source_digest'] == audit['source_digest']
    assert device['status'] == 'passed' and device['primary_storage_below_24_gib']
    assert device['free_disk_above_50_gib']
    assert device_plan['script_sha256'] == sha256_file(ROOT/'scripts/measure_auxiliary_device_memory.py')
    controls = read_sealed(ROOT/'configs/training/phase2_two_epoch_v1.json')
    rule = read_sealed(ROOT/'configs/training/week7_auxiliary_pilot_v2.json')
    registry = read_sealed(ROOT/'configs/training/phase2_future_modules_v1.json')
    base = read_sealed(ROOT/controls['action_training']['inherited_recipe_path'])
    assert initialization['both_arms_equal_original_b0_initialization']
    assert initialization['all_training_prompts_within_interface_bound']
    assert initialization['b0_task_state_digest'] == controls['initialization']['b0_initial_task_state_digest']
    assert initialization['shared_tensor_digest'] == audit['shared_initialization']
    for name, digest in initialization['source_sha256'].items():
        assert sha256_file(ROOT/name) == digest
    assert sha256_file(ROOT/'scripts/audit_auxiliary_initialization.py') == initialization['audit_script_sha256']
    results = {}
    for arm, method in [('M1', 'pixel'), ('M2', 'jepa')]:
        report = read_sealed(attempt/'run'/f'{method}-probe-report.json')
        preflight = read_sealed(attempt/'run'/f'{method}-probe-preflight.json')
        inventory = read_sealed(attempt/'run'/f'{method}-parameters.json')
        memory = device['results'][method]
        trace = device_root/f'{method}-device.csv'
        assert sha256_file(trace) == memory['trace_sha256']
        rows = list(csv.reader(trace.read_text().splitlines()))
        assert len(rows) == memory['samples']
        assert max(int(r[2]) for r in rows) == memory['peak_device_used_mib']
        assert memory['peak_device_used_mib'] < memory['device_total_mib']*.9
        assert memory['matches_primary_replay'] and memory['remote_checkpoint_and_export_hashes_reverified']
        assert memory['state_digest'] == read_sealed(attempt/'run'/f'{method}-uninterrupted.json')['state_digest']
        assert preflight['action_plan_digests'] == controls['batching_and_sampling']['mixed_plan_digests']
        module_id = registry['branch_to_module'][arm]
        module = registry['modules'][module_id]
        future_objective = ({'name': 'L1-plus-SSIM', 'ssim_weight': .1} if method == 'pixel' else
            {'name': 'cosine-plus-anchor-grouped-teacher-relative-variance', 'variance_weight': 10.,
             'variance_floor_fraction': .5, 'variance_epsilon': .0001, 'singleton_variance': 0.,
             'teacher': 'qwen35-vision-merged-ln-l2-single-image-v2', 'teacher_frozen': True})
        recipe = seal_contract({
            'artifact_type': 'eventground_wam.week7_auxiliary_policy_recipe', 'schema_version': 1,
            'recipe_id': f'week7-{arm.lower()}-auxiliary-v1', 'arm': arm, 'method': method,
            'parent_experiment_digest': rule['parent_experiment_digest'],
            'action_controls_digest': controls['digest'], 'base_recipe_digest': base['digest'],
            'base_recipe_path': controls['action_training']['inherited_recipe_path'],
            'pilot_rule_digest': rule['digest'], 'verification_digest': audit['digest'],
            'initialization_audit_digest': initialization['digest'],
            'device_memory_audit_digest': device['digest'],
            'verified_source_digest': audit['source_digest'],
            'verified_source_archive': str((attempt/'source.tar.gz').relative_to(ROOT)),
            'initialization': {'shared': controls['initialization'],
                'shared_tensor_digest': audit['shared_initialization'],
                'future_module_id': module_id, 'future_module_path': module['remote_path'],
                'future_module_sha256': module['sha256'], 'future_module_original_gate': module['original_gate'],
                'fresh_optimizer_and_scheduler': True, 'pilot_weights_may_be_loaded': False},
            'interface': rule['interface'], 'future_tokens_in_action_path': False,
            'future_module_or_teacher_at_inference': False,
            'training_seed': 17, 'epochs': 2, 'updates_per_epoch': 58566, 'total_updates': 117132,
            'action_presentations': 3412044, 'future_pairs': 2*preflight['future_pairs_per_epoch'],
            'future_pairs_per_epoch_by_horizon': preflight['future_pairs_by_horizon'],
            'mixed_plan_digests': preflight['action_plan_digests'],
            'batching': rule['batching'], 'loss': {'action': base['objective'],
                'outer_future_weight': report['calibrated_weight'], 'future': future_objective,
                'action_denominator': 'all 8x7 targets in the unchanged logical action batch',
                'future_denominator': 'all valid dual-camera pairs in the same logical batch',
                'jepa_variance_scope': 'average horizons per anchor, then population variance across anchors'},
            'optimizer': {**controls['action_training']['optimizer_common_parameters'],
                'auxiliary_learning_rate': .0001, 'groups': ['head_and_state', 'lora', 'future'],
                'step_scheduler_and_joint_clipping_once_per_logical_batch': True,
                'schedule_total_updates': 117132},
            'precision': rule['predictor_precision'], 'gradient_tolerances': rule['correctness'],
            'parameter_inventory_path': str((attempt/'run'/f'{method}-parameters.json').relative_to(ROOT)),
            'parameter_inventory_digest': inventory['digest'],
            'shared_trainable_parameters': inventory['shared_count'],
            'future_trainable_parameters': inventory['future_count'],
            'resources': {'host': 'feifei', 'gpu': 'NVIDIA GeForce RTX 5090',
                'one_substantial_gpu_job_at_a_time': True,
                'projected_hours': report['projected_training_hours'],
                'maximum_active_hours': rule['resource']['production_active_hour_cap_per_arm'],
                'maximum_projected_hours': rule['resource']['maximum_projected_hours_per_arm'],
                'gpu_peak_allocated_bytes': report['gpu_peak_allocated_bytes'],
                'gpu_peak_reserved_bytes': report['gpu_peak_reserved_bytes'],
                'host_peak_rss_bytes': report['host_peak_rss_bytes'],
                'four_update_peak_device_used_mib': memory['peak_device_used_mib'],
                'three_checkpoints_two_exports_bytes': memory['three_checkpoints_two_exports_bytes'],
                'teacher_and_decoded_caches_on_disk': False,
                'storage_scope': 'measured training-state/export footprint; 100 GiB output cap also reserves room for logs and later development rollout artifacts',
                'minimum_free_disk_bytes': 50*1024**3, 'output_limit_bytes': 100*1024**3,
                'projection_scope': 'first 128 exact epoch-0 mixed batches, cold caches, 1.2*p95/mean maximum; full-run wall time remains an estimate'},
            'recovery': {'format_extension': 'eventground-wam-auxiliary-training-state-v1',
                'checkpoint_keep_last': 3, 'checkpoint_interval_steps': 2000,
                'checkpoint_interval_seconds': 300, 'commit_partial_update': False,
                'empty_teacher_cache_on_resume': True, 'exact_process_resume_verified': True},
            'checkpoint_selection': 'fixed-epoch-2-endpoint',
            'evaluation': 'inherited Week 7 paired dev-210 protocol, with original JEPA/B0 claim limits',
            'status': 'numerical_recipe_and_feasibility_verified', 'production_training_started': False,
            'required_at_launch': ['verify source, runtime, base and module hashes',
                'recheck free GPU, memory and disk', 'enforce active-time/output limits and immutable identity',
                'initialize from original base/module, never pilot checkpoints'],
        })
        path = ROOT/f'configs/training/week7_{arm.lower()}_auxiliary_v1.json'
        if path.exists():
            assert read_sealed(path) == recipe, 'refusing to rewrite a sealed numerical recipe'
        else:
            atomic_json(path, recipe)
        results[arm] = {'path': str(path.relative_to(ROOT)), 'digest': recipe['digest']}
    atomic_json(attempt.parent/'verification.json', audit)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
