"""Seal the matched continuation design from read-only input and historical audits."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.week11_multihorizon_plan import (
    HORIZONS,
    SCHEDULE_ID,
    candidate_horizons,
    projected_seconds,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/evaluation/week11-multihorizon-plan-v1'
CONFIG = ROOT / 'configs/training/week11_multihorizon_pilot_v1.json'


def write(path, value):
    value = seal_contract(value)
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return value


def main():
    audit_path = OUT / 'verified-input-audit.json'
    audit = read_sealed(audit_path)
    batches = audit['selected_batches']
    if (len(batches) != 2000 or sum(map(len, batches)) != 58172
            or batches_digest(batches) != audit['selected_plan_digest']
            or audit['selected_training_tasks'] != 40
            or not all(p['content_hash_checked'] for e in audit['selected_episode_metadata'] for p in e['payloads'])
            or not audit['predictor_core_consistency']['bitwise_equal']):
        raise ValueError('incomplete input admission for planning')
    schedule, cursor = [], 0
    counts = Counter()
    for index, tokens in enumerate(batches):
        horizons = candidate_horizons(cursor, len(tokens), mixed=True)
        counts.update(horizons)
        schedule.append({'update_index': index, 'presentation_start': cursor,
                         'tokens': tokens, 'mixed_horizons': horizons})
        cursor += len(tokens)
    schedule = write(OUT / 'training-schedule.json', {
        'artifact_type': 'eventground_wam.week11_multihorizon_schedule',
        'schedule_id': SCHEDULE_ID, 'seed': 17, 'control_horizon': 8,
        'parent_full_epoch_plan_digest': audit['full_epoch_plan_digest'],
        'action_plan_digest': audit['selected_plan_digest'],
        'action_presentations_per_arm': cursor,
        'mixed_horizon_counts': {str(k): v for k, v in counts.items()},
        'batches': schedule,
    })
    history_path = ROOT / 'docs/evaluation/week8-m4-lambda-qualification-v1/attempt-07/run/M4-resources.json'
    history = read_sealed(history_path)
    projections = {}
    for name, seconds, margin in (
        ('mean', history['mean_step_seconds'], 1.0),
        ('conservative', max(history['mean_step_seconds'], history['p95_step_seconds']), 1.2),
    ):
        per_arm = projected_seconds(2000, seconds, history['checkpoint_seconds'], margin=margin)
        projections[name] = {'per_arm': per_arm, 'two_arm_hours': 2 * per_arm['total_seconds'] / 3600}
    estimate = write(OUT / 'compute-estimate.json', {
        'artifact_type': 'eventground_wam.week11_multihorizon_cost_proxy',
        'source': str(history_path.relative_to(ROOT)), 'source_sha256': sha256_file(history_path),
        'source_digest': history['digest'], 'source_original_time_gate_passed': history['passed'],
        'source_host': 'Lambda A100-SXM4-40GB', 'planned_host': 'Feifei RTX 5090',
        'measured_current_feifei_continuation': False,
        'mean_step_seconds': history['mean_step_seconds'], 'p95_step_seconds': history['p95_step_seconds'],
        'checkpoint_seconds': history['checkpoint_seconds'], 'projections': projections,
        'excluded': ['initialization', 'qualification', 'evaluation', 'infrastructure downtime'],
        'fresh_qualification_required': True,
    })
    gate_path = ROOT / 'docs/evaluation/week11-gate-comparison-v1/bundle/week11-gate-contract.json'
    gate = read_sealed(gate_path)
    conditions = [f'{model}/{route}' for model in ('original', 'h8-continuation', 'mixed-continuation')
                  for route in ('fixed8', 'adaptive')]
    screen = write(OUT / 'screen-manifest.json', {
        'artifact_type': 'eventground_wam.week11_multihorizon_screen',
        'parent_digest': gate['digest'], 'parent_sha256': sha256_file(gate_path),
        'cases': [{k: c[k] for k in ('id', 'panel', 'ordinal', 'task', 'outcome_class')}
                  for c in gate['cases']],
        'conditions': conditions, 'planned_episodes': 162,
        'schedule': [{'case_index': i, 'condition': conditions[(i + j) % 6]}
                     for i in range(27) for j in range(6)],
        'new_rollouts_executed': 0, 'interpretation': 'Outcome-selected development screen only',
    })
    integration = read_sealed(ROOT / 'configs/evaluation/week11_integration_v1.json')
    mapping = read_sealed(ROOT / 'configs/evaluation/week11_horizon_mapping_v1.json')
    design = read_sealed(ROOT / 'configs/training/week8_coupled_design_v1.json')
    references = [
        'docs/WEEK11_MULTIHORIZON_PILOT.md',
        'src/eventground_wam/week11_multihorizon_plan.py',
        'scripts/inspect_week11_multihorizon_inputs.py',
        'scripts/prepare_week11_multihorizon_plan.py',
        'tests/test_week11_multihorizon_plan.py',
        'src/eventground_wam/coupled_policy.py', 'src/eventground_wam/coupled_training.py',
        'src/eventground_wam/coupled_setup.py', 'src/eventground_wam/coupled_checkpoint.py',
        'src/eventground_wam/coupled_evaluation.py', 'src/eventground_wam/eventground_inference.py',
        'src/eventground_wam/auxiliary_data.py', 'scripts/train_coupled_policy.py',
        'configs/training/week4_b0_fresh_all_l1_v2.json',
        'configs/training/week7_m2_auxiliary_v1.json',
    ]
    config = write(CONFIG, {
        'artifact_type': 'eventground_wam.week11_multihorizon_pilot_design',
        'version': 'week11-multihorizon-pilot-v1', 'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'scientific_design_frozen_implementation_and_gpu_qualification_pending',
        'user_request': 'Sounds good, can you start doing it?',
        'current_step': 'Define the pilot, check training compatibility and estimate compute before launch',
        'production_launch_ready': False,
        'new_optimizer_updates': 0, 'new_rollouts': 0,
        'specification': 'docs/WEEK11_MULTIHORIZON_PILOT.md',
        'parents': {'week8_design': design['digest'], 'week11_integration': integration['digest'],
                    'week11_mapping': mapping['digest']},
        'starting_endpoints': integration['endpoints'],
        'full_predictor': audit['input_files']['full_predictor'],
        'input_audit': {'path': str(audit_path.relative_to(ROOT)), 'digest': audit['digest'],
                        'sha256': sha256_file(audit_path)},
        'training_schedule': {'path': str((OUT / 'training-schedule.json').relative_to(ROOT)),
                              'digest': schedule['digest']},
        'arms': {'h8-continuation': {'candidate_horizons': [8]},
                 'mixed-continuation': {'candidate_horizons': list(HORIZONS), 'schedule_id': SCHEDULE_ID}},
        'matched': {'updates_per_arm': 2000, 'seed': 17, 'action_presentations_per_arm': cursor,
                    'valid_future_pairs_per_arm': sum(audit['future_pairs_per_arm_by_horizon'].values()),
                    'future_pairs_by_horizon': audit['future_pairs_per_arm_by_horizon'],
                    'additional_trainable_parameters': 0, 'trainable_parameters_per_arm': 98859527,
                    'parameter_groups': ['head_and_state', 'lora', 'future', 'coupling'],
                    'inherited_action_updates': 117132, 'event_training_updates': 0,
                    'base_and_teacher_frozen': True, 'full_trained_predictor_loaded': True,
                    'original_optimizer_moments_loaded': False, 'fixed_endpoint_only': True},
        'optimizer': {'name': 'AdamW', 'peak_learning_rate_all_groups': 1e-5,
                      'betas': [0.9, 0.95], 'epsilon': 1e-8, 'weight_decay': 0.01,
                      'global_clip': 1, 'warmup_updates': 40, 'schedule': 'inherited cosine convention',
                      'total_updates': 2000, 'minimum_lr_ratio': 0.1,
                      'explicit_override_of_hardcoded_coupled_groups_required': True},
        'loss': design['arms']['M4']['loss'],
        'execution': {'candidate_microbatch': 10, 'auxiliary_microbatch': 'whole logical batch',
                      'predict_and_execute_actions': 8, 'control_hz': 20,
                      'proposal_detached': True, 'proposal_padding_steps': 24,
                      'policy_precision': 'inherited BF16', 'future_and_coupling_precision': 'FP32',
                      'event_model_separate_frozen_storage': True,
                      'allocator': 'PYTORCH_ALLOC_CONF=expandable_segments:True'},
        'qualification': {'timing_updates_per_arm': 128, 'all_weights_discarded': True,
                          'actual_device_parity_gradients_resume_reload_required': True,
                          'actual_device_checks_completed': False},
        'resources': {'qualification_hours_total': 3, 'training_hours_total': 18,
                      'screen_hours_total': 1, 'aggregate_active_hours': 22,
                      'maximum_training_projection_hours_including_startup_and_finalization': 16,
                      'qualification_output_gib': 8, 'training_output_gib': 12, 'screen_output_gib': 12,
                      'aggregate_output_gib': 32, 'minimum_free_gib': 50,
                      'host_process_tree_rss_gib': 28, 'gpu_fraction_strictly_below': 0.9,
                      'host': 'feifei', 'concurrent_arms': 1, 'cloud_provisioning': False,
                      'checkpoint_interval_seconds': 300, 'checkpoint_interval_updates': 2000,
                      'checkpoint_keep_last_per_arm': 3, 'failure': 'stop and preserve; no automatic budget revision'},
        'cost_proxy_digest': estimate['digest'],
        'screen': {'manifest_digest': screen['digest'], 'cases': 27, 'conditions': 6, 'episodes': 162,
                   'primary': 'paired mixed-adaptive minus h8-continuation-adaptive success count',
                   'progression': {'mixed_adaptive_gain_over_each_adaptive_control_at_least': 2,
                                   'mixed_fixed8_loss_against_each_fixed8_control_at_most': 1,
                                   'adaptive_minus_fixed8_interaction_gain_at_least': 1,
                                   'all_integrity_and_resource_checks_pass': True},
                   'claim': 'engineering screen, not population inference or noninferiority',
                   'untouched_panel': 'Requires separate identity/exposure/statistical freeze; no test unlock'},
        'known_evidence_packages': {
            name: read_sealed(ROOT / 'docs/evaluation' / name / 'package-index.json')['digest']
            for name in ('week11-closeout-v1', 'week11-adaptive-diagnosis-v1',
                         'week11-horizon-replay-v1', 'week11-event-review-v1',
                         'week11-gate-comparison-v1', 'week11-gated-failure-inspection-v1',
                         'week11-first-chunk-v1')},
        'source_files_sha256': {name: sha256_file(ROOT / name) for name in references},
        'limitations': ['Forecast errors unchanged', 'Long-horizon zero-padding unchanged',
                       'Single seed and selected screen', 'Original JEPA FAIL/research exception retained',
                       'Event labels lack independent QA', 'Historical source-exposure caveats retained'],
    })
    print(json.dumps({'config_digest': config['digest'], 'schedule_digest': schedule['digest'],
                      'screen_digest': screen['digest'], 'cost_proxy_digest': estimate['digest']}))


if __name__ == '__main__':
    main()
