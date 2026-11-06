"""Opt-in evidence audits; preserve historically hashed test files unchanged."""

import pytest

# Tests that read generated manifests, panels, receipts or retained run outputs.
# Keep this registry separate: historical source manifests hash some test files.
RESEARCH_ARTIFACT_TESTS = frozenset({
    'tests/test_auxiliary_evaluation.py::test_frozen_contract_and_design_parents_are_intact',
    'tests/test_auxiliary_lambda_production.py::test_amended_projection_still_enforces_480_hours',
    'tests/test_auxiliary_lambda_production.py::test_amendment_cannot_be_used_for_another_runtime',
    'tests/test_auxiliary_lambda_production.py::test_amendment_cannot_raise_budget_again',
    'tests/test_auxiliary_lambda_production.py::test_real_a10_failure_requires_explicit_amendment',
    'tests/test_auxiliary_lambda_production.py::test_time_amendment_does_not_relax_other_requirements',
    'tests/test_b0_data.py::test_committed_real_data_evidence_is_sealed_and_complete',
    'tests/test_b0_fresh.py::test_frozen_fresh_recipe_and_normalizers_are_v2_train_only',
    'tests/test_b0_fresh.py::test_v2_loader_replaces_every_membership_and_rejects_changed_groups',
    'tests/test_b0_gripper_audit.py::test_committed_ten_demo_bce_review_is_sealed_and_preserves_original_failure',
    'tests/test_b0_inference.py::test_committed_b0_evaluator_integration_evidence_is_sealed',
    'tests/test_b0_model.py::test_committed_real_model_evidence_is_sealed_and_complete',
    'tests/test_b0_model.py::test_pinned_qwen_manifest_is_complete_and_sealed_before_weight_download',
    'tests/test_b0_overfit.py::test_committed_one_batch_overfit_evidence_is_sealed_and_passes_locked_gates',
    'tests/test_b0_rollout_validation.py::test_committed_rollout_validation_and_profile_evidence_is_sealed',
    'tests/test_b0_training_task_evaluator.py::test_committed_b0_closed_loop_evidence_is_scoped_and_sealed',
    'tests/test_clean_acceptance.py::test_complete_freeze_keeps_rule_panel_runtime_and_policy_identities',
    'tests/test_clean_acceptance.py::test_frozen_panel_rebuilds_with_distinct_states_and_stable_pair_seeds',
    'tests/test_clean_acceptance.py::test_panel_rejects_duplicated_initial_states_even_if_record_is_resealed',
    'tests/test_clean_comparison.py::test_all_panel_trial_seeds_and_trial_metadata',
    'tests/test_clean_comparison.py::test_completed_artifact_recovered_after_process_interrupt',
    'tests/test_clean_comparison.py::test_completed_policy_failure_never_retried',
    'tests/test_clean_comparison.py::test_exhausted_infrastructure_kept_in_denominator',
    'tests/test_clean_comparison.py::test_input_mismatch_blocks_scientific_validity',
    'tests/test_clean_comparison.py::test_interruption_before_simulator_consumes_attempt_and_reuses_seed',
    'tests/test_clean_comparison.py::test_partial_preserved_and_single_retry',
    'tests/test_clean_comparison.py::test_receipt_detects_artifact_tampering',
    'tests/test_clean_reuse.py::test_new_addendum_identity_retains_valid_frozen_episode_metadata',
    'tests/test_clean_reuse.py::test_original_comparison_requires_every_pair_and_its_original_report',
    'tests/test_clean_reuse.py::test_reuse_preserves_original_source_identity_and_bytes',
    'tests/test_clean_reuse.py::test_reuse_rejects_changed_original_evidence',
    'tests/test_continuation_freeze.py::test_arm_selection_requires_both_arms_and_prefers_a_on_exact_tie',
    'tests/test_continuation_freeze.py::test_concentrated_goal_gains_cannot_pass_task_breadth_guard',
    'tests/test_continuation_freeze.py::test_exact_goal_boundary_validity_and_other_suite_guard',
    'tests/test_continuation_freeze.py::test_missing_and_nonboolean_outcomes_cannot_be_used',
    'tests/test_continuation_freeze.py::test_panel_is_reproducible_from_identities_and_rejects_missing_states',
    'tests/test_continuation_freeze.py::test_real_freeze_validates_without_launching',
    'tests/test_coupled_evaluation.py::test_endpoint_and_real_contract',
    'tests/test_coupled_evaluation.py::test_real_training_metadata_provenance_is_accepted',
    'tests/test_coupled_evaluation.py::test_week9_actual_panel_freeze',
    'tests/test_dataset_freeze.py::test_committed_week2_freeze_evidence_is_digest_sealed',
    'tests/test_dataset_freeze.py::test_committed_week2_human_qa_completion_evidence_is_digest_sealed',
    'tests/test_dataset_freeze.py::test_committed_week2_qa_package_evidence_is_digest_sealed',
    'tests/test_evaluation_splits.py::test_committed_manifest_identities',
    'tests/test_evaluation_splits.py::test_manifest_validation_rejects_task_drift',
    'tests/test_evaluation_splits.py::test_small_splits_balance_suites',
    'tests/test_evaluation_splits.py::test_unlabeled_difficulty_policy',
    'tests/test_event_supervision.py::test_cross_split_duplicate_episode_rejected',
    'tests/test_event_supervision.py::test_real_selection_is_disjoint_train_only_and_covers_tasks_lengths',
    'tests/test_event_supervision.py::test_selection_ignores_order_and_visual_variant_order',
    'tests/test_eventground_future_onset.py::test_actual_reconstruction_and_matched_exposure',
    'tests/test_eventground_heldout_sampling.py::test_heldout_sampler_deterministic_causal_and_capped',
    'tests/test_eventground_qa.py::test_ambiguous_predictions_preserved_in_report_and_not_negative',
    'tests/test_eventground_qa.py::test_boundary_metrics_are_separate_start_and_end',
    'tests/test_eventground_qa.py::test_censored_intervals_not_used_for_boundary_accuracy',
    'tests/test_eventground_qa.py::test_false_negatives_and_false_positives_fail_point_gates',
    'tests/test_eventground_qa.py::test_falsified_hash_rejected',
    'tests/test_eventground_qa.py::test_full_synthetic_review_measures_without_admitting',
    'tests/test_eventground_qa.py::test_generated_or_nonindependent_review_rejected',
    'tests/test_eventground_qa.py::test_incomplete_coverage_rejected',
    'tests/test_eventground_qa.py::test_low_positive_coverage_is_not_fixed_by_duplicates',
    'tests/test_eventground_qa.py::test_manifest_requires_all_sources_and_all_tasks',
    'tests/test_eventground_qa.py::test_missing_censor_flag_rejected',
    'tests/test_eventground_qa.py::test_review_without_sealed_preparation_budget_is_rejected',
    'tests/test_eventground_qa.py::test_reviewer_registration_and_generator_identity_are_external',
    'tests/test_eventground_qa.py::test_templates_are_unreviewed_and_hide_candidates',
    'tests/test_eventground_report_extension.py::test_exact_new_sources_and_parent_exclusion',
    'tests/test_eventground_report_extension.py::test_identity_failures',
    'tests/test_eventground_report_extension.py::test_no_scope_expansion',
    'tests/test_eventground_weak_evaluation.py::test_gate_missing_metrics_never_passes',
    'tests/test_eventground_weak_evaluation.py::test_panel_rejects_cross_partition_and_reordering',
    'tests/test_eventground_weak_evaluation.py::test_real_frozen_pilot_unique_caps',
    'tests/test_eventground_weak_evaluation.py::test_real_weak_rewards_rederive_targets',
    'tests/test_eventground_weak_sft.py::test_cursor_handles_84_update_epochs',
    'tests/test_eventground_weak_sft.py::test_evidence_and_resource_admission_fail_closed',
    'tests/test_eventground_weak_sft.py::test_explicit_new_admission_keeps_old_independent_gate_closed',
    'tests/test_eventground_weak_sft.py::test_launch_scope_cannot_silently_expand',
    'tests/test_eventground_weak_sft.py::test_new_checkpoint_is_exactly_recoverable_and_write_bounded',
    'tests/test_eventground_weak_sft.py::test_two_passes_cover_each_prompt_once_with_unpadded_partial_batches',
    'tests/test_eventground_weak_supervision.py::test_actual_pool_reproduces_conservative_diagnosis_and_keeps_provenance',
    'tests/test_eventground_weak_supervision.py::test_current_window_ambiguity_ongoing_target_and_input_rejections',
    'tests/test_eventground_weak_supervision.py::test_pilot_bounds_fail_closed',
    'tests/test_eventground_weak_supervision.py::test_pinned_review_bytes_cannot_be_substituted',
    'tests/test_eventground_weak_supervision.py::test_protocol_cannot_silently_change_scope',
    'tests/test_eventground_weak_supervision.py::test_thresholds_and_bitwise_state_digest',
    'tests/test_extract_week10_production_candidates.py::test_real_capture_metadata_passes_new_child_gate',
    'tests/test_extract_week10_production_candidates.py::test_rejects_lineage_scope_qualification_or_coverage_damage',
    'tests/test_future_pairs.py::test_checked_in_rule_expands_to_exact_frozen_split',
    'tests/test_instrumented_partitions.py::test_exact_approved_memberships_and_reserve_gate',
    'tests/test_instrumented_partitions.py::test_no_heldout_or_unknown_source_can_enter_training',
    'tests/test_instrumented_partitions.py::test_partition_tampering_rejected',
    'tests/test_jepa_candidate_audit.py::test_independent_audit_real_aggregates_and_tampered_cell',
    'tests/test_production_capture.py::test_launch_fails_closed_before_any_recording',
    'tests/test_production_capture.py::test_launch_selects_only_initial_two_hundred',
    'tests/test_production_capture_continuation.py::test_continuation_preserves_prefix_and_remaining_control_budget',
    'tests/test_production_capture_continuation.py::test_continuation_rejects_retries_drops_or_resource_expansion',
    'tests/test_production_capture_startup.py::test_sealed_approval_is_preserved_before_simulator_import',
    'tests/test_reference_policy_evaluator.py::test_committed_sanity_35_evidence_is_digest_sealed',
    'tests/test_stage_capture_sources.py::test_staging_boundary_checks_before_network',
    'tests/test_week10_freeze.py::test_changed_parent_weights_cannot_be_admitted_by_resealing',
    'tests/test_week10_freeze.py::test_corrupted_qa_evidence_fails_closed',
    'tests/test_week10_freeze.py::test_extra_action_exposure_is_rejected',
    'tests/test_week10_freeze.py::test_real_freeze_keeps_labels_and_training_unadmitted',
    'tests/test_week10_freeze.py::test_resealed_close_only_scope_is_rejected',
    'tests/test_week10_freeze.py::test_resealed_partition_cannot_move_a_group_between_panels',
    'tests/test_week10_v3.py::test_active_v3_matches_approved_partition_and_preserves_controls',
    'tests/test_week10_v3.py::test_v3_verifier_rejects_unapproved_changes',
    'tests/test_week11_analysis.py::test_predeclared_subsets_keep_all_rows',
    'tests/test_week11_formal.py::test_both_time_caps_include_prior_costs',
    'tests/test_week11_formal.py::test_exact_queue_and_scope',
    'tests/test_week11_formal.py::test_projection_uses_full_suite_bound_and_reserve',
    'tests/test_week11_formal.py::test_resume_subtracts_only_sealed_panels',
    'tests/test_week11_horizon_freeze.py::test_changed_mapping_helper_rejected',
    'tests/test_week11_horizon_freeze.py::test_child_and_original_package_verify',
    'tests/test_week11_horizon_freeze.py::test_original_step1_package_remains_immutable',
    'tests/test_week11_horizon_freeze.py::test_resealed_mapping_change_rejected',
    'tests/test_week11_protocol.py::test_changed_parent_endpoint_rejected',
    'tests/test_week11_protocol.py::test_copied_bundle_verifies',
    'tests/test_week11_protocol.py::test_false_claim_of_remote_verification_rejected',
    'tests/test_week11_protocol.py::test_real_freeze_and_evidence_accounting',
    'tests/test_week11_protocol.py::test_resealed_episode_seed_change_rejected',
    'tests/test_week11_protocol.py::test_resealed_scope_change_cannot_replace_freeze',
    'tests/test_week3_completion.py::test_week3_completion_evidence_closes_every_step_and_verifies_ledger',
    'tests/test_week9_execution.py::test_clean_storage_preserves_fixed_state_and_adds_binding',
})


def pytest_addoption(parser):
    parser.addoption(
        "--research-artifacts",
        action="store_true",
        default=False,
        help="Run evidence audits that require separately restored research artifacts.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "research_artifacts: requires retained research files outside the code release"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    enabled = config.getoption("--research-artifacts")
    skip = pytest.mark.skip(
        reason="Requires external research artifacts; opt in with --research-artifacts"
    )
    for item in items:
        if item.nodeid.split("[", 1)[0] in RESEARCH_ARTIFACT_TESTS:
            item.add_marker(pytest.mark.research_artifacts)
        if not enabled and item.get_closest_marker("research_artifacts") is not None:
            item.add_marker(skip)
