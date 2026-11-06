"""Fail-closed launch gate for the frozen initial 200 canonical recordings."""

from __future__ import annotations

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.instrumented_partitions import require_train_source


def initial_capture_targets(contract, split, authorization):
    for value in (contract, split, authorization):
        if value.get("digest") != contract_digest(value):
            raise ValueError("Changed sealed capture input")
    if contract["partition_digest"] != split["digest"]:
        raise ValueError("Wrong source partition contract")
    if (
        not authorization.get("approved")
        or authorization.get("scope") != "initial_200_production_capture"
        or authorization.get("capture_contract_digest") != contract["digest"]
    ):
        raise ValueError("Explicit approval of this exact initial capture contract required")
    targets = contract["targets"]
    expected = sorted(
        (r for r in split["rows"] if r["partition"] == "train"), key=lambda r: r["queue_rank"]
    )
    if [r["source_id"] for r in targets] != [r["source_id"] for r in expected]:
        raise ValueError("Capture source membership or order changed")
    for item in targets:
        row = require_train_source(split, item["source_id"], allow_reserve=True)
        if any(item[k] != row[k] for k in ("frames", "queue_role", "task_key", "demo_id")):
            raise ValueError("Capture source metadata changed")
    initial = [r for r in targets if r["queue_role"] == "initial_review"]
    if len(initial) != 200 or authorization.get("source_ids") != [r["source_id"] for r in initial]:
        raise ValueError("Approval must cover the exact initial 200, not reserves")
    if contract.get("training_authorized") or authorization.get("training_authorized"):
        raise ValueError("Capture does not authorize training")
    if contract["resources"]["control_steps"] != sum(r["frames"] + 10 for r in initial):
        raise ValueError("Wrong initial control-step ceiling")
    if (
        contract["resources"]["control_steps"] > 54033
        or contract["resources"]["wall_seconds"] > 7200
    ):
        raise ValueError("Approved capture budget exceeded")
    return initial


def continuation_capture_targets(contract, split, authorization, prior):
    """Preserve the exact completed prefix and admit only the one approved failed initialization."""
    targets = initial_capture_targets(contract, split, authorization)
    if prior.get("digest") != contract_digest(prior):
        raise ValueError("Changed prior completion")
    continuation = contract["continuation"]
    if (
        continuation["prior_completion_digest"] != prior["digest"]
        or prior["status"] != "incomplete"
    ):
        raise ValueError("Wrong prior execution")
    completed = [r for r in prior["results"] if r["status"] == "completed"]
    failed = [r for r in prior["results"] if r["status"] != "completed"]
    if len(completed) != 15 or len(failed) != 1 or failed[0]["control_steps"] != 0:
        raise ValueError("Only the preserved 15 and one zero-step failure may continue")
    if [r["recording_id"] for r in completed] != [r["recording_id"] for r in targets[:15]]:
        raise ValueError("Completed prefix cannot be dropped or repeated")
    if continuation["retained_recording_ids"] != [r["recording_id"] for r in completed]:
        raise ValueError("Wrong retained recordings")
    if failed[0]["source_id"] != targets[15]["source_id"]:
        raise ValueError("Wrong failed source")
    if authorization.get("retry_failed_zero_step_source") != failed[0]["source_id"]:
        raise ValueError("Specific failed initialization retry approval required")
    if not authorization.get("repair_completed_images_from_saved_states"):
        raise ValueError("Saved-state image repair approval required")
    resources = contract["resources"]
    if resources["wall_seconds"] + resources["prior_seconds_charged"] > 7200:
        raise ValueError("Prior execution must be charged to the time ceiling")
    if resources["output_bytes"] + resources["prior_output_bytes_charged"] > 100 * 1024**3:
        raise ValueError("Prior evidence must be charged to the output ceiling")
    if (
        resources["prior_control_steps"] != prior["control_steps"]
        or resources["remaining_control_steps"] != sum(r["frames"] + 10 for r in targets[15:])
        or resources["prior_control_steps"] + resources["remaining_control_steps"]
        != resources["control_steps"]
    ):
        raise ValueError("Continuation must share the original control-step ceiling")
    return targets
