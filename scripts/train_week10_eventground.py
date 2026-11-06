"""Explicitly admitted Week 10 SFT, RLVR and matched-prompt control runner.

This script cannot upgrade candidates, synthetic diagnostics or an automatic
probe into permission to train. Every run needs linked independent label QA,
complete data, actual-host stage qualification and an explicit launch receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path

import torch

from eventground_wam.b0_training import (
    capture_rng,
    configure_deterministic_cuda,
    cpu_tree,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_data import (
    collate_current_observations,
    load_current_observation,
    target_tokens,
)
from eventground_wam.eventground_manifest import admit_manifest, require_sealed
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_objectives import (
    CALIBRATION_REWARD,
    PRIMARY_REWARD,
    structured_reward,
)
from eventground_wam.eventground_runtime import build_eventground_from_m4
from eventground_wam.eventground_training import (
    CHECKPOINT_FORMAT,
    EventGroundTrainer,
    require_training_admission,
)
from eventground_wam.pixel_predictor import tensor_state_digest

STAGES = ("sft", "rlvr-pilot", "rlvr-main", "control")
CLASSES = ("none", "gripper_close", "contact", "grasp")
QUALIFICATION_GATES = {
    "current_observation_only",
    "independent_labels_admitted",
    "exact_parameter_allowlist",
    "finite_losses_gradients",
    "small_set_overfit",
    "logical_batch_equivalence",
    "checkpoint_reload_recovery",
    "frozen_parameters_unchanged",
    "memory_with_optimizer",
    "measured_throughput",
    "stage_budget_feasible",
}
RL_QUALIFICATION_GATES = {
    "fixed_sft_reference_verified",
    "group_four_equivalence",
    "reward_evidence_verified",
    "reference_memory_and_throughput_measured",
}
COMPLETION_RESERVE_BYTES = 256 * 1024


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def output_usage(output):
    return sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file())


def require_output_capacity(
    output, resources, additional_bytes, *, reserve=COMPLETION_RESERVE_BYTES
):
    """Reserve the complete operation and final report before allocating disk bytes."""
    if additional_bytes < 0 or reserve < 0:
        raise ValueError("Output reservations must be nonnegative")
    if output_usage(output) + additional_bytes + reserve > resources["output_bytes"]:
        raise RuntimeError("Output reservation would exceed the attempt ceiling")
    if (
        shutil.disk_usage(output).free
        < resources["minimum_free_bytes"] + additional_bytes + reserve
    ):
        raise RuntimeError("Output reservation would breach minimum free storage")


class CountingWriter:
    """Count torch serialization without allocating a checkpoint or buffering it."""

    def __init__(self, limit):
        self.count = 0
        self.limit = limit

    def write(self, data):
        if self.count + len(data) > self.limit:
            raise RuntimeError("Serialized checkpoint exceeds the reserved byte bound")
        self.count += len(data)
        return len(data)

    def flush(self):
        pass


class BoundedWriter(CountingWriter):
    """Never allow an oversized serialization chunk to reach the filesystem."""

    def __init__(self, handle, limit, *, output, resources, sidecar_reservation):
        super().__init__(limit)
        self.handle = handle
        self.output, self.resources = output, resources
        self.sidecar_reservation = sidecar_reservation

    def write(self, data):
        if self.count + len(data) > self.limit:
            raise RuntimeError("Checkpoint serialization exceeded preflight measurement")
        require_output_capacity(
            self.output,
            self.resources,
            len(data),
            reserve=COMPLETION_RESERVE_BYTES + self.sidecar_reservation,
        )
        written = self.handle.write(data)
        if written != len(data):
            raise OSError("Incomplete checkpoint write")
        self.count += written
        return written

    def flush(self):
        self.handle.flush()


def save_bounded_checkpoint(trainer, path, *, identity, batches, output, resources):
    """Persist the trainer's recovery format with an exact pre-write capacity gate.

    Serialize the same immutable CPU payload first into a counting sink. The real
    write cannot exceed that measured size. SHA, cursor and completion metadata
    are reserved as well; the previous immutable checkpoint remains untouched.
    """
    path = Path(path)
    if trainer._in_update or path.exists():
        raise ValueError("Only a fresh complete-update checkpoint may be written")
    require_sealed(identity, "training identity")
    parameters = dict(trainer.model.named_parameters())
    payload = {
        "format": CHECKPOINT_FORMAT,
        "identity": identity,
        "recipe_digest": trainer.recipe["digest"],
        "stage": trainer.stage,
        "exact_names": trainer.names,
        "parameters": {n: parameters[n].detach().cpu().clone() for n in trainer.names},
        "optimizer": cpu_tree(trainer.optimizer.state_dict()),
        "rng": capture_rng(),
        "global_step": trainer.global_step,
        "prompts_seen": trainer.prompts_seen,
    }
    cursor_placeholder = seal_contract(checkpoint_cursor(trainer, identity, batches, "0" * 64))
    cursor_size = len(json_bytes(cursor_placeholder))
    sidecar_bytes = 65 + cursor_size
    counting = CountingWriter(resources["output_bytes"])
    torch.save(payload, counting)
    require_output_capacity(output, resources, counting.count + sidecar_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb", buffering=0) as handle:
            bounded = BoundedWriter(
                handle,
                counting.count,
                output=output,
                resources=resources,
                sidecar_reservation=sidecar_bytes,
            )
            torch.save(payload, bounded)
            bounded.flush()
            os.fsync(handle.fileno())
            if bounded.count != counting.count:
                raise RuntimeError("Checkpoint serialization size changed after preflight")
        digest = sha256_file(temporary)
        if path.exists():
            raise ValueError("Checkpoint destination appeared during write")
        os.replace(temporary, path)
        cursor = seal_contract(checkpoint_cursor(trainer, identity, batches, digest))
        encoded_cursor = json_bytes(cursor)
        if len(encoded_cursor) != cursor_size:
            raise RuntimeError("Checkpoint cursor size changed after reservation")
        require_output_capacity(output, resources, sidecar_bytes)
        with path.with_suffix(path.suffix + ".sha256").open("xb") as handle:
            handle.write((digest + "\n").encode())
        with path.with_suffix(path.suffix + ".cursor.json").open("xb") as handle:
            handle.write(encoded_cursor)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def write_completion(output, report, resources):
    """Charge the completion record itself; other writes preserve its reservation."""
    prior_bytes = output_usage(output)
    payload = dict(report)
    payload["output_bytes"] = prior_bytes
    for _ in range(8):
        encoded = json_bytes(seal_contract(payload))
        total = prior_bytes + len(encoded)
        if payload["output_bytes"] == total:
            break
        payload["output_bytes"] = total
    else:
        raise RuntimeError("Completion size accounting did not converge")
    if len(encoded) > COMPLETION_RESERVE_BYTES:
        raise RuntimeError("Completion exceeded its reserved metadata bytes")
    require_output_capacity(output, resources, len(encoded), reserve=0)
    with (Path(output) / "completion.json").open("xb") as handle:
        handle.write(encoded)


def write_sealed(path, payload):
    value = seal_contract(payload)
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return value


def schedule_rows(manifest, stage):
    """Fixed seed-17 hash order; control and main RL share exact prompt batches."""
    if stage not in STAGES:
        raise ValueError("Unknown formal stage")
    rows = manifest["main_rows"]
    if len(rows) != 2000 or len({r["prompt_id"] for r in rows}) != 2000:
        raise ValueError("Exactly 2000 unique main prompts required")
    if Counter(r["target"]["next_event"] for r in rows) != dict.fromkeys(CLASSES, 500):
        raise ValueError("Main must contain exactly 500 prompts per class")
    if stage == "rlvr-pilot":
        subset = manifest["pilot_prompt_ids"]
        if len(subset) != 500 or len(set(subset)) != 500:
            raise ValueError("Exactly 500 unique nested pilot prompts required")
        rows = [row for row in rows if row["prompt_id"] in set(subset)]
        if Counter(r["target"]["next_event"] for r in rows) != dict.fromkeys(CLASSES, 125):
            raise ValueError("Pilot must be nested with exactly 125 prompts per class")
    batches = []
    for epoch in range(2 if stage == "sft" else 1):
        ordered = sorted(
            rows,
            key=lambda r: (
                hashlib.sha256(f"17:{epoch}:{r['prompt_id']}".encode()).hexdigest(),
                r["prompt_id"],
            ),
        )
        batches.extend(ordered[offset : offset + 16] for offset in range(0, len(ordered), 16))
    return batches


def schedule_digest(batches):
    return contract_digest({"batches": [[row["prompt_id"] for row in batch] for batch in batches]})


def validate_qualification(value, *, stage, recipe, manifest, qa):
    require_sealed(value, "formal stage qualification")
    gates = QUALIFICATION_GATES | (RL_QUALIFICATION_GATES if stage.startswith("rlvr-") else set())
    if (
        value.get("artifact_type") != "eventground_wam.week10_training_qualification"
        or value.get("status") != "passed"
        or value.get("stage") != stage
        or value.get("recipe_digest") != recipe["digest"]
        or value.get("manifest_digest") != manifest["digest"]
        or value.get("qa_digest") != qa["digest"]
        or value.get("actual_host_resources_passed") is not True
        or value.get("correctness_passed") is not True
        or not value.get("evidence_files_sha256")
        or set(value.get("gates", {})) != gates
        or any(value["gates"][key] is not True for key in gates)
    ):
        raise ValueError(
            "Complete actual-host stage qualification required; gradient-only probes cannot admit training"
        )
    if (
        type(value.get("active_gpu_seconds")) not in (int, float)
        or not 0 < value["active_gpu_seconds"] <= 3 * 3600
        or not value.get("host")
    ):
        raise ValueError("Qualification host and bounded measured cost required")


def validate_gate_bundle(
    stage, study, recipe, manifest, diagnostic, qa, qualification, admission, authorization
):
    for record in (
        study,
        recipe,
        manifest,
        diagnostic,
        qa,
        qualification,
        admission,
        authorization,
    ):
        require_sealed(record)
    if admit_manifest(diagnostic, qa) != manifest:
        raise ValueError("Admitted manifest does not reproduce from linked independent QA")
    if manifest.get("training_authorized") is not False:
        raise ValueError("Dataset admission must not impersonate a training launch")
    if (
        recipe["study_digest"] != study["digest"]
        or recipe["logical_batch_size"] != 16
        or recipe["microbatch_size"] != 1
        or recipe["seed"] != 17
        or recipe.get("prompt_order") != "sha256_seed_epoch_prompt_id"
        or recipe.get("checkpoint_every_updates") != 25
    ):
        raise ValueError("Frozen numerical recipe/order/checkpoint schedule changed")
    validate_qualification(qualification, stage=stage, recipe=recipe, manifest=manifest, qa=qa)
    require_training_admission(
        admission,
        study_digest=study["digest"],
        recipe_digest=recipe["digest"],
        manifest_digest=manifest["digest"],
    )
    if (
        admission["qa_digest"] != qa["digest"]
        or admission["qualification_digest"] != qualification["digest"]
        or admission["launch_authorization_digest"] != authorization["digest"]
    ):
        raise ValueError("Training admission links differ from supplied evidence")
    if (
        authorization.get("artifact_type") != "eventground_wam.week10_formal_training_launch"
        or authorization.get("approved") is not True
        or authorization.get("stage") != stage
        or authorization.get("study_digest") != study["digest"]
        or authorization.get("recipe_digest") != recipe["digest"]
        or authorization.get("manifest_digest") != manifest["digest"]
        or authorization.get("qa_digest") != qa["digest"]
        or authorization.get("qualification_digest") != qualification["digest"]
        or authorization.get("training_authorized") is not True
        or not authorization.get("user_instruction")
        or not authorization.get("execution_id")
        or authorization.get("host") != qualification["host"]
    ):
        raise ValueError("Explicit linked formal launch authorization required")
    resources = authorization["resources"]
    if (
        type(resources["wall_seconds"]) is not int
        or resources["wall_seconds"] <= 0
        or not 0 <= resources["prior_active_gpu_seconds"]
        or resources["prior_active_gpu_seconds"] + resources["wall_seconds"] > 40 * 3600
        or resources["prior_active_gpu_seconds"] < qualification["active_gpu_seconds"]
        or resources.get("prior_output_bytes", -1) < 0
        or not 0 < resources["output_bytes"] + resources["prior_output_bytes"] <= 100 * 1024**3
        or resources["minimum_free_bytes"] < 50 * 1024**3
    ):
        raise ValueError("Launch expands frozen cumulative resource limits")
    batches = schedule_rows(manifest, stage)
    if authorization.get("schedule_digest") != schedule_digest(batches):
        raise ValueError("Launch does not bind the exact deterministic prompt batches")
    expected_updates = 250 if stage == "sft" else 32 if stage == "rlvr-pilot" else 125
    expected_prompts = 4000 if stage == "sft" else 500 if stage == "rlvr-pilot" else 2000
    if (
        authorization.get("optimizer_updates") != expected_updates
        or authorization.get("prompt_presentations") != expected_prompts
    ):
        raise ValueError("Fixed endpoint/exposure changed")
    return batches


def validate_fixed_sft(completion, *, recipe, manifest):
    require_sealed(completion, "SFT completion")
    if (
        completion.get("artifact_type") != "eventground_wam.week10_training_completion"
        or completion.get("status") != "complete"
        or completion.get("stage") != "sft"
        or completion.get("recipe_digest") != recipe["digest"]
        or completion.get("manifest_digest") != manifest["digest"]
        or completion.get("global_step") != 250
        or completion.get("prompts_seen") != 4000
        or completion.get("selection") != "fixed_final_endpoint"
        or completion.get("frozen_parameters_unchanged") is not True
    ):
        raise ValueError("A verified fixed-final SFT endpoint is required")


def validate_promotion(stage, authorization, promotion, sft_completion, manifest):
    """A new sealed review gates RL expansion or the one predeclared calibration."""
    if stage == "sft":
        return
    if authorization.get("sft_completion_digest") != sft_completion["digest"]:
        raise ValueError("Launch did not bind the fixed SFT reference")
    variant = authorization.get("reward_variant", "primary")
    if variant not in ("primary", "calibration"):
        raise ValueError("Unknown reward variant")
    if stage == "rlvr-pilot" and variant == "primary":
        return  # the first nested pilot has its own formal launch/qualification gate
    if promotion is None:
        raise ValueError("Independent RL promotion/control receipt required")
    require_sealed(promotion, "RL promotion/control gate")
    if (
        promotion.get("artifact_type") != "eventground_wam.week10_rl_stage_gate"
        or promotion.get("integrity_passed") is not True
        or promotion.get("manifest_digest") != manifest["digest"]
        or promotion.get("sft_completion_digest") != sft_completion["digest"]
        or authorization.get("promotion_digest") != promotion["digest"]
        or not promotion.get("evaluation_digest")
    ):
        raise ValueError("Linked independently evaluated RL stage gate required")
    if stage == "rlvr-pilot":
        if (
            promotion.get("decision") != "one_predeclared_calibration"
            or promotion.get("prior_primary_pilot_performance_passed") is not False
            or promotion.get("calibrations_used") != 0
            or promotion.get("pilot_prompt_ids") != manifest["pilot_prompt_ids"]
        ):
            raise ValueError(
                "Only one performance-failure calibration from the same SFT/prompt set is allowed"
            )
    elif stage == "rlvr-main":
        metrics = promotion.get("metrics", {})
        if (
            promotion.get("decision") != "promote_main"
            or promotion.get("reward_variant") != variant
            or promotion.get("calibrations_used") not in (0, 1)
            or metrics.get("format_validity") != 1.0
            or metrics.get("positive_macro_iou_gain", -1) < 0.01
            or metrics.get("matched_positive_start_mae_worsening_frames", 99) > 1
            or metrics.get("none_false_positive_increase", 99) > 0.02
        ):
            raise ValueError("Frozen event-dev promotion thresholds have not passed")
    elif stage == "control" and (
        promotion.get("decision") != "matched_main_control"
        or not promotion.get("main_launch_authorization_digest")
        or promotion.get("main_schedule_digest") != authorization["schedule_digest"]
    ):
        raise ValueError("Control must match the admitted main-RL prompt batches")


def validate_reward_evidence(evidence, *, manifest, qa):
    """Check shape/lineage only; validate_bound_reward_evidence admits actual windows."""
    require_sealed(evidence, "independent offline reward evidence")
    if (
        evidence.get("artifact_type") != "eventground_wam.admitted_eventground_reward_evidence"
        or evidence.get("manifest_digest") != manifest["digest"]
        or evidence.get("qa_digest") != qa["digest"]
        or evidence.get("independently_reviewed") is not True
        or evidence.get("labels_admitted") is not True
        or not evidence.get("annotation_files_sha256")
    ):
        raise ValueError("Independent admitted reward evidence is missing")
    mapping = evidence["prompts"]
    if set(mapping) != {r["prompt_id"] for r in manifest["main_rows"]}:
        raise ValueError("Reward evidence must cover the exact main prompt universe")
    for row in manifest["main_rows"]:
        item = mapping[row["prompt_id"]]
        if (
            item.get("target") != row["target"]
            or item.get("source_id") != row["source_id"]
            or item.get("anchor") != row["anchor"]
            or item.get("complete_window") is not True
        ):
            raise ValueError("Reward evidence target/source/window mismatch")
        interval = item.get("verified_interval")
        structured_reward(
            row["target"],
            row["target"],
            verified_interval=tuple(interval) if interval is not None else None,
            evidence_digest=item["evidence_digest"],
            complete_window=True,
        )
    return mapping


def validate_bound_reward_evidence(evidence, *, manifest, qa, annotation_root):
    """Reconstruct each target from actual independent annotations before RLVR."""
    try:
        from eventground_wam.eventground_reward_admission import verify_reward_evidence_binding
    except ImportError as error:
        raise ValueError(
            "Per-prompt independent reward-evidence verifier is unavailable"
        ) from error
    structural = validate_reward_evidence(evidence, manifest=manifest, qa=qa)
    verified = verify_reward_evidence_binding(
        evidence, manifest, qa, annotation_root=annotation_root
    )
    if verified != structural:
        raise ValueError("Independently reconstructed reward evidence differs from declared rows")
    return verified


def reward_function(weights):
    def score(evidence, sampled_tokens):
        predictions = serialize_tokens(sampled_tokens)
        if len(predictions) != 4 * len(evidence):
            raise ValueError("Reward groups must remain exactly four per prompt")
        rewards = []
        for item, offset in zip(evidence, range(0, len(predictions), 4), strict=True):
            interval = item["verified_interval"]
            rewards.append(
                [
                    structured_reward(
                        prediction,
                        item["target"],
                        verified_interval=tuple(interval) if interval is not None else None,
                        evidence_digest=item["evidence_digest"],
                        complete_window=True,
                        weights=weights,
                    ).total
                    for prediction in predictions[offset : offset + 4]
                ]
            )
        return torch.tensor(rewards, dtype=torch.float32, device=sampled_tokens.device)

    return score


def load_sft_parameters(model, trainer, completion, recipe, manifest):
    validate_fixed_sft(completion, recipe=recipe, manifest=manifest)
    path = Path(completion["checkpoint_path"])
    if sha256_file(path) != completion["checkpoint_sha256"]:
        raise ValueError("SFT endpoint bytes changed")
    cursor_path = path.with_suffix(path.suffix + ".cursor.json")
    if sha256_file(cursor_path) != completion["checkpoint_cursor_sha256"]:
        raise ValueError("SFT endpoint cursor bytes changed")
    cursor = read_sealed(cursor_path)
    if (
        cursor["checkpoint_sha256"] != completion["checkpoint_sha256"]
        or cursor["identity_digest"] != completion["identity_digest"]
        or cursor["global_step"] != 250
        or cursor["prompts_seen"] != 4000
        or cursor["next_prompt_ids"]
    ):
        raise ValueError("SFT endpoint cursor is not the fixed final boundary")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved["format"] != CHECKPOINT_FORMAT
        or saved["stage"] != "sft"
        or saved["recipe_digest"] != recipe["digest"]
        or saved["global_step"] != 250
        or saved["prompts_seen"] != 4000
        or saved["identity"]["digest"] != completion["identity_digest"]
        or saved["identity"].get("digest") != contract_digest(saved["identity"])
        or tuple(saved["exact_names"]) != trainer.names
        or set(saved["parameters"]) != set(trainer.names)
    ):
        raise ValueError("SFT endpoint identity/exposure/allowlist differs")
    live = dict(model.named_parameters())
    for name, value in saved["parameters"].items():
        if (
            value.shape != live[name].shape
            or value.dtype != live[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid SFT endpoint tensor")
    with torch.no_grad():
        for name, value in saved["parameters"].items():
            live[name].copy_(value.to(live[name].device))


def validate_recovery(saved_identity, recovery, *, authorization, checkpoint, output):
    require_sealed(recovery, "explicit recovery authorization")
    require_sealed(saved_identity, "original training identity")
    if (
        recovery.get("artifact_type") != "eventground_wam.week10_training_recovery_authorization"
        or recovery.get("approved") is not True
        or recovery.get("original_launch_authorization_digest") != authorization["digest"]
        or recovery.get("identity_digest") != saved_identity["digest"]
        or recovery.get("checkpoint_sha256") != sha256_file(checkpoint)
        or recovery.get("new_output") != str(Path(output).resolve())
        or recovery.get("preserve_all_previous_attempts") is not True
        or not recovery.get("prior_failed_completion_digest")
        or not recovery.get("user_instruction")
    ):
        raise ValueError("Recovery needs an explicit identity/checkpoint-bound new attempt")
    previous = read_sealed(Path(recovery["prior_failed_completion_path"]))
    cursor = read_sealed(Path(checkpoint).with_suffix(Path(checkpoint).suffix + ".cursor.json"))
    if (
        previous["digest"] != recovery["prior_failed_completion_digest"]
        or previous["status"] != "failed_preserved"
        or previous["identity_digest"] != saved_identity["digest"]
        or cursor["identity_digest"] != saved_identity["digest"]
        or cursor["checkpoint_sha256"] != recovery["checkpoint_sha256"]
        or recovery.get("maximum_discarded_updates", -1)
        < previous["global_step"]
        - cursor["global_step"]
        + int(previous.get("partial_update_in_progress", False))
        or recovery["resources"]["prior_active_gpu_seconds"]
        < previous["cumulative_active_gpu_seconds"]
        or recovery["resources"]["prior_output_bytes"]
        < previous["prior_output_bytes"] + previous["output_bytes"]
    ):
        raise ValueError(
            "Recovery must preserve and charge the prior failed attempt and discarded updates"
        )


def checkpoint_cursor(trainer, identity, batches, checkpoint_sha256):
    """Bind a complete update boundary to exact seed, epoch and next prompt batch."""
    step = trainer.global_step
    if not 0 <= step <= len(batches) or trainer.prompts_seen != sum(map(len, batches[:step])):
        raise ValueError("Cannot checkpoint an inconsistent prompt cursor")
    return {
        "artifact_type": "eventground_wam.week10_training_cursor",
        "identity_digest": identity["digest"],
        "schedule_digest": schedule_digest(batches),
        "seed": 17,
        "global_step": step,
        "prompts_seen": trainer.prompts_seen,
        "epoch_index": step // 125 if identity["stage"] == "sft" else int(step == len(batches)),
        "batch_index_in_epoch": step % 125 if identity["stage"] == "sft" else step,
        "next_prompt_ids": [r["prompt_id"] for r in batches[step]] if step < len(batches) else [],
        "checkpoint_sha256": checkpoint_sha256,
    }


def execute_schedule(trainer, batches, make_microbatches, *, update, record_step, checkpoint):
    """Pure orchestration tested with a fake trainer; never replay completed prefixes."""
    step = trainer.global_step
    if not 0 <= step <= len(batches) or trainer.prompts_seen != sum(map(len, batches[:step])):
        raise ValueError("Recovery counters do not match the exact completed prompt prefix")
    for index in range(step, len(batches)):
        batch = batches[index]
        metrics = update(make_microbatches(batch), len(batch))
        if trainer.global_step != index + 1 or trainer.prompts_seen != sum(
            map(len, batches[: index + 1])
        ):
            raise ValueError("Trainer counters diverged from fixed schedule")
        record_step(index, batch, metrics)
        if trainer.global_step % 25 == 0 or trainer.global_step == len(batches):
            checkpoint(trainer.global_step)


def check_gpu_exclusive():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        pid, memory = (v.strip() for v in line.split(","))
        if int(pid) != os.getpid() and float(memory) > 256:
            raise ValueError("Another substantial GPU process is running")


def verify_evidence_files(record, root):
    """Evidence pins are checked from disk, not accepted as decorative strings."""
    mapping = record.get("evidence_files_sha256", record.get("annotation_files_sha256", {}))
    if not mapping:
        raise ValueError("Referenced evidence files are required")
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(root) / path
        if sha256_file(path) != expected:
            raise ValueError(f"Linked evidence bytes changed: {name}")


def validate_stage_inputs(
    stage,
    *,
    sft_completion=None,
    promotion=None,
    reward_evidence=None,
    resume=None,
    recovery_authorization=None,
):
    """Reject incompatible initialization inputs before reading any artifacts.

    Initial SFT always starts with the verified M4 child. Its only continuation
    path is a separately authorized exact recovery checkpoint, never an SFT
    endpoint that would silently add another two training passes.
    """
    if stage not in STAGES:
        raise ValueError("Unknown formal stage")
    if stage == "sft" and any(
        value is not None for value in (sft_completion, promotion, reward_evidence)
    ):
        raise ValueError("Initial SFT forbids SFT endpoint, promotion and reward-evidence inputs")
    if stage != "sft" and sft_completion is None:
        raise ValueError("Non-SFT stages require a completed fixed SFT endpoint")
    if (resume is None) != (recovery_authorization is None):
        raise ValueError(
            "Recovery checkpoint and explicit recovery authorization are required together"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    for name in (
        "root",
        "recipe",
        "manifest",
        "diagnostic",
        "qa",
        "qualification",
        "admission",
        "authorization",
        "cache",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "sft-completion",
        "promotion",
        "reward-evidence",
        "resume",
        "recovery-authorization",
    ):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    validate_stage_inputs(
        args.stage,
        sft_completion=args.sft_completion,
        promotion=args.promotion,
        reward_evidence=args.reward_evidence,
        resume=args.resume,
        recovery_authorization=args.recovery_authorization,
    )
    study = read_sealed(args.root / "configs/training/week10_eventground_experiment_v3.json")
    recipe, manifest, diagnostic, qa, qualification, admission, authorization = (
        read_sealed(getattr(args, key))
        for key in (
            "recipe",
            "manifest",
            "diagnostic",
            "qa",
            "qualification",
            "admission",
            "authorization",
        )
    )
    batches = validate_gate_bundle(
        args.stage, study, recipe, manifest, diagnostic, qa, qualification, admission, authorization
    )
    verify_evidence_files(qualification, args.qualification.parent)
    if socket.gethostname() != authorization["host"]:
        raise ValueError("Execution host differs from the qualified/authorized host")
    if sha256_file(Path(__file__)) != recipe["code_sha256"]["scripts/train_week10_eventground.py"]:
        raise ValueError("Formal runner was not frozen in the recipe")
    sft = read_sealed(args.sft_completion) if args.sft_completion else None
    promotion = read_sealed(args.promotion) if args.promotion else None
    reward = read_sealed(args.reward_evidence) if args.reward_evidence else None
    if args.stage != "sft":
        if sft is None:
            raise ValueError("Non-SFT stages require a completed fixed SFT endpoint")
        validate_fixed_sft(sft, recipe=recipe, manifest=manifest)
        validate_promotion(args.stage, authorization, promotion, sft, manifest)
    reward_rows = None
    if args.stage.startswith("rlvr-"):
        if reward is None:
            raise ValueError("RLVR requires independently reviewed offline reward evidence")
        verifier_path = "src/eventground_wam/eventground_reward_admission.py"
        if (
            verifier_path not in recipe["code_sha256"]
            or sha256_file(args.root / verifier_path) != recipe["code_sha256"][verifier_path]
        ):
            raise ValueError("Per-prompt reward-evidence verifier must be frozen in the recipe")
        reward_rows = validate_bound_reward_evidence(
            reward, manifest=manifest, qa=qa, annotation_root=args.reward_evidence.parent
        )
        if authorization.get("reward_evidence_digest") != reward["digest"]:
            raise ValueError("Launch/reward evidence link differs")
    if bool(args.resume) != bool(args.recovery_authorization):
        raise ValueError(
            "Recovery checkpoint and explicit recovery authorization are required together"
        )
    if not args.resume and str(args.output.resolve()) != authorization.get("output"):
        raise ValueError("Output differs from the unique authorized attempt")
    resources = authorization["resources"]
    recovery = read_sealed(args.recovery_authorization) if args.recovery_authorization else None
    if recovery:
        replacement = recovery["resources"]
        if (
            not 0 < replacement["wall_seconds"] <= resources["wall_seconds"]
            or replacement["prior_active_gpu_seconds"] < resources["prior_active_gpu_seconds"]
            or replacement["prior_active_gpu_seconds"] + replacement["wall_seconds"] > 40 * 3600
            or replacement["output_bytes"] > resources["output_bytes"]
            or replacement.get("prior_output_bytes", -1) < resources["prior_output_bytes"]
            or replacement["output_bytes"] + replacement["prior_output_bytes"] > 100 * 1024**3
            or replacement["minimum_free_bytes"] < resources["minimum_free_bytes"]
        ):
            raise ValueError("Recovery expands cumulative resource limits")
        resources = replacement
    check_gpu_exclusive()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "artifact_type": "eventground_wam.week10_training_completion",
        "stage": args.stage,
        "status": "failed_preserved",
        "recipe_digest": recipe["digest"],
        "manifest_digest": manifest["digest"],
        "admission_digest": admission["digest"],
        "launch_authorization_digest": authorization["digest"],
        "selection": "fixed_final_endpoint",
        "schedule_digest": schedule_digest(batches),
        "parent_policy_sha256": study["parent"]["policy_sha256"],
        "heldout_payload_reads": 0,
        "automatic_retries": 0,
        "recovery_authorization_digest": recovery["digest"] if recovery else None,
    }
    trainer = None
    latest_checkpoint = None

    def budget(*_):
        if time.monotonic() - start > resources["wall_seconds"]:
            raise TimeoutError("Formal stage wall limit exceeded")
        require_output_capacity(args.output, resources, 0)
        if torch.cuda.is_initialized() and (
            torch.cuda.max_memory_reserved()
            > 0.9 * torch.cuda.get_device_properties(0).total_memory
        ):
            raise RuntimeError("Frozen GPU memory fraction exceeded")

    def alarm(*_):
        raise TimeoutError("Formal stage active-time limit exceeded")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(resources["wall_seconds"]))
    try:
        budget()
        configure_deterministic_cuda()
        seed_training(17)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        model, builder, allowlist, parent_identity = build_eventground_from_m4(
            args.root, recipe, args.cache, device
        )
        if {k: list(v) for k, v in allowlist.items()} != recipe["exact_allowlist"]:
            raise ValueError("Actual trainable/frozen inventory differs from frozen recipe")
        trainer = EventGroundTrainer(
            model, allowlist, recipe, stage="rlvr" if args.stage.startswith("rlvr-") else args.stage
        )
        if args.stage != "sft":
            load_sft_parameters(model, trainer, sft, recipe, manifest)
        reference = None
        if args.stage.startswith("rlvr-"):
            reference = copy.deepcopy(model)
            reference.requires_grad_(False)
            reference.eval()
        identity = seal_contract(
            {
                "artifact_type": "eventground_wam.week10_training_identity",
                "stage": args.stage,
                "execution_id": authorization["execution_id"],
                "recipe_digest": recipe["digest"],
                "manifest_digest": manifest["digest"],
                "launch_authorization_digest": authorization["digest"],
                "qa_digest": qa["digest"],
                "qualification_digest": qualification["digest"],
                "schedule_digest": schedule_digest(batches),
                "seed": 17,
                "logical_batch_size": 16,
                "pass_count": 2 if args.stage == "sft" else 1,
                "parent_identity": parent_identity,
                "sft_completion_digest": sft["digest"] if sft else None,
                "reward_evidence_digest": reward["digest"] if reward else None,
                "reward_variant": authorization.get("reward_variant", "primary"),
            }
        )
        if args.resume:
            validate_recovery(
                identity,
                recovery,
                authorization=authorization,
                checkpoint=args.resume,
                output=args.output,
            )
            trainer.resume(args.resume, identity=identity)
            cursor = read_sealed(args.resume.with_suffix(args.resume.suffix + ".cursor.json"))
            expected_cursor = seal_contract(
                checkpoint_cursor(
                    trainer,
                    identity,
                    batches,
                    sha256_file(args.resume),
                )
            )
            if cursor != expected_cursor:
                raise ValueError(
                    "Recovery cursor differs from exact epoch/order/batch continuation"
                )
            if trainer.global_step >= len(batches):
                raise ValueError("Completed endpoints cannot be resumed or extended")
        require_output_capacity(args.output, resources, len(json_bytes(identity)))
        write_sealed(
            args.output / "identity.json", {k: v for k, v in identity.items() if k != "digest"}
        )
        report["identity_digest"] = identity["digest"]
        parameters = dict(model.named_parameters())
        frozen = {n: p for n, p in parameters.items() if n not in trainer.names}
        frozen_before = tensor_state_digest(frozen)
        reference_before = (
            tensor_state_digest(dict(reference.named_parameters())) if reference else None
        )
        verified_files = {}

        def microbatches(batch):
            for row in batch:  # exactly one current observation per physical microbatch
                budget()
                observation = load_current_observation(row, verified_files=verified_files)
                encoded, state = collate_current_observations([observation], builder, device)
                if reward_rows is not None:
                    yield encoded, state, [reward_rows[row["prompt_id"]]]
                else:
                    yield encoded, state, target_tokens([row], device)

        weights = (
            CALIBRATION_REWARD
            if authorization.get("reward_variant") == "calibration"
            else PRIMARY_REWARD
        )
        if args.stage.startswith("rlvr-"):
            update = lambda data, count: trainer.rlvr_update(
                data,
                prompt_count=count,
                reference_model=reference,
                reward_function=reward_function(weights),
            )
        else:
            update = lambda data, count: trainer.sft_update(data, prompt_count=count)

        def record_step(index, batch, metrics):
            budget()
            encoded = (
                json.dumps(
                    {
                        "update_index": index,
                        "prompt_ids": [r["prompt_id"] for r in batch],
                        "elapsed_seconds": time.monotonic() - start,
                        **metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            ).encode()
            require_output_capacity(args.output, resources, len(encoded))
            with (args.output / "metrics.jsonl").open("ab") as stream:
                stream.write(encoded)
                stream.flush()
            print(json.dumps({"stage": args.stage, **metrics}), flush=True)

        def checkpoint(step):
            nonlocal latest_checkpoint
            path = args.output / "checkpoints" / f"update-{step:06d}.pt"
            save_bounded_checkpoint(
                trainer,
                path,
                identity=identity,
                batches=batches,
                output=args.output,
                resources=resources,
            )
            latest_checkpoint = path
            budget()

        execute_schedule(
            trainer,
            batches,
            microbatches,
            update=update,
            record_step=record_step,
            checkpoint=checkpoint,
        )
        if tensor_state_digest(frozen) != frozen_before:
            raise ValueError("Frozen model parameters changed")
        if (
            reference
            and tensor_state_digest(dict(reference.named_parameters())) != reference_before
        ):
            raise ValueError("Fixed SFT reference parameters changed")
        if latest_checkpoint is None:
            raise ValueError("No fixed final checkpoint produced")
        report.update(
            status="complete",
            checkpoint_path=str(latest_checkpoint.resolve()),
            checkpoint_sha256=sha256_file(latest_checkpoint),
            checkpoint_cursor_sha256=sha256_file(
                latest_checkpoint.with_suffix(latest_checkpoint.suffix + ".cursor.json")
            ),
            frozen_parameters_unchanged=True,
            fixed_reference_unchanged=True if reference else None,
            sampled_completions=trainer.prompts_seen * 4 if reference else 0,
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:4096]
        raise
    finally:
        signal.alarm(0)
        report.update(
            elapsed_seconds=time.monotonic() - start,
            cumulative_active_gpu_seconds=resources["prior_active_gpu_seconds"]
            + time.monotonic()
            - start,
            output_bytes=sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
            prior_output_bytes=resources["prior_output_bytes"],
            partial_update_in_progress=trainer._in_update if trainer else False,
            global_step=trainer.global_step if trainer else 0,
            prompts_seen=trainer.prompts_seen if trainer else 0,
            last_recovery_checkpoint=str(latest_checkpoint.resolve())
            if latest_checkpoint
            else None,
        )
        write_completion(args.output, report, resources)


if __name__ == "__main__":
    main()
