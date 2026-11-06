"""Bindings for one approved preservation pilot and matched SFT-only control."""
from pathlib import Path

from eventground_wam.eventground_future_runtime import (
    verify as verify_parent, validate_training_completion as validate_run, load_endpoint,
)
from eventground_wam.eventground_preservation import preservation_weight
from eventground_wam.eventground_weak_supervision import sealed, sha

EXPERIMENT = "week10-preservation-pilot-v1"


def parent_base(base):
    return Path(base).parent / "week10-future-onset-v1"


def verify(base, *, rebuild=True):
    base = Path(base); p = sealed(base / "protocol.json"); parent = parent_base(base)
    if (p["experiment"] != EXPERIMENT or not p["user_approved"] or p["automatic_retries"] != 0
            or p["full_training_authorized"] or p["report_evaluation_authorized"]):
        raise ValueError("Pilot-only authorization changed")
    package = sealed(parent / "package-index.json")
    if package["digest"] != p["parent_package_digest"]:
        raise ValueError("Parent package changed")
    for name, digest in package["files_sha256"].items():
        if sha(parent / name) != digest:
            raise ValueError("Parent evidence changed: " + name)
    previous, index, manifests, recipe = verify_parent(parent, rebuild=rebuild)
    if previous["digest"] != p["parent_protocol_digest"] or previous["promotion"] != p["promotion"]:
        raise ValueError("Parent protocol or unchanged gates differ")
    if recipe["digest"] != p["recipe_digest"]:
        raise ValueError("Numerical recipe changed")
    for name, digest in p["source_files_sha256"].items():
        if sha(base / "source" / name) != digest:
            raise ValueError("Frozen child code changed: " + name)
    for name, digest in p["artifact_sha256"].items():
        if sha(base / name) != digest:
            raise ValueError("Frozen child artifact changed: " + name)
    if sealed(parent / "runs/sft/completion.json")["digest"] != p["sft_completion_digest"]:
        raise ValueError("Starting SFT completion changed")
    baseline = sealed(parent / "eval/dev/sft/completion.json")
    if baseline["digest"] != p["baseline_evaluation_digest"]:
        raise ValueError("Starting dev baseline changed")
    for panel, manifest in manifests.items():
        if sealed(base / f"{panel}-manifest.json") != manifest:
            raise ValueError("Panel changed: " + panel)
    a = sealed(base / "schedules/hybrid.json"); b = sealed(base / "schedules/control.json")
    previous_schedule = sealed(parent / "schedules/primary.json")
    if a["batches"] != b["batches"] or a["batches"] != previous_schedule["batches"]:
        raise ValueError("Matched 500-prompt exposure changed")
    return p, index, manifests, recipe


def selected_weight(base):
    base = Path(base); p = sealed(base / "protocol.json")
    d = sealed(base / "diagnostic/completion.json")
    if (d["status"] != "complete" or d["protocol_digest"] != p["digest"]
            or d["optimizer_updates"] != 0 or d["dev_examples_scored"] != 0
            or d["dev_metrics_used_for_weight"] or not all(d["gates"].values())
            or d["schedule_digest"] != sealed(base / "diagnostic-schedule.json")["digest"]
            or d["sft_completion_digest"] != p["sft_completion_digest"]
            or d["elapsed_seconds"] > p["diagnostic_resources"]["wall_seconds"]
            or d["peak_reserved_bytes"] > .9 * d["gpu_total_bytes"]):
        raise ValueError("Training-only diagnostic invalid")
    weight = preservation_weight(d["grpo_gradient_norms"], d["sft_gradient_norms"])
    if weight != d["selected_weight"]:
        raise ValueError("Weight selection differs from prospective rule")
    return weight, d


def validate_training_completion(base, stage):
    c = validate_run(base, stage); p = sealed(Path(base) / "protocol.json")
    weight, d = selected_weight(base)
    identity = sealed(Path(base) / "runs" / stage / "identity.json")
    if (identity["preservation_weight"] != weight or identity["diagnostic_digest"] != d["digest"]
            or identity["sft_completion_digest"] != p["sft_completion_digest"]
            or identity["mode"] != stage.split("-")[0]
            or c["identity_digest"] != identity["digest"]):
        raise ValueError("Objective or fresh SFT identity differs")
    return c


def require_stage(base, stage):
    selected_weight(base)
    if stage not in ("hybrid", "control", "hybrid-qualification", "control-qualification"):
        raise ValueError("Stage not authorized")
    if not stage.endswith("qualification"):
        validate_training_completion(base, stage + "-qualification")
