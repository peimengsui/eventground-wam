"""Trial identities, durable attempt journals and verified comparison receipts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from eventground_wam.clean_acceptance import read_sealed
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import (
    EpisodeMetadata,
    read_episode_metadata,
    sha256_array,
    sha256_file,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import protocol_sha256


class ComparisonIntegrityError(ValueError):
    """Stop instead of silently changing provenance or rerunning a policy outcome."""


@dataclass(frozen=True, kw_only=True)
class ComparisonMetadata(EpisodeMetadata):
    comparison_id: str
    acceptance_rule_digest: str
    pair_id: str
    trial_index: int
    initial_state_index: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metadata_for_pair(pair: Any, protocol: Any, identity: Any, bddl: Path) -> ComparisonMetadata:
    return ComparisonMetadata(
        protocol_id=protocol["protocol_id"], protocol_sha256=protocol_sha256(protocol),
        project_git_revision=identity["project_git_revision"],
        benchmark_revision=protocol["benchmark"]["revision"],
        checkpoint_id=identity["checkpoint_id"], checkpoint_revision=identity["checkpoint_revision"],
        suite=pair["suite"], runtime_task_index=pair["runtime_task_index"],
        classification_task_id=pair["classification_task_id"], perturbation_category="Clean",
        difficulty_level=None, instruction=pair["instruction"], bddl_path=str(bddl),
        bddl_sha256=pair["bddl_sha256"], initial_state_sha256=pair["initial_state_sha256"],
        episode_seed=pair["episode_seed"], comparison_id=protocol["protocol_id"],
        acceptance_rule_digest=identity["acceptance_rule_digest"], pair_id=pair["pair_id"],
        trial_index=pair["trial_index"], initial_state_index=pair["initial_state_index"],
    )


def paths_for_pair(root: Path, ordinal: int, pair: Any, attempt: int = 0) -> tuple[Path, Path, Path]:
    stem = (f"{ordinal:04d}-{pair['suite']}-task-{pair['runtime_task_index']:02d}"
            f"-trial-{pair['trial_index']:02d}")
    return (
        root / "episodes" / f"{stem}-attempt-{attempt}.hdf5",
        root / "attempts" / f"{stem}-attempt-{attempt}.json",
        root / "receipts" / f"{stem}.json",
    )


def inspect_artifact(path: Path, protocol: Any, expected: ComparisonMetadata) -> dict[str, Any]:
    summary = validate_episode_file(path, protocol)
    actual = read_episode_metadata(path)
    for key, value in asdict(expected).items():
        if key not in {"success", "termination_reason"} and actual.get(key) != value:
            raise ComparisonIntegrityError(f"{path}: metadata mismatch for {key}")
    settling = protocol["simulation"]["settling_steps"]
    with h5py.File(path, "r") as artifact:
        prefix = {
            key: sha256_array(artifact[f"steps/{key}"][:settling])
            for key in ("agent_rgb", "wrist_rgb", "proprioception", "mujoco_state")
        }
        first_input = None
        if summary["policy_step_count"]:
            first_input = {key: sha256_array(artifact[f"steps/{key}"][settling])
                           for key in ("agent_rgb", "wrist_rgb", "proprioception")}
        latency = artifact["steps/policy_latency_ms"][:]
    return {
        "artifact": str(path), "artifact_sha256": sha256_file(path),
        "success": summary["success"], "termination_reason": summary["termination_reason"],
        "policy_step_count": summary["policy_step_count"], "step_count": summary["step_count"],
        "initial_input_hashes": prefix, "first_policy_input_hashes": first_input,
        "policy_latencies_ms": [float(x) for x in latency if np.isfinite(x)],
    }


def verify_receipt(
    receipt: Any, *, root: Path, ordinal: int, pair: Any, identity: Any,
    protocol: Any, metadata: ComparisonMetadata,
) -> dict[str, Any]:
    if (receipt["identity_digest"] != identity["digest"] or receipt["pair_id"] != pair["pair_id"]
            or receipt["ordinal"] != ordinal or receipt["pair"] != pair):
        raise ComparisonIntegrityError("saved pair receipt identity mismatch")
    attempt = receipt["attempt_index"]
    if type(attempt) is not int or not 0 <= attempt <= 1:
        raise ComparisonIntegrityError("receipt retry allowance changed")
    if receipt["attempt_evidence"] != collect_attempt_evidence(root, ordinal, pair):
        raise ComparisonIntegrityError("attempt evidence inventory changed")
    if [x["attempt_index"] for x in receipt["attempt_evidence"]] != list(range(attempt + 1)):
        raise ComparisonIntegrityError("attempt sequence changed")
    for entry in receipt["attempt_evidence"]:
        _, journal, _ = paths_for_pair(root, ordinal, pair, entry["attempt_index"])
        validate_journal(journal, identity, pair, entry["attempt_index"])
        for name in ("journal", "partial"):
            if entry.get(name):
                evidence = entry[name]
                expected_dir = root / ("attempts" if name == "journal" else "episodes")
                if (Path(evidence["path"]).parent != expected_dir
                        or sha256_file(evidence["path"]) != evidence["sha256"]):
                    raise ComparisonIntegrityError("attempt evidence changed")
    if receipt["artifact"] is None:
        if (receipt["success"] is not False or receipt["termination_reason"] !=
                "infrastructure_failure" or receipt["assessable"] is not False
                or len(receipt["attempt_evidence"]) != 2):
            raise ComparisonIntegrityError("invalid terminal infrastructure receipt")
        return receipt
    expected_path, _, _ = paths_for_pair(root, ordinal, pair, attempt)
    if receipt["artifact"] != str(expected_path):
        raise ComparisonIntegrityError("artifact path does not identify this pair/attempt")
    actual = inspect_artifact(expected_path, protocol, metadata)
    for key, value in actual.items():
        if receipt[key] != value:
            raise ComparisonIntegrityError(f"saved artifact/receipt mismatch for {key}")
    _, journal_path, _ = paths_for_pair(root, ordinal, pair, attempt)
    journal = validate_journal(journal_path, identity, pair, attempt)
    if (journal["status"] != "complete" or
            receipt["first_query_input_hashes"] != journal["first_query_input_hashes"] or
            receipt["assessable"] != (not journal["input_mismatch"])):
        raise ComparisonIntegrityError("receipt differs from completed attempt journal")
    return receipt


def collect_attempt_evidence(root: Path, ordinal: int, pair: Any) -> list[dict[str, Any]]:
    result = []
    for attempt in range(2):
        artifact, journal, _ = paths_for_pair(root, ordinal, pair, attempt)
        partial = Path(str(artifact) + ".partial")
        if journal.exists():
            value = {"attempt_index": attempt, "journal": {
                "path": str(journal), "sha256": sha256_file(journal)}, "partial": None}
            if partial.exists():
                value["partial"] = {"path": str(partial), "sha256": sha256_file(partial)}
            result.append(value)
    return result


class FirstInputGuard:
    """Compare reference inputs with the B0 artifact before the first policy query."""

    def __init__(self, policy: Any, expected: Any = None, on_first: Any = None) -> None:
        self.policy = policy
        self.expected = expected
        self.checked = False
        self.mismatch = False
        self.on_first = on_first

    def predict_chunk(self, **kwargs: Any) -> Any:
        if not self.checked:
            self.checked = True
            actual = {key: sha256_array(kwargs[key])
                      for key in ("agent_rgb", "wrist_rgb", "proprioception")}
            if self.expected is not None:
                self.mismatch = actual != self.expected
            if self.on_first is not None:
                self.on_first(actual, self.mismatch)
            if self.mismatch:
                raise ComparisonIntegrityError("paired first policy inputs differ")
        return self.policy.predict_chunk(**kwargs)


def compare_pair_receipts(left: Any, right: Any) -> dict[str, Any]:
    if left["pair"] != right["pair"]:
        raise ComparisonIntegrityError("two policies evaluated different pair identities")
    inputs_match = (left["artifact"] is not None and right["artifact"] is not None
                    and left["initial_input_hashes"] == right["initial_input_hashes"]
                    and left["first_query_input_hashes"] == right["first_query_input_hashes"])
    return {
        "pair_id": left["pair_id"], "ordinal": left["ordinal"],
        "b0_receipt_digest": left["digest"], "reference_receipt_digest": right["digest"],
        "b0_success": left["success"], "reference_success": right["success"],
        "initial_inputs_exactly_matched": inputs_match,
        "assessable": bool(inputs_match and left["assessable"] and right["assessable"]),
    }


def publish_receipt(
    *, root: Path, ordinal: int, pair: Any, identity: Any, attempt: int,
    result: dict[str, Any], assessable: bool,
) -> dict[str, Any]:
    _, _, target = paths_for_pair(root, ordinal, pair, attempt)
    value = seal_contract({
        "artifact_type": "eventground_wam.clean_comparison_episode", "schema_version": 1,
        "identity_digest": identity["digest"], "pair_id": pair["pair_id"],
        "ordinal": ordinal, "pair": pair, "attempt_index": attempt,
        "assessable": assessable, "completed_at_utc": utc_now(),
        "attempt_evidence": collect_attempt_evidence(root, ordinal, pair), **result,
    })
    atomic_json(target, value)
    return value


def load_pair_receipt(root: Path, ordinal: int, pair: Any) -> dict[str, Any]:
    return read_sealed(paths_for_pair(root, ordinal, pair)[2])


def validate_journal(path: Path, identity: Any, pair: Any, attempt: int) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (value["identity_digest"] != identity["digest"] or value["pair"] != pair
            or value["attempt_index"] != attempt):
        raise ComparisonIntegrityError("attempt journal identity mismatch")
    return value


def execute_pair(
    *, root: Path, ordinal: int, pair: Any, identity: Any, protocol: Any,
    metadata: ComparisonMetadata, initial_state: Any, policy: Any,
    environment_factory: Any, episode_runner: Any, seed_episode: Any,
    expected_input: Any = None,
) -> dict[str, Any]:
    """Persist intent before any simulator work; reuse outcomes and allow one retry."""
    receipt_path = paths_for_pair(root, ordinal, pair)[2]
    if receipt_path.exists():
        return verify_receipt(read_sealed(receipt_path), root=root, ordinal=ordinal, pair=pair,
                              identity=identity, protocol=protocol, metadata=metadata)
    for attempt in range(2):
        artifact, journal_path, _ = paths_for_pair(root, ordinal, pair, attempt)
        partial = Path(str(artifact) + ".partial")
        if journal_path.exists():
            journal = validate_journal(journal_path, identity, pair, attempt)
            if not artifact.exists():
                continue  # A started attempt is consumed, including before recorder creation.
        else:
            if artifact.exists() or partial.exists():
                raise ComparisonIntegrityError("artifact exists without an attempt journal")
            journal = {"identity_digest": identity["digest"], "pair": pair,
                       "attempt_index": attempt, "started_at_utc": utc_now(),
                       "status": "started", "first_query_input_hashes": None,
                       "input_mismatch": False}
            atomic_json(journal_path, journal)

            def on_first(hashes: Any, mismatch: bool, journal: Any = journal,
                         journal_path: Path = journal_path) -> None:
                journal.update(first_query_input_hashes=hashes, input_mismatch=mismatch)
                atomic_json(journal_path, journal)

            guarded = FirstInputGuard(policy, expected_input, on_first)
            environment = None
            try:
                seed_episode(pair["episode_seed"])
                environment = environment_factory()
                episode_runner(environment=environment, policy=guarded, protocol=protocol,
                               metadata=metadata, initial_state=initial_state,
                               artifact_path=artifact)
            except Exception as error:  # noqa: BLE001 - infrastructure boundary
                journal.update(status="interrupted", exception=f"{type(error).__name__}: {error}")
                atomic_json(journal_path, journal)
                if not artifact.exists():
                    continue
            finally:
                if environment is not None:
                    try:
                        environment.close()
                    except Exception as error:  # noqa: BLE001 - preserve completed outcomes
                        journal["close_exception"] = f"{type(error).__name__}: {error}"
                        atomic_json(journal_path, journal)
        # Publication can be recovered after a crash, without querying the policy again.
        result = inspect_artifact(artifact, protocol, metadata)
        query = journal["first_query_input_hashes"]
        if result["first_policy_input_hashes"] is not None and (
            result["first_policy_input_hashes"] != query
        ):
            raise ComparisonIntegrityError("recorded first input differs from query journal")
        result["first_query_input_hashes"] = query
        journal.update(status="complete", completed_at_utc=utc_now())
        atomic_json(journal_path, journal)
        return publish_receipt(root=root, ordinal=ordinal, pair=pair, identity=identity,
                               attempt=attempt, result=result,
                               assessable=not journal["input_mismatch"])
    return publish_receipt(
        root=root, ordinal=ordinal, pair=pair, identity=identity, attempt=1, assessable=False,
        result={"artifact": None, "artifact_sha256": None, "success": False,
                "termination_reason": "infrastructure_failure", "policy_step_count": 0,
                "step_count": 0, "initial_input_hashes": None, "first_policy_input_hashes": None,
                "first_query_input_hashes": None, "policy_latencies_ms": []},
    )
