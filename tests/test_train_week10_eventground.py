import copy
import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import torch

from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "formal_eventground", ROOT / "scripts/train_week10_eventground.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def reseal(value):
    return seal_contract({k: v for k, v in value.items() if k != "digest"})


def manifest():
    rows = []
    for index in range(2000):
        event = runner.CLASSES[index % 4]
        rows.append(
            {
                "prompt_id": f"prompt-{index:04d}",
                "source_id": f"source-{index // 10}",
                "anchor": index % 10,
                "target": {
                    "next_event": event,
                    "interval": None if event == "none" else [1, 3],
                    "horizon": None if event == "none" else 1,
                },
            }
        )
    return seal_contract(
        {
            "main_rows": rows,
            "pilot_prompt_ids": [r["prompt_id"] for r in rows[:500]],
            "labels_admitted": True,
            "training_authorized": False,
        }
    )


def gate_bundle(stage="sft"):
    study = seal_contract({"study": "synthetic guard test"})
    recipe = seal_contract(
        {
            "study_digest": study["digest"],
            "logical_batch_size": 16,
            "microbatch_size": 1,
            "seed": 17,
            "prompt_order": "sha256_seed_epoch_prompt_id",
            "checkpoint_every_updates": 25,
        }
    )
    data = manifest()
    diagnostic = seal_contract({"diagnostic": "synthetic guard fixture"})
    qa = seal_contract({"independent_review": "synthetic guard fixture only"})
    gates = runner.QUALIFICATION_GATES | (
        runner.RL_QUALIFICATION_GATES if stage.startswith("rlvr-") else set()
    )
    qualification = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_training_qualification",
            "stage": stage,
            "status": "passed",
            "recipe_digest": recipe["digest"],
            "manifest_digest": data["digest"],
            "qa_digest": qa["digest"],
            "actual_host_resources_passed": True,
            "correctness_passed": True,
            "evidence_files_sha256": {"fixture.json": "f" * 64},
            "gates": dict.fromkeys(gates, True),
            "active_gpu_seconds": 10,
            "host": "synthetic",
        }
    )
    authorization = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_formal_training_launch",
            "approved": True,
            "stage": stage,
            "study_digest": study["digest"],
            "recipe_digest": recipe["digest"],
            "manifest_digest": data["digest"],
            "qa_digest": qa["digest"],
            "qualification_digest": qualification["digest"],
            "training_authorized": True,
            "user_instruction": "SYNTHETIC UNIT TEST ONLY",
            "execution_id": "unit-test",
            "host": "synthetic",
            "resources": {
                "wall_seconds": 100,
                "prior_active_gpu_seconds": 10,
                "output_bytes": 1024,
                "minimum_free_bytes": 50 * 1024**3,
                "prior_output_bytes": 0,
            },
            "schedule_digest": runner.schedule_digest(runner.schedule_rows(data, stage)),
            "optimizer_updates": 250 if stage == "sft" else 32 if stage == "rlvr-pilot" else 125,
            "prompt_presentations": 4000
            if stage == "sft"
            else 500
            if stage == "rlvr-pilot"
            else 2000,
        }
    )
    admission = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_training_admission",
            "status": "passed",
            "study_digest": study["digest"],
            "recipe_digest": recipe["digest"],
            "manifest_digest": data["digest"],
            "labels_admitted": True,
            "human_review_gate_passed": True,
            "correctness_passed": True,
            "actual_host_resources_passed": True,
            "formal_launch_approved": True,
            "qa_digest": qa["digest"],
            "qualification_digest": qualification["digest"],
            "launch_authorization_digest": authorization["digest"],
        }
    )
    return [stage, study, recipe, data, diagnostic, qa, qualification, admission, authorization]


def test_sft_fixed_two_passes_and_main_control_same_batch_order():
    data = manifest()
    sft = runner.schedule_rows(data, "sft")
    main = runner.schedule_rows(data, "rlvr-main")
    control = runner.schedule_rows(data, "control")
    pilot = runner.schedule_rows(data, "rlvr-pilot")
    assert len(sft) == 250 and all(len(b) == 16 for b in sft)
    assert Counter(r["prompt_id"] for b in sft for r in b) == dict.fromkeys(
        (r["prompt_id"] for r in data["main_rows"]), 2
    )
    assert main == control == sft[:125]
    assert len(pilot) == 32 and len(pilot[-1]) == 4
    assert sum(map(len, pilot)) == 500
    assert {r["prompt_id"] for b in pilot for r in b} == set(data["pilot_prompt_ids"])
    reversed_data = copy.deepcopy(data)
    reversed_data["main_rows"].reverse()
    assert runner.schedule_rows(reversed_data, "sft") == sft


@pytest.mark.parametrize(
    "damage",
    ["duplicate", "shortfall", "class_substitution", "pilot_duplicate", "pilot_not_nested"],
)
def test_schedule_never_fills_shortfalls_or_repeats_prompts(damage):
    data = copy.deepcopy(manifest())
    stage = "sft"
    if damage == "duplicate":
        data["main_rows"][1] = data["main_rows"][0]
    elif damage == "shortfall":
        data["main_rows"].pop()
    elif damage == "class_substitution":
        data["main_rows"][1]["target"]["next_event"] = "none"
    else:
        stage = "rlvr-pilot"
        data["pilot_prompt_ids"][0] = (
            data["pilot_prompt_ids"][1] if damage == "pilot_duplicate" else "unknown"
        )
    with pytest.raises(ValueError):
        runner.schedule_rows(data, stage)


def test_linked_gate_allows_only_explicit_formal_stage(monkeypatch):
    bundle = gate_bundle()
    monkeypatch.setattr(runner, "admit_manifest", lambda diagnostic, qa: bundle[3])
    batches = runner.validate_gate_bundle(*bundle)
    assert len(batches) == 250


@pytest.mark.parametrize(
    "damage",
    [
        "probe",
        "empty_gates",
        "wrong_stage",
        "unapproved",
        "wrong_qa",
        "output_budget",
        "time_budget",
        "oversampling",
        "recipe_order",
        "manifest_self_authorizes",
    ],
)
def test_formal_gate_denies_incomplete_or_changed_authority(monkeypatch, damage):
    bundle = gate_bundle()
    monkeypatch.setattr(runner, "admit_manifest", lambda diagnostic, qa: bundle[3])
    qual, auth = bundle[6], bundle[8]
    if damage == "probe":
        qual["artifact_type"] = "eventground_wam.week10_gradient_resource_diagnostic"
        qual["status"] = "passed_gradient_only_probe_not_training_admission"
    elif damage == "empty_gates":
        qual["gates"] = {}
    elif damage == "wrong_stage":
        qual["stage"] = "control"
    elif damage == "unapproved":
        auth["approved"] = False
    elif damage == "wrong_qa":
        auth["qa_digest"] = "0" * 64
    elif damage == "output_budget":
        auth["resources"]["prior_output_bytes"] = 100 * 1024**3
    elif damage == "time_budget":
        auth["resources"]["prior_active_gpu_seconds"] = 40 * 3600
    elif damage == "oversampling":
        auth["optimizer_updates"] = 251
    elif damage == "recipe_order":
        bundle[2]["prompt_order"] = "random_replacement"
        bundle[2] = reseal(bundle[2])
    elif damage == "manifest_self_authorizes":
        bundle[3]["training_authorized"] = True
        bundle[3] = reseal(bundle[3])
    bundle[6] = reseal(qual)
    auth["qualification_digest"] = bundle[6]["digest"]
    bundle[8] = reseal(auth)
    bundle[7]["launch_authorization_digest"] = bundle[8]["digest"]
    bundle[7]["qualification_digest"] = bundle[6]["digest"]
    bundle[7] = reseal(bundle[7])
    with pytest.raises(ValueError):
        runner.validate_gate_bundle(*bundle)


class FakeTrainer:
    def __init__(self, step=0, seen=0):
        self.global_step, self.prompts_seen = step, seen

    def update(self, batches, count):
        assert len(list(batches)) == count
        self.global_step += 1
        self.prompts_seen += count
        return {"loss": 0.5}


def test_fake_execution_checkpoints_and_resume_do_not_replay_prefix():
    batches = runner.schedule_rows(manifest(), "sft")
    fake = FakeTrainer(25, 400)
    updated, checkpoints = [], []
    runner.execute_schedule(
        fake,
        batches,
        lambda batch: iter(batch),
        update=fake.update,
        record_step=lambda i, batch, metrics: updated.append(i),
        checkpoint=checkpoints.append,
    )
    assert updated == list(range(25, 250))
    assert checkpoints == list(range(50, 251, 25))
    assert fake.global_step == 250 and fake.prompts_seen == 4000
    bad = FakeTrainer(25, 399)
    with pytest.raises(ValueError, match="prefix"):
        runner.execute_schedule(
            bad,
            batches,
            iter,
            update=bad.update,
            record_step=lambda *args: None,
            checkpoint=lambda step: None,
        )


def test_fixed_sft_endpoint_rejects_intermediate_selection():
    completion = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_training_completion",
            "status": "complete",
            "stage": "sft",
            "recipe_digest": "r",
            "manifest_digest": "m",
            "global_step": 250,
            "prompts_seen": 4000,
            "selection": "fixed_final_endpoint",
            "frozen_parameters_unchanged": True,
        }
    )
    runner.validate_fixed_sft(completion, recipe={"digest": "r"}, manifest={"digest": "m"})
    for key, value in (
        ("global_step", 249),
        ("selection", "best_dev"),
        ("status", "failed_preserved"),
    ):
        bad = reseal(completion | {key: value})
        with pytest.raises(ValueError):
            runner.validate_fixed_sft(bad, recipe={"digest": "r"}, manifest={"digest": "m"})


def test_reward_grouping_uses_separate_admitted_evidence():
    evidence = [
        {
            "target": {"next_event": "none", "interval": None, "horizon": None},
            "verified_interval": None,
            "evidence_digest": "a" * 64,
        }
    ]
    tokens = torch.tensor([[0, 0, 0], [1, 1, 3], [2, 5, 8], [3, 12, 18]], dtype=torch.long)
    rewards = runner.reward_function(runner.PRIMARY_REWARD)(evidence, tokens)
    assert rewards.shape == (1, 4)
    assert rewards[0, 0] == 1 and torch.allclose(rewards[0, 1:], torch.full((3,), 0.15))
    with pytest.raises(ValueError, match="four"):
        runner.reward_function(runner.PRIMARY_REWARD)(evidence, tokens[:3])


def test_main_and_calibration_require_distinct_explicit_stage_gates():
    data = manifest()
    sft = {"digest": "sft"}
    primary = {"sft_completion_digest": "sft", "reward_variant": "primary"}
    runner.validate_promotion("rlvr-pilot", primary, None, sft, data)
    with pytest.raises((ValueError, AttributeError)):
        runner.validate_promotion("rlvr-main", primary, None, sft, data)
    promotion = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_rl_stage_gate",
            "integrity_passed": True,
            "manifest_digest": data["digest"],
            "sft_completion_digest": "sft",
            "evaluation_digest": "eval",
            "decision": "promote_main",
            "reward_variant": "primary",
            "calibrations_used": 0,
            "metrics": {
                "format_validity": 1.0,
                "positive_macro_iou_gain": 0.01,
                "matched_positive_start_mae_worsening_frames": 1.0,
                "none_false_positive_increase": 0.02,
            },
        }
    )
    auth = primary | {"promotion_digest": promotion["digest"]}
    runner.validate_promotion("rlvr-main", auth, promotion, sft, data)
    bad = reseal(promotion | {"integrity_passed": False})
    with pytest.raises(ValueError):
        runner.validate_promotion(
            "rlvr-main", auth | {"promotion_digest": bad["digest"]}, bad, sft, data
        )


def test_linked_qualification_evidence_bytes_checked(tmp_path):
    file = tmp_path / "evidence.json"
    file.write_text("independent fixture")
    record = {"evidence_files_sha256": {file.name: runner.sha256_file(file)}}
    runner.verify_evidence_files(record, tmp_path)
    file.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_evidence_files(record, tmp_path)


def test_checkpoint_cursor_binds_epoch_and_next_batch_exactly():
    data = manifest()
    batches = runner.schedule_rows(data, "sft")
    identity = seal_contract({"stage": "sft", "seed": 17})
    cursor = runner.checkpoint_cursor(FakeTrainer(125, 2000), identity, batches, "c" * 64)
    assert cursor["epoch_index"] == 1 and cursor["batch_index_in_epoch"] == 0
    assert cursor["next_prompt_ids"] == [r["prompt_id"] for r in batches[125]]
    assert cursor["schedule_digest"] == runner.schedule_digest(batches)
    final = runner.checkpoint_cursor(FakeTrainer(250, 4000), identity, batches, "d" * 64)
    assert final["epoch_index"] == 2 and final["next_prompt_ids"] == []
    with pytest.raises(ValueError, match="inconsistent"):
        runner.checkpoint_cursor(FakeTrainer(125, 1999), identity, batches, "c" * 64)


def test_recovery_requires_preserved_failure_and_cost_and_discard_bounds(tmp_path):
    batches = runner.schedule_rows(manifest(), "sft")
    identity = seal_contract({"stage": "sft", "seed": 17})
    authorization = {"digest": "a" * 64}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes; no optimizer or model")
    digest = runner.sha256_file(checkpoint)
    runner.write_sealed(
        checkpoint.with_suffix(".pt.cursor.json"),
        runner.checkpoint_cursor(
            FakeTrainer(25, 400),
            identity,
            batches,
            digest,
        ),
    )
    previous_path = tmp_path / "failed-completion.json"
    previous = runner.write_sealed(
        previous_path,
        {
            "status": "failed_preserved",
            "identity_digest": identity["digest"],
            "global_step": 27,
            "partial_update_in_progress": True,
            "cumulative_active_gpu_seconds": 50,
            "prior_output_bytes": 20,
            "output_bytes": 100,
        },
    )
    output = tmp_path / "attempt-2"
    recovery = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_training_recovery_authorization",
            "approved": True,
            "original_launch_authorization_digest": authorization["digest"],
            "identity_digest": identity["digest"],
            "checkpoint_sha256": digest,
            "new_output": str(output.resolve()),
            "preserve_all_previous_attempts": True,
            "prior_failed_completion_digest": previous["digest"],
            "prior_failed_completion_path": str(previous_path),
            "user_instruction": "UNIT TEST ONLY",
            "maximum_discarded_updates": 3,
            "resources": {"prior_active_gpu_seconds": 50, "prior_output_bytes": 120},
        }
    )
    runner.validate_recovery(
        identity, recovery, authorization=authorization, checkpoint=checkpoint, output=output
    )
    for damage in ("discarded", "time", "output", "prior", "wrong_attempt"):
        bad = copy.deepcopy(recovery)
        if damage == "discarded":
            bad["maximum_discarded_updates"] = 2
        elif damage == "time":
            bad["resources"]["prior_active_gpu_seconds"] = 49
        elif damage == "output":
            bad["resources"]["prior_output_bytes"] = 119
        elif damage == "prior":
            bad["prior_failed_completion_digest"] = "0" * 64
        else:
            bad["new_output"] = str(tmp_path / "unapproved-attempt")
        with pytest.raises(ValueError):
            runner.validate_recovery(
                identity,
                reseal(bad),
                authorization=authorization,
                checkpoint=checkpoint,
                output=output,
            )


@pytest.mark.parametrize("option", ["sft_completion", "promotion", "reward_evidence"])
def test_initial_sft_rejects_later_stage_inputs_even_during_recovery(option):
    for recovery in (
        {},
        {"resume": Path("checkpoint.pt"), "recovery_authorization": Path("recovery.json")},
    ):
        with pytest.raises(ValueError, match="Initial SFT forbids"):
            runner.validate_stage_inputs("sft", **{option: Path("later-stage.json")}, **recovery)


def test_initial_sft_and_explicit_sft_recovery_keep_m4_initialization_route():
    runner.validate_stage_inputs("sft")
    runner.validate_stage_inputs(
        "sft", resume=Path("checkpoint.pt"), recovery_authorization=Path("recovery.json")
    )
    with pytest.raises(ValueError, match="required together"):
        runner.validate_stage_inputs("sft", resume=Path("checkpoint.pt"))
    with pytest.raises(ValueError, match="fixed SFT endpoint"):
        runner.validate_stage_inputs("control")
    runner.validate_stage_inputs("control", sft_completion=Path("completed-sft.json"))


@pytest.mark.parametrize("option", ["--sft-completion", "--promotion", "--reward-evidence"])
def test_cli_rejects_sft_initialization_bypass_before_any_artifact_read(monkeypatch, option):
    argv = ["train_week10_eventground.py", "--stage", "sft"]
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
        argv.extend([f"--{name}", f"/unused/{name}"])
    argv.extend([option, "/unused/previous-sft-endpoint.json"])
    monkeypatch.setattr("sys.argv", argv)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Incompatible stage inputs must be rejected before artifact reads")

    monkeypatch.setattr(runner, "read_sealed", forbidden_read)
    with pytest.raises(ValueError, match="Initial SFT forbids"):
        runner.main()


class FakeOptimizerState:
    def state_dict(self):
        return {"state": {}, "param_groups": []}


class FakeCheckpointTrainer(FakeTrainer):
    def __init__(self):
        super().__init__(25, 400)
        self.model = torch.nn.Linear(4, 1, bias=False)
        self.names = ("weight",)
        self._in_update = False
        self.recipe = {"digest": "a" * 64}
        self.stage = "sft"
        self.optimizer = FakeOptimizerState()


def test_checkpoint_reserves_serialization_and_all_metadata_before_writing(tmp_path):
    fake = FakeCheckpointTrainer()
    identity = seal_contract({"stage": "sft", "seed": 17})
    batches = runner.schedule_rows(manifest(), "sft")
    path = tmp_path / "checkpoints" / "update-000025.pt"
    too_small = {"output_bytes": runner.COMPLETION_RESERVE_BYTES + 1, "minimum_free_bytes": 0}
    with pytest.raises(RuntimeError, match="reservation"):
        runner.save_bounded_checkpoint(
            fake, path, identity=identity, batches=batches, output=tmp_path, resources=too_small
        )
    assert list(tmp_path.iterdir()) == []  # rejection precedes any large/partial file
    enough = {"output_bytes": 32 * 1024**2, "minimum_free_bytes": 0}
    digest = runner.save_bounded_checkpoint(
        fake, path, identity=identity, batches=batches, output=tmp_path, resources=enough
    )
    assert digest == runner.sha256_file(path)
    assert path.with_suffix(".pt.sha256").read_text() == digest + "\n"
    cursor = runner.read_sealed(path.with_suffix(".pt.cursor.json"))
    assert cursor["checkpoint_sha256"] == digest and cursor["global_step"] == 25
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["format"] == runner.CHECKPOINT_FORMAT
    assert torch.equal(loaded["parameters"]["weight"], fake.model.weight)
    assert not list(tmp_path.rglob("*.partial"))
    runner.write_completion(tmp_path, {"status": "fake_test_only"}, enough)
    complete = runner.read_sealed(tmp_path / "completion.json")
    assert complete["output_bytes"] == runner.output_usage(tmp_path)


def test_checkpoint_preflight_preserves_minimum_free_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    fake = FakeCheckpointTrainer()
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=1000))
    with pytest.raises(RuntimeError, match="minimum free"):
        runner.save_bounded_checkpoint(
            fake,
            tmp_path / "checkpoint.pt",
            identity=seal_contract({"stage": "sft"}),
            batches=runner.schedule_rows(manifest(), "sft"),
            output=tmp_path,
            resources={"output_bytes": 32 * 1024**2, "minimum_free_bytes": 0},
        )
    assert list(tmp_path.iterdir()) == []


def test_bounded_writer_rejects_oversized_chunk_before_output(tmp_path):
    import io

    output = io.BytesIO()
    writer = runner.BoundedWriter(
        output,
        3,
        output=tmp_path,
        resources={"output_bytes": 1024**2, "minimum_free_bytes": 0},
        sidecar_reservation=65,
    )
    with pytest.raises(RuntimeError, match="preflight"):
        writer.write(b"1234")
    assert output.getvalue() == b""


def test_checkpoint_size_drift_aborts_partial_without_replacing_endpoint(tmp_path, monkeypatch):
    original_save = runner.torch.save
    calls = 0

    def drift(payload, handle):
        nonlocal calls
        calls += 1
        original_save(payload, handle)
        if calls == 2:
            handle.write(b"unexpected additional bytes")

    monkeypatch.setattr(runner.torch, "save", drift)
    with pytest.raises(RuntimeError, match="preflight"):
        runner.save_bounded_checkpoint(
            FakeCheckpointTrainer(),
            tmp_path / "checkpoint.pt",
            identity=seal_contract({"stage": "sft"}),
            batches=runner.schedule_rows(manifest(), "sft"),
            output=tmp_path,
            resources={"output_bytes": 32 * 1024**2, "minimum_free_bytes": 0},
        )
    assert not (tmp_path / "checkpoint.pt").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_bound_reward_validation_cannot_accept_merely_declared_hashes(tmp_path, monkeypatch):
    import eventground_wam.eventground_reward_admission as binding

    calls = []
    monkeypatch.setattr(
        runner, "validate_reward_evidence", lambda *args, **kwargs: {"p": {"claimed": True}}
    )

    def reject_unbound(evidence, manifest, qa, *, annotation_root):
        calls.append(annotation_root)
        raise ValueError("No independent annotation binds this declared digest")

    monkeypatch.setattr(binding, "verify_reward_evidence_binding", reject_unbound)
    with pytest.raises(ValueError, match="independent annotation"):
        runner.validate_bound_reward_evidence(
            {"claimed_digest": "a" * 64}, manifest={}, qa={}, annotation_root=tmp_path
        )
    assert calls == [tmp_path]


def test_rlvr_cli_requires_concrete_reward_binding_before_model_or_gpu(monkeypatch):
    recipe = {
        "digest": "recipe",
        "code_sha256": {
            "scripts/train_week10_eventground.py": "a" * 64,
            "src/eventground_wam/eventground_reward_admission.py": "a" * 64,
        },
    }
    authorization = {
        "host": "synthetic",
        "sft_completion_digest": "sft",
        "reward_variant": "primary",
    }
    values = {
        "recipe": recipe,
        "manifest": {},
        "diagnostic": {},
        "qa": {},
        "qualification": {},
        "admission": {},
        "authorization": authorization,
        "completed-sft": {"digest": "sft"},
        "reward": {"digest": "reward"},
    }
    argv = ["train_week10_eventground.py", "--stage", "rlvr-pilot"]
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
        argv.extend([f"--{name}", f"/unused/{name}"])
    argv.extend(
        ["--sft-completion", "/unused/completed-sft", "--reward-evidence", "/unused/reward"]
    )
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(runner, "read_sealed", lambda path: values.get(Path(path).name, {}))
    monkeypatch.setattr(runner, "validate_gate_bundle", lambda *args: [])
    monkeypatch.setattr(runner, "verify_evidence_files", lambda *args: None)
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "synthetic")
    monkeypatch.setattr(runner, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(runner, "validate_fixed_sft", lambda *args, **kwargs: None)

    def no_binding(*args, **kwargs):
        raise ValueError("Concrete independent reward binding required")

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU/model activity occurred before independent reward admission")

    monkeypatch.setattr(runner, "validate_bound_reward_evidence", no_binding)
    monkeypatch.setattr(runner, "check_gpu_exclusive", forbidden)
    monkeypatch.setattr(runner, "build_eventground_from_m4", forbidden)
    with pytest.raises(ValueError, match="Concrete independent reward binding"):
        runner.main()
