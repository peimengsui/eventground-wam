import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eventground_wam.b0_training import sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_model import (
    ENCODING_KEYS,
    EventGroundHead,
    EventGroundModel,
    configure_eventground_trainables,
)
from eventground_wam.eventground_training import (
    EventGroundTrainer,
    require_training_admission,
)


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(8, 96)
        self.lora_A = nn.ModuleDict({"b0_oft": nn.Linear(8, 4, bias=False)})
        self.lora_B = nn.ModuleDict({"b0_oft": nn.Linear(4, 96, bias=False)})

    def forward(self, state):
        return self.base_layer(state) + self.lora_B["b0_oft"](self.lora_A["b0_oft"](state))


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.language_model = nn.ModuleDict({"layer": TinyLoRA()})
        self.backbone.visual = nn.ModuleDict({"layer": TinyLoRA()})

    def forward(self, encoded, state):
        assert set(encoded) == ENCODING_KEYS
        features = self.backbone.language_model["layer"](state).reshape(-1, 8, 12)
        return SimpleNamespace(tokens=features,
                               key_padding_mask=torch.zeros_like(state, dtype=torch.bool))


def setup(stage="sft", model=None):
    if model is None:
        with torch.random.fork_rng():
            torch.manual_seed(23)
            model = EventGroundModel(TinyAdapter(), EventGroundHead(input_dim=12, hidden_dim=16))
    inherited = tuple(name for name, _ in model.named_parameters() if ".lora_" in name)
    allowlist = configure_eventground_trainables(model, inherited)
    recipe = seal_contract({
        "artifact_type": "eventground_wam.synthetic_unit_test_recipe",
        "study_digest": "a" * 64,
        "logical_batch_size": 8,
        "optimization": {
            "sft_learning_rates": {"head": 3e-4, "language_lora": 1e-5},
            "rlvr_learning_rates": {"head": 1e-5, "language_lora": 1e-6},
            "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": .01,
            "maximum_gradient_norm": 1.,
        },
        "grpo": {"temperature": 1., "clip_epsilon": .2, "kl_beta": .02,
                 "advantage_epsilon": 1e-4},
    })
    return EventGroundTrainer(model, allowlist, recipe, stage=stage)


def batches(sizes=(5,), *, rl=False):
    state = torch.linspace(-1, 1, 40).reshape(5, 8)
    tokens = torch.tensor([[0, 0, 0], [1, 2, 7], [2, 32, 33], [3, 8, 13], [0, 0, 0]])
    result, offset = [], 0
    for count in sizes:
        encoded = {key: torch.zeros(count, 2, dtype=torch.int64) for key in ENCODING_KEYS}
        result.append((encoded, state[offset:offset+count],
                       list(range(offset, offset+count)) if rl else tokens[offset:offset+count]))
        offset += count
    return result


def state_copy(trainer):
    return {name: value.clone() for name, value in trainer.model.state_dict().items()}


def test_sft_update_preserves_frozen_weights_and_unequal_microbatch_equivalence():
    whole, micro = setup(), setup()
    before = state_copy(whole)
    full_metrics = whole.sft_update(batches(), prompt_count=5)
    micro_metrics = micro.sft_update(batches((2, 3)), prompt_count=5)
    assert full_metrics["loss"] == pytest.approx(micro_metrics["loss"], abs=2e-7)
    assert full_metrics["global_step"] == 1 and full_metrics["prompts_seen"] == 5
    for name, value in whole.model.state_dict().items():
        if name not in whole.names:
            assert torch.equal(value, before[name])
        torch.testing.assert_close(value, micro.model.state_dict()[name], atol=2e-7, rtol=2e-5)


def test_checkpoint_exact_resume_and_rng(tmp_path):
    original = setup()
    original.sft_update(batches(), prompt_count=5)
    identity = seal_contract({"artifact_type": "synthetic_test_identity", "manifest": "b" * 64})
    checkpoint = tmp_path / "state.pt"
    original.save(checkpoint, identity=identity)
    expected_random = torch.rand(5)
    expected = original.sft_update(batches(), prompt_count=5)
    resumed = setup()
    resumed.resume(checkpoint, identity=identity)
    assert torch.equal(torch.rand(5), expected_random)
    actual = resumed.sft_update(batches(), prompt_count=5)
    assert actual == expected
    assert all(torch.equal(value, resumed.model.state_dict()[name])
               for name, value in original.model.state_dict().items())
    with pytest.raises(ValueError, match="fresh"):
        resumed.save(checkpoint, identity=identity)


def test_checkpoint_checksum_corruption_rejected_without_model_mutation(tmp_path):
    trainer = setup()
    identity = seal_contract({"artifact_type": "synthetic_test_identity"})
    checkpoint = tmp_path / "state.pt"
    trainer.save(checkpoint, identity=identity)
    with checkpoint.open("ab") as stream:
        stream.write(b"corrupted")
    before = state_copy(trainer)
    with pytest.raises(ValueError, match="checksum"):
        trainer.resume(checkpoint, identity=identity)
    assert all(torch.equal(value, before[name]) for name, value in trainer.model.state_dict().items())


@pytest.mark.parametrize("corruption", [
    "parameter_nan", "identity", "optimizer_nan", "lr", "counter", "rng",
])
def test_recovery_schema_corruption_rejected_atomically(tmp_path, corruption):
    trainer = setup()
    trainer.sft_update(batches(), prompt_count=5)
    identity = seal_contract({"artifact_type": "synthetic_test_identity"})
    checkpoint = tmp_path / "state.pt"
    trainer.save(checkpoint, identity=identity)
    saved = torch.load(checkpoint, weights_only=True)
    if corruption == "parameter_nan":
        saved["parameters"][trainer.names[0]].flatten()[0] = float("nan")
    elif corruption == "identity":
        saved["identity"] = seal_contract({"artifact_type": "different_identity"})
    elif corruption == "optimizer_nan":
        first = next(iter(saved["optimizer"]["state"].values()))
        first["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "lr":
        saved["optimizer"]["param_groups"][0]["lr"] *= 100
    elif corruption == "counter":
        saved["global_step"] = -1
    else:
        saved["rng"]["torch"] = torch.tensor([1, 2], dtype=torch.uint8)
    torch.save(saved, checkpoint)
    checkpoint.with_suffix(".pt.sha256").write_text(sha256_file(checkpoint) + "\n")
    target = setup()
    before = state_copy(target)
    with pytest.raises((ValueError, FloatingPointError)):
        target.resume(checkpoint, identity=identity)
    assert all(torch.equal(value, before[name]) for name, value in target.model.state_dict().items())


def reward_function(evidence, sampled):
    assert sampled.shape == (len(evidence) * 4, 3)
    return ((sampled[:, 0].float() / 3 + sampled[:, 1].float() / 32) / 2).reshape(-1, 4)


def reference_copy(trainer):
    reference = copy.deepcopy(trainer.model)
    reference.requires_grad_(False)
    return reference


def test_rlvr_whole_groups_and_fixed_reference_with_microbatch_equivalence():
    whole, micro = setup("rlvr"), setup("rlvr")
    first_reference, second_reference = reference_copy(whole), reference_copy(micro)
    before = {name: value.clone() for name, value in first_reference.state_dict().items()}
    torch.manual_seed(109)
    full = whole.rlvr_update(batches(rl=True), prompt_count=5,
                             reference_model=first_reference, reward_function=reward_function)
    torch.manual_seed(109)
    pieces = micro.rlvr_update(batches((2, 3), rl=True), prompt_count=5,
                              reference_model=second_reference, reward_function=reward_function)
    assert full["loss"] == pytest.approx(pieces["loss"], abs=2e-7)
    assert full["policy_loss"] == pytest.approx(pieces["policy_loss"], abs=2e-7)
    assert full["global_step"] == 1 and full["prompts_seen"] == 5
    for name, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, micro.model.state_dict()[name], atol=2e-7, rtol=3e-5)
    assert all(torch.equal(value, before[name]) for name, value in first_reference.state_dict().items())
    assert all(parameter.grad is None for parameter in first_reference.parameters())


def test_rlvr_rejects_trainable_or_shared_reference():
    trainer = setup("rlvr")
    for reference in (trainer.model, copy.deepcopy(trainer.model)):
        with pytest.raises(ValueError, match="reference|Reference"):
            trainer.rlvr_update(batches(rl=True), prompt_count=5, reference_model=reference,
                                reward_function=reward_function)


def test_rlvr_checkpoint_restores_candidate_rng_and_exact_next_update(tmp_path):
    trainer = setup("rlvr")
    reference = reference_copy(trainer)
    trainer.rlvr_update(batches(rl=True), prompt_count=5, reference_model=reference,
                        reward_function=reward_function)
    identity = seal_contract({"artifact_type": "synthetic_rlvr_identity", "reference": "d" * 64})
    checkpoint = tmp_path / "rlvr.pt"
    trainer.save(checkpoint, identity=identity)
    expected = trainer.rlvr_update(batches(rl=True), prompt_count=5, reference_model=reference,
                                   reward_function=reward_function)
    resumed = setup("rlvr")
    resumed.resume(checkpoint, identity=identity)
    actual = resumed.rlvr_update(batches(rl=True), prompt_count=5, reference_model=reference,
                                 reward_function=reward_function)
    assert actual == expected
    assert all(torch.equal(value, resumed.model.state_dict()[name])
               for name, value in trainer.model.state_dict().items())


def test_partial_update_must_be_recovered_before_save_or_next_update(tmp_path):
    trainer = setup()
    bad = batches((2, 3))
    bad[1][2][0, 1] = -1
    with pytest.raises(ValueError):
        trainer.sft_update(bad, prompt_count=5)
    with pytest.raises(ValueError, match="(?i)partial|recover|in.progress"):
        trainer.sft_update(batches(), prompt_count=5)
    with pytest.raises(ValueError):
        trainer.save(tmp_path / "state.pt", identity=seal_contract({"kind": "test"}))


def admission(**changes):
    return seal_contract({
        "artifact_type": "eventground_wam.week10_training_admission", "status": "passed",
        "study_digest": "a" * 64, "recipe_digest": "b" * 64, "manifest_digest": "c" * 64,
        "labels_admitted": True, "human_review_gate_passed": True, "correctness_passed": True,
        "actual_host_resources_passed": True, "formal_launch_approved": True,
        "qa_digest": "d" * 64, "qualification_digest": "e" * 64,
        "launch_authorization_digest": "f" * 64, **changes,
    })


def test_training_admission_is_fail_closed_for_each_gate_and_identity():
    expected = {"study_digest": "a" * 64, "recipe_digest": "b" * 64, "manifest_digest": "c" * 64}
    require_training_admission(admission(), **expected)
    for key in ("labels_admitted", "human_review_gate_passed", "correctness_passed",
                "actual_host_resources_passed", "formal_launch_approved"):
        with pytest.raises(ValueError):
            require_training_admission(admission(**{key: False}), **expected)
    for key in expected:
        with pytest.raises(ValueError):
            require_training_admission(admission(**{key: "0" * 64}), **expected)
    tampered = admission()
    tampered["labels_admitted"] = False
    with pytest.raises(ValueError):
        require_training_admission(tampered, **expected)
