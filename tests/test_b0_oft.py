from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eventground_wam.b0_model import B0ActionPolicy, B0ModelError, B0ObjectiveConfig
from eventground_wam.b0_oft import (
    TRAINABLE_PREFIXES,
    OFTActionHead,
    OFTConditioningAdapter,
    marker_positions,
    normalize_state,
)
from eventground_wam.b0_training import (
    CheckpointStore,
    TrainingContractError,
    TrainingEngine,
    load_recipe,
)

ROOT = Path(__file__).parents[1]


def recipe():
    return load_recipe(ROOT / "configs/training/week4_b0_oft_v1.json")


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 2048)
        self.embedding.weight.requires_grad_(False)
        self.lora_scale = nn.Parameter(torch.ones(2048))
        self.seen = None

    def get_input_embeddings(self):
        return self.embedding

    def get_rope_index(self, input_ids, **kwargs):
        assert "mm_token_type_ids" in kwargs
        positions = torch.arange(input_ids.shape[1]).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1)

    def forward(self, inputs_embeds, position_ids, **kwargs):
        assert "input_ids" not in kwargs
        assert position_ids.shape == (3, *inputs_embeds.shape[:2])
        self.seen = inputs_embeds.detach().clone()
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1) * self.lora_scale)


def inputs():
    ids = torch.tensor([[1, 2, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 0],
                        [1, 2, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4]])
    return {"input_ids": ids, "attention_mask": (ids != 0).long(),
            "mm_token_type_ids": torch.zeros_like(ids), "image_grid_thw": torch.ones(4, 3)}


def policy():
    config = recipe()
    adapter = OFTConditioningAdapter(ToyCore(), 7, config)
    head = OFTActionHead(seed=11, hidden=16)
    model = B0ActionPolicy(adapter, head, objective_config=B0ObjectiveConfig(
        "continuous-l1-plus-gripper-bce-v1"))
    model.trainable_parameter_prefixes = TRAINABLE_PREFIXES
    return model


def batch(index):
    targets = torch.zeros(2, 8, 7)
    targets[..., 6] = -1
    return {"index": index, "inputs": inputs(), "proprioception": torch.ones(2, 8) * .1,
            "targets": targets, "sample_count": 2, "identity_digest": "a" * 64}


def test_state_injection_preserves_other_embeddings_and_changes_action_features():
    model = policy()
    core = model.conditioning_adapter.backbone
    data = inputs()
    before = core.embedding(data["input_ids"]).detach()
    first = model(data, torch.zeros(2, 8))
    seen = core.seen.clone()
    positions = marker_positions(data["input_ids"], data["attention_mask"], 7)
    for row in range(2):
        keep = torch.ones(data["input_ids"].shape[1], dtype=torch.bool)
        keep[positions[row, 0]] = False
        assert torch.equal(seen[row, keep], before[row, keep])
    second = model(data, torch.ones(2, 8))
    assert first.normalized_actions.shape == (2, 8, 7)
    assert not torch.equal(first.raw_action_predictions, second.raw_action_predictions)


def test_hybrid_objective_has_state_head_and_backbone_gradients_without_target_leakage():
    model = policy()
    targets = batch(0)["targets"]
    output = model(inputs(), torch.ones(2, 8) * .1, normalized_action_targets=targets)
    output.loss.backward()
    assert model.conditioning_adapter.backbone.embedding.weight.grad is None
    assert model.conditioning_adapter.backbone.lora_scale.grad.abs().sum() > 0
    assert model.conditioning_adapter.proprioception_projector[0].weight.grad.abs().sum() > 0
    assert model.action_decoder.network[-1].weight.grad.abs().sum() > 0
    assert set(output.normalized_actions[..., 6].unique().tolist()) <= {-1.0, 1.0}
    changed = model(inputs(), torch.ones(2, 8) * .1, normalized_action_targets=-targets)
    assert torch.equal(changed.raw_action_predictions, output.raw_action_predictions)


def test_marker_alignment_rejects_missing_slots_and_padding_slots():
    data = inputs()
    data["attention_mask"][0, 2] = 0
    with pytest.raises(B0ModelError, match="eight action"):
        marker_positions(data["input_ids"], data["attention_mask"], 7)


def test_state_quantile_scaling_is_dimensionwise_and_clipped():
    state = torch.tensor([[1., 5., 9., -3.]])
    result = normalize_state(state, torch.tensor([0., 0., 9., 0.]),
                             torch.tensor([2., 2., 9., 2.]))
    assert torch.equal(result, torch.tensor([[0., 1., 0., -1.]]))


def test_new_policy_resume_preserves_next_update(tmp_path):
    torch.manual_seed(71)
    model = policy()
    names = ("conditioning_adapter.backbone.lora_scale",)
    engine = TrainingEngine(model, names, recipe(), identity_digest="b" * 64,
                            steps_per_epoch=10, device=torch.device("cpu"))
    engine.step(batch(0))
    store = CheckpointStore(tmp_path, "b" * 64)
    store.save(engine.snapshot())
    engine.step(batch(1))
    expected = {name: value.clone() for name, value in engine.snapshot()["state_dict"].items()}
    engine.restore(store.load())
    engine.step(batch(1))
    assert all(torch.equal(value, engine.snapshot()["state_dict"][name])
               for name, value in expected.items())


def test_recipe_rejects_modified_state_statistics_and_unbounded_budget(tmp_path):
    config = recipe()
    config["state_normalization"]["lower_q01"][0] += 1
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config))
    with pytest.raises(TrainingContractError, match="digest"):
        load_recipe(path)
    config = recipe()
    config["maximum_active_hours"] = 72
    path.write_text(json.dumps(config))
    with pytest.raises(TrainingContractError, match="budget"):
        load_recipe(path)
