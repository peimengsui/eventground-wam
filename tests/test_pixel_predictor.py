from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from eventground_wam.b0_data import ActionNormalizer
from eventground_wam.pixel_data import FuturePairDataset, collate_future_samples, make_future_sample
from eventground_wam.pixel_predictor import (
    FrozenQwenFeatures,
    PixelCondition,
    PixelPredictor,
    PixelPredictorConfig,
    load_pixel_checkpoint,
    pixel_reconstruction_loss,
    save_pixel_checkpoint,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture()
def condition():
    g = torch.Generator().manual_seed(51)
    return PixelCondition(torch.rand(2, 5, 8, generator=g),
                          torch.tensor([[False, False, False, True, True], [False]*5]),
                          .2+.6*torch.rand(2, 2, 3, 128, 128, generator=g),
                          torch.rand(2, 8, generator=g), torch.rand(2, 32, 7, generator=g),
                          torch.tensor([2, 32]))


@pytest.fixture()
def model():
    return PixelPredictor(PixelPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))


def test_default_budget_is_bounded_and_reports_decoder_separately():
    budget = PixelPredictor().parameter_budget()
    assert budget["total_trainable"] < 5_000_000
    assert budget["future_core"] + budget["pixel_decoder"] == budget["total_trainable"]
    assert budget["trainable_base_parameters"] == 0
    assert (budget["future_tokens"], budget["future_token_dimension"]) == (128, 256)


def test_initial_prediction_is_exact_copy_with_stable_two_camera_tokens(model, condition):
    result = model(condition)
    torch.testing.assert_close(result.images, condition.current_images, rtol=0, atol=0)
    assert result.future_tokens.shape == (2, 128, 32)
    assert not torch.equal(result.future_tokens[:, :64], result.future_tokens[:, 64:])
    torch.testing.assert_close(model.decode(result.future_tokens, condition.current_images), result.images)


@pytest.mark.parametrize("ablation", ["zero-actions", "reverse-actions", "zero-horizon", "zero-features"])
def test_action_order_horizon_and_features_reach_future_tokens(model, condition, ablation):
    normal = model(condition).future_tokens
    changed = model(condition, ablation=ablation).future_tokens
    assert (normal-changed).abs().max().item() > 1e-5


def test_action_tail_and_padding_values_cannot_enter_prediction(model, condition):
    actions, features = condition.actions.clone(), condition.features.clone()
    actions[0, 2:] = float("nan")
    features[0, 3:] = float("nan")
    altered = replace(condition, actions=actions, features=features)
    for ablation in ("none", "reverse-actions"):
        original = model(condition, ablation=ablation)
        result = model(altered, ablation=ablation)
        torch.testing.assert_close(result.future_tokens, original.future_tokens, rtol=0, atol=0)
        torch.testing.assert_close(result.images, original.images, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["horizon", "float-horizon", "all-padding", "active-nan", "image-range"])
def test_bad_condition_is_rejected(model, condition, bad):
    if bad == "horizon":
        condition = replace(condition, horizons=torch.tensor([3, 32]))
    elif bad == "float-horizon":
        condition = replace(condition, horizons=condition.horizons.float())
    elif bad == "all-padding":
        condition = replace(condition, feature_padding=torch.ones_like(condition.feature_padding))
    elif bad == "active-nan":
        actions = condition.actions.clone(); actions[0, 0, 0] = float("nan")
        condition = replace(condition, actions=actions)
    else:
        condition = replace(condition, current_images=condition.current_images+2)
    with pytest.raises(ValueError):
        model(condition)


def test_future_targets_are_not_a_forward_argument(model, condition):
    with pytest.raises(TypeError):
        model(condition, target_images=condition.current_images)
    with pytest.raises(TypeError, match="PixelCondition"):
        model({"condition": condition, "targets": condition.current_images})


def test_losses_backpropagate_to_conditioning_after_copy_head_activation(model, condition):
    features = condition.features.clone().requires_grad_(True)
    condition = replace(condition, features=features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    target = condition.current_images*.8
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = pixel_reconstruction_loss(model(condition).images, target)
        loss.backward()
        optimizer.step()
    for module in (model.action_projection, model.action_sequence_projection,
                   model.horizon_projection, model.feature_projection,
                   model.state_projection, model.current_patches):
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert torch.equal(features.grad[0, 3:], torch.zeros_like(features.grad[0, 3:]))


def test_l1_ssim_loss_is_differentiable_and_zero_for_identity():
    target = torch.full((1, 2, 3, 128, 128), .4)
    prediction = torch.full_like(target, .5, requires_grad=True)
    loss = pixel_reconstruction_loss(prediction, target, ssim_weight=.1)
    loss.backward()
    assert loss.item() > .1 and prediction.grad.abs().sum() > 0
    assert abs(pixel_reconstruction_loss(target, target, ssim_weight=.1).item()) < 1e-10
    with pytest.raises(ValueError):
        pixel_reconstruction_loss(prediction, target, ssim_weight=float("nan"))


def test_weight_reload_is_exact_and_rejects_corruption(model, condition, tmp_path):
    path = tmp_path / "diagnostic.pt"
    save_pixel_checkpoint(path, model, metadata={"purpose": "unit-test-not-production"})
    restored, header = load_pixel_checkpoint(path)
    assert header["metadata"]["purpose"] == "unit-test-not-production"
    torch.testing.assert_close(restored(condition).future_tokens, model(condition).future_tokens,
                               rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_pixel_checkpoint(path, model, metadata={})
    artifact = torch.load(path, weights_only=True)
    artifact["state_dict"]["residual_head.bias"][0] += 1
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="changed"):
        load_pixel_checkpoint(path)


def test_initialization_does_not_advance_global_rng():
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    PixelPredictor(PixelPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("kwargs", [{"heads": 0}, {"layers": 100}, {"dimension": 35}, {"seed": -1}])
def test_invalid_architecture_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PixelPredictorConfig(**kwargs)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)
        self.dropout = nn.Dropout(.9)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def test_frozen_extractor_has_no_gradients_or_stochastic_training_mode():
    wrapper = FrozenQwenFeatures(FakeCore(), dimension=8).train()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 0]])}
    a, padding = wrapper(inputs); b, _ = wrapper(inputs)
    assert not wrapper.core.training and not a.requires_grad
    assert not any(p.requires_grad for p in wrapper.parameters())
    assert padding.tolist() == [[False, False, True]]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        wrapper({**inputs, "future_images": torch.ones(1)})


def test_frozen_extractor_rejects_adapted_backbone():
    core = FakeCore(); core.lora_A = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="without adapters"):
        FrozenQwenFeatures(core)


@pytest.fixture()
def episode():
    length = 40
    source = {"episode_index": 0, "episode_length": length, "instruction": "move the object",
              "split": "train", "digest": "a"*64, "group_digest": "b"*64, "task_key": "task"}
    actions = np.arange(length*7, dtype=np.float32).reshape(length, 7)/100
    states = np.arange(length*8, dtype=np.float32).reshape(length, 8)/100
    videos = np.zeros((length, 2, 256, 256, 3), np.uint8)
    for t in range(length):
        videos[t, 0, :, :, 0] = t
        videos[t, 1, :, :, 2] = 100+t
    bounds = {"lower_q01": [0]*8, "upper_q99": [4]*8}
    return source, actions, states, videos, bounds


def test_sample_camera_target_alignment_and_action_stop_are_exact(episode):
    source, actions, states, videos, bounds = episode
    sample = make_future_sample(source, 3, 8, actions, states, videos, ActionNormalizer.identity(), bounds)
    x = sample.observation
    np.testing.assert_array_equal(x.current_rgb, videos[3])
    np.testing.assert_allclose(x.current_images[0, 0], 3/255, atol=1e-8)
    np.testing.assert_allclose(sample.target_images[0, 0], 11/255, atol=1e-8)
    np.testing.assert_allclose(sample.target_images[1, 2], 111/255, atol=1e-8)
    np.testing.assert_array_equal(x.actions[:8], actions[3:11])
    assert x.actions[8:].count_nonzero() == 0
    assert sample.identity["action_stop_exclusive"] == 11


def test_future_images_states_and_target_action_cannot_change_inputs(episode):
    source, actions, states, videos, bounds = episode
    normalizer = ActionNormalizer.identity()
    before = make_future_sample(source, 3, 8, actions, states, videos, normalizer, bounds)
    videos[11] = 255
    states[4:] = float("nan")
    actions[11:] = float("nan")
    after = make_future_sample(source, 3, 8, actions, states, videos, normalizer, bounds)
    for name in ("current_images", "state", "actions"):
        torch.testing.assert_close(getattr(before.observation, name), getattr(after.observation, name),
                                   rtol=0, atol=0)
    np.testing.assert_array_equal(before.observation.current_rgb, after.observation.current_rgb)
    assert not torch.equal(before.target_images, after.target_images)


def test_collation_keeps_labels_outside_model_condition(episode):
    source, actions, states, videos, bounds = episode
    samples = [make_future_sample(source, 3, h, actions, states, videos, ActionNormalizer.identity(), bounds)
               for h in (2, 8)]
    batch = collate_future_samples(samples, torch.ones(2, 5, 8), torch.zeros(2, 5, dtype=torch.bool))
    assert batch.condition.horizons.tolist() == [2, 8]
    assert batch.targets.shape == (2, 2, 3, 128, 128)
    assert not hasattr(batch.condition, "targets") and not hasattr(batch.condition, "identities")


def test_dataset_test_split_rejected_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match="train/validation"):
        FuturePairDataset(tmp_path/"missing", tmp_path/"missing", split="test",
                          expected_manifest_sha256="a"*64, normalizer=ActionNormalizer.identity(),
                          state_bounds={})


@pytest.mark.parametrize("anchor,h", [(-1, 2), (38, 2), (0, 3), (8, 32)])
def test_sample_boundaries_are_enforced(episode, anchor, h):
    source, actions, states, videos, bounds = episode
    with pytest.raises(ValueError, match="out-of-episode"):
        make_future_sample(source, anchor, h, actions, states, videos, ActionNormalizer.identity(), bounds)
