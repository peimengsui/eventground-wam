from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eventground_wam.jepa_data import JepaPairDataset, JepaTrainingSample, collate_jepa_samples
from eventground_wam.jepa_features import normalize_visual_targets
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    JepaPredictorConfig,
    jepa_prediction_loss,
    load_jepa_checkpoint,
    normalize_predictions,
    save_jepa_checkpoint,
)
from eventground_wam.pixel_data import FutureObservation
from eventground_wam.pixel_predictor import PixelCondition, PixelPredictor


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture()
def condition():
    g = torch.Generator().manual_seed(56)
    return PixelCondition(torch.randn(2, 5, 8, generator=g),
                          torch.tensor([[False, False, False, True, True], [False]*5]),
                          torch.rand(2, 2, 3, 128, 128, generator=g),
                          torch.randn(2, 8, generator=g), torch.randn(2, 32, 7, generator=g),
                          torch.tensor([2, 32]))


@pytest.fixture()
def model():
    return JepaPredictor(JepaPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))


def test_exact_default_core_budget_initialization_and_separate_head():
    jepa = JepaPredictor()
    pixel = PixelPredictor(jepa.config.pixel_config())
    assert jepa.parameter_budget() == {"future_core": 4238592, "jepa_projection": 526336,
                                      "total_trainable": 4764928, "trainable_base_parameters": 0,
                                      "future_tokens": 128, "future_token_dimension": 256,
                                      "target_dimension": 2048}
    for key, value in pixel.state_dict().items():
        if not key.startswith(("image_decoder.", "residual_head.")):
            torch.testing.assert_close(value, jepa.state_dict()[key], atol=0, rtol=0)
    assert not hasattr(jepa, "image_decoder") and not hasattr(jepa, "residual_head")


def test_pixel_future_tokens_match_exactly_and_prediction_is_normalized(model, condition):
    pixel = PixelPredictor(model.config.pixel_config())
    result = model(condition)
    torch.testing.assert_close(pixel(condition).future_tokens, result.future_tokens, atol=0, rtol=0)
    assert result.features.shape == (2, 128, 2048)
    assert (result.features.norm(dim=-1)-1).abs().max() < 1e-6
    assert result.features.mean(dim=-1).abs().max() < 1e-7
    assert not torch.equal(result.features[:, :64], result.features[:, 64:])


def test_prediction_normalization_matches_teacher_but_keeps_gradients():
    raw = torch.randn(2, 128, 2048, requires_grad=True)
    predicted = normalize_predictions(raw)
    torch.testing.assert_close(predicted, normalize_visual_targets(raw), rtol=0, atol=0)
    predicted[..., 0].sum().backward()
    assert raw.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="degenerate"):
        normalize_predictions(torch.ones_like(raw))


@pytest.mark.parametrize("ablation", ["zero-actions", "reverse-actions", "zero-horizon", "zero-features"])
def test_conditioning_changes_latent_predictions(model, condition, ablation):
    assert (model(condition).features-model(condition, ablation=ablation).features).abs().max() > 1e-5


def test_padded_action_and_feature_poison_cannot_enter_prediction(model, condition):
    actions, features = condition.actions.clone(), condition.features.clone()
    actions[0, 2:] = float("nan")
    features[0, 3:] = float("nan")
    changed = replace(condition, actions=actions, features=features)
    for ablation in ("none", "reverse-actions"):
        torch.testing.assert_close(model(condition, ablation=ablation).features,
                                   model(changed, ablation=ablation).features, atol=0, rtol=0)


def test_targets_are_separate_and_teacher_gradients_are_stopped(model, condition):
    features = condition.features.clone().requires_grad_(True)
    prediction = model(replace(condition, features=features)).features
    target = normalize_predictions(torch.randn_like(prediction)).detach().requires_grad_(True)
    loss = jepa_prediction_loss(prediction, target, torch.tensor([0, 1]))
    loss.total.backward()
    assert target.grad is None and features.grad.abs().sum() > 0
    assert features.grad[0, 3:].abs().sum() == 0
    for module in (model.feature_projection, model.action_projection, model.action_sequence_projection,
                   model.state_projection, model.horizon_projection, model.current_patches, model.output_projection):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    with pytest.raises(TypeError):
        model(condition, targets=target)
    with pytest.raises(TypeError):
        model({"condition": condition, "targets": target})


def test_variance_uses_independent_anchors_not_spatial_texture_or_repeated_horizons():
    target = normalize_predictions(torch.randn(2, 128, 2048))
    identical = target[:1].repeat(2, 1, 1).requires_grad_(True)
    penalty = jepa_prediction_loss(identical, target, torch.tensor([0, 1]))
    assert penalty.variance > .1 and penalty.independent_anchors == 2
    repeat = jepa_prediction_loss(identical.repeat_interleave(5, 0), target.repeat_interleave(5, 0),
                                  torch.tensor([0]*5+[1]*5))
    torch.testing.assert_close(repeat.variance, penalty.variance)
    singleton = jepa_prediction_loss(identical, target, torch.tensor([0, 0]))
    assert singleton.variance == 0 and singleton.independent_anchors == 1
    assert jepa_prediction_loss(target, target, torch.tensor([0, 1])).variance == 0
    assert jepa_prediction_loss(target, target, torch.tensor([0, 1])).cosine < 1e-6


@pytest.mark.parametrize("bad", ["nan", "zero", "nonunit", "groups", "weight"])
def test_invalid_loss_inputs_rejected(bad):
    target = normalize_predictions(torch.randn(2, 128, 2048))
    prediction, groups, weight = target.clone(), torch.tensor([0, 1]), .1
    if bad == "nan":
        prediction[0, 0, 0] = float("nan")
    elif bad == "zero":
        prediction.zero_()
    elif bad == "nonunit":
        prediction *= 2
    elif bad == "groups":
        groups = groups.float()
    else:
        weight = -1
    with pytest.raises(ValueError):
        jepa_prediction_loss(prediction, target, groups, variance_weight=weight)


def test_checkpoint_exact_reload_and_corruption_rejection(model, condition, tmp_path):
    path = tmp_path/"jepa.pt"
    save_jepa_checkpoint(path, model, metadata={"purpose": "diagnostic-only"})
    restored, header = load_jepa_checkpoint(path)
    assert header["metadata"]["purpose"] == "diagnostic-only"
    torch.testing.assert_close(restored(condition).features, model(condition).features, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_jepa_checkpoint(path, model, metadata={})
    saved = torch.load(path, weights_only=True)
    saved["state_dict"]["output_projection.bias"][0] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="changed"):
        load_jepa_checkpoint(path)


def test_rng_and_core_seed_are_independent_of_output_projection_seed():
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    a = JepaPredictor(JepaPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))
    b = JepaPredictor(replace(a.config, projection_seed=55))
    assert torch.equal(before, torch.get_rng_state())
    for key, value in a.state_dict().items():
        if not key.startswith("output_projection."):
            assert torch.equal(value, b.state_dict()[key])
    assert not torch.equal(a.output_projection.weight, b.output_projection.weight)


def test_native_targets_and_group_ids_never_enter_condition(condition, model):
    rgb = np.zeros((2, 256, 256, 3), np.uint8)
    samples = []
    for i in range(2):
        observation = FutureObservation("task", rgb.copy(), condition.current_images[i],
                                        condition.state[i], condition.actions[i], int(condition.horizons[i]))
        samples.append(JepaTrainingSample(observation, torch.zeros(2, 3, 128, 128),
                                          {"episode_index": 8, "anchor": 9, "target": 9+observation.horizon}, rgb.copy()))
    targets = normalize_predictions(torch.randn(2, 128, 2048)).requires_grad_(True)
    batch = collate_jepa_samples(samples, condition.features, condition.feature_padding, targets)
    assert not batch.targets.requires_grad and batch.anchor_groups.tolist() == [0, 0]
    before = model(batch.condition).features
    samples[0].target_rgb[:] = 255
    altered = collate_jepa_samples(samples, condition.features, condition.feature_padding, -targets)
    torch.testing.assert_close(before, model(altered.condition).features, atol=0, rtol=0)


def test_test_split_rejected_before_filesystem_access(tmp_path):
    with pytest.raises(ValueError, match="train/validation"):
        JepaPairDataset(tmp_path/"missing", tmp_path/"missing", split="test",
                        expected_manifest_sha256="", normalizer=None, state_bounds={})


def test_independent_numpy_loss_includes_grouped_variance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from verify_jepa_predictor import numpy_metrics

    target = normalize_predictions(torch.randn(6, 128, 2048))
    prediction = normalize_predictions(torch.ones_like(target)*0 + torch.randn(1, 128, 2048))
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    rule = {"variance_weight": .1, "variance_floor_ratio": .5, "variance_epsilon": 1e-4}
    result = jepa_prediction_loss(prediction, target, groups, **rule)
    reference = numpy_metrics(prediction.numpy(), target.numpy(), groups.numpy(), rule)
    assert result.variance > 0
    for field in ("total", "cosine", "variance"):
        assert abs(getattr(result, field).item()-reference[field]) < 2e-7
