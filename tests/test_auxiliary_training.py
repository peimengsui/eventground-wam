from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_b0_oft import batch as action_batch
from test_b0_oft import policy, recipe

from eventground_wam.auxiliary_data import AuxiliaryTargetCache, future_references
from eventground_wam.auxiliary_losses import auxiliary_backward, joint_backward
from eventground_wam.auxiliary_policy import (
    AuxiliaryInputs,
    ObservationFeatures,
    policy_with_features,
)
from eventground_wam.auxiliary_training import AuxiliaryTrainingEngine
from eventground_wam.b0_training import CheckpointStore, cpu_tree, seed_training
from eventground_wam.demonstration_contracts import canonical_json_bytes
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    JepaPredictorConfig,
    jepa_prediction_loss,
    normalize_predictions,
)
from eventground_wam.jepa_training import LOSS_RULE
from eventground_wam.pixel_predictor import (
    PixelPredictor,
    PixelPredictorConfig,
    pixel_reconstruction_loss,
)
from eventground_wam.pixel_training import tree_digest


def predictor(method, feature_dimension=8):
    kwargs = {"feature_dimension": feature_dimension, "dimension": 32, "heads": 4, "layers": 1}
    if method == "jepa":
        return JepaPredictor(JepaPredictorConfig(**kwargs))
    model = PixelPredictor(PixelPredictorConfig(**kwargs))
    # CPT has already trained the initially zero residual head.
    with torch.no_grad():
        model.residual_head.weight.normal_(std=.01)
    return model


def future(feature_dimension=8):
    values = torch.randn(2, 13, feature_dimension, requires_grad=True)
    padding = torch.zeros(2, 13, dtype=torch.bool)
    padding[0, -1] = True
    return ObservationFeatures(values, padding), AuxiliaryInputs(
        torch.tensor([0, 0, 1]), torch.rand(2, 2, 3, 128, 128), torch.rand(2, 8),
        torch.rand(3, 32, 7), torch.tensor([2, 4, 2]))


def targets(method):
    return (torch.rand(3, 2, 3, 128, 128) if method == "pixel" else
            normalize_predictions(torch.randn(3, 128, 2048))).requires_grad_(True)


@pytest.mark.parametrize("method", ["pixel", "jepa"])
@pytest.mark.parametrize("microbatch", [1, 2, 3])
def test_full_loss_matches_accumulated_predictor_and_shared_feature_gradients(method, microbatch):
    seed_training(12)
    model = predictor(method)
    features, inputs = future()
    truth = targets(method)
    predicted = model(inputs.condition(features))
    loss = (pixel_reconstruction_loss(predicted.images, truth.detach(), ssim_weight=.1)
            if method == "pixel" else jepa_prediction_loss(
                predicted.features, truth, inputs.anchor_indices, **LOSS_RULE).total)
    (loss*.13).backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    hidden_gradient = features.values.grad.clone()
    model.zero_grad(set_to_none=True)
    metrics, derivative = auxiliary_backward(model, method, features, inputs, truth,
                                             microbatch=microbatch, loss_weight=.13)
    assert truth.grad is None
    assert abs(metrics["future_loss"]-loss.item()) < 1e-6
    actual = [p.grad for p in model.parameters()]+[derivative]
    expected = gradients+[hidden_gradient]
    errors = torch.cat([(a-b).flatten() for a, b in zip(actual, expected, strict=True)])
    reference = torch.cat([b.flatten() for b in expected])
    # Pixel's clamped RGB/L1 path has nonsmooth points; convolution execution
    # shapes can change their FP32 rounding. Keep a separate absolute bound.
    assert (errors.norm()/reference.norm()).item() < (1e-3 if method == "pixel" else 1e-5)
    assert errors.abs().max() < 1e-5
    assert derivative[~features.padding].abs().sum() > 0
    assert derivative[features.padding].count_nonzero() == 0


@pytest.mark.parametrize("method", ["pixel", "jepa"])
def test_auxiliary_only_reaches_policy_state_and_backbone_but_not_action_head(method):
    seed_training(6)
    p = policy()
    b = action_batch(0)
    original = p(b["inputs"], b["proprioception"], normalized_action_targets=b["targets"])
    output, features = policy_with_features(p, b["inputs"], b["proprioception"], -b["targets"])
    assert torch.equal(original.raw_action_predictions, output.raw_action_predictions)
    _, inputs = future(2048)
    model = predictor(method, 2048)
    _, derivative = auxiliary_backward(model, method, features, inputs, targets(method), microbatch=2)
    features.values.backward(derivative)
    assert p.conditioning_adapter.backbone.lora_scale.grad.abs().sum() > 0
    assert p.conditioning_adapter.proprioception_projector[0].weight.grad.abs().sum() > 0
    assert all(v.grad is None for v in p.action_decoder.parameters())
    assert p.conditioning_adapter.backbone.embedding.weight.grad is None
    assert not p.conditioning_adapter.backbone._forward_hooks
    mutated = replace(inputs, actions=inputs.actions*100, current_images=1-inputs.current_images)
    model(mutated.condition(ObservationFeatures(features.values.detach(), features.padding)))
    after = p(b["inputs"], b["proprioception"])
    assert torch.equal(original.raw_action_predictions, after.raw_action_predictions)


@pytest.mark.parametrize("method", ["pixel", "jepa"])
def test_joint_vjp_matches_direct_end_to_end_backward(method):
    seed_training(6)
    p = policy()
    b = action_batch(0)
    model = predictor(method, 2048)
    _, inputs = future(2048)
    truth = targets(method)
    output, features = policy_with_features(p, b["inputs"], b["proprioception"], b["targets"])
    pred = model(inputs.condition(features))
    loss = (pixel_reconstruction_loss(pred.images, truth.detach(), ssim_weight=.1) if method == "pixel"
            else jepa_prediction_loss(pred.features, truth, inputs.anchor_indices, **LOSS_RULE).total)
    (output.loss+.13*loss).backward()
    trainable = [v for v in p.parameters() if v.requires_grad]+list(model.parameters())
    expected = [v.grad.clone() for v in trainable]
    p.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    output, features = policy_with_features(p, b["inputs"], b["proprioception"], b["targets"])
    joint_backward(output, features, model, method, inputs, truth, loss_weight=.13, microbatch=2)
    error = torch.cat([(v.grad-g).flatten() for v, g in zip(trainable, expected, strict=True)])
    ref = torch.cat([g.flatten() for g in expected])
    assert (error.norm()/ref.norm()).item() < 1e-5


def test_future_masks_keep_every_action_anchor_and_reject_split_or_boundary_errors():
    sources = {4: {"split": "train", "episode_length": 40}}
    refs, mask = future_references([(0, 4, 0), (0, 4, 32)], sources)
    assert mask.tolist() == [[True]*5, [True, True, False, False, False]]
    assert len(refs) == 7
    for tokens in ([(0, 4, 33)], [(0, 4, -1)], [(0, 5, 0)], [(0, 4, 0)]*2):
        with pytest.raises(ValueError):
            future_references(tokens, sources)
    with pytest.raises(ValueError):
        future_references([(0, 4, 0)], sources, split="validation")
    with pytest.raises(ValueError):
        future_references([(0, 4, 0)], sources, split="test")


def complete_batch(index):
    b = action_batch(index)
    _, inputs = future(2048)
    identities = [{"split": "train", "episode_index": 4, "anchor": a,
                   "horizon": h, "target": a+h} for a, h in [(0, 2), (0, 4), (1, 2)]]
    return {**b, "tokens": [(0, 4, 0), (0, 4, 1)], "future_inputs": inputs,
            "future_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool),
            "future_identities": identities,
            "future_digest": hashlib.sha256(canonical_json_bytes(identities)).hexdigest()}


def engine(method):
    config = {**recipe(), "auxiliary_learning_rate": 1e-4}
    return AuxiliaryTrainingEngine(policy(), ("conditioning_adapter.backbone.lora_scale",),
        predictor(method, 2048), method, config, identity_digest="a"*64,
        steps_per_epoch=8, device=torch.device("cpu"), loss_weight=.13, microbatch=2)


@pytest.mark.parametrize("method", ["pixel", "jepa"])
def test_disk_recovery_restores_optimizer_scheduler_both_models_rng_and_cursors(method, tmp_path):
    seed_training(17)
    ref = engine(method)
    ref.step(complete_batch(0), targets(method))
    saved = cpu_tree(ref.snapshot())
    store = CheckpointStore(tmp_path, "a"*64)
    record = store.save(saved)
    ref.step(complete_batch(1), targets(method))
    expected = tree_digest(ref.snapshot())
    # Recreate the same frozen base, then disturb RNG before restoring it.
    seed_training(17)
    resumed = engine(method)
    seed_training(899)
    resumed.restore(store.load())
    resumed.step(complete_batch(1), targets(method))
    assert tree_digest(resumed.snapshot()) == expected
    assert resumed.samples_seen == 4 and resumed.future_pairs_seen == 6
    committed = cpu_tree(resumed.snapshot())
    def kill(a, b):
        raise RuntimeError("partial")
    with pytest.raises(RuntimeError):
        resumed.step(complete_batch(2), targets(method), hook=kill)
    with pytest.raises(ValueError, match="partial"):
        resumed.snapshot()
    with pytest.raises(ValueError, match="partial"):
        resumed.step(complete_batch(2), targets(method))
    resumed.restore(committed)
    assert tree_digest(resumed.snapshot()) == tree_digest(committed)
    damaged = store.save(committed)
    (tmp_path/damaged["checkpoint"]).write_bytes(b"interrupted write")
    assert store.load()["global_step"] == record["global_step"]
    assert store.recovery_notes


def test_teacher_cache_is_detached_bounded_and_reproducible_after_restart():
    def teacher(rgb):
        return normalize_predictions(torch.stack([torch.randn(128, 2048,
            generator=torch.Generator().manual_seed(int(v[0, 0, 0, 0]))) for v in rgb]))
    b = {"future_identities": [{"split": "train", "source_digest": "a", "target": t}
                              for t in [2, 2, 4]],
         "target_rgb": np.stack([np.full((2, 256, 256, 3), t, np.uint8) for t in [2, 2, 4]])}
    c = AuxiliaryTargetCache(teacher, maximum_bytes=128*2048*4)
    values, info = c.encode(b, "cpu")
    again, _ = AuxiliaryTargetCache(teacher).encode(b, "cpu")
    assert torch.equal(values, again) and not values.requires_grad
    assert info["target_cache_hits"] == 1 and len(c.values) == 1
    b["future_identities"][0]["split"] = "test"
    with pytest.raises(ValueError):
        c.encode(b, "cpu")


def test_capture_hook_is_removed_when_original_policy_rejects_targets():
    p = policy()
    b = action_batch(0)
    bad = b['targets'].clone()
    bad[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        policy_with_features(p, b['inputs'], b['proprioception'], bad)
    assert not p.conditioning_adapter.backbone._forward_hooks
    output, _ = policy_with_features(p, b['inputs'], b['proprioception'], b['targets'])
    assert torch.isfinite(output.loss)


@pytest.mark.parametrize('field', ['split', 'anchor', 'digest', 'cursor'])
def test_invalid_future_batch_rejected_before_starting_update(field):
    e = engine('jepa')
    b = complete_batch(0)
    if field == 'split':
        b['future_identities'][0]['split'] = 'validation'
    elif field == 'anchor':
        b['future_identities'][0]['anchor'] = 2
    elif field == 'digest':
        b['future_digest'] = 'f'*64
    else:
        b['index'] = 1
    before = tree_digest(e.snapshot())
    with pytest.raises(ValueError):
        e.step(b, targets('jepa'))
    # Batch/target construction consumes RNG; model, optimizer and cursors must not change.
    after = e.snapshot()
    assert e.global_step == 0 and not e._in_update
    assert after['optimizer']['state'] == {}
    assert before and after['future_chain'] == '0'*64
