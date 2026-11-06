"""Exact full-batch auxiliary losses with bounded predictor graphs and shared VJPs."""
from __future__ import annotations

import math

import torch

from eventground_wam.auxiliary_policy import ObservationFeatures
from eventground_wam.jepa_predictor import jepa_prediction_loss
from eventground_wam.jepa_training import LOSS_RULE
from eventground_wam.pixel_predictor import pixel_reconstruction_loss


@torch.backends.cudnn.flags(allow_tf32=False)
def auxiliary_backward(model, method, features, inputs, targets, *, microbatch=10,
                       loss_weight=1., hook=None):
    """Accumulate predictor gradients and return d(weighted loss)/d(shared hidden).

    Shared hidden states are FP32 leaves during the small-predictor backwards.
    The caller backpropagates the returned derivative into the original policy
    once, together with its action loss. JEPA differentiates one full-batch output
    bank and replays deterministic chunks, preserving cross-anchor variance.
    """
    if (method not in {"pixel", "jepa"} or type(microbatch) is not int or microbatch < 1
            or not math.isfinite(loss_weight) or loss_weight < 0):
        raise ValueError("invalid auxiliary method, microbatch or weight")
    count = len(inputs.horizons)
    if len(targets) != count:
        raise ValueError("target and future-pair cardinalities differ")
    leaf = features.values.detach().float().requires_grad_(True)
    source = ObservationFeatures(leaf, features.padding)
    if not count:
        return {"future_loss": 0., "future_pairs": 0}, torch.zeros_like(leaf)
    ranges = [(a, min(a+microbatch, count)) for a in range(0, count, microbatch)]
    truth = targets.detach()
    if method == "jepa":
        with torch.no_grad():
            bank = torch.cat([model(inputs.condition(source, a, b)).features for a, b in ranges])
        bank.requires_grad_(True)
        loss = jepa_prediction_loss(bank, truth, inputs.anchor_indices, **LOSS_RULE)
        derivative, = torch.autograd.grad(loss.total*loss_weight, bank)
        metrics = {"future_loss": float(loss.total.detach()), "cosine": float(loss.cosine.detach()),
                   "variance": float(loss.variance.detach()),
                   "independent_anchors": loss.independent_anchors, "future_pairs": count}
        for a, b in ranges:
            predicted = model(inputs.condition(source, a, b)).features
            if not torch.equal(predicted.detach(), bank[a:b]):
                raise ValueError("JEPA predictor replay changed within the same update")
            predicted.backward(derivative[a:b])
            if hook is not None:
                hook(a, b)
    else:
        total = 0.
        for a, b in ranges:
            prediction = model(inputs.condition(source, a, b)).images
            loss = pixel_reconstruction_loss(prediction, truth[a:b], ssim_weight=.1)
            fraction = (b-a)/count
            (loss*loss_weight*fraction).backward()
            total += float(loss.detach())*fraction
            if hook is not None:
                hook(a, b)
        metrics = {"future_loss": total, "future_pairs": count}
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("auxiliary loss did not produce a finite shared-feature gradient")
    if not math.isfinite(metrics["future_loss"]):
        raise ValueError("nonfinite auxiliary loss")
    return metrics, leaf.grad.detach()


def joint_backward(action_output, features, model, method, inputs, targets, *,
                   microbatch=10, loss_weight=1., hook=None):
    metrics, derivative = auxiliary_backward(
        model, method, features, inputs, targets, microbatch=microbatch,
        loss_weight=loss_weight, hook=hook)
    loss = action_output.loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("joint training requires a finite action loss")
    torch.autograd.backward((loss, features.values),
                            (None, derivative.to(features.values.dtype)))
    return {**metrics, "action_loss": float(loss.detach()),
            "joint_loss": float(loss.detach())+loss_weight*metrics["future_loss"],
            "weighted_auxiliary_feature_gradient_norm": float(derivative.norm())}
