"""Training-only access to the unchanged OFT policy's current-observation features."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from eventground_wam.pixel_predictor import PixelCondition

INTERFACE_VERSION = "oft-final-hidden-auxiliary-v1"


@dataclass(frozen=True)
class ObservationFeatures:
    values: Tensor
    padding: Tensor


@contextmanager
def capture_observation_features(policy):
    """Tap the single original backbone call without changing its inputs or output.

    The hook is removed even on failure and is absent during checkpoint backward
    recomputation. It never detaches the tensor. Captures belong to this call,
    not to persistent model state. This interface is for serial training only.
    """
    captured = []

    def capture(_module, _args, output):
        hidden = output.last_hidden_state
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise ValueError("OFT backbone must return [batch,tokens,channels] hidden states")
        captured.append(hidden)

    handle = policy.conditioning_adapter.backbone.register_forward_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


def policy_with_features(policy, inputs, state, targets=None):
    """Run the original B0 path once; supervision enters only its loss argument."""
    with capture_observation_features(policy) as captured:
        output = policy(inputs, state, normalized_action_targets=targets)
    if len(captured) != 1:
        raise ValueError("auxiliary interface requires exactly one backbone forward")
    hidden = captured[0]
    padding = ~inputs["attention_mask"].bool()
    if hidden.shape[:2] != padding.shape or hidden.shape[1] > 512:
        raise ValueError("OFT sequence differs from the bounded predictor interface")
    if not torch.isfinite(hidden[~padding]).all():
        raise ValueError("nonfinite active policy features")
    return output, ObservationFeatures(hidden, padding)


@dataclass(frozen=True)
class AuxiliaryInputs:
    """Separate training-only actions; no future target is a conditioning field."""
    anchor_indices: Tensor
    current_images: Tensor
    normalized_state: Tensor
    actions: Tensor
    horizons: Tensor

    def condition(self, features: ObservationFeatures, start=0, stop=None):
        stop = len(self.horizons) if stop is None else stop
        indices = self.anchor_indices[start:stop]
        if (indices.dtype != torch.int64 or indices.ndim != 1
                or (indices < 0).any() or (indices >= len(features.values)).any()):
            raise ValueError("future pairs must reference existing action anchors")
        return PixelCondition(
            features.values.index_select(0, indices),
            features.padding.index_select(0, indices),
            self.current_images.index_select(0, indices),
            self.normalized_state.index_select(0, indices),
            self.actions[start:stop], self.horizons[start:stop],
        )

    def to(self, device):
        return AuxiliaryInputs(**{key: value.to(device) for key, value in vars(self).items()})
