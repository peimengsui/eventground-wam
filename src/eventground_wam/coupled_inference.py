"""Observation-only deployment adapter for the frozen M3/M4 coupled policies."""
from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from eventground_wam.b0_inference import B0InferenceError, B0InferencePolicy, decode_b0_action_chunk
from eventground_wam.pixel_baselines import box_rgb


class CoupledInferencePolicy(B0InferencePolicy):
    """Reuse B0 action decoding, counters and recorder interface for both arms."""

    def predict_chunk(
        self,
        *,
        agent_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        proprioception: np.ndarray,
        instruction: str,
        suite: str,
    ) -> NDArray[np.float64]:
        """Use canonical recorder observations only; never accept demonstrated futures."""

        for name, image in (("agent_rgb", agent_rgb), ("wrist_rgb", wrist_rgb)):
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise B0InferenceError(f"{name} must be uint8 HWC 256x256")
        state = np.asarray(proprioception, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise B0InferenceError("proprioception must be a finite 8-vector")
        if not isinstance(instruction, str) or not instruction:
            raise B0InferenceError("instruction must be a nonempty string")
        if not isinstance(suite, str) or not suite:
            raise B0InferenceError("suite must be a nonempty string")
        # Recorder images already have the canonical orientation. box_rgb is the
        # exact training reducer and already returns FP32 [0,1] CHW values.
        current = torch.from_numpy(np.stack([box_rgb(agent_rgb), box_rgb(wrist_rgb)]))
        current = current.unsqueeze(0).to(device=self._device, dtype=torch.float32)
        encoding = self._input_builder([instruction], [agent_rgb], [wrist_rgb])
        inputs = {
            name: value.to(
                device=self._device,
                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            )
            for name, value in encoding.items()
        }
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device=self._device)
        self._policy.eval()
        with torch.inference_mode():
            output = self._policy(inputs, state_tensor, current)
        normalized = output.normalized_actions.detach().float().cpu().numpy()
        if normalized.shape != (1, 8, 7):
            raise B0InferenceError(
                f"B0 policy output has shape {normalized.shape}, expected (1,8,7)"
            )
        actions, decode_summary = decode_b0_action_chunk(normalized[0], self._normalizer)
        self._prediction_count += 1
        self._clipped_continuous_value_count += decode_summary[
            "clipped_continuous_value_count"
        ]
        self._maximum_continuous_clipping_delta = max(
            self._maximum_continuous_clipping_delta,
            decode_summary["maximum_continuous_clipping_delta"],
        )
        return actions

