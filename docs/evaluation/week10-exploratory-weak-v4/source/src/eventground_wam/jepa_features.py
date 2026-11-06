"""Frozen, image-only JEPA targets; no predictor or policy training in this module."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

TARGET_VERSION = "qwen35-vision-merged-ln-l2-single-image-v2"
TARGET_DIMENSION = 2048
TOKENS_PER_CAMERA = 64
LAYER_NORM_EPS = 1e-5
L2_EPS = 1e-6


def normalize_visual_targets(raw: Tensor) -> Tensor:
    """FP32, per-token non-affine LayerNorm then unit L2; no fitted statistics."""
    if (raw.ndim != 3 or raw.shape[1:] != (128, TARGET_DIMENSION)
            or len(raw) == 0 or not raw.is_floating_point() or not torch.isfinite(raw).all()):
        raise ValueError("expected finite [batch,128,2048] visual tokens")
    values = F.layer_norm(raw.detach().float(), (TARGET_DIMENSION,), eps=LAYER_NORM_EPS)
    if (torch.linalg.vector_norm(values, dim=-1) <= L2_EPS).any():
        raise ValueError("degenerate constant-channel visual target")
    return F.normalize(values, dim=-1, eps=L2_EPS).detach()


class FrozenQwenVisualTargets(nn.Module):
    """Encode [B,front/wrist,256,256,RGB] uint8 without language or future metadata.

    The caller loads and verifies the pinned base weights. Only its vision tower
    is retained here. Qwen's merger returns 8x8 row-major tokens per camera;
    front precedes wrist. This teacher must remain independent of policy LoRA.
    """

    def __init__(self, visual: nn.Module, image_processor):
        super().__init__()
        if any("lora" in name.lower() or "adapter" in name.lower()
               for name, _ in visual.named_parameters()):
            raise ValueError("JEPA targets require an unadapted base vision tower")
        expected = {"patch_size": 16, "spatial_merge_size": 2,
                    "temporal_patch_size": 2, "out_hidden_size": 2048,
                    "hidden_size": 1024, "depth": 24}
        if any(getattr(visual.config, key, None) != value for key, value in expected.items()):
            raise ValueError("vision configuration differs from the pinned Qwen target")
        processor_values = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
                            "backend": "torchvision", "rescale_factor": 1/255}
        if (any(getattr(image_processor, key, None) != value
                for key, value in processor_values.items())
                or tuple(image_processor.image_mean) != (.5, .5, .5)
                or tuple(image_processor.image_std) != (.5, .5, .5)):
            raise ValueError("target image processor differs from the pinned contract")
        self.visual = visual.requires_grad_(False).eval()
        self.image_processor = image_processor

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self

    def preprocess(self, rgb: np.ndarray) -> dict:
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 5 or rgb.shape[1:] != (2, 256, 256, 3)
                or not 1 <= len(rgb) <= 16):
            raise ValueError("expected 1..16 dual-camera uint8 RGB observations at 256x256")
        images = [np.ascontiguousarray(frame) for frame in rgb.reshape(-1, 256, 256, 3)]
        encoded = self.image_processor(
            images=images, do_resize=False, do_rescale=True, do_normalize=True,
            do_convert_rgb=True, return_tensors="pt")
        if set(encoded) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("image-only target preprocessing returned unexpected fields")
        pixels, grid = encoded["pixel_values"], encoded["image_grid_thw"]
        expected_grid = torch.tensor([1, 16, 16], device=grid.device).expand(2*len(rgb), 3)
        if (grid.shape != expected_grid.shape or not torch.equal(grid, expected_grid)
                or grid.dtype != torch.int64
                or pixels.shape != (2*len(rgb)*256, 1536)
                or not pixels.is_floating_point() or not torch.isfinite(pixels).all()
                or pixels.min() < -1.000001 or pixels.max() > 1.000001):
            raise ValueError("target patch values or grid differ from the frozen layout")
        return dict(encoded)

    @torch.no_grad()
    def encode_raw(self, rgb: np.ndarray) -> Tensor:
        encoded = self.preprocess(rgb)
        self.visual.eval()
        if any(parameter.requires_grad for parameter in self.visual.parameters()):
            raise ValueError("target encoder was unfrozen")
        parameter = next(self.visual.parameters())
        if parameter.dtype != torch.bfloat16:
            raise ValueError("target encoder must use the pinned BF16 execution path")
        # The pinned BF16 tower failed batch-composition invariance. Canonical
        # single-image calls fix its execution shape and exclude all companions.
        pixels = encoded["pixel_values"].to(device=parameter.device, dtype=parameter.dtype)
        grid = encoded["image_grid_thw"].to(parameter.device)
        outputs = []
        for index in range(2*len(rgb)):
            result = self.visual(pixels[index*256:(index+1)*256],
                                 grid_thw=grid[index:index+1], return_dict=True)
            if result.pooler_output.shape != (64, TARGET_DIMENSION):
                raise ValueError("single-image vision merger returned an invalid layout")
            outputs.append(result.pooler_output)
        raw = torch.cat(outputs)
        if raw.shape != (len(rgb)*128, TARGET_DIMENSION) or not torch.isfinite(raw).all():
            raise ValueError("vision merger did not return 64 finite tokens per image")
        return raw.reshape(len(rgb), 128, TARGET_DIMENSION).detach()

    def forward(self, rgb: np.ndarray) -> Tensor:
        return normalize_visual_targets(self.encode_raw(rgb))


def anchor_feature_diagnostics(features: Tensor) -> dict:
    """Across independent observations at fixed camera; spatial variance is separate.

    Pool the 64 tokens only for the across-example rank statistic. Its purpose is
    teacher sanity, not a trained predictor acceptance test or retrieval metric.
    """
    if (features.ndim != 3 or features.shape[1:] != (64, TARGET_DIMENSION)
            or len(features) < 2 or not torch.isfinite(features).all()):
        raise ValueError("diagnostics require at least two finite camera observations")
    values = features.detach().cpu().double()
    pooled = values.mean(dim=1)
    centered = pooled-pooled.mean(dim=0, keepdim=True)
    energy = torch.linalg.svdvals(centered).square()
    total = energy.sum()
    if total <= 1e-24:
        rank = 0.
    else:
        probabilities = energy/total
        active = probabilities[probabilities > 0]
        rank = float(torch.exp(-(active*active.log()).sum()))
    return {"observations": len(values), "pooled_effective_rank": rank,
            "pooled_across_example_std_rms": float(centered.square().mean().sqrt()),
            "fixed_position_across_example_std_rms":
                float(values.var(dim=0, correction=0).mean().sqrt()),
            "within_image_spatial_std_rms":
                float(values.var(dim=1, correction=0).mean().sqrt())}
