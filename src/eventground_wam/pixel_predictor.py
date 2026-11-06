"""Action-conditioned pixel futures with a bounded, reusable future-token interface."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.future_pairs import HORIZONS
from eventground_wam.pixel_baselines import PixelMetrics

ARCHITECTURE = "pixel-action-horizon-transformer-v2"


@dataclass(frozen=True)
class PixelPredictorConfig:
    architecture: str = ARCHITECTURE
    feature_dimension: int = 2048
    dimension: int = 256
    heads: int = 8
    layers: int = 3
    feedforward_multiplier: int = 4
    maximum_feature_tokens: int = 512
    seed: int = 20260925

    def __post_init__(self):
        if self.architecture != ARCHITECTURE:
            raise ValueError("unsupported pixel predictor architecture")
        integers = [self.feature_dimension, self.dimension, self.heads, self.layers,
                    self.feedforward_multiplier, self.maximum_feature_tokens, self.seed]
        if any(type(x) is not int for x in integers):
            raise ValueError("pixel configuration requires integer dimensions/seed")
        if (not 1 <= self.feature_dimension <= 2048 or not 32 <= self.dimension <= 512
                or not 1 <= self.heads <= 16 or self.dimension % self.heads
                or not 1 <= self.layers <= 6 or not 1 <= self.feedforward_multiplier <= 4
                or not 1 <= self.maximum_feature_tokens <= 512 or self.seed < 0):
            raise ValueError("invalid or unbounded pixel predictor configuration")


@dataclass(frozen=True)
class PixelCondition:
    """Only current observations, base features, and requested intervening actions.

    Feature padding is True for ignored tokens. Actions are padded to 32; only
    the first h enter the model. Images are the frozen 128x128 box preprocessing.
    State/actions use the pinned train-only state/action normalizers.
    """

    features: Tensor
    feature_padding: Tensor
    current_images: Tensor
    state: Tensor
    actions: Tensor
    horizons: Tensor

    def to(self, device):
        return PixelCondition(**{field.name: getattr(self, field.name).to(device)
                                 for field in fields(self)})


@dataclass(frozen=True)
class PixelPrediction:
    images: Tensor
    future_tokens: Tensor


class FrozenQwenFeatures(nn.Module):
    """A base-only feature extractor; no action slots, targets, LoRA or policy weights."""

    def __init__(self, core: nn.Module, dimension=2048):
        super().__init__()
        if any("lora" in name.lower() or "adapter" in name.lower()
               for name, _ in core.named_parameters()):
            raise ValueError("CPT features must use the pinned base core without adapters")
        self.core = core.requires_grad_(False).eval()
        self.dimension = dimension

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        return self

    @torch.no_grad()
    def forward(self, encoding):
        allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                   "mm_token_type_ids", "token_type_ids"}
        if set(encoding) - allowed or "input_ids" not in encoding or "attention_mask" not in encoding:
            raise ValueError("unexpected base-feature inputs; current-image prompt only")
        self.core.eval()
        result = self.core(**encoding, use_cache=False, return_dict=True)
        features = result.last_hidden_state
        mask = encoding["attention_mask"]
        if (features.shape != (*mask.shape, self.dimension) or mask.ndim != 2
                or not torch.isfinite(features).all() or not mask.bool().any(dim=1).all()):
            raise ValueError("invalid base feature output")
        return features.detach(), ~mask.bool()


class PixelPredictor(nn.Module):
    """128 spatial future tokens (front 64, wrist 64), plus a shared residual decoder."""

    def __init__(self, config: PixelPredictorConfig | None = None):
        super().__init__()
        self.config = config or PixelPredictorConfig()
        c, d = self.config, self.config.dimension
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed)
            self.feature_projection = nn.Sequential(nn.LayerNorm(c.feature_dimension),
                                                    nn.Linear(c.feature_dimension, d), nn.LayerNorm(d))
            self.current_patches = nn.Conv2d(3, d, kernel_size=16, stride=16)
            self.action_projection = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
            # Fixed temporal slots make order directly available, even when two
            # action sequences have the same multiset. Padded slots stay zero.
            self.action_sequence_projection = nn.Sequential(nn.Linear(32*7, d), nn.GELU(),
                                                            nn.Linear(d, d))
            self.state_projection = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
            self.horizon_projection = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
            self.spatial_position = nn.Parameter(torch.randn(1, 1, 64, d) * .02)
            self.camera_position = nn.Parameter(torch.randn(1, 2, 1, d) * .02)
            self.action_position = nn.Parameter(torch.randn(1, 32, d) * .02)
            self.memory_type = nn.Parameter(torch.randn(3, d) * .02)
            self.blocks = nn.ModuleList([
                nn.TransformerDecoderLayer(d, c.heads, d*c.feedforward_multiplier,
                                           dropout=0., activation="gelu", batch_first=True,
                                           norm_first=True) for _ in range(c.layers)])
            self.future_norm = nn.LayerNorm(d)
            channels = [d, 128, 64, 32, 16]
            self.image_decoder = nn.ModuleList([
                nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU())
                for a, b in pairwise(channels)])
            self.residual_head = nn.Conv2d(16, 3, 3, padding=1)
            # Begin exactly at the measured copy baseline. The first update
            # activates the residual path; later updates reach all future modules.
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    def _validate(self, x: PixelCondition):
        if not isinstance(x, PixelCondition):
            raise TypeError("predictor accepts PixelCondition, never a training batch with targets")
        tensors = [getattr(x, field.name) for field in fields(x)]
        if any(not isinstance(value, Tensor) for value in tensors):
            raise ValueError("all conditioning fields must be tensors")
        if x.horizons.ndim != 1 or x.horizons.dtype != torch.int64 or len(x.horizons) == 0:
            raise ValueError("horizons must be a nonempty int64 vector")
        b = len(x.horizons)
        if (x.features.ndim != 3 or x.features.shape[0] != b
                or x.features.shape[2] != self.config.feature_dimension
                or not 1 <= x.features.shape[1] <= self.config.maximum_feature_tokens
                or x.feature_padding.shape != x.features.shape[:2]
                or x.feature_padding.dtype != torch.bool or x.feature_padding.all(dim=1).any()
                or x.current_images.shape != (b, 2, 3, 128, 128)
                or x.state.shape != (b, 8) or x.actions.shape != (b, 32, 7)):
            raise ValueError("invalid pixel conditioning shapes or masks")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("conditioning tensors must share a device")
        if not all(value.is_floating_point() for value in
                   (x.features, x.current_images, x.state, x.actions)):
            raise ValueError("images, features, state and actions must be floating point")
        if not torch.isin(x.horizons, x.horizons.new_tensor(HORIZONS)).all():
            raise ValueError("horizon is outside the frozen five-horizon contract")
        valid_actions = torch.arange(32, device=x.actions.device)[None] < x.horizons[:, None]
        if (not torch.isfinite(x.features[~x.feature_padding]).all()
                or not torch.isfinite(x.actions[valid_actions]).all()
                or not torch.isfinite(x.current_images).all() or not torch.isfinite(x.state).all()
                or x.current_images.min() < 0 or x.current_images.max() > 1):
            raise ValueError("invalid active conditioning values")
        return valid_actions

    def forward(self, x: PixelCondition, *, ablation="none") -> PixelPrediction:
        valid_actions = self._validate(x)
        if ablation not in {"none", "zero-actions", "reverse-actions", "zero-horizon", "zero-features"}:
            raise ValueError("unknown diagnostic ablation")
        b, d = len(x.horizons), self.config.dimension
        dtype = self.current_patches.weight.dtype
        features = x.features.masked_fill(x.feature_padding[..., None], 0).to(dtype)
        if ablation == "zero-features":
            features = torch.zeros_like(features)
        feature_tokens = self.feature_projection(features) + self.memory_type[0]
        actions = x.actions.masked_fill(~valid_actions[..., None], 0).to(dtype)
        if ablation == "zero-actions":
            actions = torch.zeros_like(actions)
        elif ablation == "reverse-actions":
            indices = (x.horizons[:, None] - 1 - torch.arange(32, device=actions.device)[None]).clamp_min(0)
            actions = actions.gather(1, indices[..., None].expand(-1, -1, 7))
        actions = actions.masked_fill(~valid_actions[..., None], 0)
        action_sequence = self.action_sequence_projection(actions.flatten(1))[:, None]
        action_tokens = self.action_projection(actions) + self.action_position + self.memory_type[1]
        state_token = self.state_projection(x.state.to(dtype))[:, None] + self.memory_type[2]
        memory = torch.cat((feature_tokens, state_token, action_tokens), dim=1)
        padding = torch.cat((x.feature_padding, torch.zeros(b, 1, dtype=torch.bool, device=memory.device),
                             ~valid_actions), dim=1)
        time = torch.stack((x.horizons/32, x.horizons/20), dim=-1).to(dtype)
        if ablation == "zero-horizon":
            time = torch.zeros_like(time)
        horizon_token = self.horizon_projection(time)[:, None]
        current = x.current_images.to(dtype)
        queries = self.current_patches(current.reshape(b*2, 3, 128, 128))
        queries = queries.flatten(2).transpose(1, 2).reshape(b, 2, 64, d)
        queries = (queries + self.camera_position + self.spatial_position).reshape(b, 128, d)
        queries = queries + horizon_token + action_sequence
        for block in self.blocks:
            queries = block(queries, memory, memory_key_padding_mask=padding)
        tokens = self.future_norm(queries)
        images = self.decode(tokens, current)
        return PixelPrediction(images=images, future_tokens=tokens)

    def decode(self, future_tokens, current_images):
        b, d = len(future_tokens), self.config.dimension
        if future_tokens.shape != (b, 128, d) or current_images.shape != (b, 2, 3, 128, 128):
            raise ValueError("invalid future token/image interface")
        grid = future_tokens.reshape(b*2, 8, 8, d).permute(0, 3, 1, 2)
        for block in self.image_decoder:
            grid = block(F.interpolate(grid, scale_factor=2, mode="bilinear", align_corners=False))
        delta = torch.tanh(self.residual_head(grid)).reshape(b, 2, 3, 128, 128)
        return (current_images + delta).clamp(0, 1)

    def parameter_budget(self):
        decoder = sum(p.numel() for p in self.image_decoder.parameters())
        decoder += sum(p.numel() for p in self.residual_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total_trainable": total, "future_core": total-decoder,
                "pixel_decoder": decoder, "trainable_base_parameters": 0,
                "future_tokens": 128, "future_token_dimension": self.config.dimension}


def pixel_reconstruction_loss(prediction, target, *, ssim_weight=0.):
    """Differentiable reconstruction objective; Step 4 freezes production weights."""
    if (prediction.shape != target.shape or prediction.ndim != 5
            or prediction.shape[1:] != (2, 3, 128, 128) or not math.isfinite(ssim_weight) or ssim_weight < 0
            or not torch.isfinite(target).all() or target.min() < 0 or target.max() > 1):
        raise ValueError("invalid paired pixel loss inputs")
    l1 = F.l1_loss(prediction.float(), target.float())
    if ssim_weight == 0:
        return l1
    score = PixelMetrics(prediction.device).ssim(prediction.flatten(0, 1), target.flatten(0, 1))
    return l1 + ssim_weight * (1-score.mean()).to(l1.dtype)


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_pixel_checkpoint(path: Path, model: PixelPredictor, *, metadata: dict):
    """Atomic weight/interface snapshot, not a production optimizer-resume checkpoint."""
    if path.exists():
        raise FileExistsError(path)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    header = seal_contract({"artifact_type": "eventground_wam.pixel_predictor_weights",
                            "config": asdict(model.config), "metadata": metadata,
                            "state_sha256": tensor_state_digest(state)})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"header": header, "state_dict": state}, temporary)
    temporary.replace(path)
    return header


def load_pixel_checkpoint(path: Path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    header = artifact["header"]
    if (header["digest"] != contract_digest(header)
            or header["artifact_type"] != "eventground_wam.pixel_predictor_weights"
            or tensor_state_digest(artifact["state_dict"]) != header["state_sha256"]):
        raise ValueError("pixel checkpoint identity or weights changed")
    model = PixelPredictor(PixelPredictorConfig(**header["config"]))
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model, header
