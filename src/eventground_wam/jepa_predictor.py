"""Pixel-matched future tokens with a differentiable frozen-space JEPA head."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.jepa_features import L2_EPS, LAYER_NORM_EPS, TARGET_DIMENSION
from eventground_wam.pixel_predictor import (
    ARCHITECTURE as PIXEL_ARCHITECTURE,
)
from eventground_wam.pixel_predictor import (
    PixelCondition,
    PixelPredictor,
    PixelPredictorConfig,
    tensor_state_digest,
)

ARCHITECTURE = "jepa-pixel-matched-transformer-v1"


@dataclass(frozen=True)
class JepaPredictorConfig(PixelPredictorConfig):
    architecture: str = ARCHITECTURE
    target_dimension: int = TARGET_DIMENSION
    projection_seed: int = 20260926

    def __post_init__(self):
        if (self.architecture != ARCHITECTURE or self.target_dimension != TARGET_DIMENSION
                or type(self.target_dimension) is not int
                or type(self.projection_seed) is not int or self.projection_seed < 0):
            raise ValueError("unsupported JEPA architecture or target space")
        self.pixel_config()

    def pixel_config(self):
        values = asdict(self)
        values.pop("target_dimension")
        values.pop("projection_seed")
        return PixelPredictorConfig(**{**values, "architecture": PIXEL_ARCHITECTURE})


@dataclass(frozen=True)
class JepaPrediction:
    features: Tensor
    future_tokens: Tensor


def normalize_predictions(raw: Tensor) -> Tensor:
    """Same geometry as the teacher, while retaining prediction gradients."""
    if (raw.ndim != 3 or raw.shape[1:] != (128, TARGET_DIMENSION) or len(raw) == 0
            or not raw.is_floating_point() or not torch.isfinite(raw).all()):
        raise ValueError("expected finite [batch,128,2048] predicted fields")
    values = F.layer_norm(raw.float(), (TARGET_DIMENSION,), eps=LAYER_NORM_EPS)
    if (torch.linalg.vector_norm(values, dim=-1) <= L2_EPS).any():
        raise ValueError("degenerate constant-channel prediction")
    return F.normalize(values, dim=-1, eps=L2_EPS)


class JepaPredictor(PixelPredictor):
    """Execute the accepted Pixel core verbatim, with no RGB decoder or skip.

    The parent forward's decode hook is an identity. Its temporary image field
    is unused; only its future_tokens cross this internal compatibility bridge.
    Public JEPA outputs are normalized target-space features and future tokens.
    """

    def __init__(self, config: JepaPredictorConfig | None = None):
        config = config or JepaPredictorConfig()
        super().__init__(config.pixel_config())
        del self.image_decoder, self.residual_head
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.projection_seed)
            self.output_projection = nn.Linear(config.dimension, config.target_dimension)

    def decode(self, future_tokens, current_images):
        return future_tokens

    def forward(self, x: PixelCondition, *, ablation="none") -> JepaPrediction:
        tokens = super().forward(x, ablation=ablation).future_tokens
        return JepaPrediction(normalize_predictions(self.output_projection(tokens)), tokens)

    def parameter_budget(self):
        projection = sum(p.numel() for p in self.output_projection.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total_trainable": total, "future_core": total-projection,
                "jepa_projection": projection, "trainable_base_parameters": 0,
                "future_tokens": 128, "future_token_dimension": self.config.dimension,
                "target_dimension": self.config.target_dimension}


@dataclass(frozen=True)
class JepaLoss:
    total: Tensor
    cosine: Tensor
    variance: Tensor
    independent_anchors: int


def jepa_prediction_loss(prediction, target, anchor_groups, *, variance_weight=.1,
                         variance_floor_ratio=.5, variance_epsilon=1e-4):
    """Cosine plus teacher-relative diversity across distinct current anchors.

    Average repeated horizons of an anchor first, then population variance
    across anchors at each fixed camera/spatial/channel coordinate. Scale unit
    vectors by sqrt(2048) before computing std. A singleton contributes zero
    variance penalty and explicitly reports one independent anchor. Production
    accumulation must assemble the entire logical batch before this calculation.
    """
    if (prediction.shape != target.shape or prediction.ndim != 3
            or prediction.shape[1:] != (128, TARGET_DIMENSION) or len(prediction) == 0
            or prediction.device != target.device
            or anchor_groups.shape != (len(prediction),) or anchor_groups.dtype != torch.int64
            or anchor_groups.device != prediction.device or (anchor_groups < 0).any()
            or not all(math.isfinite(x) for x in (variance_weight, variance_floor_ratio, variance_epsilon))
            or variance_weight < 0 or not 0 <= variance_floor_ratio <= 1 or variance_epsilon <= 0):
        raise ValueError("invalid JEPA loss inputs or variance rule")
    for value in (prediction, target):
        if (not value.is_floating_point() or not torch.isfinite(value).all()
                or (torch.linalg.vector_norm(value.float(), dim=-1)-1).abs().max() > 1e-5):
            raise ValueError("JEPA loss requires finite unit feature vectors")
    predicted, truth = prediction.float(), target.detach().float()
    cosine = (1-(predicted*truth).sum(dim=-1)).clamp(0, 2).mean()
    groups = torch.unique(anchor_groups, sorted=True).tolist()
    variance = predicted.sum()*0
    if len(groups) > 1:
        p = torch.stack([predicted[anchor_groups == group].mean(dim=0) for group in groups])
        t = torch.stack([truth[anchor_groups == group].mean(dim=0) for group in groups])
        p_std = (p.var(dim=0, correction=0)*TARGET_DIMENSION+variance_epsilon).sqrt()
        t_std = (t.var(dim=0, correction=0)*TARGET_DIMENSION+variance_epsilon).sqrt()
        variance = F.relu(variance_floor_ratio*t_std-p_std).mean()
    return JepaLoss(cosine+variance_weight*variance, cosine, variance, len(groups))


def save_jepa_checkpoint(path: Path, model: JepaPredictor, *, metadata: dict):
    """Atomic diagnostic weight snapshot; production recovery belongs to Step 4."""
    if path.exists():
        raise FileExistsError(path)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    header = seal_contract({"artifact_type": "eventground_wam.jepa_predictor_weights",
                            "config": asdict(model.config), "metadata": metadata,
                            "state_sha256": tensor_state_digest(state)})
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save({"header": header, "state_dict": state}, temporary)
    temporary.replace(path)
    return header


def load_jepa_checkpoint(path: Path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    header = artifact["header"]
    if (header["digest"] != contract_digest(header)
            or header["artifact_type"] != "eventground_wam.jepa_predictor_weights"
            or tensor_state_digest(artifact["state_dict"]) != header["state_sha256"]):
        raise ValueError("JEPA checkpoint identity or weights changed")
    model = JepaPredictor(JepaPredictorConfig(**header["config"]))
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model, header
