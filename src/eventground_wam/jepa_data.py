"""Native future RGB stays outside the unchanged Pixel conditioning interface."""
from dataclasses import dataclass

import numpy as np
import torch

from eventground_wam.pixel_data import (
    FuturePairDataset,
    FutureTrainingSample,
    collate_future_samples,
)
from eventground_wam.pixel_predictor import PixelCondition


@dataclass(frozen=True)
class JepaTrainingSample(FutureTrainingSample):
    target_rgb: np.ndarray


class JepaPairDataset(FuturePairDataset):
    def __getitem__(self, ordinal):
        sample = super().__getitem__(ordinal)
        source = self.sources[sample.identity["episode_index"]]
        rgb = self._episode(source)[2][sample.identity["target"]].copy()
        return JepaTrainingSample(sample.observation, sample.target_images, sample.identity, rgb)


@dataclass(frozen=True)
class JepaTrainingBatch:
    condition: PixelCondition
    targets: torch.Tensor
    anchor_groups: torch.Tensor
    identities: tuple[dict, ...]


def collate_jepa_samples(samples, features, feature_padding, target_features):
    if any(not isinstance(s, JepaTrainingSample) for s in samples):
        raise TypeError("JEPA samples with separate native RGB targets required")
    batch = collate_future_samples(samples, features, feature_padding)
    if (target_features.shape != (len(samples), 128, 2048)
            or not torch.isfinite(target_features).all()
            or (torch.linalg.vector_norm(target_features.float(), dim=-1)-1).abs().max() > 1e-5):
        raise ValueError("teacher features must be finite normalized paired targets")
    keys = [(s.identity["episode_index"], s.identity["anchor"]) for s in samples]
    unique = sorted(set(keys))
    groups = torch.tensor([unique.index(k) for k in keys], dtype=torch.int64,
                          device=target_features.device)
    return JepaTrainingBatch(batch.condition, target_features.detach(), groups, batch.identities)
