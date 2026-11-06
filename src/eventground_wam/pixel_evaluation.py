"""Full-coverage accumulation for the frozen learned-pixel validation gate."""
from __future__ import annotations

import numpy as np
import torch

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.pixel_baselines import (
    FIELDS,
    PixelMetrics,
    group_weights,
    grouped_arrays,
    summarize_arrays,
)


class EpisodeAccumulator:
    def __init__(self, source, device):
        if source["split"] != "validation":
            raise ValueError("candidate evaluation requires validation episodes")
        self.source = source
        self.seen = set()
        self.sums = np.zeros((2, 5, len(FIELDS)), np.float64)
        self.counts = np.zeros(5, np.int64)
        self.scorer = PixelMetrics(device)

    def add(self, identities, current, target, prediction):
        if (current.shape != target.shape or prediction.shape != target.shape
                or prediction.ndim != 5 or prediction.shape[:3] != (len(identities), 2, 3)
                or not torch.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1):
            raise ValueError("invalid candidate image batch")
        keys = []
        for item in identities:
            if any(item[k] != self.source[s] for k, s in (("episode_index", "episode_index"),
                   ("source_digest", "digest"), ("group_digest", "group_digest"), ("task_key", "task_key"))):
                raise ValueError("candidate source identity changed")
            t, h = item["anchor"], item["horizon"]
            if (item["split"] != "validation" or h not in HORIZONS or not 0 <= t < self.source["episode_length"] - h
                    or item["target"] != t + h or (t, h) in self.seen or (t, h) in keys):
                raise ValueError("invalid, duplicate or out-of-split validation pair")
            keys.append((t, h))
        values = self.scorer.paired(current.flatten(0, 1), target.flatten(0, 1), prediction.flatten(0, 1))
        values = torch.stack([values[k] for k in FIELDS], dim=-1).double().cpu().numpy().reshape(len(keys), 2, -1)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite candidate metrics")
        for i, (_, horizon) in enumerate(keys):
            j = HORIZONS.index(horizon)
            self.sums[:, j] += values[i]
            self.counts[j] += 1
        self.seen.update(keys)

    def finish(self):
        expected = np.array([self.source["episode_length"] - h for h in HORIZONS])
        if not np.array_equal(expected, self.counts):
            raise ValueError("incomplete episode/horizon coverage")
        return [{"episode_index": self.source["episode_index"], "source_digest": self.source["digest"],
                 "group_digest": self.source["group_digest"], "task_key": self.source["task_key"],
                 "camera": camera, "horizon": h, "pairs": int(self.counts[j]),
                 "prediction": dict(zip(FIELDS, (self.sums[c, j] / self.counts[j]).tolist(), strict=True))}
                for c, camera in enumerate(CAMERAS) for j, h in enumerate(HORIZONS)]


def prediction_summary(rows):
    groups, array = grouped_arrays(rows, "prediction")
    point, bootstrap = group_weights(groups)
    cells = [{"camera": camera, "horizon": h,
              "prediction": summarize_arrays(array[:, c:c+1, j:j+1], point, bootstrap)}
             for c, camera in enumerate(CAMERAS) for j, h in enumerate(HORIZONS)]
    weights = np.array([r["pairs"] for r in rows], np.float64)
    values = np.average([[r["prediction"][k] for k in FIELDS] for r in rows], axis=0, weights=weights)
    return {"overall": summarize_arrays(array, point, bootstrap),
            "motion_h8_h16_h32": summarize_arrays(array[:, :, 2:], point, bootstrap),
            "by_camera_horizon": cells,
            "micro": {"l1": float(values[0]), "ssim": float(values[1]),
                      "motion_l1": float(values[2] / values[4]), "static_l1": float(values[3] / values[5])},
            "numeric_groups": len(groups), "tasks": len({g[0] for g in groups})}
