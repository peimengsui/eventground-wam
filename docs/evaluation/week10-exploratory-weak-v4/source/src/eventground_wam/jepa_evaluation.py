"""Full-coverage candidate scoring in the frozen JEPA measurement space."""
from __future__ import annotations

import numpy as np

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.jepa_baselines import (
    FIELDS,
    RETRIEVAL_HORIZONS,
    cosine_errors,
    grouped_measurements,
    retrieval_metrics,
    retrieval_scores,
    retrieval_summary,
    scalar_metrics,
    summary,
    token_motion_weights,
)
from eventground_wam.pixel_baselines import group_weights


class EpisodeAccumulator:
    def __init__(self, source, *, split):
        if split not in ("train", "validation") or source["split"] != split:
            raise ValueError("forbidden or cross-split candidate episode")
        self.source, self.split = source, split
        self.seen = set()
        self.sums = np.zeros((2, 5, len(FIELDS)), np.float64)
        self.counts = np.zeros(5, np.int64)

    def add(self, identities, current, future, prediction, target):
        count = len(identities)
        prediction, target = np.asarray(prediction), np.asarray(target)
        if (prediction.shape != (count, 128, 2048) or target.shape != prediction.shape
                or current.shape != (count, 2, 3, 128, 128) or future.shape != current.shape):
            raise ValueError("invalid candidate feature or image shapes")
        for values in (prediction, target):
            if (not np.isfinite(values).all()
                    or np.max(np.abs(np.linalg.norm(values, axis=-1)-1)) > 1e-5):
                raise ValueError("candidate and target features must be finite unit vectors")
        keys = []
        for item in identities:
            if any(item[k] != self.source[s] for k, s in (("episode_index", "episode_index"),
                   ("source_digest", "digest"), ("group_digest", "group_digest"), ("task_key", "task_key"))):
                raise ValueError("candidate source identity changed")
            t, h = item["anchor"], item["horizon"]
            if (item["split"] != self.split or h not in HORIZONS
                    or not 0 <= t < self.source["episode_length"]-h or item["target"] != t+h
                    or (t, h) in self.seen or (t, h) in keys):
                raise ValueError("invalid, duplicate or out-of-split pair")
            keys.append((t, h))
        errors = cosine_errors(prediction, target).reshape(count, 2, 64)
        weights = token_motion_weights(current.reshape(-1, 3, 128, 128),
                                       future.reshape(-1, 3, 128, 128)).reshape(count, 2, 64)
        values = np.stack((errors.mean(-1), (errors*weights).mean(-1),
                           (errors*(1-weights)).mean(-1), weights.mean(-1, dtype=np.float64),
                           (1-weights).mean(-1, dtype=np.float64)), axis=-1)
        for i, (_, h) in enumerate(keys):
            j = HORIZONS.index(h)
            self.sums[:, j] += values[i]
            self.counts[j] += 1
        self.seen.update(keys)

    def finish(self):
        expected = np.array([self.source["episode_length"]-h for h in HORIZONS])
        if not np.array_equal(self.counts, expected):
            raise ValueError("incomplete episode/horizon coverage")
        return [{"episode_index": self.source["episode_index"], "source_digest": self.source["digest"],
                 "group_digest": self.source["group_digest"], "task_key": self.source["task_key"],
                 "camera": camera, "horizon": h, "pairs": int(self.counts[j]),
                 "prediction": dict(zip(FIELDS, (self.sums[c, j]/self.counts[j]).tolist(), strict=True))}
                for c, camera in enumerate(CAMERAS) for j, h in enumerate(HORIZONS)]


def score_queries(queries, predictions, features):
    """Use the already frozen pool verbatim, including ineligible queries."""
    output = []
    for q in queries:
        key = q["anchor"], q["horizon"]
        if key not in predictions or q["target"] != sum(key):
            raise ValueError("missing or mismatched retrieval prediction")
        c = CAMERAS.index(q["camera"])
        row = dict(q)
        if q["pool"]["eligible"]:
            frames, slot = q["pool"]["frames"], q["pool"]["target_slot"]
            if len(frames) != 8 or frames[slot] != q["target"]:
                raise ValueError("invalid frozen retrieval pool")
            scores = retrieval_scores(predictions[key][c*64:(c+1)*64],
                                      features[frames, c*64:(c+1)*64])
            row["prediction"] = {**retrieval_metrics(scores, slot), "scores": scores.tolist()}
        output.append(row)
    return output


def prediction_summary(rows, queries):
    groups, array = grouped_measurements(rows, "prediction")
    point, bootstrap = group_weights(groups)
    values = np.average([[r["prediction"][f] for f in FIELDS] for r in rows],
                        axis=0, weights=[r["pairs"] for r in rows])
    return {"overall": summary(array, point, bootstrap),
            "medium_long": summary(array[:, :, 2:], point, bootstrap),
            "by_camera_horizon": [{"camera": c, "horizon": h,
                "prediction": summary(array[:, ci:ci+1, hi:hi+1], point, bootstrap)}
                for ci, c in enumerate(CAMERAS) for hi, h in enumerate(HORIZONS)],
            "micro": {k: float(v) if np.isfinite(v) else None for k, v in scalar_metrics(values).items()},
            "retrieval": retrieval_summary(queries, "prediction"),
            "retrieval_by_camera_horizon": [{"camera": c, "horizon": h,
                **retrieval_summary([q for q in queries if q["camera"] == c and q["horizon"] == h], "prediction")}
                for c in CAMERAS for h in RETRIEVAL_HORIZONS],
            "numeric_groups": len(groups), "tasks": len({g[0] for g in groups})}
