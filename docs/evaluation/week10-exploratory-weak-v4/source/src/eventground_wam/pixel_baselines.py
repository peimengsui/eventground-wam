"""Pixel-CPT measurements, trajectory-group uncertainty and prospective gates."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from eventground_wam.future_pairs import CAMERAS, HORIZONS

VERSION = "week5-pixel-measurement-v2"
FIELDS = ("l1", "ssim", "motion_error_mass", "static_error_mass",
          "motion_area", "static_area")


def measurement_rule() -> dict:
    return {
        "version": VERSION,
        "resolution": [128, 128],
        "preprocessing": "Decode RGB24 at 256x256; exact 2x2 box mean /255; float32 RGB NCHW",
        "augmentation": "none",
        "horizons": list(HORIZONS), "cameras": list(CAMERAS),
        "baselines": {
            "copy": "Current image at t for both cameras, without using t+h",
            "background": "Per-task/per-camera constant training mean image: equal frames "
                          "within episode, equal variants within numeric group, equal groups "
                          "within task; no validation or test fitting",
        },
        "l1": "Mean absolute RGB error on [0,1], all pixels/channels",
        "ssim": {"window": 11, "sigma": 1.5, "K1": 0.01, "K2": 0.03,
                 "data_range": 1.0, "covariance": "population", "border": "valid",
                 "channel_aggregation": "equal RGB channel mean", "precision": "float64 SSIM moments and reductions; float32 normalized images, L1 and masks",
                 "reference_absolute_tolerance": 0.00005},
        "motion": {
            "threshold": 10 / 255,
            "mask": "Mean absolute target-current RGB change strictly above threshold",
            "exact_threshold": "Recover integer 2x2 RGB box sums by round(image*1020); sum_RGB(abs(target_sum-current_sum)) > 120; eligible pixel count*100 >= H*W",
            "minimum_area": 0.01,
            "small_masks": "Set mask to zero when changed area is below minimum_area",
            "error": "Hierarchical mean(mask * RGB absolute error) / hierarchical mean(mask)",
            "static_error": "Same ratio over the complement of the eligible motion mask",
            "interpretation": "Image-change proxy, not object segmentation or physical motion labels",
            "gate_horizons": [8, 16, 32],
        },
        "aggregation": "Equal anchors within episode/horizon/camera; equal variants within "
                       "numeric group; equal groups within task; equal 40 tasks, two cameras "
                       "and five horizons. Motion ratios use aggregated error/area masses.",
        "micro": "Also report ordinary pair-weighted metrics; these do not determine acceptance",
        "uncertainty": {"unit": "numeric trajectory group", "strata": "fixed canonical task",
                        "method": "paired percentile bootstrap, resample groups within each task",
                        "replicates": 10000, "seed": 20260925, "interval": [0.025, 0.975],
                        "scope": "Conditional on these 40 tasks and the frozen validation split"},
        "exit_gate": {
            "all_required": True,
            "l1_relative_gain_vs_copy_minimum": 0.05,
            "l1_gain_paired_95_lower_strictly_above": 0.0,
            "motion_l1_relative_gain_vs_copy_minimum": 0.10,
            "motion_gain_paired_95_lower_strictly_above": 0.0,
            "ssim_delta_vs_copy_minimum": 0.0,
            "ssim_delta_paired_95_lower_minimum": -0.005,
            "l1_gain_vs_background_point_and_95_lower_strictly_above": 0.0,
            "maximum_per_camera_horizon_l1_ratio_vs_copy": 1.02,
            "minimum_motion_groups": 120,
            "required_motion_task_count": 40,
            "minimum_baseline_denominator": 1e-12,
            "threshold_rationale": "Prospective practical-effect thresholds for this study, "
                                   "not literature-derived universal cutoffs",
            "selection": "Apply to the predeclared final CPT checkpoint; schedule/compute "
                         "budget must be frozen in step 4. Validation is development evidence.",
        },
        "qualitative": {
            "namespace": "week5-pixel-qualitative-v1", "task_count": 40,
            "group": "Lowest SHA256(namespace + ':' + group_digest) per validation task",
            "episode": "Lowest SHA256(namespace + ':' + episode_index) within selected group",
            "horizons": [8, 32], "anchor_fractions": [0.25, 0.5, 0.75],
            "anchor": "floor(fraction * (episode_length - horizon - 1))",
            "cameras": list(CAMERAS), "selection_uses_errors": False,
        },
        "test_payloads": "forbidden", "predictor_training": "not part of this step",
    }


def box_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected source RGB24 256x256")
    image = rgb.astype(np.uint16)
    sums = image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]
    return np.ascontiguousarray(np.moveaxis(sums.astype(np.float32) / 1020.0, 2, 0))


class PixelMetrics:
    """Reusable image moments reduce work across overlapping future pairs."""

    def __init__(self, device="cpu", dtype=torch.float64):
        axis = torch.arange(-5, 6, device=device, dtype=dtype)
        kernel = torch.exp(-axis.square() / (2 * 1.5 ** 2))
        kernel /= kernel.sum()
        self.horizontal = kernel.view(1, 1, 1, 11).repeat(3, 1, 1, 1)
        self.vertical = kernel.view(1, 1, 11, 1).repeat(3, 1, 1, 1)

    def blur(self, images):
        images = images.to(self.horizontal.dtype)
        return F.conv2d(F.conv2d(images, self.horizontal, groups=3), self.vertical, groups=3)

    def moments(self, images):
        images = images.to(self.horizontal.dtype)
        mean = self.blur(images)
        return mean, self.blur(images.square()) - mean.square()

    def ssim(self, left, right, left_moments=None, right_moments=None):
        left, right = left.to(self.horizontal.dtype), right.to(self.horizontal.dtype)
        mu_x, var_x = self.moments(left) if left_moments is None else left_moments
        mu_y, var_y = self.moments(right) if right_moments is None else right_moments
        cov = self.blur(left * right) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)
                 / ((mu_x.square() + mu_y.square() + 0.01 ** 2)
                    * (var_x + var_y + 0.03 ** 2)))
        return score.mean(dim=(1, 2, 3))

    def paired(self, current, target, prediction):
        """Per-pair vectors with the same truth-defined mask for every method."""
        if current.shape != target.shape or prediction.shape != target.shape:
            raise ValueError("prediction, current and target shapes must match")
        mask = motion_mask(current, target)
        error = (prediction - target).abs().mean(dim=1)
        area = mask.float().mean(dim=(1, 2))
        return {
            "l1": error.mean(dim=(1, 2)), "ssim": self.ssim(prediction, target),
            "motion_error_mass": (error * mask).mean(dim=(1, 2)),
            "static_error_mass": (error * ~mask).mean(dim=(1, 2)),
            "motion_area": area, "static_area": 1 - area,
        }


def motion_mask(current, target):
    """Exact strict threshold for the frozen 2x2 box-averaged RGB24 inputs."""
    current_sum = torch.round(current * 1020).to(torch.int32)
    target_sum = torch.round(target * 1020).to(torch.int32)
    mask = (target_sum - current_sum).abs().sum(dim=1) > 120
    eligible = mask.sum(dim=(1, 2)) * 100 >= mask.shape[1] * mask.shape[2]
    return mask & eligible[:, None, None]


@torch.inference_mode()
def measure_video(images, background, batch_size=64) -> dict:
    """Score every valid anchor, caching single-frame moments across horizons."""
    if (images.ndim != 4 or images.shape[1] != 3 or min(images.shape[2:]) < 11
            or background.shape != images.shape[1:] or len(images) <= max(HORIZONS)
            or not torch.isfinite(images).all() or not torch.isfinite(background).all()
            or images.min() < 0 or images.max() > 1
            or background.min() < 0 or background.max() > 1 or batch_size < 1):
        raise ValueError("invalid image sequence/background")
    scorer = PixelMetrics(images.device)
    moment_parts = [scorer.moments(images[i:i+batch_size])
                    for i in range(0, len(images), batch_size)]
    moments = tuple(torch.cat([part[k] for part in moment_parts]) for k in (0, 1))
    del moment_parts
    background = background.unsqueeze(0)
    background_moments = scorer.moments(background)
    background_error = (images - background).abs().mean(dim=1)
    background_ssim = torch.cat([
        scorer.ssim(background.expand_as(images[i:i+batch_size]), images[i:i+batch_size],
                    background_moments, tuple(value[i:i+batch_size] for value in moments))
        for i in range(0, len(images), batch_size)])
    result = {}
    for horizon in HORIZONS:
        count = len(images) - horizon
        sums = {method: torch.zeros(len(FIELDS), dtype=torch.float64, device=images.device)
                for method in ("copy", "background")}
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            current, target = images[start:stop], images[start+horizon:stop+horizon]
            copy_error = (target - current).abs().mean(dim=1)
            mask = motion_mask(current, target)
            area = mask.float().mean(dim=(1, 2))
            copy_ssim = scorer.ssim(
                current, target, tuple(value[start:stop] for value in moments),
                tuple(value[start+horizon:stop+horizon] for value in moments))
            bg_error = background_error[start+horizon:stop+horizon]
            bg_ssim = background_ssim[start+horizon:stop+horizon]
            for method, error, ssim in (("copy", copy_error, copy_ssim),
                                         ("background", bg_error, bg_ssim)):
                values = torch.stack([error.mean(dim=(1, 2)), ssim,
                                      (error * mask).mean(dim=(1, 2)),
                                      (error * ~mask).mean(dim=(1, 2)), area, 1 - area], dim=1)
                sums[method] += values.double().sum(dim=0)
        result[horizon] = {method: dict(zip(FIELDS, (values / count).tolist(), strict=True))
                           for method, values in sums.items()}
    return result


def fit_templates(sources: list[dict], episode_means) -> tuple[np.ndarray, list[str], dict]:
    """Fit constants with explicit group weights; never accept non-training input."""
    ordered = sorted(sources, key=lambda source: source["episode_index"])
    if any(source["split"] != "train" for source in ordered):
        raise ValueError("background fitting is training-only")
    if len({source["episode_index"] for source in ordered}) != len(ordered):
        raise ValueError("duplicate training episode")
    group_sums = {}
    group_counts = defaultdict(int)
    group_tasks = {}
    count = 0
    for source, mean in episode_means:
        if count >= len(ordered) or source != ordered[count]:
            raise ValueError("training mean source order/identity mismatch")
        mean = np.asarray(mean, dtype=np.float64)
        if (mean.shape != (2, 3, 128, 128) or not np.isfinite(mean).all()
                or mean.min() < 0 or mean.max() > 1):
            raise ValueError("invalid episode mean")
        group = source["group_digest"]
        if group_tasks.setdefault(group, source["task_key"]) != source["task_key"]:
            raise ValueError("numeric group crosses tasks")
        if group not in group_sums:
            group_sums[group] = mean.copy()
        else:
            group_sums[group] += mean
        group_counts[group] += 1
        count += 1
    if count != len(ordered) or not count:
        raise ValueError("incomplete training mean coverage")
    tasks = sorted(set(group_tasks.values()))
    templates = []
    task_counts = {}
    for task in tasks:
        groups = sorted(group for group in group_sums if group_tasks[group] == task)
        template = np.zeros((2, 3, 128, 128), dtype=np.float64)
        for group in groups:
            template += group_sums.pop(group) / group_counts[group]
        templates.append((template / len(groups)).astype(np.float32))
        task_counts[task] = {"groups": len(groups),
                             "episodes": sum(group_counts[group] for group in groups)}
    return np.stack(templates), tasks, task_counts


def fixed_examples(sources: list[dict]) -> list[dict]:
    if any(source["split"] != "validation" for source in sources):
        raise ValueError("qualitative panel must use validation only")
    namespace = measurement_rule()["qualitative"]["namespace"]

    def rank(value):
        return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()

    by_task = defaultdict(list)
    for source in sources:
        by_task[source["task_key"]].append(source)
    examples = []
    for task, selected in sorted(by_task.items()):
        group = min({source["group_digest"] for source in selected}, key=rank)
        source = min((source for source in selected if source["group_digest"] == group),
                     key=lambda source: rank(source["episode_index"]))
        for h in (8, 32):
            for fraction in (0.25, 0.5, 0.75):
                anchor = int(fraction * (source["episode_length"] - h - 1))
                if anchor < 0:
                    raise ValueError("qualitative horizon out of bounds")
                for camera in CAMERAS:
                    examples.append({"task_key": task, "group_digest": group,
                                     "episode_index": source["episode_index"],
                                     "source_digest": source["digest"], "horizon": h,
                                     "anchor_fraction": fraction, "anchor": anchor,
                                     "target": anchor + h, "camera": camera})
    return examples


def grouped_arrays(rows: list[dict], method: str) -> tuple[list, np.ndarray]:
    """Produce group x camera x horizon x metric values, retaining paired identities."""
    by_group = defaultdict(list)
    seen = set()
    episode_identities = {}
    for row in rows:
        identity = (row["task_key"], row["group_digest"], row["source_digest"])
        if episode_identities.setdefault(row["episode_index"], identity) != identity:
            raise ValueError("episode identity differs between measurements")
        key = (row["episode_index"], row["camera"], row["horizon"])
        if key in seen or row["camera"] not in CAMERAS or row["horizon"] not in HORIZONS:
            raise ValueError("duplicate or unsupported measurement row")
        seen.add(key)
        if type(row["pairs"]) is not int or row["pairs"] <= 0:
            raise ValueError("invalid pair count")
        values = np.asarray([row[method][field] for field in FIELDS], dtype=np.float64)
        if (not np.isfinite(values).all() or values[0] < 0 or values[0] > 1
                or not -1.0001 <= values[1] <= 1.0001
                or np.any(values[2:] < -1e-8) or np.any(values[2:] > 1.0001)
                or not np.isclose(values[4] + values[5], 1, atol=1e-6)
                or not np.isclose(values[2] + values[3], values[0], atol=1e-6)
                or values[2] > values[4] + 1e-6 or values[3] > values[5] + 1e-6):
            raise ValueError("invalid metric values")
        by_group[(row["task_key"], row["group_digest"])].append(row)
    groups = sorted(by_group)
    array = np.empty((len(groups), 2, 5, len(FIELDS)), dtype=np.float64)
    for i, group in enumerate(groups):
        selected = by_group[group]
        episodes = {row["episode_index"] for row in selected}
        for c, camera in enumerate(CAMERAS):
            for j, horizon in enumerate(HORIZONS):
                cell = [row for row in selected if row["camera"] == camera
                        and row["horizon"] == horizon]
                if {row["episode_index"] for row in cell} != episodes:
                    raise ValueError("missing episode/camera/horizon measurement")
                array[i, c, j] = np.mean(
                    [[row[method][field] for field in FIELDS] for row in cell], axis=0)
    if not groups:
        raise ValueError("empty measurements")
    if len({group[1] for group in groups}) != len(groups):
        raise ValueError("numeric group crosses tasks")
    return groups, array


def group_weights(groups, replicates=10000, seed=20260925):
    by_task = defaultdict(list)
    for i, (task, _) in enumerate(groups):
        by_task[task].append(i)
    point = np.zeros(len(groups))
    bootstrap = np.zeros((replicates, len(groups)))
    rng = np.random.default_rng(seed)
    for task in sorted(by_task):
        indices = by_task[task]
        n = len(indices)
        point[indices] = 1 / (len(by_task) * n)
        samples = rng.integers(n, size=(replicates, n))
        for j, index in enumerate(indices):
            bootstrap[:, index] = (samples == j).sum(axis=1) / (len(by_task) * n)
    return point, bootstrap


def _metric_values(masses):
    return {"l1": masses[..., 0], "ssim": masses[..., 1],
            "motion_l1": np.divide(masses[..., 2], masses[..., 4],
                                   out=np.full_like(masses[..., 2], np.nan),
                                   where=masses[..., 4] > 1e-12),
            "static_l1": np.divide(masses[..., 3], masses[..., 5],
                                   out=np.full_like(masses[..., 3], np.nan),
                                   where=masses[..., 5] > 1e-12),
            "motion_area": masses[..., 4]}


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _interval(values):
    if not np.isfinite(values).all():
        return None
    return np.quantile(values, [0.025, 0.975]).tolist()


def summarize_arrays(array, point, bootstrap):
    # Average the requested cameras/horizons before task/group weighting.
    values = array.mean(axis=(1, 2))
    estimate = _metric_values(point @ values)
    draws = _metric_values(bootstrap @ values)
    return {key: {"value": _finite(value), "group_bootstrap_95": _interval(draws[key])}
            for key, value in estimate.items()}


def paired_effect(reference, candidate, point, bootstrap, *, motion=False, ssim=False):
    base = reference.mean(axis=(1, 2))
    new = candidate.mean(axis=(1, 2))
    if ssim:
        delta = new[:, 1] - base[:, 1]
        return {"value": float(point @ delta), "paired_95": _interval(bootstrap @ delta)}
    field = 2 if motion else 0
    denominator = point @ base[:, field]
    draw_denominator = bootstrap @ base[:, field]
    if denominator <= 1e-12 or np.any(draw_denominator <= 1e-12):
        return {"value": None, "paired_95": None}
    return {"value": float((point @ (base[:, field] - new[:, field])) / denominator),
            "paired_95": _interval((bootstrap @ (base[:, field] - new[:, field]))
                                   / draw_denominator)}


def baseline_report(rows: list[dict]) -> dict:
    groups, copy = grouped_arrays(rows, "copy")
    other_groups, background = grouped_arrays(rows, "background")
    if groups != other_groups or not np.array_equal(copy[..., 4:], background[..., 4:]):
        raise ValueError("baseline pairing or truth masks differ")
    point, bootstrap = group_weights(groups)
    overall, cells, micro = {}, [], {}
    for name, array in (("copy", copy), ("background", background)):
        overall[name] = summarize_arrays(array, point, bootstrap)
        weights = np.asarray([row["pairs"] for row in rows], dtype=np.float64)
        values = np.asarray([[row[name][field] for field in FIELDS] for row in rows])
        micro[name] = {key: _finite(value) for key, value in
                       _metric_values(np.average(values, axis=0, weights=weights)).items()}
    for c, camera in enumerate(CAMERAS):
        for j, h in enumerate(HORIZONS):
            cells.append({"camera": camera, "horizon": h,
                          **{name: summarize_arrays(array[:, c:c+1, j:j+1], point, bootstrap)
                             for name, array in (("copy", copy), ("background", background))}})
    motion = copy[:, :, 2:]
    supported = motion[..., 4].mean(axis=(1, 2)) > 0
    return {"overall": overall, "by_camera_horizon": cells, "micro": micro,
            "numeric_groups": len(groups), "tasks": len({group[0] for group in groups}),
            "motion_groups": int(supported.sum()),
            "motion_tasks": len({groups[i][0] for i in np.flatnonzero(supported)}),
            "background_relative_l1_gain_vs_copy": paired_effect(copy, background, point, bootstrap),
            "prediction_gate": "NOT_ASSESSED_NO_PREDICTOR"}


def assess_candidate(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    """Apply the locked gate to a complete, identically paired predictor evaluation."""
    def key(row):
        return row["episode_index"], row["camera"], row["horizon"]

    baseline = sorted(baseline_rows, key=key)
    candidate = sorted(candidate_rows, key=key)
    identity = ("episode_index", "source_digest", "camera", "horizon", "pairs",
                "group_digest", "task_key")
    if len(baseline) != len(candidate) or any(
        any(a[field] != b[field] for field in identity) for a, b in zip(baseline, candidate)
    ):
        raise ValueError("candidate evaluation identities or coverage differ")
    for a, b in zip(baseline, candidate):
        if any(abs(a["copy"][field] - b["prediction"][field]) > 1e-7
               for field in ("motion_area", "static_area")):
            raise ValueError("candidate changed truth-defined motion mask")
    groups, copy = grouped_arrays(baseline, "copy")
    _, background = grouped_arrays(baseline, "background")
    candidate_groups, predicted = grouped_arrays(candidate, "prediction")
    if groups != candidate_groups:
        raise ValueError("candidate groups differ")
    point, bootstrap = group_weights(groups)
    effects = {"l1_vs_copy": paired_effect(copy, predicted, point, bootstrap),
               "motion_l1_vs_copy": paired_effect(copy[:, :, 2:], predicted[:, :, 2:],
                                                   point, bootstrap, motion=True),
               "ssim_vs_copy": paired_effect(copy, predicted, point, bootstrap, ssim=True),
               "l1_vs_background": paired_effect(background, predicted, point, bootstrap)}
    criteria = measurement_rule()["exit_gate"]
    if any(value["value"] is None or value["paired_95"] is None for value in effects.values()):
        return {"status": "NOT_ASSESSED_DEGENERATE_BASELINE", "effects": effects}
    supported = copy[:, :, 2:, 4].mean(axis=(1, 2)) > 0
    tasks = {groups[i][0] for i in np.flatnonzero(supported)}
    if (supported.sum() < criteria["minimum_motion_groups"]
            or len(tasks) != criteria["required_motion_task_count"]):
        return {"status": "NOT_ASSESSED_MOTION_COVERAGE", "effects": effects}
    cell_copy = np.einsum("g,gch->ch", point, copy[..., 0])
    cell_prediction = np.einsum("g,gch->ch", point, predicted[..., 0])
    checks = {
        "overall_l1": effects["l1_vs_copy"]["value"] >= criteria["l1_relative_gain_vs_copy_minimum"]
                      and effects["l1_vs_copy"]["paired_95"][0] > 0,
        "motion_l1": effects["motion_l1_vs_copy"]["value"] >= criteria["motion_l1_relative_gain_vs_copy_minimum"]
                     and effects["motion_l1_vs_copy"]["paired_95"][0] > 0,
        "ssim": effects["ssim_vs_copy"]["value"] >= criteria["ssim_delta_vs_copy_minimum"]
                and effects["ssim_vs_copy"]["paired_95"][0] >= criteria["ssim_delta_paired_95_lower_minimum"],
        "beats_background": effects["l1_vs_background"]["value"] > 0
                            and effects["l1_vs_background"]["paired_95"][0] > 0,
        "all_camera_horizon_cells": bool(np.all(cell_prediction <= criteria[
            "maximum_per_camera_horizon_l1_ratio_vs_copy"] * cell_copy)),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "effects": effects, "per_camera_horizon_l1": {
                "copy": cell_copy.tolist(), "prediction": cell_prediction.tolist()}}
