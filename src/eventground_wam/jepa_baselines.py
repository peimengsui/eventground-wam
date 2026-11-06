"""Prospective JEPA measurements, deterministic temporal retrieval and exit gates."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.pixel_baselines import group_weights, motion_mask

VERSION = "week6-jepa-measurement-v1"
FIELDS = ("cosine", "motion_mass", "static_mass", "motion_area", "static_area")
RETRIEVAL_HORIZONS = (8, 16, 32)


def measurement_rule():
    return {
        "version": VERSION,
        "target_contract": "week6-jepa-targets-v2",
        "cosine": "1 minus FP64 dot product of unit FP32 vectors, clipped to [0,2]; mean over 64 spatial tokens per camera",
        "training_mean": {"groups": "all 1341 training numeric groups",
                          "episode": "one variant/group, minimum SHA256(week6-jepa-mean-v1:episode_index)",
                          "frames": "unique floor(linspace(0,L-1,16)) per selected episode",
                          "averaging": "equal selected frames within group, equal groups within task; final per-position unit L2",
                          "label": "sampled train-only task/camera/position mean; not the full training empirical mean"},
        "coverage": "all 1109994 validation pairs, 1528 episodes, 180 groups, 40 tasks, two cameras",
        "motion": "Week 5 exact 128x128 truth-defined change mask; each 8x8 token gets the fraction of changed pixels in its aligned 16x16 mask tile; aggregate error masses/areas",
        "aggregation": "equal anchors/episode/camera/horizon, equal variants/group, equal groups/task, equal tasks/cameras/horizons",
        "uncertainty": {"replicates": 10000, "seed": 20260925,
                        "unit": "numeric groups within each fixed task; paired draws",
                        "interval": [0.025, 0.975], "scope": "conditional on these tasks and frozen development split"},
        "retrieval": {"episode": "one validation variant per numeric group, minimum SHA256(week6-jepa-retrieval-v1:episode_index)",
                      "horizons": list(RETRIEVAL_HORIZONS), "anchor_fractions": [0.25, 0.5, 0.75],
                      "anchor": "floor(fraction*(L-h-1))", "candidate_count": 8,
                      "candidate_proposals": "true target plus unique floor(linspace(0,L-1,33)); same episode/camera only",
                      "candidate_order": "SHA256(week6-jepa-retrieval-v1:episode:camera:h:anchor:frame)",
                      "minimum_frame_separation": 4, "minimum_pairwise_cosine_distance": 0.01,
                      "duplicates": "reject equal RGB SHA256 or cosine distance <=0.01 against any retained candidate; never relax separation",
                      "incomplete_pool": "mark ineligible with reason/count; no smaller pool or replacement query",
                      "pool_freeze": "retain exact candidates and target slot before predictor training; content filtering uses frozen teacher only",
                      "score": "mean same-position cosine similarity over all 64 tokens; joint cameras are not mixed",
                      "ties": "within 1e-8 of true score: expected Recall@1/@3 and reciprocal rank under uniform tie breaking",
                      "chance": {"recall1": 0.125, "recall3": 0.375,
                                 "mrr": sum(1/k for k in range(1, 9))/8},
                      "aggregation": "equal eligible queries within group, equal supported groups/task, equal tasks; also report camera/horizon coverage and scores"},
        "collapse": {"panel": "all 9 fixed query targets per representative validation group, including retrieval-ineligible queries",
                     "independent_unit": "mean of query features within each numeric group",
                     "fixed_position_std": "task/group-weighted RMS std across group means at corresponding spatial positions",
                     "effective_rank": "entropy effective rank of eigenvalues of weighted covariance of 64-token pooled group means",
                     "similarity": "mean cosine between normalized pooled group means within each task, then equal tasks",
                     "compare": "prediction versus corresponding frozen teacher targets per camera; no spatial-only variance shortcut"},
        "gate": {"overall_cosine_gain_vs_copy_min": 0.05, "motion_cosine_gain_vs_copy_min": 0.10,
                 "gain_paired_95_lower_strictly_above": 0., "overall_gain_vs_mean_point_and_lower_strictly_above": 0.,
                 "per_camera_horizon_ratio_vs_copy_max": 1.02,
                 "motion_groups_min": 120, "motion_tasks_required": 40,
                 "retrieval_groups_min": 120, "retrieval_tasks_required": 40,
                 "retrieval_recall1_delta_vs_copy_min": 0.05,
                 "retrieval_recall1_min": 0.225,
                 "retrieval_delta_paired_95_lower_strictly_above": 0.,
                 "collapse_std_ratio_min": 0.5, "collapse_rank_ratio_min": 0.5,
                 "collapse_rank_min": 5., "collapse_within_task_cosine_excess_max": 0.05,
                 "collapse_within_task_cosine_max": 0.999,
                 "minimum_denominator": 1e-12,
                 "all_required": True,
                 "selection": "fixed final CPT endpoint from Step 4; no best-checkpoint search or automatic extension",
                 "qualitative": "review all 480 fixed Week 5 example identities with feature/prediction diagnostics",
                 "rationale": "prospective practical-effect thresholds for this study, not universal literature cutoffs"},
        "no_predictor": "NOT_ASSESSED_NO_PREDICTOR",
        "test_payloads": "forbidden", "optimizer_updates": 0,
    }


def rank(namespace, value):
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def representative_sources(sources, split, namespace):
    if split not in ("train", "validation") or any(s["split"] != split for s in sources):
        raise ValueError("representative selection requires one explicit allowed split")
    groups = defaultdict(list)
    for source in sources:
        groups[(source["task_key"], source["group_digest"])].append(source)
    return [min(group, key=lambda s: rank(namespace, s["episode_index"]))
            for _, group in sorted(groups.items())]


def cosine_errors(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("cosine operands must be finite, shape-matched feature arrays")
    return np.clip(1-np.einsum("...d,...d->...", left, right, dtype=np.float64), 0, 2)


def token_motion_weights(current_box, target_box):
    if current_box.shape != target_box.shape or current_box.shape[-3:] != (3, 128, 128):
        raise ValueError("motion inputs must be matched 128x128 RGB")
    mask = motion_mask(torch.as_tensor(current_box), torch.as_tensor(target_box))
    return mask.reshape(-1, 8, 16, 8, 16).float().mean(dim=(2, 4)).reshape(-1, 64).numpy()


def error_masses(errors, weights):
    if errors.shape != weights.shape or errors.ndim != 2 or errors.shape[1] != 64:
        raise ValueError("expected pair x 64 error/weight arrays")
    if not np.isfinite(errors).all() or not np.isfinite(weights).all() or np.any((weights < 0)|(weights > 1)):
        raise ValueError("invalid error or truth-mask weights")
    values = [errors.mean(), (errors*weights).mean(), (errors*(1-weights)).mean(),
              weights.mean(dtype=np.float64), (1-weights).mean(dtype=np.float64)]
    return dict(zip(FIELDS, map(float, values), strict=True))


def retrieval_pool(features, image_hashes, *, episode, camera, anchor, horizon):
    """Freeze hard temporal negatives; membership never depends on a predictor."""
    length = len(features)
    target = anchor+horizon
    if len(image_hashes) != length or not 0 <= anchor < target < length:
        raise ValueError("invalid retrieval source or bounds")
    proposals = sorted(set(np.linspace(0, length-1, 33).astype(int).tolist()))
    namespace = f"week6-jepa-retrieval-v1:{episode}:{camera}:{horizon}:{anchor}"
    proposals.sort(key=lambda frame: rank(namespace, frame))
    selected = [target]
    rejected = {"near_time": 0, "duplicate_rgb": 0, "near_feature": 0}
    for frame in proposals:
        if any(abs(frame-other) < 4 for other in selected):
            rejected["near_time"] += 1
            continue
        if any(image_hashes[frame] == image_hashes[other] for other in selected):
            rejected["duplicate_rgb"] += 1
            continue
        if any(float(cosine_errors(features[frame], features[other]).mean()) <= .01 for other in selected):
            rejected["near_feature"] += 1
            continue
        selected.append(frame)
        if len(selected) == 8:
            break
    if len(selected) < 8:
        return {"eligible": False, "reason": "fewer_than_eight_distinct_candidates",
                "retained_count": len(selected), "rejected": rejected}
    selected.sort(key=lambda frame: rank(namespace+":slots", frame))
    return {"eligible": True, "frames": selected, "target_slot": selected.index(target),
            "image_sha256": [image_hashes[i] for i in selected], "rejected": rejected}


def retrieval_metrics(scores, target_slot):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (8,) or not np.isfinite(scores).all() or not 0 <= target_slot < 8:
        raise ValueError("retrieval requires eight finite scores and a valid target slot")
    truth = scores[target_slot]
    better = int(np.sum(scores > truth+1e-8))
    tied = int(np.sum(np.abs(scores-truth) <= 1e-8))
    return {"recall1": float(np.clip(1-better, 0, tied)/tied),
            "recall3": float(np.clip(3-better, 0, tied)/tied),
            "mrr": float(np.mean(1/np.arange(better+1, better+tied+1))),
            "strictly_better": better, "tied": tied}


def retrieval_scores(query, candidates):
    if candidates.shape != (8, *query.shape) or query.shape != (64, 2048):
        raise ValueError("retrieval must use eight aligned 64x2048 candidate fields")
    return np.einsum("td,ntd->n", query, candidates, dtype=np.float64)/64


def grouped_measurements(rows, method):
    groups = sorted({(r["task_key"], r["group_digest"]) for r in rows})
    mapping = {g: i for i, g in enumerate(groups)}
    buckets = defaultdict(list)
    seen = set()
    episodes = defaultdict(set)
    for row in rows:
        key = row["episode_index"], row["camera"], row["horizon"]
        if key in seen or row["camera"] not in CAMERAS or row["horizon"] not in HORIZONS or row["pairs"] <= 0:
            raise ValueError("duplicate or invalid measurement identity")
        seen.add(key)
        g = row["task_key"], row["group_digest"]
        episodes[g].add(row["episode_index"])
        values = [row[method][f] for f in FIELDS]
        if not np.isfinite(values).all():
            raise ValueError("nonfinite measurements")
        buckets[(g, row["camera"], row["horizon"])].append((row["episode_index"], values))
    array = np.zeros((len(groups), 2, 5, len(FIELDS)), np.float64)
    for g in groups:
        for c, camera in enumerate(CAMERAS):
            for j, h in enumerate(HORIZONS):
                cell = buckets[(g, camera, h)]
                if {ep for ep, _ in cell} != episodes[g]:
                    raise ValueError("incomplete episode/camera/horizon coverage")
                array[mapping[g], c, j] = np.mean([v for _, v in cell], axis=0)
    return groups, array


def scalar_metrics(masses):
    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 1e-12)
    return {"cosine": masses[..., 0], "motion_cosine": ratio(masses[..., 1], masses[..., 3]),
            "static_cosine": ratio(masses[..., 2], masses[..., 4]), "motion_area": masses[..., 3]}


def interval(values):
    return np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None


def summary(array, point, bootstrap):
    reduced = array.mean(axis=(1, 2))
    estimates, draws = scalar_metrics(point@reduced), scalar_metrics(bootstrap@reduced)
    return {key: {"value": float(value) if np.isfinite(value) else None,
                  "group_bootstrap_95": interval(draws[key])} for key, value in estimates.items()}


def retrieval_group_values(queries, method):
    buckets = defaultdict(list)
    seen = set()
    for row in queries:
        key = row["episode_index"], row["camera"], row["horizon"], row["anchor"]
        if key in seen:
            raise ValueError("duplicate retrieval query")
        seen.add(key)
        if row["pool"]["eligible"]:
            buckets[(row["task_key"], row["group_digest"])].append(
                [row[method][f] for f in ("recall1", "recall3", "mrr")])
    groups = sorted(buckets)
    return groups, np.asarray([np.mean(buckets[g], axis=0) for g in groups])


def retrieval_summary(queries, method):
    groups, array = retrieval_group_values(queries, method)
    if not groups:
        return {"groups": 0, "tasks": 0, "eligible_queries": 0, "metrics": None}
    point, bootstrap = group_weights(groups)
    return {"groups": len(groups), "tasks": len({g[0] for g in groups}),
            "eligible_queries": sum(q["pool"]["eligible"] for q in queries),
            "total_queries": len(queries),
            "metrics": {field: {"value": float(point@array[:, i]),
                                 "group_bootstrap_95": interval(bootstrap@array[:, i])}
                        for i, field in enumerate(("recall1", "recall3", "mrr"))}}


def collapse_diagnostics(group_means, groups):
    """Across numeric groups, with task-balanced weights; both cameras separately."""
    values = np.asarray(group_means)
    if values.shape != (len(groups), 128, 2048) or not np.isfinite(values).all():
        raise ValueError("collapse panel must contain each group's complete 128x2048 mean")
    point, _ = group_weights(groups, replicates=1)
    result = {}
    for c, camera in enumerate(CAMERAS):
        x = values[:, c*64:(c+1)*64].astype(np.float64)
        center = np.einsum("g,gtd->td", point, x)
        variance = float(np.einsum("g,gtd->", point, (x-center)**2)/(64*2048))
        pooled = x.mean(axis=1)
        centered = (pooled-point@pooled)*np.sqrt(point[:, None])
        energies = np.maximum(np.linalg.eigvalsh(centered@centered.T), 0)
        if energies.sum() <= 1e-24:
            effective_rank = 0.
        else:
            p = energies/energies.sum()
            p = p[p > 1e-15]
            effective_rank = float(np.exp(-np.sum(p*np.log(p))))
        norms = np.linalg.norm(pooled, axis=1)
        similarities = []
        if np.all(norms > 1e-12):
            unit = pooled/norms[:, None]
            for task in sorted({g[0] for g in groups}):
                ix = [i for i, g in enumerate(groups) if g[0] == task]
                if len(ix) > 1:
                    matrix = unit[ix]@unit[ix].T
                    similarities.append(float(matrix[np.triu_indices(len(ix), 1)].mean()))
        result[camera] = {"fixed_position_std_rms": float(np.sqrt(variance)),
                          "pooled_effective_rank": effective_rank,
                          "within_task_group_cosine": float(np.mean(similarities)) if similarities else None,
                          "nonzero_pooled_vectors": bool(np.all(norms > 1e-12))}
    return result


def baseline_report(rows, queries, collapse):
    groups, copy = grouped_measurements(rows, "copy")
    other, mean = grouped_measurements(rows, "mean")
    if groups != other or not np.array_equal(copy[..., 3:], mean[..., 3:]):
        raise ValueError("baseline masks or groups disagree")
    point, bootstrap = group_weights(groups)
    arrays = {"copy": copy, "mean": mean}
    supported = copy[:, :, 2:, 3].mean(axis=(1, 2)) > 0
    return {"overall": {m: summary(a, point, bootstrap) for m, a in arrays.items()},
            "medium_long": {m: summary(a[:, :, 2:], point, bootstrap) for m, a in arrays.items()},
            "by_camera_horizon": [{"camera": camera, "horizon": h,
                                   **{m: summary(a[:, c:c+1, j:j+1], point, bootstrap) for m, a in arrays.items()}}
                                  for c, camera in enumerate(CAMERAS) for j, h in enumerate(HORIZONS)],
            "micro": {m: {k: float(v) if np.isfinite(v) else None for k, v in scalar_metrics(
                np.average([[r[m][f] for f in FIELDS] for r in rows], axis=0,
                           weights=[r["pairs"] for r in rows])).items()} for m in arrays},
            "retrieval": {m: retrieval_summary(queries, m) for m in arrays},
            "retrieval_by_camera_horizon": [{"camera": c, "horizon": h,
                **{m: retrieval_summary([q for q in queries if q["camera"] == c and q["horizon"] == h], m)
                   for m in arrays}} for c in CAMERAS for h in RETRIEVAL_HORIZONS],
            "collapse": collapse, "numeric_groups": len(groups), "tasks": len({g[0] for g in groups}),
            "motion_groups": int(supported.sum()),
            "motion_tasks": len({groups[i][0] for i in np.flatnonzero(supported)}),
            "prediction_gate": "NOT_ASSESSED_NO_PREDICTOR"}


def assess_candidate(rows, queries, candidate_rows, candidate_queries, target_collapse, candidate_collapse,
                     *, reviewed_examples=0):
    """Require exact evaluation pairing; candidate losses cannot alter masks/pools."""
    def key(r):
        return r["episode_index"], r["camera"], r["horizon"]
    a, b = sorted(rows, key=key), sorted(candidate_rows, key=key)
    identities = ("episode_index", "camera", "horizon", "source_digest", "group_digest", "task_key", "pairs")
    if len(a) != len(b) or any(any(x[f] != y[f] for f in identities) for x, y in zip(a, b)):
        raise ValueError("candidate pair coverage or provenance changed")
    if any(any(abs(x["copy"][f]-y["prediction"][f]) > 1e-10 for f in FIELDS[3:]) for x, y in zip(a, b)):
        raise ValueError("candidate truth masks changed")
    def qkey(q):
        return (*key(q), q["anchor"])
    qa, qb = sorted(queries, key=qkey), sorted(candidate_queries, key=qkey)
    if len(qa) != len(qb) or any(any(x[f] != y[f] for f in
                                   ("episode_index", "camera", "horizon", "anchor", "target", "task_key", "group_digest", "source_digest", "pool"))
                                for x, y in zip(qa, qb)):
        raise ValueError("candidate retrieval identities or candidate pools changed")
    groups, copy = grouped_measurements(a, "copy")
    _, mean = grouped_measurements(a, "mean")
    cg, predicted = grouped_measurements(b, "prediction")
    if cg != groups:
        raise ValueError("candidate numeric groups changed")
    point, bootstrap = group_weights(groups)
    def gain(base, new, motion=False):
        idx = 1 if motion else 0
        base, new = base.mean(axis=(1, 2))[:, idx], new.mean(axis=(1, 2))[:, idx]
        denom, draws = point@base, bootstrap@base
        if denom <= 1e-12 or np.any(draws <= 1e-12):
            return None
        return {"value": float(point@(base-new)/denom), "paired_95": interval((bootstrap@(base-new))/draws)}
    effects = {"cosine_vs_copy": gain(copy, predicted), "cosine_vs_mean": gain(mean, predicted),
               "motion_vs_copy": gain(copy[:, :, 2:], predicted[:, :, 2:], True)}
    if any(v is None for v in effects.values()):
        return {"status": "NOT_ASSESSED_DEGENERATE_BASELINE", "effects": effects}
    rg, rv = retrieval_group_values(qa, "copy")
    pg, pv = retrieval_group_values(qb, "prediction")
    if rg != pg:
        raise ValueError("candidate eligible retrieval groups changed")
    support = copy[:, :, 2:, 3].mean(axis=(1, 2)) > 0
    gate = measurement_rule()["gate"]
    if (support.sum() < 120 or len({groups[i][0] for i in np.flatnonzero(support)}) != 40
            or len(rg) < 120 or len({g[0] for g in rg}) != 40):
        return {"status": "NOT_ASSESSED_INSUFFICIENT_COVERAGE", "effects": effects}
    rp, rb = group_weights(rg)
    delta = pv[:, 0]-rv[:, 0]
    effects["retrieval_recall1"] = {"value": float(rp@delta), "paired_95": interval(rb@delta)}
    checks = {"overall_cosine": effects["cosine_vs_copy"]["value"] >= .05 and effects["cosine_vs_copy"]["paired_95"][0] > 0,
              "motion_cosine": effects["motion_vs_copy"]["value"] >= .1 and effects["motion_vs_copy"]["paired_95"][0] > 0,
              "mean_control": effects["cosine_vs_mean"]["value"] > 0 and effects["cosine_vs_mean"]["paired_95"][0] > 0,
              "camera_horizon": bool(np.all(np.einsum("g,gch->ch", point, predicted[..., 0]) <=
                                              1.02*np.einsum("g,gch->ch", point, copy[..., 0]))),
              "retrieval": float(rp@delta) >= .05 and effects["retrieval_recall1"]["paired_95"][0] > 0 and float(rp@pv[:, 0]) >= .225,
              "qualitative_review": reviewed_examples == 480}
    for camera in CAMERAS:
        target, candidate = target_collapse[camera], candidate_collapse[camera]
        numbers = (candidate["fixed_position_std_rms"], candidate["pooled_effective_rank"], candidate["within_task_group_cosine"])
        checks[f"collapse/{camera}"] = bool(
            all(n is not None and np.isfinite(n) for n in numbers)
            and candidate["nonzero_pooled_vectors"]
            and candidate["fixed_position_std_rms"] >= gate["collapse_std_ratio_min"]*target["fixed_position_std_rms"]
            and candidate["pooled_effective_rank"] >= max(5, .5*target["pooled_effective_rank"])
            and candidate["within_task_group_cosine"] <= min(.999, target["within_task_group_cosine"]+.05))
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "effects": effects}
