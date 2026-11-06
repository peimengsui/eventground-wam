#!/usr/bin/env python3
"""Independent coverage, mean, uncertainty, retrieval and real-image JEPA audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from run_jepa_baselines import decode, encode

from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.jepa_features import FrozenQwenVisualTargets

FIELDS = ("cosine", "motion_mass", "static_mass", "motion_area", "static_area")


def draws(groups, values):
    """Explicit resampled group arrays; no production bootstrap-weight matrix."""
    rng = np.random.default_rng(20260925)
    tasks = sorted({g[0] for g in groups})
    samples = np.zeros((10000, *values.shape[1:]), np.float64)
    point = np.zeros(values.shape[1:], np.float64)
    for task in tasks:
        ix = np.asarray([i for i, g in enumerate(groups) if g[0] == task])
        choice = rng.integers(len(ix), size=(10000, len(ix)))
        samples += values[ix[choice]].mean(axis=1)/len(tasks)
        point += values[ix].mean(axis=0)/len(tasks)
    return point, samples


def scalars(array):
    return {"cosine": array[..., 0], "motion_cosine": array[..., 1]/array[..., 3],
            "static_cosine": array[..., 2]/array[..., 4], "motion_area": array[..., 3]}


def tie_scores(scores, target):
    scores = np.asarray(scores, np.float64)
    above = sum(float(x) > float(scores[target])+1e-8 for x in scores)
    equal = sum(abs(float(x)-float(scores[target])) <= 1e-8 for x in scores)
    ranks = list(range(above+1, above+equal+1))
    return {"recall1": sum(r <= 1 for r in ranks)/equal,
            "recall3": sum(r <= 3 for r in ranks)/equal,
            "mrr": sum(1/r for r in ranks)/equal}


def audit(args):
    root, out = args.source, args.run
    config = read_sealed(root/"configs/evaluation/week6_jepa_baselines_v1.json")
    identity = read_sealed(out/"identity.json")
    report = read_sealed(out/"report.json")
    completion = read_sealed(out/"completion.json")
    templates_record = read_sealed(out/"templates.json")
    rows = read_sealed(out/"episode-metrics.json")["rows"]
    queries = read_sealed(out/"retrieval-panel.json")["queries"]
    for name, expected in config["input_sha256"].items():
        assert sha256_file(root/name) == expected, name
    for name, expected in identity["source_sha256"].items():
        assert sha256_file(root/name) == expected, name
    for name, expected in completion["files_sha256"].items():
        assert sha256_file(out/name) == expected, name
    assert report["identity_digest"] == identity["digest"] == completion["identity_digest"]
    assert report["config_digest"] == identity["config_digest"] == config["digest"]
    assert completion["report_digest"] == report["digest"]
    assert report["prediction_gate"] == "NOT_ASSESSED_NO_PREDICTOR"
    manifest = read_sealed(args.pairs/"manifest.json")
    assert sha256_file(args.pairs/"manifest.json") == config["pair_manifest_sha256"]
    assert sha256_file(args.pairs/"sources.jsonl") == manifest["outputs"]["sources.jsonl"]["sha256"]
    sources = [json.loads(line) for line in (args.pairs/"sources.jsonl").read_text().splitlines()]
    train = [s for s in sources if s["split"] == "train"]
    validation = [s for s in sources if s["split"] == "validation"]
    training_by_group = defaultdict(list)
    for s in train:
        training_by_group[(s["task_key"], s["group_digest"])].append(s)
    rank = lambda ns, value: hashlib.sha256(f"{ns}:{value}".encode()).hexdigest()
    selected = [min(items, key=lambda s: rank("week6-jepa-mean-v1", s["episode_index"]))
                for _, items in sorted(training_by_group.items())]
    assert len(selected) == 1341
    assert len(templates_record["training_records"]) == 1341
    means, counts = {}, defaultdict(int)
    for source, receipt in zip(selected, templates_record["training_records"], strict=True):
        assert receipt["episode_index"] == source["episode_index"] and receipt["source_digest"] == source["digest"]
        assert receipt["split"] == "train" and receipt["group_digest"] == source["group_digest"]
        assert receipt["task_key"] == source["task_key"] and receipt["identity_digest"] == identity["digest"]
        assert receipt["frames"] == np.unique(np.linspace(0, source["episode_length"]-1, 16).astype(int)).tolist()
        p = out/"train-means"/f"{source['episode_index']:06d}.npy"
        assert sha256_file(p) == receipt["array_sha256"]
        assert read_sealed(p.with_suffix(".json")) == receipt
        x = np.load(p, allow_pickle=False)
        task = source["task_key"]
        counts[task] += 1
        if task not in means:
            means[task] = x.copy()
        else:
            means[task] += (x-means[task])/counts[task]
    expected_templates = np.stack([means[t] for t in templates_record["task_order"]])
    expected_templates /= np.linalg.norm(expected_templates, axis=-1, keepdims=True)
    templates = np.load(out/"templates.npy", allow_pickle=False)
    mean_error = float(np.max(np.abs(expected_templates-templates)))
    assert mean_error < 1e-7
    assert sum(len(r["frames"]) for r in templates_record["training_records"]) == 21456
    print(json.dumps({"phase": "training-mean-audited", "maximum_error": mean_error}), flush=True)
    expected = {(s["episode_index"], c, h): s for s in validation for c in CAMERAS for h in HORIZONS}
    per_episode = defaultdict(list)
    buckets = defaultdict(list)
    for row in rows:
        source = expected.pop((row["episode_index"], row["camera"], row["horizon"]))
        assert row["source_digest"] == source["digest"] and row["group_digest"] == source["group_digest"]
        assert row["task_key"] == source["task_key"] and row["pairs"] == source["episode_length"]-row["horizon"]
        assert row["copy"]["motion_area"] == row["mean"]["motion_area"]
        per_episode[source["episode_index"]].append(row)
        buckets[(row["task_key"], row["group_digest"], row["camera"], row["horizon"])].append(row)
    assert not expected and len(rows) == 15280 and sum(r["pairs"] for r in rows)//2 == 1109994
    for source in validation:
        saved = read_sealed(out/"episodes"/f"{source['episode_index']:06d}.json")
        assert saved["rows"] == per_episode[source["episode_index"]]
        assert saved["identity_digest"] == identity["digest"] and saved["source_digest"] == source["digest"]
        for name, expected_sha in saved["arrays"].items():
            assert sha256_file(out/name) == expected_sha
    groups = sorted({(s["task_key"], s["group_digest"]) for s in validation})
    array = np.zeros((180, 2, 2, 5, 5), np.float64)
    for g, (task, group) in enumerate(groups):
        for c, camera in enumerate(CAMERAS):
            for j, h in enumerate(HORIZONS):
                cell = buckets[(task, group, camera, h)]
                for m, method in enumerate(("copy", "mean")):
                    array[g, m, c, j] = np.mean([[r[method][f] for f in FIELDS] for r in cell], axis=0)
    point, bootstrap = draws(groups, array)
    errors = []
    def check_summary(saved, estimate, resampled):
        estimates = scalars(estimate)
        samples = scalars(resampled)
        for field in estimates:
            errors.append(abs(float(estimates[field])-saved[field]["value"]))
            errors.extend(np.abs(np.quantile(samples[field], [.025, .975])-saved[field]["group_bootstrap_95"]).tolist())
    for m, method in enumerate(("copy", "mean")):
        check_summary(report["overall"][method], point[m].mean(axis=(0, 1)), bootstrap[:, m].mean(axis=(1, 2)))
        check_summary(report["medium_long"][method], point[m, :, 2:].mean(axis=(0, 1)), bootstrap[:, m, :, 2:].mean(axis=(1, 2)))
        for c, camera in enumerate(CAMERAS):
            for j, h in enumerate(HORIZONS):
                saved = next(x for x in report["by_camera_horizon"] if x["camera"] == camera and x["horizon"] == h)
                check_summary(saved[method], point[m, c, j], bootstrap[:, m, c, j])
    assert max(errors) < 1e-10
    print(json.dumps({"phase": "full-metrics-bootstrap-audited", "maximum_error": max(errors)}), flush=True)
    # Independent retrieval scoring from every retained score vector.
    retrieval_error = 0.
    for query in queries:
        if not query["pool"]["eligible"]:
            continue
        pool = query["pool"]
        assert len(pool["frames"]) == len(set(pool["frames"])) == 8
        assert pool["frames"][pool["target_slot"]] == query["target"]
        assert len(set(pool["image_sha256"])) == 8
        assert all(abs(a-b) >= 4 for i, a in enumerate(pool["frames"]) for b in pool["frames"][i+1:])
        for method in ("copy", "mean"):
            recomputed = tie_scores(query[method]["scores"], pool["target_slot"])
            retrieval_error = max(retrieval_error, max(abs(v-query[method][k]) for k, v in recomputed.items()))
    def check_retrieval(selected_queries, saved):
        grouped = defaultdict(list)
        for q in selected_queries:
            if q["pool"]["eligible"]:
                grouped[(q["task_key"], q["group_digest"])].append(
                    [[q[m][f] for f in ("recall1", "recall3", "mrr")] for m in ("copy", "mean")])
        ordered = sorted(grouped)
        p, b = draws(ordered, np.asarray([np.mean(grouped[g], axis=0) for g in ordered]))
        for m, method in enumerate(("copy", "mean")):
            assert saved[method]["groups"] == len(ordered)
            assert saved[method]["tasks"] == len({g[0] for g in ordered})
            assert saved[method]["eligible_queries"] == sum(q["pool"]["eligible"] for q in selected_queries)
            for j, field in enumerate(("recall1", "recall3", "mrr")):
                actual = saved[method]["metrics"][field]
                errors.append(abs(float(p[m, j])-actual["value"]))
                errors.extend(np.abs(np.quantile(b[:, m, j], [.025, .975])-actual["group_bootstrap_95"]).tolist())
    check_retrieval(queries, report["retrieval"])
    for saved in report["retrieval_by_camera_horizon"]:
        check_retrieval([q for q in queries if q["camera"] == saved["camera"] and q["horizon"] == saved["horizon"]], saved)
    assert retrieval_error < 1e-12 and max(errors) < 1e-10
    # Replay two complete episodes from saved features and exact integer masks.
    replay_error = 0.
    for ep in completion["replay_episode_indices"]:
        features = np.load(out/"replay"/f"{ep:06d}-features.npy", mmap_mode="r", allow_pickle=False)
        boxes = np.load(out/"replay"/f"{ep:06d}-boxes.npy", mmap_mode="r", allow_pickle=False)
        source = next(s for s in validation if s["episode_index"] == ep)
        template = templates[templates_record["task_order"].index(source["task_key"])]
        for row in per_episode[ep]:
            c, h = CAMERAS.index(row["camera"]), row["horizon"]
            sums = {m: np.zeros(5) for m in ("copy", "mean")}
            for t in range(len(features)-h):
                a, b = np.rint(boxes[t, c]*1020).astype(np.int32), np.rint(boxes[t+h, c]*1020).astype(np.int32)
                mask = np.abs(b-a).sum(axis=0) > 120
                if mask.sum()*100 < 128*128:
                    mask[:] = False
                weights = mask.reshape(8, 16, 8, 16).mean(axis=(1, 3)).reshape(64)
                target = features[t+h, c*64:(c+1)*64].astype(np.float64)
                for m, value in (("copy", features[t, c*64:(c+1)*64]), ("mean", template[c*64:(c+1)*64])):
                    error = np.clip(1-(value.astype(np.float64)*target).sum(axis=-1), 0, 2)
                    sums[m] += [error.mean(), (error*weights).mean(), (error*(1-weights)).mean(), weights.mean(), (1-weights).mean()]
            for m, value in sums.items():
                expected_values = np.asarray([row[m][f] for f in FIELDS])
                replay_error = max(replay_error, float(np.max(np.abs(value/(len(features)-h)-expected_values))))
    assert replay_error < 1e-10
    # Re-encode independent real RGB observations for two fixed retrieval episodes.
    val_groups = defaultdict(list)
    for s in validation:
        val_groups[(s["task_key"], s["group_digest"])].append(s)
    panel = [min(v, key=lambda s: rank("week6-jepa-retrieval-v1", s["episode_index"])) for _, v in sorted(val_groups.items())]
    expected_queries = {(s["episode_index"], c, h, int(f*(s["episode_length"]-h-1))): s
                        for s in panel for c in CAMERAS for h in (8, 16, 32) for f in (.25, .5, .75)}
    for q in queries:
        s = expected_queries.pop((q["episode_index"], q["camera"], q["horizon"], q["anchor"]))
        assert q["source_digest"] == s["digest"] and q["group_digest"] == s["group_digest"]
    assert not expected_queries and len(queries) == 3240
    # Independently recompute cross-group collapse statistics using Torch SVD
    # rather than the production NumPy Gram-matrix eigendecomposition.
    torch.set_num_threads(1)
    panel_values = np.stack([np.load(out/"panels"/f"{s['episode_index']:06d}.npy", allow_pickle=False) for s in panel])
    panel_groups = [(s["task_key"], s["group_digest"]) for s in panel]
    tasks = sorted({g[0] for g in panel_groups})
    weights = np.asarray([1/(len(tasks)*sum(other[0] == g[0] for other in panel_groups)) for g in panel_groups])
    collapse_error = 0.
    for m, method in enumerate(("copy", "target", "mean")):
        for c, camera in enumerate(CAMERAS):
            x = panel_values[:, m, c*64:(c+1)*64].astype(np.float64)
            mean = sum(weights[i]*x[i] for i in range(len(x)))
            variance = sum(weights[i]*np.mean((x[i]-mean)**2) for i in range(len(x)))
            pooled = x.mean(axis=1)
            center = sum(weights[i]*pooled[i] for i in range(len(x)))
            weighted = (pooled-center)*np.sqrt(weights[:, None])
            energy = torch.linalg.svdvals(torch.from_numpy(weighted)).numpy()**2
            if energy.sum() <= 1e-24:
                effective_rank = 0.
            else:
                probabilities = energy/energy.sum()
                active = probabilities[probabilities > 1e-15]
                effective_rank = float(np.exp(-np.sum(active*np.log(active))))
            unit = pooled/np.linalg.norm(pooled, axis=1, keepdims=True)
            similarities = []
            for task in tasks:
                ix = [i for i, g in enumerate(panel_groups) if g[0] == task]
                if len(ix) > 1:
                    similarities.append(np.mean([np.dot(unit[i], unit[j]) for pos, i in enumerate(ix) for j in ix[pos+1:]]))
            stats = {"fixed_position_std_rms": float(np.sqrt(variance)), "pooled_effective_rank": effective_rank,
                     "within_task_group_cosine": float(np.mean(similarities))}
            for field, value in stats.items():
                collapse_error = max(collapse_error, abs(value-report["collapse"][method][camera][field]))
    assert collapse_error < 1e-8
    del panel_values
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True)
    visual = full.model.visual
    del full
    teacher = FrozenQwenVisualTargets(visual.to("cuda"), load_pinned_qwen_processor(
        cache_dir=str(args.cache), local_files_only=True).image_processor).eval()
    repository = args.data_root/f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo"
    real_score_error = 0.
    replayed = []
    for source in (panel[0], max(panel, key=lambda s: s["episode_length"])):
        selected_queries = [q for q in queries if q["episode_index"] == source["episode_index"]]
        proposals = set(np.linspace(0, source["episode_length"]-1, 33).astype(int).tolist())
        frames = sorted(proposals|{q["anchor"] for q in selected_queries}|{q["target"] for q in selected_queries})
        rgb = decode(repository, source, "validation", frames)
        encoded = encode(teacher, rgb)
        vectors = dict(zip(frames, encoded, strict=True))
        image_hashes = {frame: [hashlib.sha256(rgb[i, c].tobytes()).hexdigest() for c in range(2)] for i, frame in enumerate(frames)}
        template = templates[templates_record["task_order"].index(source["task_key"])]
        for q in selected_queries:
            c = CAMERAS.index(q["camera"])
            ns = f"week6-jepa-retrieval-v1:{source['episode_index']}:{q['camera']}:{q['horizon']}:{q['anchor']}"
            candidates = [q["target"]]
            for frame in sorted(proposals, key=lambda x: rank(ns, x)):
                if any(abs(frame-other) < 4 or image_hashes[frame][c] == image_hashes[other][c] for other in candidates):
                    continue
                a = vectors[frame][c*64:(c+1)*64].astype(np.float64)
                if any(np.clip(1-(a*vectors[other][c*64:(c+1)*64].astype(np.float64)).sum(axis=-1), 0, 2).mean() <= .01 for other in candidates):
                    continue
                candidates.append(frame)
                if len(candidates) == 8:
                    break
            assert q["pool"]["eligible"] == (len(candidates) == 8)
            if len(candidates) != 8:
                continue
            candidates.sort(key=lambda x: rank(ns+":slots", x))
            assert candidates == q["pool"]["frames"]
            for m, prediction in (("copy", vectors[q["anchor"]][c*64:(c+1)*64]), ("mean", template[c*64:(c+1)*64])):
                scores = [float((prediction.astype(np.float64)*vectors[f][c*64:(c+1)*64].astype(np.float64)).sum()/64) for f in candidates]
                real_score_error = max(real_score_error, float(np.max(np.abs(np.asarray(scores)-q[m]["scores"]))))
        replayed.append(source["episode_index"])
    assert real_score_error < 1e-10
    return seal_contract({"artifact_type": "eventground_wam.jepa_baseline_independent_verification", "passed": True,
                          "report_digest": report["digest"], "protocol_digest": config["digest"],
                          "auditor_sha256": sha256_file(Path(__file__)), "complete_pair_count": 1109994,
                          "training_groups": 1341, "training_frames": 21456,
                          "retrieval_proposed_queries": 3240, "mean_template_max_error": mean_error,
                          "primary_and_bootstrap_max_error": max(errors), "retrieval_rank_max_error": retrieval_error,
                          "collapse_statistics_max_error": collapse_error,
                          "saved_full_episode_replay_max_error": replay_error,
                          "real_image_retrieval_replay_max_error": real_score_error,
                          "real_image_replayed_episode_indices": replayed,
                          "test_payload_reads": 0, "optimizer_updates": 0,
                          "limits": "Complete scalar and retrieval-score audit; full saved-feature replay for two episodes and fresh RGB retrieval replay for two panel episodes, not a full second encoding pass."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, audit(args))
    print("Independent baseline verification passed", flush=True)
