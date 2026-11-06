#!/usr/bin/env python3
"""Pilot, then score the fixed JEPA endpoint under the unchanged Week 6 protocol."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import resource
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files
from run_jepa_baselines import atomic_array, encode, episode_retrieval
from train_jepa_cpt import configure

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file
from eventground_wam.jepa_baselines import (
    assess_candidate,
    collapse_diagnostics,
    measurement_rule,
    representative_sources,
)
from eventground_wam.jepa_evaluation import EpisodeAccumulator, prediction_summary, score_queries
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import load_jepa_checkpoint
from eventground_wam.pixel_baselines import box_rgb, fixed_examples
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import FrozenQwenFeatures, tensor_state_digest
from eventground_wam.pixel_training import encode_training_batch, slice_condition

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = {"anchor_batch": 8, "feature_microbatch": 4, "prediction_microbatch": 10,
             "pilot_projection_margin": 1.5, "maximum_projected_hours": 6,
             "maximum_process_hours": 8, "maximum_host_bytes": 24*1024**3,
             "maximum_gpu_fraction": .85, "minimum_free_disk_bytes": 12*1024**3,
             "maximum_new_artifact_bytes": 8*1024**3,
             "pilot_replay_max_feature_error": 1e-5, "pilot_scalar_reference_tolerance": 1e-10,
             "pilot_warmup_anchors": 8, "aggregation_allowance_seconds": 900,
             "projection_timing": "recurring-path-plus-once-only-pilot-overhead-v2"}


def source_pins():
    paths = sorted(ROOT.glob("src/eventground_wam/*.py"))
    paths += [ROOT/"scripts"/name for name in ("evaluate_jepa_cpt.py", "run_jepa_baselines.py",
              "check_jepa_targets.py", "check_pixel_predictor.py", "train_jepa_cpt.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def verify_pilot(pilot, pilot_identity, identity):
    if (pilot["status"] != "passed" or not all(pilot["checks"].values())
            or pilot["identity_digest"] != pilot_identity["digest"]
            or pilot_identity["phase"] != "pilot" or identity["phase"] != "full"
            or any(pilot_identity[k] != identity[k] for k in identity if k not in ("phase", "digest"))):
        raise ValueError("passed pilot for the exact candidate/code/protocol/runtime is required")


def project_runtime(measurements, prelude_seconds, total_pairs):
    recurring = max((m["seconds"]-m["pilot_only_seconds"])/m["pairs"] for m in measurements)
    once = prelude_seconds + sum(m["pilot_only_seconds"] for m in measurements)
    return (recurring*total_pairs+once)*EXECUTION["pilot_projection_margin"]+EXECUTION["aggregation_allowance_seconds"]


def check_resources(output, started):
    free, total = torch.cuda.mem_get_info()
    if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= EXECUTION["maximum_host_bytes"]
            or (total-free)/total >= EXECUTION["maximum_gpu_fraction"]
            or shutil.disk_usage(output).free < EXECUTION["minimum_free_disk_bytes"]
            or time.monotonic()-started > EXECUTION["maximum_process_hours"]*3600):
        raise RuntimeError("candidate evaluation exceeded a frozen execution limit")


def reference_errors(prediction, target, current, future, observed_rows, identities):
    """Independent scalar arithmetic and integer truth masks on a real pilot batch."""
    expected = np.zeros((len(prediction), 2, 5), np.float64)
    for i in range(len(prediction)):
        for c in range(2):
            a = np.rint(current[i, c]*1020).astype(np.int32)
            b = np.rint(future[i, c]*1020).astype(np.int32)
            mask = np.abs(b-a).sum(axis=0) > 120
            if mask.sum()*100 < 128*128:
                mask[:] = False
            weights = mask.reshape(8, 16, 8, 16).mean(axis=(1, 3)).reshape(64)
            p = prediction[i, c*64:(c+1)*64].astype(np.float64)
            t = target[i, c*64:(c+1)*64].astype(np.float64)
            error = np.clip(1-np.sum(p*t, axis=-1), 0, 2)
            expected[i, c] = [error.mean(), (error*weights).mean(), (error*(1-weights)).mean(),
                              weights.mean(), (1-weights).mean()]
    accumulated = np.zeros((2, 5, 5), np.float64)
    for i, item in enumerate(identities):
        accumulated[:, HORIZONS.index(item["horizon"])] += expected[i]
    return float(np.max(np.abs(accumulated-observed_rows)))


def evaluate_episode(source, dataset, model, encoder, builder, teacher, output, *, queries,
                     examples, replay, pilot=False):
    started = time.monotonic()
    ep = source["episode_index"]
    rgb = dataset._episode(source)[2]
    features = encode(teacher, rgb)
    boxes = np.stack([[box_rgb(image) for image in frame] for frame in rgb])
    pilot_only_seconds = 0.
    if pilot:
        probe_start = time.monotonic()
        template = features.mean(axis=0, dtype=np.float64)
        template = (template/np.linalg.norm(template, axis=-1, keepdims=True)).astype(np.float32)
        queries, _ = episode_retrieval(source, rgb, features, template)
        pilot_only_seconds += time.monotonic()-probe_start
    for q in queries:
        if (q["episode_index"] != ep or q["source_digest"] != source["digest"]
                or q["group_digest"] != source["group_digest"] or q["task_key"] != source["task_key"]):
            raise ValueError("frozen retrieval source changed")
        if q["pool"]["eligible"]:
            c = CAMERAS.index(q["camera"])
            assert [hashlib.sha256(rgb[f, c].tobytes()).hexdigest() for f in q["pool"]["frames"]] == q["pool"]["image_sha256"]
    query_keys = sorted({(q["anchor"], q["horizon"]) for q in queries})
    example_keys = sorted({(e["anchor"], e["horizon"]) for e in examples})
    wanted = set(query_keys+example_keys)
    selected = {}
    accumulator = EpisodeAccumulator(source, split=dataset.split)
    pairs = sum(source["episode_length"]-h for h in HORIZONS)
    archive = output/"replay"/f"{ep:06d}-predictions.npy"
    buffer = np.lib.format.open_memmap(archive.with_suffix(".partial"), mode="w+", dtype=np.float32,
                                       shape=(pairs, 128, 2048)) if replay else None
    pair_keys, cursor = [], 0
    reference_error, replay_error = 0., 0.
    for begin in range(0, source["episode_length"]-2, 8):
        samples = [dataset[dataset.ordinal(ep, anchor, h)]
                   for anchor in range(begin, min(begin+8, source["episode_length"]-2))
                   for h in HORIZONS if anchor+h < source["episode_length"]]
        condition, _, _ = encode_training_batch(samples, encoder, builder, torch.device("cuda"), feature_microbatch=4)
        for offset in range(0, len(samples), 10):
            stop = min(offset+10, len(samples))
            part = slice_condition(condition, offset, stop)
            prediction = model(part).features.cpu().numpy()
            identities = [s.identity for s in samples[offset:stop]]
            anchors = [s["anchor"] for s in identities]
            targets = [s["target"] for s in identities]
            current, future, target = boxes[anchors], boxes[targets], features[targets]
            previous = accumulator.sums.copy()
            accumulator.add(identities, current, future, prediction, target)
            if pilot and cursor == 0:
                probe_start = time.monotonic()
                reference_error = reference_errors(prediction, target, current, future,
                                                    accumulator.sums-previous, identities)
                single = model(slice_condition(part, 0, 1)).features.cpu().numpy()
                replay_error = float(np.max(np.abs(single[0]-prediction[0])))
                again = model(part).features.cpu().numpy()
                assert np.array_equal(again, prediction), "fixed-shape replay is not exact"
                pilot_only_seconds += time.monotonic()-probe_start
            for i, item in enumerate(identities):
                key = item["anchor"], item["horizon"]
                if key in wanted:
                    selected[key] = prediction[i].copy()
                if replay:
                    pair_keys.append(key)
            if replay:
                buffer[cursor:cursor+len(prediction)] = prediction
            cursor += len(prediction)
        del condition, samples, part, prediction, target
    assert cursor == pairs and set(selected) == wanted
    rows = accumulator.finish()
    scored = score_queries(queries, selected, features)
    arrays = {}
    group_mean = None
    if query_keys:
        bank = np.stack([selected[key] for key in query_keys])
        target_mean = np.mean([features[t+h] for t, h in query_keys], axis=0, dtype=np.float64).astype(np.float32)
        group_mean = bank.mean(axis=0, dtype=np.float64).astype(np.float32)
        for label, value in (("query-predictions", bank), ("means", np.stack([group_mean, target_mean]))):
            name = f"panels/{ep:06d}-{label}.npy"
            arrays[name] = atomic_array(output/name, value)
    if example_keys:
        name = f"qualitative/{ep:06d}-predictions.npy"
        arrays[name] = atomic_array(output/name, np.stack([selected[key] for key in example_keys]))
    if replay:
        buffer.flush()
        del buffer
        with archive.with_suffix(".partial").open("rb") as handle:
            os.fsync(handle.fileno())
        archive.with_suffix(".partial").replace(archive)
        arrays[str(archive.relative_to(output))] = sha256_file(archive)
        for label, value in (("features", features), ("boxes", boxes), ("pair-keys", np.asarray(pair_keys))):
            name = f"replay/{ep:06d}-{label}.npy"
            arrays[name] = atomic_array(output/name, value)
    torch.cuda.synchronize()
    return {"rows": rows, "queries": scored, "pair_count": pairs, "arrays": arrays,
            "query_keys": query_keys, "qualitative_examples": examples, "qualitative_keys": example_keys,
            "elapsed_seconds": time.monotonic()-started, "scalar_reference_error": reference_error,
            "pilot_only_seconds": pilot_only_seconds,
            "single_vs_batch_feature_max_error": replay_error}, group_mean


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    runtime = configure()
    protocol = read_sealed(ROOT/"configs/evaluation/week6_jepa_baselines_v1.json")
    assert protocol["measurement"] == measurement_rule()
    for name, digest in protocol["input_sha256"].items():
        assert sha256_file(ROOT/name) == digest, name
    verified = read_sealed(args.verification)
    training = read_sealed(args.run/"report.json")
    assert verified["status"] == "passed" and verified["report_digest"] == training["digest"]
    assert training["runtime"] == runtime and training["global_step"] == 221633
    for name, digest in training["source_sha256"].items():
        assert sha256_file(ROOT/name) == digest, name
    candidate = args.run/"C_jepa-candidate.pt"
    candidate_sha = sha256_file(candidate)
    assert candidate_sha == verified["candidate_sha256"]
    baseline_completion = read_sealed(args.baselines/"completion.json")
    baseline = read_sealed(args.baselines/"report.json")
    frozen_baseline = read_sealed(ROOT/"docs/evaluation/week6-jepa-baselines-v1/run/report.json")
    assert baseline["digest"] == baseline_completion["report_digest"] == frozen_baseline["digest"]
    assert baseline["config_digest"] == protocol["digest"]
    for name, digest in baseline_completion["files_sha256"].items():
        assert sha256_file(args.baselines/name) == digest, name
    baseline_rows = read_sealed(args.baselines/"episode-metrics.json")["rows"]
    baseline_queries = read_sealed(args.baselines/"retrieval-panel.json")["queries"]
    rows_by_ep, queries_by_ep = defaultdict(list), defaultdict(list)
    for row in baseline_rows:
        rows_by_ep[row["episode_index"]].append(row)
    for query in baseline_queries:
        queries_by_ep[query["episode_index"]].append(query)
    split = "train" if args.phase == "pilot" else "validation"
    dataset = FuturePairDataset(args.pairs, args.data_root/f"datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo",
                               split=split, expected_manifest_sha256=protocol["pair_manifest_sha256"], cache_episodes=1,
                               normalizer=load_b0_action_normalizer(ROOT/"docs/model/week4-fresh-v2/action-statistics.json"),
                               state_bounds=read_sealed(ROOT/"configs/training/week4_b0_fresh_all_l1_v2.json")["state_normalization"])
    all_sources = [json.loads(line) for line in (args.pairs/"sources.jsonl").read_text().splitlines()]
    val_sources = sorted([s for s in all_sources if s["split"] == "validation"], key=lambda s: s["episode_index"])
    examples_record = read_sealed(ROOT/"configs/evaluation/week5_pixel_qualitative_v2.json")
    assert examples_record["examples"] == fixed_examples(val_sources)
    examples_by_ep = defaultdict(list)
    for example in examples_record["examples"]:
        examples_by_ep[example["episode_index"]].append(example)
    code = source_pins()
    execution = read_sealed(ROOT/"configs/evaluation/week6_jepa_candidate_v2.json")
    assert execution["execution"] == EXECUTION and execution["source_sha256"] == code
    assert execution["candidate_sha256"] == candidate_sha and execution["protocol_digest"] == protocol["digest"]
    identity = seal_contract({"artifact_type": "eventground_wam.jepa_candidate_evaluation_identity",
                              "phase": args.phase, "candidate_sha256": candidate_sha,
                              "training_report_digest": training["digest"], "verification_digest": verified["digest"],
                              "protocol_digest": protocol["digest"], "pair_manifest_digest": dataset.manifest["digest"],
                              "baseline_report_digest": baseline["digest"], "baseline_completion_digest": baseline_completion["digest"],
                              "qualitative_panel_digest": examples_record["digest"], "source_sha256": code,
                              "runtime": runtime, "execution": EXECUTION, "execution_digest": execution["digest"],
                              "checkpoint_selection": "fixed-final-update-221633"})
    identity_path = args.output/"identity.json"
    if identity_path.exists():
        assert read_sealed(identity_path) == identity
        if (args.output/"completion.json").exists() or (args.output/"pilot.json").exists():
            raise FileExistsError("completed evaluation output is immutable")
    else:
        assert all(p.name in ("owner.lock", "service.log") for p in args.output.iterdir())
        atomic_json(identity_path, identity)
    if args.phase == "full":
        verify_pilot(read_sealed(args.pilot), read_sealed(args.pilot.parent/"identity.json"), identity)
    for name in ("episodes", "panels", "replay", "qualitative"):
        (args.output/name).mkdir(exist_ok=True)
    assert shutil.disk_usage(args.output).free > EXECUTION["minimum_free_disk_bytes"]
    verify_base_files(args.cache, read_sealed(ROOT/"docs/model/Qwen3.5-2B-manifest.json"))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to("cuda")
    encoder = FrozenQwenFeatures(full.model).to("cuda")
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    builder = FutureInputBuilder(processor)
    teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor).eval()
    accepted = read_sealed(ROOT/"docs/evaluation/week6-jepa-targets-v2/report.json")
    for label, obj in (("vision", type(teacher.visual)), ("processor", type(teacher.image_processor))):
        assert sha256_file(Path(inspect.getfile(obj))) == accepted["implementation_files"][label]["sha256"]
    base_before = tensor_state_digest(encoder.core.state_dict())
    teacher_before = tensor_state_digest(teacher.visual.state_dict())
    assert base_before == training["base_state_sha256_after"]
    assert teacher_before == training["teacher_state_sha256_after"] == accepted["vision_state_after"]
    model, header = load_jepa_checkpoint(candidate)
    assert header == verified["candidate_header"]
    model = model.eval().requires_grad_(False).to("cuda")
    predictor_before = tensor_state_digest(model.state_dict())
    torch.cuda.reset_peak_memory_stats()
    sources = sorted(dataset.sources.values(), key=lambda s: s["episode_index"])
    if args.phase == "pilot":
        train = representative_sources(sources, "train", "week6-jepa-mean-v1")
        chosen = [min(train, key=lambda s: s["episode_length"]), max(train, key=lambda s: s["episode_length"]),
                  max(sources, key=lambda s: len(s["instruction"]))]
        sources = list({s["episode_index"]: s for s in chosen}.values())
        # Warm up fixed execution shapes on TRAIN only; charge this cost once.
        source = sources[0]
        warmup = [dataset[dataset.ordinal(source["episode_index"], anchor, h)]
                  for anchor in range(EXECUTION["pilot_warmup_anchors"]) for h in HORIZONS]
        encode(teacher, dataset._episode(source)[2][:1])
        condition, _, _ = encode_training_batch(warmup, encoder, builder, torch.device("cuda"), feature_microbatch=4)
        model(slice_condition(condition, 0, 10))
        torch.cuda.synchronize()
        del warmup, condition
        # Re-read episode payload during measurement, matching the full path.
        dataset._cache.clear()
    else:
        assert len(sources) == 1528 and len(dataset) == 1109994
    replay_ids = set(baseline_completion["replay_episode_indices"]) if args.phase == "full" else set()
    rows, queries, receipts, measurements, panel_values, panel_groups = [], [], [], [], [], []
    pairs_done, new_pairs = 0, 0
    scoring_start = time.monotonic()
    prelude_seconds = scoring_start-started
    for source in sources:
        ep = source["episode_index"]
        path = args.output/"episodes"/f"{ep:06d}.json"
        if path.exists():
            saved = read_sealed(path)
            assert saved["identity_digest"] == identity["digest"] and saved["source_digest"] == source["digest"]
            for name, digest in saved["arrays"].items():
                assert sha256_file(args.output/name) == digest, name
        else:
            measured, _ = evaluate_episode(source, dataset, model, encoder, builder, teacher, args.output,
                                           queries=queries_by_ep[ep] if args.phase == "full" else [],
                                           examples=examples_by_ep[ep] if args.phase == "full" else [],
                                           replay=ep in replay_ids, pilot=args.phase == "pilot")
            if args.phase == "full":
                expected = {(r["camera"], r["horizon"]): r for r in rows_by_ep[ep]}
                for row in measured["rows"]:
                    ref = expected[row["camera"], row["horizon"]]
                    assert all(row[k] == ref[k] for k in ("episode_index", "source_digest", "group_digest", "task_key", "pairs"))
                    assert all(abs(row["prediction"][k]-ref["copy"][k]) < 1e-10 for k in ("motion_area", "static_area"))
                if measured["query_keys"]:
                    baseline_receipt = read_sealed(args.baselines/"episodes"/f"{ep:06d}.json")
                    name = f"panels/{ep:06d}.npy"
                    assert sha256_file(args.baselines/name) == baseline_receipt["arrays"][name]
                    baseline_mean = np.load(args.baselines/name, allow_pickle=False)[1]
                    current_mean = np.load(args.output/f"panels/{ep:06d}-means.npy", allow_pickle=False)[1]
                    assert np.array_equal(current_mean, baseline_mean), "canonical teacher panel changed"
            saved = seal_contract({"artifact_type": "eventground_wam.jepa_candidate_episode",
                                   "identity_digest": identity["digest"], "source_digest": source["digest"],
                                   "split": split, **measured})
            atomic_json(path, saved)
            new_pairs += saved["pair_count"]
        assert saved["pair_count"] == sum(source["episode_length"]-h for h in HORIZONS)
        rows.extend(saved["rows"])
        queries.extend(saved["queries"])
        receipts.append({"episode_index": ep, "digest": saved["digest"], "sha256": sha256_file(path)})
        measurements.append({"episode": ep, "frames": source["episode_length"], "pairs": saved["pair_count"],
                             "seconds": saved["elapsed_seconds"], "scalar_reference_error": saved["scalar_reference_error"],
                             "pilot_only_seconds": saved["pilot_only_seconds"],
                             "single_vs_batch_feature_max_error": saved["single_vs_batch_feature_max_error"]})
        if saved["query_keys"]:
            panel_values.append(np.load(args.output/f"panels/{ep:06d}-means.npy", allow_pickle=False)[0])
            panel_groups.append((source["task_key"], source["group_digest"]))
        pairs_done += saved["pair_count"]
        check_resources(args.output, started)
        progress = {"status": "running", "phase": args.phase, "episodes": len(receipts), "total_episodes": len(sources),
                    "pairs": pairs_done, "elapsed_seconds": time.monotonic()-started,
                    "projected_full_scoring_hours": (time.monotonic()-scoring_start)/max(1, new_pairs)*1109994/3600}
        atomic_json(args.output/"progress.json", progress)
        print(json.dumps(progress), flush=True)
    assert tensor_state_digest(model.state_dict()) == predictor_before
    assert tensor_state_digest(encoder.core.state_dict()) == base_before
    assert tensor_state_digest(teacher.visual.state_dict()) == teacher_before
    assert source_pins() == code and sha256_file(candidate) == candidate_sha
    resources = {"total_process_seconds": time.monotonic()-started,
                 "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 "artifact_bytes": sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())}
    assert resources["artifact_bytes"] < EXECUTION["maximum_new_artifact_bytes"]
    if args.phase == "pilot":
        projected = project_runtime(measurements, prelude_seconds, 1109994)
        replay_pairs = sum(sum(s["episode_length"]-h for h in HORIZONS) for s in val_sources if s["episode_index"] in baseline_completion["replay_episode_indices"])
        projected_disk = (replay_pairs+sum(s["episode_length"] for s in val_sources if s["episode_index"] in baseline_completion["replay_episode_indices"])+180*11+480)*128*2048*4+128*1024**2
        checks = {"runtime_projection": projected < EXECUTION["maximum_projected_hours"]*3600,
                  "host_memory": resources["host_peak_rss_bytes"] < EXECUTION["maximum_host_bytes"],
                  "persistent_artifacts": projected_disk < EXECUTION["maximum_new_artifact_bytes"],
                  "scalar_reference": max(m["scalar_reference_error"] for m in measurements) < EXECUTION["pilot_scalar_reference_tolerance"],
                  "single_vs_batch_prediction": max(m["single_vs_batch_feature_max_error"] for m in measurements) < EXECUTION["pilot_replay_max_feature_error"],
                  "frozen_models_unchanged": True, "fixed_shape_prediction_replay_exact": True}
        pilot = seal_contract({"artifact_type": "eventground_wam.jepa_candidate_pilot", "status": "passed" if all(checks.values()) else "failed",
                               "identity_digest": identity["digest"], "checks": checks, "measurements": measurements,
                               "projected_hours_with_margin": projected/3600, "projected_artifact_bytes": projected_disk,
                               "prelude_seconds": prelude_seconds, "warmup_training_pairs": EXECUTION["pilot_warmup_anchors"]*len(HORIZONS),
                               "resources": resources, "validation_payload_reads": 0, "test_payload_reads": 0,
                               "optimizer_updates": 0, "pilot_training_pairs_evaluated": pairs_done})
        atomic_json(args.output/"pilot.json", pilot)
        assert pilot["status"] == "passed", pilot
        return
    assert pairs_done == 1109994 and len(rows) == 15280 and len(queries) == 3240 and len(panel_groups) == 180
    collapse = collapse_diagnostics(np.stack(panel_values), panel_groups)
    gate = assess_candidate(baseline_rows, baseline_queries, rows, queries, baseline["collapse"]["target"], collapse)
    atomic_json(args.output/"episode-metrics.json", seal_contract({"artifact_type": "eventground_wam.jepa_candidate_rows", "rows": rows}))
    atomic_json(args.output/"retrieval-panel.json", seal_contract({"artifact_type": "eventground_wam.jepa_candidate_retrieval", "queries": queries}))
    report = seal_contract({"artifact_type": "eventground_wam.jepa_candidate_validation", "status": "complete",
                            "identity_digest": identity["digest"], "candidate_sha256": candidate_sha,
                            "protocol_digest": protocol["digest"], "baseline_report_digest": baseline["digest"],
                            **prediction_summary(rows, queries), "collapse": collapse, "gate_with_review_pending": gate,
                            "quantitative_checks_passed": all(v for k, v in gate.get("checks", {}).items() if k != "qualitative_review") if "checks" in gate else False,
                            "dual_camera_pairs": pairs_done, "episodes": len(sources), "optimizer_updates": 0,
                            "test_payload_reads": 0, "qualitative_examples_saved": sum(len(read_sealed(args.output/'episodes'/f"{s['episode_index']:06d}.json")["qualitative_examples"]) for s in sources),
                            "qualitative_review": "pending", "independent_final_metric_audit": "pending",
                            "appearance_comparison": "pending", "C_jepa_accepted": False})
    atomic_json(args.output/"report.json", report)
    completion = seal_contract({"artifact_type": "eventground_wam.jepa_candidate_evaluation_completion", "status": "complete",
                                "identity_digest": identity["digest"], "report_digest": report["digest"],
                                "candidate_sha256": candidate_sha, "episode_receipts": receipts, "panel_groups": panel_groups,
                                "replay_episode_indices": sorted(replay_ids), "predictor_base_teacher_unchanged": True,
                                "resources": resources, "optimizer_updates": 0, "test_payload_reads": 0,
                                "files_sha256": {name: sha256_file(args.output/name) for name in ("episode-metrics.json", "retrieval-panel.json", "report.json")}})
    atomic_json(args.output/"completion.json", completion)
    atomic_json(args.output/"progress.json", {"status": "complete", "report_digest": report["digest"],
                                            "quantitative_checks_passed": report["quantitative_checks_passed"]})
    print(json.dumps({"status": "complete", "report_digest": report["digest"], "gate_with_review_pending": gate}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "baselines", "run", "verification", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path.home()/"data/eventground-wam/libero-plus")
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    args = parser.parse_args()
    if args.phase == "full" and args.pilot is None:
        parser.error("full evaluation requires --pilot")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
