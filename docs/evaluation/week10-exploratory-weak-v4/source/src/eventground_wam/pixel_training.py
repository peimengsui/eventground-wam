"""Frozen common CPT order, logical-batch updates and durable pixel recovery."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_training import (
    atomic_json,
    atomic_torch_save,
    capture_rng,
    cpu_tree,
    restore_rng,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import HORIZONS, read_sealed, sha256_file
from eventground_wam.pixel_data import collate_future_samples
from eventground_wam.pixel_predictor import PixelCondition, pixel_reconstruction_loss

ORDER_VERSION = "trajectory-v2-cpt-order-v1"
RECOVERY_FORMAT = "eventground-wam-pixel-full-training-state-v1"


def order_rule():
    return {"version": ORDER_VERSION, "epochs": 1, "seed": 20260926,
            "episode_order": "ascending SHA256(version:seed:episode_index)",
            "pool_episodes": 4, "anchors_per_visit": 2, "anchors_per_batch": 8,
            "anchor_order": "PCG64(SeedSequence([seed,episode_index])) permutation of [0,L-2)",
            "mixing": "round-robin two anchors per episode in each four-episode pool until exhausted",
            "horizon_order": list(HORIZONS), "targets": "every h with anchor+h<L; both cameras",
            "tail_rule": "retain all short batches and valid near-end horizons; never pad/drop/repeat",
            "augmentation": "none", "weighting": "equal future pairs and equal camera/RGB pixels"}


def build_order(sources, output: Path, *, pair_manifest_sha256: str):
    """Metadata-only full coverage plan shared by Pixel and JEPA; never reads frames."""
    rule = order_rule()
    sources = list(sources)
    if (not sources or any(s["split"] != "train" or s["episode_length"] <= 32 for s in sources)
            or len({s["episode_index"] for s in sources}) != len(sources)):
        raise ValueError("order requires unique frozen train episodes with all horizons")
    ordered = sorted(sources, key=lambda s: hashlib.sha256(
        f"{ORDER_VERSION}:{rule['seed']}:{s['episode_index']}".encode()).hexdigest())
    anchors, offsets, pools, pair_counts = [], [0], [], []
    counts = {str(h): 0 for h in HORIZONS}
    pool_starts = []
    for pool_index, start in enumerate(range(0, len(ordered), 4)):
        group = ordered[start:start+4]
        permutations = [np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [rule["seed"], s["episode_index"]]))).permutation(s["episode_length"]-2) for s in group]
        positions, pending = [0]*len(group), []
        pool_starts.append(len(pair_counts))
        while any(p < len(a) for p, a in zip(positions, permutations, strict=True)):
            for j, (source, permutation) in enumerate(zip(group, permutations, strict=True)):
                for anchor in permutation[positions[j]:positions[j]+2]:
                    pending.append((source["episode_index"], int(anchor), source["episode_length"]))
                positions[j] += 2
            while len(pending) >= 8:
                emit, pending = pending[:8], pending[8:]
                anchors.extend((ep, t) for ep, t, _ in emit)
                offsets.append(len(anchors)); pools.append(pool_index)
                pair_counts.append(sum(t+h < length for _, t, length in emit for h in HORIZONS))
                for _, t, length in emit:
                    for h in HORIZONS:
                        counts[str(h)] += int(t+h < length)
        if pending:
            anchors.extend((ep, t) for ep, t, _ in pending)
            offsets.append(len(anchors)); pools.append(pool_index)
            pair_counts.append(sum(t+h < length for _, t, length in pending for h in HORIZONS))
            for _, t, length in pending:
                for h in HORIZONS:
                    counts[str(h)] += int(t+h < length)
    expected = {str(h): sum(s["episode_length"]-h for s in sources) for h in HORIZONS}
    if counts != expected or sum(pair_counts) != sum(expected.values()):
        raise ValueError("CPT plan lost or repeated future targets")
    output.mkdir(parents=True, exist_ok=False)
    arrays = {"anchors": np.asarray(anchors, dtype="<u4"), "offsets": np.asarray(offsets, dtype="<u8"),
              "pools": np.asarray(pools, dtype="<u4"), "pair_counts": np.asarray(pair_counts, dtype="<u2")}
    pins = {}
    for name, array in arrays.items():
        path = output/f"{name}.npy"; np.save(path, array, allow_pickle=False)
        pins[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                           "shape": list(array.shape), "dtype": str(array.dtype)}
    record = seal_contract({"artifact_type": "eventground_wam.common_cpt_order", "rule": rule,
                             "pair_manifest_sha256": pair_manifest_sha256,
                             "train_source_digests_sha256": hashlib.sha256(
                                 "\n".join(s["digest"] for s in ordered).encode()).hexdigest(),
                             "episodes": len(sources), "anchors": len(anchors),
                             "updates": len(pair_counts), "pair_presentations": sum(pair_counts),
                             "pairs_by_horizon": counts, "maximum_pairs_per_update": max(pair_counts),
                             "minimum_pairs_per_update": min(pair_counts),
                             "pool_first_updates": pool_starts, "files": pins})
    atomic_json(output/"manifest.json", record)
    return record


class CPTOrder:
    def __init__(self, root: Path, expected_sha256: str):
        if sha256_file(root/"manifest.json") != expected_sha256:
            raise ValueError("common CPT order manifest changed")
        self.manifest = read_sealed(root/"manifest.json")
        if self.manifest["rule"] != order_rule():
            raise ValueError("unsupported CPT sample ordering")
        for name, pin in self.manifest["files"].items():
            path = root/name
            if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
                raise ValueError("common CPT order array changed")
        self.anchors = np.load(root/"anchors.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(root/"offsets.npy", mmap_mode="r", allow_pickle=False)
        self.pair_counts = np.load(root/"pair_counts.npy", mmap_mode="r", allow_pickle=False)
        self.pools = np.load(root/"pools.npy", mmap_mode="r", allow_pickle=False)
        if (len(self.offsets) != len(self.pair_counts)+1 or self.offsets[0] != 0
                or self.offsets[-1] != len(self.anchors)
                or np.any(np.diff(self.offsets) > 8) or np.any(np.diff(self.offsets) == 0)):
            raise ValueError("invalid logical-batch boundaries")

    def __len__(self):
        return len(self.pair_counts)

    def references(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        return [(int(ep), int(t)) for ep, t in self.anchors[self.offsets[index]:self.offsets[index+1]]]

    def samples(self, dataset, index):
        result = []
        for ep, t in self.references(index):
            length = dataset.sources[ep]["episode_length"]
            result.extend(dataset[dataset.ordinal(ep, t, h)] for h in HORIZONS if t+h < length)
        if len(result) != int(self.pair_counts[index]):
            raise ValueError("CPT sample count does not match the frozen logical batch")
        return result


def pilot_indices(order, sources):
    """32 disjoint four-update windows, including longest episode/prompt pools."""
    starts = order.manifest["pool_first_updates"]
    selected = set(np.linspace(0, len(starts)-1, min(30, len(starts)), dtype=int).tolist())
    for key in (lambda s: s["episode_length"], lambda s: len(s["instruction"])):
        ep = max(sources.values(), key=key)["episode_index"]
        row = int(np.flatnonzero(order.anchors[:, 0] == ep)[0])
        index = int(np.searchsorted(order.offsets, row, side="right")-1)
        selected.add(int(order.pools[index]))
    # Fill to 32 pools even when extreme cases coincide with uniform selections.
    for pool in range(len(starts)):
        if len(selected) >= min(32, len(starts)):
            break
        selected.add(pool)
    result = []
    for pool in sorted(selected):
        end = starts[pool+1] if pool+1 < len(starts) else len(order)
        result.extend(range(starts[pool], min(starts[pool]+4, end)))
    return result


def encode_training_batch(samples, encoder, builder, device, *, feature_microbatch=4):
    """Encode each current observation once, then share it across its valid horizons."""
    unique, mapping, positions = [], {}, []
    for sample in samples:
        key = sample.identity["episode_index"], sample.identity["anchor"]
        if key not in mapping:
            mapping[key] = len(unique); unique.append(sample.observation)
        positions.append(mapping[key])
    features = []
    for start in range(0, len(unique), feature_microbatch):
        observations = unique[start:start+feature_microbatch]
        encoding = builder([x.instruction for x in observations], [x.current_rgb[0] for x in observations],
                           [x.current_rgb[1] for x in observations])
        encoded, padding = encoder({k: v.to(device) for k, v in encoding.items()})
        features.extend(encoded[j, ~padding[j]].detach() for j in range(len(observations)))
    repeated = [features[j] for j in positions]
    padded = torch.nn.utils.rnn.pad_sequence(repeated, batch_first=True)
    padding = torch.arange(padded.shape[1], device=device)[None] >= torch.tensor(
        [len(f) for f in repeated], device=device)[:, None]
    batch = collate_future_samples(samples, padded, padding)
    return batch.condition.to(device), batch.targets.to(device), {
        "unique_current_observations": len(unique), "maximum_feature_tokens": padded.shape[1],
        "feature_bytes": sum(f.numel()*f.element_size() for f in features)}


def slice_condition(condition, start, stop):
    return PixelCondition(**{field.name: getattr(condition, field.name)[start:stop]
                             for field in fields(condition)})


def accumulate_loss(model, condition, targets, *, microbatch, ssim_weight, hook=None):
    """Pair-count weighting preserves a single loss over the full logical batch."""
    total = 0.
    for start in range(0, len(targets), microbatch):
        stop = min(start+microbatch, len(targets))
        prediction = model(slice_condition(condition, start, stop)).images
        loss = pixel_reconstruction_loss(prediction, targets[start:stop], ssim_weight=ssim_weight)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite pixel objective")
        weighted = loss * ((stop-start)/len(targets))
        weighted.backward(); total += float(weighted.detach())
        if hook is not None:
            hook(start, stop)
    return total


def rate_factor(step, total, warmup, minimum):
    if step < warmup:
        return (step+1)/warmup
    progress = min(1., (step-warmup)/max(1, total-warmup-1))
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))


class PixelTrainingEngine:
    def __init__(self, model, recipe, *, identity_digest, indices, order):
        self.model, self.recipe, self.identity_digest = model, recipe, identity_digest
        self.indices, self.order = list(indices), order
        o = recipe["optimizer"]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=o["learning_rate"],
                                           betas=tuple(o["betas"]), eps=o["epsilon"],
                                           weight_decay=o["weight_decay"], foreach=False, fused=False)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: rate_factor(
            step, recipe["total_updates"], o["warmup_updates"], o["minimum_lr_ratio"]))
        self.global_step, self.pairs_seen = 0, 0
        self.sample_chain = "0"*64
        self.loss_sum = 0.
        self.in_update = False

    def step(self, condition, targets, identities, *, hook=None):
        if self.in_update or self.global_step >= len(self.indices):
            raise ValueError("update is partial or exceeds the frozen endpoint")
        plan_index = self.indices[self.global_step]
        if len(targets) != int(self.order.pair_counts[plan_index]) or len(identities) != len(targets):
            raise ValueError("logical batch pair count changed")
        expected = self.order.references(plan_index)
        actual = list(dict.fromkeys((s["episode_index"], s["anchor"]) for s in identities))
        if expected != actual or any(s["split"] != "train" for s in identities):
            raise ValueError("sample order or split disagrees with the frozen cursor")
        batch_digest = hashlib.sha256("\n".join(s["pair_digest"] for s in identities).encode()).hexdigest()
        self.in_update = True
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        loss = accumulate_loss(self.model, condition, targets,
                               microbatch=self.recipe["predictor_microbatch"],
                               ssim_weight=self.recipe["loss"]["ssim_weight"], hook=hook)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.recipe["optimizer"]["gradient_clip"],
                                             error_if_nonfinite=True)
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step(); self.scheduler.step()
        self.global_step += 1; self.pairs_seen += len(targets)
        self.loss_sum += loss*len(targets)
        self.sample_chain = hashlib.sha256(bytes.fromhex(self.sample_chain)+bytes.fromhex(batch_digest)).hexdigest()
        self.in_update = False
        return {"global_step": self.global_step, "plan_index": plan_index, "pairs": len(targets),
                "pairs_seen": self.pairs_seen, "batch_digest": batch_digest,
                "sample_chain": self.sample_chain, "loss": loss, "learning_rate": lr,
                "gradient_norm": float(norm)}

    def snapshot(self):
        if self.in_update:
            raise ValueError("partial accumulation cannot be checkpointed")
        return {"format": RECOVERY_FORMAT, "identity_digest": self.identity_digest,
                "model": cpu_tree(self.model.state_dict()), "optimizer": cpu_tree(self.optimizer.state_dict()),
                "scheduler": self.scheduler.state_dict(), "rng": capture_rng(),
                "global_step": self.global_step, "pairs_seen": self.pairs_seen,
                "sample_chain": self.sample_chain, "loss_sum": self.loss_sum}

    def restore(self, payload):
        payload = cpu_tree(payload)
        step = payload.get("global_step")
        if (payload.get("format") != RECOVERY_FORMAT or payload.get("identity_digest") != self.identity_digest
                or type(step) is not int or not 0 <= step <= len(self.indices)
                or payload["pairs_seen"] != sum(int(self.order.pair_counts[j]) for j in self.indices[:step])
                or payload["scheduler"]["last_epoch"] != step):
            raise ValueError("checkpoint identity, schedule, exposure or cursor changed")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.global_step, self.pairs_seen = step, payload["pairs_seen"]
        self.sample_chain, self.loss_sum = payload["sample_chain"], payload["loss_sum"]
        restore_rng(payload["rng"])


class PixelCheckpointStore:
    """Fsync tensor, then sealed commit; recover newest intact committed generation."""
    def __init__(self, root, identity_digest, *, keep=3):
        if type(keep) is not int or keep < 2:
            raise ValueError("retain at least two complete recovery generations")
        self.root, self.identity_digest, self.keep = Path(root), identity_digest, keep
        self.root.mkdir(parents=True, exist_ok=True)
        self.notes = []

    def save(self, payload):
        if payload.get("identity_digest") != self.identity_digest or payload.get("format") != RECOVERY_FORMAT:
            raise ValueError("checkpoint identity mismatch")
        name = f"checkpoint-{payload['global_step']:012d}-{uuid.uuid4().hex}"
        path = self.root/f"{name}.pt"
        digest = atomic_torch_save(path, payload)
        record = seal_contract({"artifact_type": "eventground_wam.pixel_recovery_checkpoint",
                                "identity_digest": self.identity_digest, "global_step": payload["global_step"],
                                "checkpoint": path.name, "sha256": digest, "bytes": path.stat().st_size})
        atomic_json(path.with_suffix(".json"), record)
        atomic_json(self.root/"latest.json", record)
        committed = sorted(self.root.glob("checkpoint-*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for old in committed[self.keep:]:
            old.with_suffix(".pt").unlink(missing_ok=True); old.unlink()
        # Uncommitted tensor/temporary files left by a crash never authorize progress.
        # The writer owns this directory under an exclusive process lock.
        known = {p.stem for p in self.root.glob("checkpoint-*.json")}
        for old in self.root.glob("checkpoint-*.pt"):
            if old.stem not in known:
                old.unlink()
        for old in self.root.glob(".*.partial"):
            old.unlink()
        return record

    def load(self):
        records = sorted(self.root.glob("checkpoint-*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for path in records:
            try:
                record = read_sealed(path)
            except (ValueError, OSError) as error:
                self.notes.append(f"ignored invalid commit {path.name}: {error}"); continue
            if record["identity_digest"] != self.identity_digest:
                raise ValueError("refusing a different training identity")
            filename = record["checkpoint"]
            if Path(filename).name != filename or not filename.endswith(".pt"):
                raise ValueError("unsafe checkpoint path")
            try:
                tensor = self.root/filename
                if sha256_file(tensor) != record["sha256"]:
                    raise ValueError("checkpoint checksum mismatch")
                payload = torch.load(tensor, map_location="cpu", weights_only=True)
            except (ValueError, OSError, RuntimeError, EOFError) as error:
                self.notes.append(f"ignored damaged checkpoint {filename}: {error}"); continue
            if (payload.get("format") != RECOVERY_FORMAT or payload.get("identity_digest") != self.identity_digest
                    or payload.get("global_step") != record["global_step"]):
                raise ValueError("invalid checkpoint envelope")
            return payload
        if records:
            raise ValueError("no intact committed checkpoint remains")
        return None


def verify_recipe(path, root):
    recipe = read_sealed(path)
    if (recipe["artifact_type"] != "eventground_wam.pixel_cpt_recipe" or recipe["epochs"] != 1
            or recipe["order_rule"] != order_rule() or recipe["checkpoint_selection"] != "fixed-final-update"
            or not 0 < recipe["predictor_microbatch"] <= 40
            or recipe["loss"] != {"l1_weight": 1., "ssim_weight": .1}
            or recipe["feature_microbatch"] != 4):
        raise ValueError("unsupported frozen pixel recipe")
    for name, digest in recipe["input_sha256"].items():
        if sha256_file(root/name) != digest:
            raise ValueError(f"frozen recipe input changed: {name}")
    return recipe


def tree_digest(value):
    """Content digest independent of torch archive serialization and dictionary order."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}".encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, allow_nan=False).encode())
        digest.update(b"\0")
    visit(value)
    return digest.hexdigest()


def append_event(path, event):
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
