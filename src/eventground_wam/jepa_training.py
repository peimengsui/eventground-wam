"""Matched CPT batches, exact logical-batch JEPA gradients and bounded targets."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from eventground_wam.future_pairs import read_sealed, sha256_file
from eventground_wam.jepa_predictor import JepaPredictorConfig, jepa_prediction_loss
from eventground_wam.pixel_training import (
    PixelTrainingEngine,
    order_rule,
    slice_condition,
)
from eventground_wam.pixel_training import (
    encode_training_batch as encode_current_batch,
)

TARGET_BYTES = 128*2048*4
LOSS_RULE = {"variance_weight": 10., "variance_floor_ratio": .5, "variance_epsilon": 1e-4}


def anchor_groups(identities, device):
    keys = [(s["episode_index"], s["anchor"]) for s in identities]
    mapping = {key: i for i, key in enumerate(sorted(set(keys)))}
    return torch.tensor([mapping[key] for key in keys], dtype=torch.int64, device=device)


def accumulate_loss(model, condition, targets, groups, *, microbatch, loss_rule, hook=None):
    """Differentiate one logical loss, then replay deterministic model chunks.

    The leaf feature bank contains the entire batch, so grouped variance never
    becomes an average of microbatch penalties. Only one model-chunk graph is
    live during backward. There is no optimizer or RNG advance between passes.
    """
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("positive integer microbatch required")
    ranges = [(start, min(start+microbatch, len(targets)))
              for start in range(0, len(targets), microbatch)]
    with torch.no_grad():
        bank = torch.cat([model(slice_condition(condition, a, b)).features for a, b in ranges])
    bank.requires_grad_(True)
    loss = jepa_prediction_loss(bank, targets, groups, **loss_rule)
    derivative, = torch.autograd.grad(loss.total, bank)
    if not torch.isfinite(derivative).all():
        raise ValueError("nonfinite logical-batch feature gradient")
    for start, stop in ranges:
        predicted = model(slice_condition(condition, start, stop)).features
        if not torch.equal(predicted.detach(), bank[start:stop]):
            raise ValueError("two-pass predictor is not deterministic")
        predicted.backward(derivative[start:stop])
        if hook is not None:
            hook(start, stop)
    return {"loss": float(loss.total.detach()), "cosine": float(loss.cosine.detach()),
            "variance": float(loss.variance.detach()), "independent_anchors": loss.independent_anchors}


class PoolTargetCache:
    """Canonical FP32 teacher targets in RAM for one frozen four-episode pool.

    The enclosing run pins the teacher and source identities. No disk cache is
    read or written; restarting recomputes identical targets. Changing a pool
    clears every entry, even if consecutive pilot windows skip many pools.
    """

    def __init__(self, teacher, *, maximum_bytes=4*1024**3):
        self.teacher, self.maximum_bytes = teacher, maximum_bytes
        self.pool = None
        self.values = {}
        self.episodes = set()
        self.peak_bytes = 0

    def clear(self):
        self.values.clear()
        self.episodes.clear()
        self.pool = None

    def encode(self, samples, *, pool, device):
        if pool != self.pool:
            self.clear(); self.pool = pool
        if any(s.identity["split"] != "train" for s in samples):
            raise ValueError("target cache accepts training samples only")
        self.episodes.update(s.identity["episode_index"] for s in samples)
        if len(self.episodes) > 4:
            raise ValueError("target cache crossed a four-episode pool boundary")
        keys = [(s.identity["episode_index"], s.identity["target"]) for s in samples]
        missing = {}
        for key, sample in zip(keys, samples, strict=True):
            if key not in self.values:
                missing.setdefault(key, sample.target_rgb)
        if (len(self.values)+len(missing))*TARGET_BYTES > self.maximum_bytes:
            raise ValueError("target RAM budget exceeded")
        pairs = list(missing.items())
        for start in range(0, len(pairs), 16):
            chunk = pairs[start:start+16]
            values = self.teacher(np.stack([rgb for _, rgb in chunk])).detach().cpu()
            if (values.dtype != torch.float32 or values.shape != (len(chunk), 128, 2048)
                    or not torch.isfinite(values).all()
                    or (values.norm(dim=-1)-1).abs().max() > 1e-5):
                raise ValueError("teacher cache requires canonical finite FP32 unit targets")
            for (key, _), value in zip(chunk, values, strict=True):
                # Own each frame's storage; never retain a larger temporary batch.
                self.values[key] = value.clone()
        self.peak_bytes = max(self.peak_bytes, len(self.values)*TARGET_BYTES)
        targets = torch.stack([self.values[key] for key in keys]).to(device)
        return targets, {"target_cache_misses": len(missing), "target_cache_hits": len(keys)-len(missing),
                         "target_cache_bytes": len(self.values)*TARGET_BYTES,
                         "target_cache_peak_bytes": self.peak_bytes}


def encode_training_batch(samples, encoder, builder, cache, device, *, pool, feature_microbatch=4):
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    condition, _, info = encode_current_batch(samples, encoder, builder, device,
                                               feature_microbatch=feature_microbatch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    current_done = time.monotonic()
    targets, cache_info = cache.encode(samples, pool=pool, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    info.update(cache_info)
    info.update(current_feature_seconds=current_done-started,
                target_seconds=time.monotonic()-current_done)
    return condition, targets, anchor_groups([s.identity for s in samples], device), info


class JepaTrainingEngine(PixelTrainingEngine):
    """Reuse the tested optimizer/scheduler and full-state recovery envelope.

    Recovery keeps the legacy Pixel serializer format, with a JEPA-specific
    sealed identity. A Pixel or pilot checkpoint can never resume JEPA production.
    """

    def step(self, condition, targets, identities, *, groups, hook=None):
        if self.in_update or self.global_step >= len(self.indices):
            raise ValueError("update is partial or exceeds the frozen endpoint")
        index = self.indices[self.global_step]
        actual = list(dict.fromkeys((s["episode_index"], s["anchor"]) for s in identities))
        if (len(targets) != int(self.order.pair_counts[index]) or len(identities) != len(targets)
                or actual != self.order.references(index) or any(s["split"] != "train" for s in identities)
                or not torch.equal(groups, anchor_groups(identities, targets.device))):
            raise ValueError("logical batch, anchor groups, order or split changed")
        batch_digest = hashlib.sha256("\n".join(s["pair_digest"] for s in identities).encode()).hexdigest()
        self.in_update = True
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        metrics = accumulate_loss(self.model, condition, targets, groups,
                                  microbatch=self.recipe["predictor_microbatch"],
                                  loss_rule=self.recipe["loss"], hook=hook)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("missing or nonfinite trainable gradient")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.recipe["optimizer"]["gradient_clip"],
                                             error_if_nonfinite=True)
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step(); self.scheduler.step()
        self.global_step += 1; self.pairs_seen += len(targets)
        self.loss_sum += metrics["loss"]*len(targets)
        self.sample_chain = hashlib.sha256(bytes.fromhex(self.sample_chain)+bytes.fromhex(batch_digest)).hexdigest()
        self.in_update = False
        return {"global_step": self.global_step, "plan_index": index, "pairs": len(targets),
                "pairs_seen": self.pairs_seen, "batch_digest": batch_digest,
                "sample_chain": self.sample_chain, **metrics, "learning_rate": lr,
                "gradient_norm": float(norm)}


def verify_recipe(path, root):
    from dataclasses import asdict

    recipe = read_sealed(path)
    if (recipe["artifact_type"] != "eventground_wam.jepa_cpt_recipe" or recipe["epochs"] != 1
            or recipe["order_rule"] != order_rule() or recipe["checkpoint_selection"] != "fixed-final-update"
            or recipe["predictor_microbatch"] != 10 or recipe["feature_microbatch"] != 4
            or recipe["loss"] != LOSS_RULE or recipe["architecture"] != asdict(JepaPredictorConfig())
            or recipe["target_cache"] != {"mode": "online-canonical-fp32-pool-ram", "maximum_bytes": 4*1024**3,
                                          "persistent": False, "reset_on_pool_change": True}
            or recipe["pair_presentations"] != 8221365 or recipe["total_updates"] != 221633):
        raise ValueError("unsupported frozen JEPA recipe")
    for name, digest in recipe["input_sha256"].items():
        if sha256_file(root/name) != digest:
            raise ValueError(f"frozen recipe input changed: {name}")
    return recipe
