"""Versioned all-L1 continuation: matched mixed batches, child state and budget."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from eventground_wam.b0_data import collate_b0_samples
from eventground_wam.b0_diagnostics import gradient_statistics, tensor_tree_digest
from eventground_wam.b0_model import load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder
from eventground_wam.b0_training import (
    TrainingContractError,
    atomic_json,
    build_batch_plan,
    cpu_tree,
    move_batch,
    plan_digest,
    read_sealed,
)
from eventground_wam.continuation_freeze import batches_digest, exposure_digest, mix_batches
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract

OBJECTIVE = "all-action-l1-binary-gripper-v2"
EXPERIMENT = "week4-oft-all-l1-v2"


def continuation_plans(train: Any, parent_manifest: Any, audit: Any) -> tuple[Any, Any]:
    plan = build_batch_plan(train.ordered_episodes(1), batch_size=32, seed=17,
                            epoch=1, shuffle=True)
    if plan_digest(plan) != parent_manifest["epoch_plan_digests"][1]:
        raise TrainingContractError("parent source epoch plan changed")
    blocked = [[(1, episode, anchor) for anchor in anchors]
               for episode, anchors in plan[13063:13063 + 15000]]
    mixed = mix_batches(blocked)
    expected = next(r for r in audit["checkpoints"] if r["additional_updates"] == 15000)
    if (len(blocked) != 15000 or sum(map(len, blocked)) != 435945
            or batches_digest(blocked) != expected["arm_a_plan_digest"]
            or batches_digest(mixed) != expected["arm_b_plan_digest"]
            or exposure_digest(mixed) != expected["shared_exposure_multiset_digest"]
            or exposure_digest(blocked) != exposure_digest(mixed)
            or list(map(len, blocked)) != list(map(len, mixed))):
        raise TrainingContractError("continuation sample identities differ from predeclared plans")
    return blocked, mixed


class MixedBatches(Dataset):
    """Pure batch keys with an LRU decoded-frame cache bounded per worker."""

    def __init__(self, dataset: Any, plan: Any, cache_dir: str, maximum_frames: int = 8192):
        self.dataset, self.plan, self.cache_dir = dataset, plan, cache_dir
        self.maximum_frames = maximum_frames
        self.episodes = {e.episode_index: e for e in dataset.episodes}
        self.cache: OrderedDict[int, Any] = OrderedDict()
        self.cached_frames = 0
        self.builder = None
        for batch in plan:
            if not batch or len(batch) > 32 or len(set(batch)) != len(batch):
                raise TrainingContractError("invalid mixed batch tokens")
            for epoch, episode, anchor in batch:
                if (epoch != 1 or episode not in self.episodes
                        or not 0 <= anchor < self.episodes[episode].length - 7
                        or self.episodes[episode].length > maximum_frames):
                    raise TrainingContractError("mixed token lies outside frozen train chunks")

    def __len__(self):
        return len(self.plan)

    def sample(self, episode: int, anchor: int):
        if episode not in self.cache:
            size = self.episodes[episode].length
            while self.cache and self.cached_frames + size > self.maximum_frames:
                evicted, _ = self.cache.popitem(last=False)
                self.cached_frames -= self.episodes[evicted].length
            self.cache[episode] = list(self.dataset.iter_episode_samples(self.episodes[episode]))
            self.cached_frames += size
        self.cache.move_to_end(episode)
        return self.cache[episode][anchor]

    def __getitem__(self, index: int) -> dict[str, Any]:
        tokens = self.plan[index]
        samples = [self.sample(episode, anchor) for _, episode, anchor in tokens]
        batch = collate_b0_samples(samples)
        if self.builder is None:
            torch.set_num_threads(1)
            self.builder = OFTInputBuilder(load_pinned_qwen_processor(
                cache_dir=self.cache_dir, local_files_only=True))
        inputs = self.builder(batch.instructions, list(batch.front_rgb), list(batch.wrist_rgb))
        return {"index": index,
                "inputs": {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                           for k, v in inputs.items()},
                "proprioception": torch.from_numpy(batch.proprioception),
                "targets": torch.from_numpy(batch.actions), "sample_count": len(tokens),
                "unique_task_count": len(set(batch.instructions)),
                "token_digest": batches_digest([tokens]),
                "identity_digest": hashlib.sha256(canonical_json_bytes(
                    [tokens, [identity.digest for identity in batch.identities]])).hexdigest()}


def mixed_loader(dataset, plan, cache_dir, recipe, *, start=0):
    batches = MixedBatches(dataset, plan, str(cache_dir), recipe["maximum_cached_episode_frames"])
    return DataLoader(batches, batch_size=None, sampler=range(start, len(plan)),
                      num_workers=recipe["workers"], prefetch_factor=recipe["prefetch_factor"],
                      multiprocessing_context="spawn", pin_memory=True,
                      generator=torch.Generator().manual_seed(recipe["seed"]), timeout=300)


def adopt_parent(engine: Any, parent: Any) -> dict[str, Any]:
    """Keep model/Adam/RNG; start child counters and constant-LR scheduler at zero."""
    if (parent["global_step"] != 72000 or parent["epoch"] != 1
            or parent["next_batch"] != 13063 or parent["samples_seen"] != 2096581
            or parent["trainable_names"] != list(engine.names)):
        raise TrainingContractError("wrong parent recovery cursor or trainable names")
    value = cpu_tree(parent)
    before = tensor_tree_digest(value["optimizer"]["state"])
    value.update(identity_digest=engine.identity_digest, epoch=0, next_batch=0, global_step=0,
                 samples_seen=0, sample_chain="0" * 64, epoch_loss_sum=0., epoch_sample_count=0,
                 history=[], best=None, scheduler=cpu_tree(engine.scheduler.state_dict()))
    for group in value["optimizer"]["param_groups"]:
        group["lr"] = group["initial_lr"] = 1e-5
    engine.restore(value)
    after = tensor_tree_digest(engine.optimizer.state_dict()["state"])
    if after != before:
        raise TrainingContractError("parent Adam state changed during adoption")
    return {"model_reset": False, "gripper_head_reset": False, "adam_state_reset": False,
            "adam_state_digest": before, "parent_rng_restored": True,
            "learning_rates": [g["lr"] for g in engine.optimizer.param_groups]}


def action_metrics(output: Any, targets: torch.Tensor) -> dict[str, float]:
    raw = output.raw_action_predictions.detach().float()
    error = (raw - targets.detach().float()).abs()
    decoded = output.normalized_actions.detach().float()
    return {"motion_l1": float(error[..., :6].mean()),
            "gripper_l1": float(error[..., 6].mean()),
            "all_action_l1": float(error.mean()),
            "gripper_accuracy": float((decoded[..., 6] == targets[..., 6]).float().mean()),
            "decoded_action_mae": float((decoded - targets.float()).abs().mean()),
            "raw_gripper_min": float(raw[..., 6].min()),
            "raw_gripper_max": float(raw[..., 6].max())}


def validation_metrics(policy: Any, loader: Any, device: Any) -> dict[str, Any]:
    # Preserve train/eval modes and RNG; validation must not change continuation state.
    from eventground_wam.b0_training import capture_rng, restore_rng

    rng = capture_rng()
    modes = {name: module.training for name, module in policy.named_modules()}
    totals: dict[str, float] = {}
    count = 0
    try:
        policy.eval()
        with torch.no_grad():
            for batch in loader:
                inputs, state, targets = move_batch(batch, device)
                output = policy(inputs, state, normalized_action_targets=targets)
                metrics = action_metrics(output, targets)
                if not torch.isfinite(output.loss):
                    raise TrainingContractError("nonfinite validation loss")
                n = batch["sample_count"]
                for name, value in metrics.items():
                    if name not in {"raw_gripper_min", "raw_gripper_max"}:
                        totals[name] = totals.get(name, 0.) + n * value
                count += n
    finally:
        restore_rng(rng)
        for name, module in policy.named_modules():
            module.training = modes[name]
    if not count:
        raise TrainingContractError("empty validation panel")
    return {"sample_count": count, **{k: v / count for k, v in totals.items()}}


def l1_gradient_panel(policy, loader, device):
    from eventground_wam.b0_training import capture_rng, restore_rng

    rng = capture_rng()
    modes = {name: module.training for name, module in policy.named_modules()}
    parameters = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    names, tensors = zip(*parameters, strict=True)
    before = tensor_tree_digest(policy.state_dict())
    gradients_before = [p.grad for _, p in parameters]
    rows = []
    try:
        policy.train()
        for batch in loader:
            inputs, state, targets = move_batch(batch, device)
            out = policy(inputs, state, normalized_action_targets=targets)
            motion = torch.autograd.grad(out.continuous_l1_loss * (6 / 7), tensors,
                                         retain_graph=True, allow_unused=True)
            gripper = torch.autograd.grad(out.gripper_l1_loss / 7, tensors, allow_unused=True)
            statistics = gradient_statistics(names, motion, gripper)
            # The shared numeric helper has historical BCE names; relabel explicitly.
            rows.append({group: {key.replace("bce", "gripper_l1"): value
                                 for key, value in values.items()}
                         for group, values in statistics.items()})
            del out, motion, gripper, inputs, state, targets
    finally:
        restore_rng(rng)
        for name, module in policy.named_modules():
            module.training = modes[name]
    if before != tensor_tree_digest(policy.state_dict()):
        raise TrainingContractError("gradient panel changed model tensors")
    if any(p.grad is not old for (_, p), old in zip(parameters, gradients_before, strict=True)):
        raise TrainingContractError("gradient panel populated parameter gradients")
    import statistics

    ratios = [r["backbone_lora"]["gripper_l1_motion_ratio"] for r in rows]
    return {"batch_count": len(rows), "rows": rows, "state_preserved": True,
            "weighted_loss_terms": {"motion": 6 / 7, "gripper": 1 / 7},
            "backbone_median_gripper_motion_ratio": statistics.median(
                [x for x in ratios if x is not None]),
            "scope": "Endpoint gradients before clipping, not Adam updates or causal evidence"}


class ActiveBudget:
    """Crash-persistent heartbeat ledger; unfinished sessions charged 10 s extra."""

    def __init__(self, path: Path, identity: str):
        self.path, self.identity = path, identity
        saved = read_sealed(path) if path.exists() else None
        if saved and saved["identity"] != identity:
            raise TrainingContractError("budget ledger identity changed")
        self.prior = (saved["active_seconds"] + (10. if saved["running"] else 0.)) if saved else 0.
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.failure: Exception | None = None
        self.write(True)
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    @property
    def seconds(self):
        if self.failure:
            raise TrainingContractError(f"budget persistence failed: {self.failure}")
        return self.prior + time.monotonic() - self.started

    def write(self, running):
        atomic_json(self.path, seal_contract({"identity": self.identity, "running": running,
                                              "active_seconds": self.seconds}))

    def heartbeat(self):
        while not self.stop.wait(10):
            try:
                self.write(True)
            except (OSError, ValueError, TrainingContractError) as error:
                self.failure = error
                return

    def close(self):
        self.stop.set()
        self.thread.join()
        self.write(False)
