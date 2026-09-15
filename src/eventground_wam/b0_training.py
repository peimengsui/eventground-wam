"""Full-data B0 batching, recipe validation, and crash-consistent recovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from eventground_wam.b0_data import B0StreamingDataset, collate_b0_samples
from eventground_wam.b0_model import (
    QWEN_REVISION,
    QwenB0InputBuilder,
    b0_trainable_state_dict,
    load_b0_trainable_state_dict,
    load_pinned_qwen_processor,
    validate_b0_trainable_parameters,
)
from eventground_wam.b0_overfit import validate_training_gradients
from eventground_wam.demonstration_contracts import (
    canonical_json_bytes,
    contract_digest,
    seal_contract,
)

TRAINING_FORMAT = "eventground-wam-b0-full-training-state-v1"
POLICY_FORMAT = "eventground-wam-b0-full-policy-v1"


class TrainingContractError(ValueError):
    """A non-retryable recipe, identity, or saved-state failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("digest") != contract_digest(value):
        raise TrainingContractError(f"invalid artifact digest: {path}")
    return value


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text())
    locked = {
        "schema_version": 1,
        "epochs": 5,
        "qwen_revision": QWEN_REVISION,
        "objective": "continuous-l1-plus-gripper-bce-v1",
        "gripper_bce_weight": 1.0,
        "lora_rank": 16,
        "action_chunk_length": 8,
        "action_dimension": 7,
        "sample_order": "frozen-episode-shuffle-and-pcg64-anchor-shuffle-v1",
        "drop_last": False,
        "augmentation": "none",
        "validation": "all-frozen-validation-chunks-after-each-epoch",
        "checkpoint_selection": "minimum-validation-hybrid-objective-earliest-epoch-tie",
        "training_sample_count": 1_715_954,
    }
    for key, expected in locked.items():
        if recipe.get(key) != expected:
            raise TrainingContractError(f"recipe {key} must be {expected!r}")
    for key in (
        "batch_size", "workers", "prefetch_factor", "maximum_cached_episode_frames",
        "checkpoint_interval_seconds", "checkpoint_interval_steps", "checkpoint_keep_last",
        "log_interval_steps",
    ):
        if type(recipe.get(key)) is not int or recipe[key] <= 0:
            raise TrainingContractError(f"{key} must be a positive integer")
    if recipe["checkpoint_keep_last"] < 2:
        raise TrainingContractError("retain at least two recovery checkpoints")
    for key in ("seed", "projector_initialization_seed", "decoder_initialization_seed"):
        if type(recipe.get(key)) is not int or not 0 <= recipe[key] < 2**32:
            raise TrainingContractError(f"{key} must fit uint32")
    for key in (
        "decoder_projector_learning_rate", "lora_learning_rate", "adam_epsilon",
        "maximum_gradient_norm", "maximum_projected_training_hours", "pilot_projection_margin",
    ):
        if not isinstance(recipe.get(key), (float, int)) or not (
            math.isfinite(recipe[key]) and recipe[key] > 0
        ):
            raise TrainingContractError(f"{key} must be finite and positive")
    if recipe["pilot_projection_margin"] < 1:
        raise TrainingContractError("pilot projection margin must be at least one")
    if not 0 <= recipe["warmup_fraction"] < 1:
        raise TrainingContractError("warmup fraction must be in [0,1)")
    if not 0 < recipe["minimum_learning_rate_ratio"] <= 1:
        raise TrainingContractError("minimum LR ratio must be in (0,1]")
    if len(recipe["adam_betas"]) != 2 or not all(0 <= x < 1 for x in recipe["adam_betas"]):
        raise TrainingContractError("invalid Adam betas")
    if not math.isfinite(recipe["weight_decay"]) or recipe["weight_decay"] < 0:
        raise TrainingContractError("invalid weight decay")
    return recipe


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise TrainingContractError("CUDA device count changed since checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def learning_rate_factor(step: int, total_steps: int, recipe: Mapping[str, Any]) -> float:
    """LambdaLR factor for the next update, including warmup from the first update."""

    warmup = max(1, int(total_steps * recipe["warmup_fraction"]))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    minimum = recipe["minimum_learning_rate_ratio"]
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            torch.save(cpu_tree(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointStore:
    """Publish the tensor file before its sealed commit record; ignore incomplete writes."""

    def __init__(self, root: Path, identity_digest: str, keep: int = 3) -> None:
        self.root = root
        self.identity_digest = identity_digest
        self.keep = keep
        root.mkdir(parents=True, exist_ok=True)
        self.recovery_notes: list[str] = []

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("identity_digest") != self.identity_digest:
            raise TrainingContractError("checkpoint identity mismatch")
        name = f"checkpoint-{payload['global_step']:012d}-{uuid.uuid4().hex}"
        tensor_path = self.root / f"{name}.pt"
        digest = atomic_torch_save(tensor_path, payload)
        record = seal_contract({
            "artifact_type": "eventground_wam.b0_recovery_checkpoint",
            "identity_digest": self.identity_digest,
            "global_step": payload["global_step"],
            "epoch": payload["epoch"],
            "next_batch": payload["next_batch"],
            "checkpoint": tensor_path.name,
            "sha256": digest,
        })
        atomic_json(self.root / f"{name}.json", record)
        atomic_json(self.root / "latest.json", record)
        # Only prune committed checkpoints after a new durable commit exists.
        records = sorted(self.root.glob("checkpoint-*.json"),
                         key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for old in records[self.keep:]:
            old.with_suffix(".pt").unlink(missing_ok=True)
            old.unlink()
        return record

    def load(self) -> dict[str, Any] | None:
        records = sorted(self.root.glob("checkpoint-*.json"),
                         key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for path in records:
            try:
                record = read_sealed(path)
            except (ValueError, OSError) as error:
                self.recovery_notes.append(f"ignored invalid commit {path.name}: {error}")
                continue
            if record["identity_digest"] != self.identity_digest:
                raise TrainingContractError("refusing to resume a different training identity")
            checkpoint = self.root / record["checkpoint"]
            if checkpoint.parent != self.root or checkpoint.suffix != ".pt":
                raise TrainingContractError("unsafe checkpoint path")
            try:
                if sha256_file(checkpoint) != record["sha256"]:
                    raise ValueError("checkpoint SHA-256 mismatch")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except (ValueError, OSError, RuntimeError, EOFError) as error:
                self.recovery_notes.append(f"ignored damaged checkpoint {path.name}: {error}")
                continue
            if (payload.get("format") != TRAINING_FORMAT
                    or payload.get("identity_digest") != self.identity_digest
                    or any(payload.get(key) != record[key]
                           for key in ("global_step", "epoch", "next_batch"))):
                raise TrainingContractError("invalid recovery tensor envelope")
            return payload
        if records:
            raise TrainingContractError("no intact committed recovery checkpoint remains")
        return None


def build_batch_plan(
    episodes: Any, *, batch_size: int, seed: int, epoch: int, shuffle: bool,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Each valid chunk appears once; an episode's short final batch is preserved."""

    plan = []
    for episode in episodes:
        anchors = np.arange(max(0, episode.length - 7))
        if shuffle:
            payload = f"{seed}|{epoch}|{episode.episode_index}".encode()
            episode_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            anchors = np.random.Generator(np.random.PCG64(episode_seed)).permutation(anchors)
        for start in range(0, len(anchors), batch_size):
            plan.append((episode.episode_index, tuple(int(x) for x in
                                                    anchors[start:start + batch_size])))
    return tuple(plan)


def plan_digest(plan: Any) -> str:
    digest = hashlib.sha256()
    for episode_index, anchors in plan:
        digest.update(canonical_json_bytes([episode_index, anchors]) + b"\n")
    return digest.hexdigest()


class EpochBatches(Dataset):
    """Ordered map-style batches with one decoded episode cached per CPU worker.

    Independent batch keys make worker prefetch disposable on interruption: the
    trainer saves the next consumed batch, not the worker's advanced iterator.
    """

    def __init__(
        self, dataset: B0StreamingDataset, plan: Any, *, cache_dir: str,
        maximum_cached_episode_frames: int,
    ) -> None:
        self.dataset = dataset
        self.plan = plan
        self.cache_dir = cache_dir
        self.maximum_frames = maximum_cached_episode_frames
        self.episodes = {episode.episode_index: episode for episode in dataset.episodes}
        if any(episode.length > self.maximum_frames for episode in self.episodes.values()):
            raise TrainingContractError("episode exceeds declared CPU cache bound")
        self._cached_episode: int | None = None
        self._samples: Any = None
        self._builder: QwenB0InputBuilder | None = None

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, anchors = self.plan[index]
        if self._cached_episode != episode_index:
            self._samples = None
            self._samples = list(self.dataset.iter_episode_samples(self.episodes[episode_index]))
            self._cached_episode = episode_index
        batch = collate_b0_samples([self._samples[anchor] for anchor in anchors])
        if self._builder is None:
            torch.set_num_threads(1)
            self._builder = QwenB0InputBuilder(load_pinned_qwen_processor(
                cache_dir=self.cache_dir, local_files_only=True,
            ))
        encoding = self._builder(batch.instructions, list(batch.front_rgb), list(batch.wrist_rgb))
        return {
            "index": index,
            "inputs": {key: value.to(torch.bfloat16) if value.is_floating_point() else value
                       for key, value in encoding.items()},
            "proprioception": torch.from_numpy(batch.proprioception),
            "targets": torch.from_numpy(batch.actions),
            "sample_count": len(anchors),
            "identity_digest": hashlib.sha256(canonical_json_bytes(
                [identity.digest for identity in batch.identities],
            )).hexdigest(),
        }


def move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any]:
    return (
        {key: value.to(device, non_blocking=True) for key, value in batch["inputs"].items()},
        batch["proprioception"].to(device, non_blocking=True),
        batch["targets"].to(device, non_blocking=True),
    )


class TrainingEngine:
    """One acknowledged batch per optimizer update; all continuation state is explicit."""

    def __init__(
        self, policy: Any, exact_lora_names: tuple[str, ...], recipe: dict[str, Any],
        *, identity_digest: str, steps_per_epoch: int, device: torch.device,
    ) -> None:
        self.policy = policy
        self.exact_lora_names = exact_lora_names
        self.recipe = recipe
        self.identity_digest = identity_digest
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.names = validate_b0_trainable_parameters(policy, allowed_exact_names=exact_lora_names)
        parameters = dict(policy.named_parameters())
        exact = set(exact_lora_names)
        self.parameters = [parameters[name] for name in self.names]
        self.optimizer = torch.optim.AdamW([
            {"params": [parameters[n] for n in self.names if n not in exact],
             "lr": recipe["decoder_projector_learning_rate"]},
            {"params": [parameters[n] for n in self.names if n in exact],
             "lr": recipe["lora_learning_rate"]},
        ], betas=tuple(recipe["adam_betas"]), eps=recipe["adam_epsilon"],
            weight_decay=recipe["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: learning_rate_factor(
                step, steps_per_epoch * recipe["epochs"], recipe,
            ),
        )
        self.epoch = 0
        self.next_batch = 0
        self.global_step = 0
        self.samples_seen = 0
        self.sample_chain = "0" * 64
        self.epoch_loss_sum = 0.0
        self.epoch_sample_count = 0
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None

    def step(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        if batch["index"] != self.next_batch:
            raise TrainingContractError("batch order disagrees with acknowledged cursor")
        inputs, state, targets = move_batch(batch, self.device)
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.policy(inputs, state, normalized_action_targets=targets)
        if output.loss is None or not torch.isfinite(output.loss):
            raise TrainingContractError("nonfinite training objective")
        output.loss.backward()
        if self.global_step == 0 or self.global_step % 1000 == 0:
            validate_training_gradients(self.policy, trainable_names=self.names,
                                        exact_lora_parameter_names=self.exact_lora_names)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters,
                                             self.recipe["maximum_gradient_norm"],
                                             error_if_nonfinite=True)
        rates = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.step()
        self.scheduler.step()
        # Cursor only advances after a complete optimizer/scheduler update.
        self.global_step += 1
        self.next_batch += 1
        count = int(batch["sample_count"])
        self.samples_seen += count
        loss = float(output.loss.detach())
        self.epoch_loss_sum += loss * count
        self.epoch_sample_count += count
        self.sample_chain = hashlib.sha256(
            bytes.fromhex(self.sample_chain) + bytes.fromhex(batch["identity_digest"]),
        ).hexdigest()
        return {"epoch": self.epoch + 1, "global_step": self.global_step,
                "next_batch": self.next_batch, "samples_seen": self.samples_seen,
                "loss": loss, "gradient_norm": float(norm), "learning_rates": rates}

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": TRAINING_FORMAT,
            "identity_digest": self.identity_digest,
            "state_dict": b0_trainable_state_dict(
                self.policy, allowed_exact_names=self.exact_lora_names,
            ),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": capture_rng(),
            "trainable_names": list(self.names),
            "epoch": self.epoch, "next_batch": self.next_batch,
            "global_step": self.global_step, "samples_seen": self.samples_seen,
            "sample_chain": self.sample_chain,
            "epoch_loss_sum": self.epoch_loss_sum,
            "epoch_sample_count": self.epoch_sample_count,
            "history": self.history, "best": self.best,
        }

    def restore(self, value: Mapping[str, Any]) -> None:
        if (value["format"] != TRAINING_FORMAT
                or value["identity_digest"] != self.identity_digest
                or value["trainable_names"] != list(self.names)):
            raise TrainingContractError("recovery identity or parameter allowlist changed")
        epoch, cursor = value["epoch"], value["next_batch"]
        if (not 0 <= epoch <= self.recipe["epochs"]
                or not 0 <= cursor <= self.steps_per_epoch
                or value["global_step"] != epoch * self.steps_per_epoch + cursor
                or (epoch == self.recipe["epochs"] and cursor != 0)):
            raise TrainingContractError("invalid saved epoch/update cursor")
        load_b0_trainable_state_dict(self.policy, value["state_dict"],
                                    allowed_exact_names=self.exact_lora_names)
        self.optimizer.load_state_dict(value["optimizer"])
        self.scheduler.load_state_dict(value["scheduler"])
        for name in ("epoch", "next_batch", "global_step", "samples_seen", "sample_chain",
                     "epoch_loss_sum", "epoch_sample_count", "history", "best"):
            setattr(self, name, value[name])
        restore_rng(value["rng"])

    def finish_epoch(self, validation: dict[str, Any]) -> None:
        if self.next_batch != self.steps_per_epoch:
            raise TrainingContractError("cannot finish an incomplete training epoch")
        self.history.append({"epoch": self.epoch + 1, "global_step": self.global_step,
                             "training_samples": self.epoch_sample_count,
                             "training_objective": self.epoch_loss_sum / self.epoch_sample_count,
                             "validation": validation})
        self.epoch += 1
        self.next_batch = 0
        self.epoch_loss_sum = 0.0
        self.epoch_sample_count = 0


def evaluate_validation(policy: Any, batches: Any, device: torch.device) -> dict[str, Any]:
    count = 0
    loss_sum = mae_sum = continuous_sum = 0.0
    gripper_correct = 0
    policy.eval()
    with torch.no_grad():
        for batch in batches:
            inputs, state, targets = move_batch(batch, device)
            result = policy(inputs, state, normalized_action_targets=targets)
            if result.loss is None or not torch.isfinite(result.loss):
                raise TrainingContractError("nonfinite validation objective")
            size = int(batch["sample_count"])
            error = (result.normalized_actions.float() - targets.float()).abs()
            loss_sum += float(result.loss) * size
            mae_sum += float(error.sum())
            continuous_sum += float(error[..., :6].sum())
            gripper_correct += int((result.normalized_actions[..., 6] == targets[..., 6]).sum())
            count += size
    if count == 0:
        raise TrainingContractError("validation stream is empty")
    return {"sample_count": count, "objective": loss_sum / count,
            "decoded_action_mae": mae_sum / (count * 8 * 7),
            "continuous_action_mae": continuous_sum / (count * 8 * 6),
            "gripper_accuracy": gripper_correct / (count * 8)}
