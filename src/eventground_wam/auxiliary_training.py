"""Matched OFT/auxiliary updates with atomic, complete continuation state."""
from __future__ import annotations

import hashlib
import math

import torch

from eventground_wam.auxiliary_losses import joint_backward
from eventground_wam.auxiliary_policy import INTERFACE_VERSION, policy_with_features
from eventground_wam.b0_training import (
    TrainingContractError,
    TrainingEngine,
    cpu_tree,
    learning_rate_factor,
    move_batch,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes

AUXILIARY_FORMAT = "eventground-wam-auxiliary-training-state-v1"


class AuxiliaryTrainingEngine(TrainingEngine):
    def __init__(self, policy, exact_lora_names, predictor, method, recipe, *,
                 identity_digest, steps_per_epoch, device, loss_weight, microbatch=10):
        if method not in {"pixel", "jepa"} or not math.isfinite(loss_weight) or loss_weight <= 0:
            raise TrainingContractError("invalid frozen auxiliary objective")
        if type(microbatch) is not int or microbatch < 1:
            raise TrainingContractError("invalid predictor microbatch")
        super().__init__(policy, exact_lora_names, recipe, identity_digest=identity_digest,
                         steps_per_epoch=steps_per_epoch, device=device)
        self.predictor, self.method = predictor, method
        self.loss_weight, self.microbatch = loss_weight, microbatch
        future = list(predictor.parameters())
        if not future or not all(p.requires_grad for p in future):
            raise TrainingContractError("all future-module parameters must train")
        if set(map(id, future)) & set(map(id, policy.parameters())):
            raise TrainingContractError("future module cannot own policy parameters")
        groups = self.optimizer.param_groups
        groups[0]["lr"] = recipe["decoder_projector_learning_rate"]
        groups[1]["lr"] = recipe["lora_learning_rate"]
        self.optimizer = torch.optim.AdamW([
            {"params": groups[0]["params"], "lr": groups[0]["lr"]},
            {"params": groups[1]["params"], "lr": groups[1]["lr"]},
            {"params": future, "lr": recipe["auxiliary_learning_rate"]},
        ], betas=tuple(recipe["adam_betas"]), eps=recipe["adam_epsilon"],
            weight_decay=recipe["weight_decay"])
        total = min(steps_per_epoch*recipe["epochs"], recipe["maximum_updates"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: learning_rate_factor(step, total, recipe))
        self.parameters += future
        self.future_pairs_seen = 0
        self.future_chain = "0"*64
        self._in_update = False

    def step(self, batch, future_targets, *, hook=None):
        if self._in_update:
            raise TrainingContractError("partial update requires recovery from a committed snapshot")
        if batch["index"] != self.next_batch:
            raise TrainingContractError("batch order disagrees with acknowledged cursor")
        identities = batch["future_identities"]
        future = batch["future_inputs"]
        tokens = batch["tokens"]
        if (len(tokens) != batch["sample_count"] or any(t[0] != self.epoch for t in tokens)
                or len(identities) != len(future.horizons)
                or len(identities) != len(future_targets)
                or len(identities) != int(batch["future_mask"].sum())
                or any(i["split"] != "train" for i in identities)):
            raise TrainingContractError("auxiliary batch cardinality, epoch or split mismatch")
        for row, anchor, horizon in zip(identities, future.anchor_indices.tolist(),
                                       future.horizons.tolist(), strict=True):
            if (not 0 <= anchor < len(tokens) or row["episode_index"] != tokens[anchor][1]
                    or row["anchor"] != tokens[anchor][2] or row["horizon"] != horizon
                    or row["target"] != row["anchor"]+horizon):
                raise TrainingContractError("future identity disagrees with action anchor")
        digest = hashlib.sha256(canonical_json_bytes(list(identities))).hexdigest()
        if digest != batch["future_digest"]:
            raise TrainingContractError("future supervision identity changed")
        inputs, state, targets = move_batch(batch, self.device)
        future = future.to(self.device)
        future_targets = future_targets.to(self.device)
        self.policy.train()
        self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Once work starts, any failure leaves this object unusable until restore.
        self._in_update = True
        output, features = policy_with_features(self.policy, inputs, state, targets)
        metrics = joint_backward(output, features, self.predictor, self.method, future,
                                 future_targets, microbatch=self.microbatch,
                                 loss_weight=self.loss_weight, hook=hook)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters):
            raise TrainingContractError("missing or nonfinite trainable gradient")
        norm = torch.nn.utils.clip_grad_norm_(self.parameters,
            self.recipe["maximum_gradient_norm"], error_if_nonfinite=True)
        rates = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        self.next_batch += 1
        count = int(batch["sample_count"])
        self.samples_seen += count
        self.epoch_sample_count += count
        self.epoch_loss_sum += metrics["action_loss"]*count
        self.future_pairs_seen += len(identities)
        self.sample_chain = hashlib.sha256(bytes.fromhex(self.sample_chain)
            + bytes.fromhex(batch["identity_digest"])).hexdigest()
        self.future_chain = hashlib.sha256(bytes.fromhex(self.future_chain)
            + bytes.fromhex(digest)).hexdigest()
        self._in_update = False
        return {**metrics, "epoch": self.epoch+1, "global_step": self.global_step,
                "next_batch": self.next_batch, "samples_seen": self.samples_seen,
                "future_pairs_seen": self.future_pairs_seen,
                "gradient_norm": float(norm), "learning_rates": rates}

    def snapshot(self):
        if self._in_update:
            raise TrainingContractError("cannot commit a partial auxiliary update")
        return {**super().snapshot(), "auxiliary_format": AUXILIARY_FORMAT,
                "interface": INTERFACE_VERSION, "method": self.method,
                "loss_weight": self.loss_weight, "microbatch": self.microbatch,
                "predictor": cpu_tree(self.predictor.state_dict()),
                "future_pairs_seen": self.future_pairs_seen, "future_chain": self.future_chain}

    def restore(self, value):
        if (value.get("auxiliary_format") != AUXILIARY_FORMAT
                or value.get("interface") != INTERFACE_VERSION or value.get("method") != self.method
                or value.get("loss_weight") != self.loss_weight
                or value.get("microbatch") != self.microbatch):
            raise TrainingContractError("auxiliary recovery recipe changed")
        super().restore(value)
        self.predictor.load_state_dict(value["predictor"], strict=True)
        self.future_pairs_seen = value["future_pairs_seen"]
        self.future_chain = value["future_chain"]
        self._in_update = False
