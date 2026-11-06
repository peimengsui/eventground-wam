"""One coupled logical update, with complete same-device recovery state."""
from __future__ import annotations

import hashlib

import torch

from eventground_wam.b0_training import (
    TrainingContractError,
    TrainingEngine,
    cpu_tree,
    learning_rate_factor,
    move_batch,
)
from eventground_wam.coupled_policy import (
    DESIGN_DIGEST,
    CANDIDATE_MICROBATCH,
    FULL_BATCH_EXECUTION,
    INTERFACE_VERSION,
    LOSS_WEIGHTS,
    METHODS,
    coupled_full_batch_backward,
    coupled_parameter_groups,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes

COUPLED_FORMAT = 'eventground-wam-coupled-training-state-v2'


class CoupledTrainingEngine(TrainingEngine):
    def __init__(self, model, exact_lora_names, predictor, recipe, *,
                 identity_digest, steps_per_epoch, device, microbatch=None):
        if microbatch is not None:
            raise TrainingContractError('this execution recipe requires the whole logical future batch')
        super().__init__(model.policy, exact_lora_names, recipe,
                         identity_digest=identity_digest, steps_per_epoch=steps_per_epoch, device=device)
        self.model, self.predictor = model, predictor
        self.method, self.loss_weight = METHODS[model.arm], LOSS_WEIGHTS[model.arm]
        self.microbatch = microbatch
        groups = coupled_parameter_groups(model, predictor, exact_lora_names)
        self.parameters = [p for group in groups for p in group['params']]
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(recipe['adam_betas']),
                                          eps=recipe['adam_epsilon'], weight_decay=recipe['weight_decay'])
        total = min(steps_per_epoch*recipe['epochs'], recipe['maximum_updates'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: learning_rate_factor(step, total, recipe))
        self.future_pairs_seen = 0
        self.future_chain = '0'*64
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
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Once work starts, any failure leaves this object unusable until restore.
        self._in_update = True
        result = self.model.forward_with_features(inputs, state, future.current_images,
                                                  normalized_action_targets=targets)
        metrics = coupled_full_batch_backward(self.model, result, self.predictor, future,
                                              future_targets, hook=hook)
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
            raise TrainingContractError('cannot commit a partial coupled update')
        # The full predictor owns all shared future-core tensors exactly once.
        return cpu_tree({**super().snapshot(), 'coupled_format': COUPLED_FORMAT,
                'design_digest': DESIGN_DIGEST, 'interface': INTERFACE_VERSION,
                'arm': self.model.arm, 'method': self.method,
                'loss_weight': self.loss_weight, 'microbatch': self.microbatch,
                'backward_execution': FULL_BATCH_EXECUTION,
                'candidate_microbatch': CANDIDATE_MICROBATCH,
                'predictor': self.predictor.state_dict(),
                'coupling': self.model.coupling.state_dict(),
                'future_pairs_seen': self.future_pairs_seen, 'future_chain': self.future_chain})

    def restore(self, value):
        if (value.get('coupled_format') != COUPLED_FORMAT
                or value.get('design_digest') != DESIGN_DIGEST
                or value.get('interface') != INTERFACE_VERSION
                or value.get('arm') != self.model.arm
                or value.get('method') != self.method
                or value.get('loss_weight') != self.loss_weight
                or value.get('backward_execution') != FULL_BATCH_EXECUTION
                or value.get('candidate_microbatch') != CANDIDATE_MICROBATCH
                or value.get('microbatch') != self.microbatch):
            raise TrainingContractError('coupled recovery recipe changed')
        for key, module in [('predictor', self.predictor), ('coupling', self.model.coupling)]:
            reference = module.state_dict()
            supplied = value.get(key, {})
            if set(reference) != set(supplied) or any(
                v.shape != reference[n].shape or v.dtype != reference[n].dtype
                or not torch.isfinite(v).all() for n, v in supplied.items()
            ):
                raise TrainingContractError(f'invalid coupled recovery tensors: {key}')
        if (type(value.get('future_pairs_seen')) is not int or value['future_pairs_seen'] < 0
                or len(bytes.fromhex(value.get('future_chain', ''))) != 32):
            raise TrainingContractError('invalid coupled exposure counters')
        super().restore(value)
        self.predictor.load_state_dict(value['predictor'], strict=True)
        self.model.coupling.load_state_dict(value['coupling'], strict=True)
        self.future_pairs_seen = value['future_pairs_seen']
        self.future_chain = value['future_chain']
        self._in_update = False
