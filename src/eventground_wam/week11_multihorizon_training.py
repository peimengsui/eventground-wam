"""Versioned M4 continuation: candidate-only routing and complete recovery."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import replace

import torch

from eventground_wam.b0_training import TrainingContractError, learning_rate_factor
from eventground_wam.coupled_policy import CoupledPolicy, coupled_parameter_groups
from eventground_wam.coupled_training import CoupledTrainingEngine
from eventground_wam.demonstration_contracts import canonical_json_bytes
from eventground_wam.week11_multihorizon_plan import HORIZONS, SCHEDULE_ID, candidate_horizons

FORMAT = 'eventground-wam-week11-multihorizon-continuation-v1'
ARMS = ('h8-continuation', 'mixed-continuation')


@contextmanager
def route_candidates(model, horizons):
    """Replace only candidate horizons, respecting inherited microbatch order."""
    if not 1 <= len(horizons) <= 32 or any(type(h) is not int or h not in HORIZONS for h in horizons):
        raise TrainingContractError('invalid candidate horizon vector')
    cursor = 0

    def route(_module, args):
        nonlocal cursor
        condition, = args
        n = len(condition.horizons)
        if (not 1 <= n <= 10 or cursor + n > len(horizons)
                or not torch.all(condition.horizons == 8)
                or condition.actions.shape != (n, 32, 7)
                or condition.actions.requires_grad or torch.count_nonzero(condition.actions[:, 8:])):
            raise TrainingContractError('candidate proposal or microbatch boundary changed')
        selected = torch.tensor(horizons[cursor:cursor + n], device=condition.horizons.device,
                                dtype=condition.horizons.dtype)
        cursor += n
        return (replace(condition, horizons=selected),)

    handle = model.future_core.register_forward_pre_hook(route)
    try:
        yield
        if cursor != len(horizons):
            raise TrainingContractError('candidate routing did not cover every action anchor')
    finally:
        handle.remove()


class ContinuationPolicy(CoupledPolicy):
    """Original coupled forward, with a scoped per-anchor candidate route."""

    candidate_horizons = None

    def forward_with_features(self, *args, **kwargs):
        if self.candidate_horizons is None:
            return super().forward_with_features(*args, **kwargs)
        with route_candidates(self, self.candidate_horizons):
            return super().forward_with_features(*args, **kwargs)


class ContinuationEngine(CoupledTrainingEngine):
    def __init__(self, model, names, predictor, recipe, *, identity_digest,
                 device, continuation_arm, design_digest, plan):
        if continuation_arm not in ARMS or not isinstance(model, ContinuationPolicy) or model.arm != 'M4':
            raise TrainingContractError('wrong continuation model or arm')
        self.continuation_arm, self.continuation_design = continuation_arm, design_digest
        self.plan = [[tuple(t) for t in batch] for batch in plan]
        if not self.plan or any(not b or len(b) > 32 for b in self.plan):
            raise TrainingContractError('invalid continuation plan')
        self.offsets = [0]
        for batch in self.plan:
            self.offsets.append(self.offsets[-1] + len(batch))
        config = {**recipe, 'epochs': 1, 'maximum_updates': len(plan),
                  'decoder_projector_learning_rate': 1e-5, 'lora_learning_rate': 1e-5,
                  'auxiliary_learning_rate': 1e-5, 'warmup_fraction': .02,
                  'minimum_learning_rate_ratio': .1}
        super().__init__(model, names, predictor, config, identity_digest=identity_digest,
                         steps_per_epoch=len(plan), device=device)
        groups = coupled_parameter_groups(model, predictor, names)
        for group in groups:
            group['lr'] = 1e-5
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(config['adam_betas']),
                                          eps=config['adam_epsilon'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: learning_rate_factor(step, len(self.plan), config))
        self.horizon_chain = '0' * 64

    def horizons(self, index):
        return candidate_horizons(self.offsets[index], len(self.plan[index]),
                                  mixed=self.continuation_arm == 'mixed-continuation')

    def step(self, batch, future_targets, *, hook=None):
        i = self.global_step
        if (not 0 <= i < len(self.plan) or batch['index'] != i
                or [tuple(t) for t in batch['tokens']] != self.plan[i]
                or batch['sample_count'] != len(self.plan[i])
                or self.samples_seen != self.offsets[i]):
            raise TrainingContractError('continuation exposure or cursor changed')
        selected = self.horizons(i)
        self.model.candidate_horizons = selected
        try:
            row = super().step(batch, future_targets, hook=hook)
        finally:
            self.model.candidate_horizons = None
        self.horizon_chain = hashlib.sha256(bytes.fromhex(self.horizon_chain)
                                            + canonical_json_bytes(selected)).hexdigest()
        return {**row, 'continuation_arm': self.continuation_arm,
                'candidate_horizons': selected, 'horizon_chain': self.horizon_chain}

    def snapshot(self):
        return {**super().snapshot(), 'continuation_format': FORMAT,
                'continuation_arm': self.continuation_arm, 'continuation_design': self.continuation_design,
                'horizon_schedule': SCHEDULE_ID, 'horizon_chain': self.horizon_chain}

    def restore(self, value):
        if (value.get('continuation_format') != FORMAT
                or value.get('continuation_arm') != self.continuation_arm
                or value.get('continuation_design') != self.continuation_design
                or value.get('horizon_schedule') != SCHEDULE_ID):
            raise TrainingContractError('continuation recovery identity changed')
        step = value['global_step']
        if (type(step) is not int or not 0 <= step <= len(self.plan)
                or value['epoch'] != 0 or value['next_batch'] != step
                or value['samples_seen'] != self.offsets[step]):
            raise TrainingContractError('continuation recovery cursor changed')
        chain = '0' * 64
        for index in range(step):
            chain = hashlib.sha256(bytes.fromhex(chain) + canonical_json_bytes(self.horizons(index))).hexdigest()
        expected_rate = 1e-5 * learning_rate_factor(step, len(self.plan), self.recipe)
        if (chain != value['horizon_chain'] or value['scheduler']['base_lrs'] != [1e-5] * 4
                or value['scheduler']['last_epoch'] != step
                or any(g['lr'] != expected_rate for g in value['optimizer']['param_groups'])):
            raise TrainingContractError('continuation horizon stream or learning rates changed')
        super().restore(value)
        self.horizon_chain = chain
        self.model.candidate_horizons = None
