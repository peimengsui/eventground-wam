"""Frozen H8 reference with an independently trained non-8 motion correction."""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, replace
from threading import Lock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eventground_wam.b0_training import (
    TRAINING_FORMAT,
    capture_rng,
    cpu_tree,
    learning_rate_factor,
    move_batch,
    restore_rng,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes
from eventground_wam.eventground_data import CurrentObservation
from eventground_wam.eventground_horizon import select_horizon
from eventground_wam.eventground_inference import require_disjoint_models, synchronize

FORMAT = 'week11-frozen-h8-motion-adapter-v1'
ARMS = ('horizon-motion', 'h8-information-control')
NON8 = (2, 4, 16, 32)


class MotionAdapter(nn.Module):
    """Four-head, width-64 attention; zero six-dimensional output, no dropout."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261031)
            self.query_norm = nn.LayerNorm(2048, dtype=torch.float32)
            self.future_norm = nn.LayerNorm(256, dtype=torch.float32)
            self.query = nn.Linear(2048, 64, dtype=torch.float32)
            self.key = nn.Linear(256, 64, dtype=torch.float32)
            self.value = nn.Linear(256, 64, dtype=torch.float32)
            self.output = nn.Linear(64, 6, dtype=torch.float32)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, query, future):
        n = len(query)
        if query.shape != (n, 8, 2048) or future.shape != (n, 128, 256):
            raise ValueError('adapter feature shape changed')
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = self.query(self.query_norm(query.float())).reshape(n, 8, 4, 16).transpose(1, 2)
            f = self.future_norm(future.float())
            k = self.key(f).reshape(n, 128, 4, 16).transpose(1, 2)
            v = self.value(f).reshape(n, 128, 4, 16).transpose(1, 2)
            context = (((q @ k.transpose(-1, -2)) * .25).softmax(-1) @ v)
            return self.output(context.transpose(1, 2).reshape(n, 8, 64))


class FrozenMotionPolicy(nn.Module):
    def __init__(self, reference, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown motion arm')
        self.reference = reference.requires_grad_(False).eval()
        self.adapter = MotionAdapter().to(next(reference.parameters()).device)
        self.arm, self.horizons = arm, None

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, inputs, state, images):
        n = len(state)
        hs = self.horizons if self.horizons is not None else [8] * n
        if len(hs) != n or any(type(h) is not int or h not in (2, 4, 8, 16, 32) for h in hs):
            raise ValueError('invalid motion horizon vector')
        # Execute the exact original path at H8, with no adapter or extra future call.
        if all(h == 8 for h in hs):
            with torch.no_grad():
                return self.reference(inputs, state, images)
        conditions, queries = [], []
        handles = [self.reference.future_core.register_forward_pre_hook(
            lambda _m, a: conditions.append(a[0])),
            self.reference.coupling.register_forward_pre_hook(
                lambda _m, a: queries.append(a[0]))]
        try:
            with torch.no_grad():
                result = self.reference.forward_with_features(inputs, state, images)
        finally:
            for handle in handles:
                handle.remove()
        if len(queries) != 1 or sum(len(c.horizons) for c in conditions) != n:
            raise ValueError('reference capture boundary changed')
        offset, tokens = 0, []
        with torch.no_grad(), torch.autocast(device_type=state.device.type, enabled=False), \
                torch.backends.cudnn.flags(allow_tf32=False):
            for c in conditions:
                if not torch.all(c.horizons == 8) or torch.count_nonzero(c.actions[:, 8:]):
                    raise ValueError('reference H8 proposal changed')
                count = len(c.horizons)
                selected = torch.tensor(hs[offset:offset+count], device=state.device)
                tokens.append(self.reference.future_core(replace(c, horizons=selected)))
                offset += count
            routed = torch.cat(tokens)
        # Both arms execute the same extra core work; the control discards routed tokens.
        future = routed if self.arm == ARMS[0] else result.future_tokens
        delta = self.adapter(queries[0].detach(), future.detach())
        base = result.action_output
        raw = base.raw_action_predictions
        mask = torch.tensor([h != 8 for h in hs], device=raw.device)[:, None, None]
        motion = torch.where(mask, raw[..., :6] + delta, raw[..., :6])
        corrected = torch.cat((motion, raw[..., 6:]), dim=-1)
        normalized = torch.cat((motion, base.normalized_actions[..., 6:]), dim=-1)
        if not torch.isfinite(corrected).all():
            raise ValueError('nonfinite motion correction')
        return replace(base, raw_action_predictions=corrected, normalized_actions=normalized)


class MotionInference:
    """One frozen event prediction, then the current-observation motion policy."""

    def __init__(self, action_policy, event, *, guard=lambda: None):
        self.action_policy, self.event, self.guard = action_policy, event, guard
        self.checkpoint = action_policy.checkpoint
        self.last_query = None
        self.lock = Lock()
        require_disjoint_models(action_policy._policy, event.model)

    def predict_chunk(self, *, agent_rgb, wrist_rgb, proprioception, instruction, suite):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('concurrent motion queries')
        record = {'exception': None, 'execute_actions': 8, 'proposal_actions': 8,
                  'zero_padded_actions': 24, 'adapter_arm': self.action_policy._policy.arm}
        self.last_query = record
        start = time.perf_counter()
        try:
            self.guard()
            observation = CurrentObservation(agent_rgb, wrist_rgb,
                np.asarray(proprioception, dtype=np.float32), instruction)
            prediction, details = self.event.predict(observation)
            decision = select_horizon(prediction)
            record.update(prediction=prediction, event=details, decision=asdict(decision),
                          used_horizon=decision.selected_horizon,
                          future_calls=1+int(decision.selected_horizon != 8))
            self.action_policy._policy.horizons = [decision.selected_horizon]
            actions = self.action_policy.predict_chunk(agent_rgb=agent_rgb, wrist_rgb=wrist_rgb,
                proprioception=proprioception, instruction=instruction, suite=suite)
            synchronize(self.action_policy._device)
            self.guard()
            return actions
        except BaseException as exc:
            record['exception'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.action_policy._policy.horizons = None
            record['wrapper_wall_ms'] = (time.perf_counter()-start)*1000
            self.lock.release()


class MotionEngine:
    def __init__(self, model, plan, identity, *, learning_rate=1e-4):
        self.model, self.plan, self.identity = model, plan, identity
        self.device = next(model.parameters()).device
        self.parameters = list(model.adapter.parameters())
        if any(p.requires_grad for p in model.reference.parameters()):
            raise ValueError('reference must be frozen')
        self.recipe = {'warmup_fraction': .02, 'minimum_learning_rate_ratio': .1}
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate,
            betas=(.9, .95), eps=1e-8, weight_decay=.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda i: learning_rate_factor(i, len(plan), self.recipe))
        self.offsets = [0]
        for batch in plan:
            self.offsets.append(self.offsets[-1] + len(batch))
        self.global_step = self.samples_seen = 0
        self.sample_chain = self.horizon_chain = '0' * 64
        self.partial = False

    def horizons(self, i):
        return [NON8[(self.offsets[i]+j) % 4] for j in range(len(self.plan[i]))]

    def step(self, batch, *, hook=None):
        i = self.global_step
        if (self.partial or not 0 <= i < len(self.plan) or batch['index'] != i
                or [tuple(t) for t in batch['tokens']] != [tuple(t) for t in self.plan[i]]
                or batch['sample_count'] != len(self.plan[i]) or self.samples_seen != self.offsets[i]):
            raise ValueError('training exposure/cursor changed or partial update')
        self.partial = True
        inputs, state, truth = move_batch(batch, self.device)
        images = batch['current_images'].to(self.device)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        hs = self.horizons(i); self.model.horizons = hs
        try:
            output = self.model(inputs, state, images)
            if truth.shape != output.raw_action_predictions.shape or not torch.isfinite(truth).all():
                raise ValueError('invalid motion targets')
            loss = F.l1_loss(output.raw_action_predictions[..., :6], truth[..., :6].detach())
            if not torch.isfinite(loss):
                raise ValueError('nonfinite motion loss')
            loss.backward()
            if any(p.grad is not None for p in self.model.reference.parameters()):
                raise ValueError('gradient escaped into frozen reference')
            norm = nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
            rate = self.optimizer.param_groups[0]['lr']
            if hook is not None:
                hook()
            self.optimizer.step(); self.scheduler.step()
        finally:
            self.model.horizons = None
        self.global_step += 1; self.samples_seen += len(hs)
        self.sample_chain = hashlib.sha256(bytes.fromhex(self.sample_chain)
            + bytes.fromhex(batch['identity_digest'])).hexdigest()
        self.horizon_chain = hashlib.sha256(bytes.fromhex(self.horizon_chain)
            + canonical_json_bytes(hs)).hexdigest()
        self.partial = False
        return {'global_step': self.global_step, 'samples_seen': self.samples_seen,
            'motion_l1': float(loss.detach()), 'gradient_norm': float(norm), 'learning_rate': rate,
            'sample_chain': self.sample_chain, 'horizon_chain': self.horizon_chain,
            'candidate_horizons': hs}

    def snapshot(self):
        if self.partial:
            raise ValueError('cannot save a partial update')
        return cpu_tree({'format': TRAINING_FORMAT, 'adapter_format': FORMAT,
            'identity_digest': self.identity, 'arm': self.model.arm,
            'state_dict': self.model.adapter.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), 'rng': capture_rng(), 'epoch': 0,
            'next_batch': self.global_step, 'global_step': self.global_step,
            'samples_seen': self.samples_seen, 'sample_chain': self.sample_chain,
            'horizon_chain': self.horizon_chain})

    def restore(self, value):
        value = cpu_tree(value)
        if (value['format'] != TRAINING_FORMAT or value['adapter_format'] != FORMAT
                or value['identity_digest'] != self.identity or value['arm'] != self.model.arm):
            raise ValueError('adapter recovery identity changed')
        i = value['global_step']
        if (type(i) is not int or not 0 <= i <= len(self.plan) or value['epoch'] != 0
                or value['next_batch'] != i or value['samples_seen'] != self.offsets[i]):
            raise ValueError('invalid adapter recovery cursor')
        chain = '0' * 64
        for j in range(i):
            chain = hashlib.sha256(bytes.fromhex(chain)+canonical_json_bytes(self.horizons(j))).hexdigest()
        expected = self.scheduler.base_lrs[0]*learning_rate_factor(i, len(self.plan), self.recipe)
        if (value['horizon_chain'] != chain or value['scheduler']['last_epoch'] != i
                or value['scheduler']['base_lrs'] != self.scheduler.base_lrs
                or value['optimizer']['param_groups'][0]['lr'] != expected):
            raise ValueError('recovery horizon stream or learning rate changed')
        if not all(torch.isfinite(v).all() for v in value['state_dict'].values()):
            raise ValueError('nonfinite adapter checkpoint')
        self.model.adapter.load_state_dict(value['state_dict'], strict=True)
        self.optimizer.load_state_dict(value['optimizer']); self.scheduler.load_state_dict(value['scheduler'])
        for name in ('global_step', 'samples_seen', 'sample_chain', 'horizon_chain'):
            setattr(self, name, value[name])
        restore_rng(value['rng']); self.partial = False; self.model.horizons = None
