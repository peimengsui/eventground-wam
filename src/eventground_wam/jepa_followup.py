"""Isolated training-only JEPA follow-up; the Week 6 predictor stays unchanged."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn

from eventground_wam.future_pairs import HORIZONS
from eventground_wam.jepa_predictor import (
    JepaPrediction,
    JepaPredictor,
    jepa_prediction_loss,
    normalize_predictions,
)
from eventground_wam.pixel_predictor import PixelPredictor
from eventground_wam.pixel_training import slice_condition


class ResidualJepaPredictor(JepaPredictor):
    """Same core/budget; detached CURRENT teacher features plus a learned delta.

    The extra current-image teacher path is an explicit treatment difference.
    Future targets are never an argument. The output interface still carries
    the core's 128x256 tokens; their predictive utility needs separate testing.
    """

    def __init__(self, config=None):
        super().__init__(config)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def from_tokens(self, tokens, current_features):
        if (tokens.ndim != 3 or tokens.shape[1:] != (128, self.config.dimension)
                or current_features.shape != (len(tokens), 128, self.config.target_dimension)
                or current_features.device != tokens.device
                or not current_features.is_floating_point()
                or not torch.isfinite(current_features).all()
                or (current_features.float().norm(dim=-1)-1).abs().max() > 1e-5
                or current_features.float().mean(dim=-1).abs().max() > 1e-5):
            raise ValueError('finite centered unit CURRENT teacher features required')
        delta = self.output_projection(tokens)/math.sqrt(self.config.target_dimension)
        return normalize_predictions(current_features.detach().float()+delta)

    def forward(self, condition, current_features, *, ablation='none'):
        tokens = PixelPredictor.forward(self, condition, ablation=ablation).future_tokens
        return JepaPrediction(self.from_tokens(tokens, current_features), tokens)


def predict(model, condition, current, *, ablation='none'):
    if isinstance(model, ResidualJepaPredictor):
        return model(condition, current, ablation=ablation)
    return model(condition, ablation=ablation)


def from_tokens(model, tokens, current):
    if isinstance(model, ResidualJepaPredictor):
        return model.from_tokens(tokens, current)
    return normalize_predictions(model.output_projection(tokens))


def backward_batch(model, condition, current, target, groups, loss_rule, microbatch=10):
    """One logical variance loss; no optimizer advance during exact replay."""
    if microbatch < 1:
        raise ValueError('positive microbatch required')
    ranges = [(i, min(i+microbatch, len(target))) for i in range(0, len(target), microbatch)]
    with torch.no_grad():
        fields = torch.cat([predict(model, slice_condition(condition, a, b), current[a:b]).features
                            for a, b in ranges])
    fields.requires_grad_(True)
    loss = jepa_prediction_loss(fields, target, groups, **loss_rule)
    derivative, = torch.autograd.grad(loss.total, fields)
    if not torch.isfinite(derivative).all():
        raise ValueError('nonfinite feature gradient')
    for a, b in ranges:
        value = predict(model, slice_condition(condition, a, b), current[a:b]).features
        if not torch.equal(value.detach(), fields[a:b]):
            raise ValueError('microbatch replay changed')
        value.backward(derivative[a:b])
    return loss


def select_groups(groups, seed=20261002):
    """Metadata-only selection. Entire numeric groups have one pilot role."""
    rank = lambda kind, value: hashlib.sha256(f'jepa-followup:{seed}:{kind}:{value}'.encode()).hexdigest()
    train = [g for g in groups if g['split'] == 'train']
    tasks = sorted({g['task_key'] for g in train})
    selected = []
    for task in tasks:
        candidates = sorted([g for g in train if g['task_key'] == task and g['episode_length'] >= 40],
                            key=lambda g: rank('group', g['group_digest']))
        if len(candidates) < 4:
            raise ValueError('four separate training numeric groups required per task')
        for i, group in enumerate(candidates[:4]):
            anchors = [int(f*(group['episode_length']-33)) for f in (.2, .4, .6, .8)]
            if len(set(anchors)) != 4:
                raise ValueError('anchor selection must be distinct')
            selected.append({'task_key': task, 'group_digest': group['group_digest'],
                'split': 'train', 'role': 'fit' if i < 2 else 'pilot_eval',
                'episode_index': min(group['episode_indices'], key=lambda e: rank('episode', e)),
                'episode_length': group['episode_length'], 'anchors': anchors,
                'horizons': list(HORIZONS)})
    validate_selection(selected)
    return selected


def validate_selection(selected):
    seen, episodes = set(), set()
    for row in selected:
        if (row['split'] != 'train' or row['role'] not in ('fit', 'pilot_eval')
                or row['group_digest'] in seen or row['episode_index'] in episodes
                or row['horizons'] != list(HORIZONS)
                or len(row['anchors']) != 4 or len(set(row['anchors'])) != 4
                or min(row['anchors']) < 0 or max(row['anchors'])+32 >= row['episode_length']):
            raise ValueError('leaking, duplicate or invalid pilot membership')
        seen.add(row['group_digest']); episodes.add(row['episode_index'])
    for task in {r['task_key'] for r in selected}:
        if any(sum(r['task_key'] == task and r['role'] == role for r in selected) != 2
               for role in ('fit', 'pilot_eval')):
            raise ValueError('each task requires two groups in each role')


def training_plan(anchor_count, updates=6000, seed=20261002):
    if anchor_count < 8 or anchor_count % 8 or updates < 1:
        raise ValueError('complete eight-anchor batches required')
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < updates:
        order = rng.permutation(anchor_count).reshape(-1, 8)
        result.extend(order.tolist())
    return result[:updates]


def head_span(model):
    """Optimistic bound: centered head columns AND a freely scaled bias.

    Actual affine outputs form a subset, so this cannot overstate an
    irreducible error. Future features may be used only for this diagnostic.
    """
    weight = model.output_projection.weight.detach().double()
    bias = model.output_projection.bias.detach().double()
    matrix = torch.cat((weight-weight.mean(0), (bias-bias.mean())[:, None]), 1)
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    keep = singular > singular.max()*1e-10
    return u[:, keep]


def span_lower_bound(unit_features, basis):
    projected = unit_features.double() @ basis
    return (1-projected.square().sum(-1).sqrt()).clamp(0, 2)


def errors(prediction, target):
    return (1-(prediction.double()*target.double()).sum(-1)).clamp(0, 2).reshape(-1, 2, 64).mean(-1)


def motion_bin(value):
    return 'low' if value <= .01 else 'medium' if value <= .05 else 'high'
