"""Frozen Week 10 reward semantics and explicit logical-batch SFT/GRPO math."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from eventground_wam.eventground_model import EVENT_TYPES, DecisionScores, validate_tokens

PRIMARY_REWARD = (.40, .25, .20, .15)
CALIBRATION_REWARD = (.30, .35, .20, .15)


def valid_prediction(value) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"next_event", "interval", "horizon"}:
        return False
    event, interval, horizon = value["next_event"], value["interval"], value["horizon"]
    if event not in EVENT_TYPES:
        return False
    if event == "none":
        return interval is None and horizon is None
    return (isinstance(interval, list) and len(interval) == 2
            and all(type(offset) is int for offset in interval)
            and 1 <= interval[0] < interval[1] <= 33
            and type(horizon) is int and horizon == interval[0])


@dataclass(frozen=True)
class RewardResult:
    total: float
    event_iou: float
    time_score: float
    predicate: float
    format: float
    valid: bool


def structured_reward(prediction, target, *, verified_interval: tuple[int, int] | None,
                      evidence_digest: str, complete_window: bool,
                      weights: tuple[float, ...] = PRIMARY_REWARD) -> RewardResult:
    """Reward against a separately admitted offline evidence record, never model inputs.

    ``verified_interval`` is the independently verified event interval in anchor
    offsets, possibly extending beyond 33. A target's clipped interval must be its
    exact intersection with the window. None needs a complete, reviewed negative
    window and no positive interval. Evidence admission/checksum lookup belongs to
    the dataset loader; this function cannot turn an unreviewed detector into QA.
    """
    if tuple(weights) not in (PRIMARY_REWARD, CALIBRATION_REWARD):
        raise ValueError("only the primary or single predeclared reward calibration is allowed")
    if (not valid_prediction(target) or complete_window is not True
            or not isinstance(evidence_digest, str) or len(evidence_digest) != 64):
        raise ValueError("reward requires a valid target and complete admitted evidence window")
    try:
        decoded_digest = bytes.fromhex(evidence_digest)
        if len(decoded_digest) != 32 or decoded_digest.hex() != evidence_digest:
            raise ValueError("evidence digest must be canonical SHA-256")
    except ValueError as error:
        raise ValueError("invalid independent evidence digest") from error
    if target["next_event"] == "none":
        if verified_interval is not None:
            raise ValueError("none evidence must contain no selected positive interval")
    elif (not isinstance(verified_interval, tuple) or len(verified_interval) != 2
          or any(type(value) is not int for value in verified_interval)
          or not 1 <= verified_interval[0] < verified_interval[1]
          or target["interval"] != [verified_interval[0], min(33, verified_interval[1])]):
        raise ValueError("target and independently verified event interval disagree")
    if not valid_prediction(prediction):
        return RewardResult(0., 0., 0., 0., 0., False)
    iou = timing = predicate = 0.
    if prediction["next_event"] == target["next_event"]:
        if target["next_event"] == "none":
            iou = timing = predicate = 1.
        else:
            start, end = prediction["interval"]
            true_start, true_end = target["interval"]
            intersection = max(0, min(end, true_end) - max(start, true_start))
            union = end - start + true_end - true_start - intersection
            iou = intersection / union
            timing = max(0., 1. - abs(start - true_start) / 32.)
            predicate = float(verified_interval[0] <= start < verified_interval[1])
    components = iou, timing, predicate, 1.
    return RewardResult(sum(weight * component for weight, component
                            in zip(weights, components, strict=True)), *components, True)


def _logical_denominator(rows: int, logical_size: int | None) -> int:
    logical_size = rows if logical_size is None else logical_size
    if type(logical_size) is not int or logical_size < rows or rows <= 0:
        raise ValueError("logical batch must include this nonempty microbatch")
    return logical_size


def structured_sft_loss(scores: DecisionScores, *, logical_batch_size: int | None = None) -> Tensor:
    """Mean active CE per example, sum microbatch, divide by full logical batch.

    Calling on a complete logical batch defaults to its mean. For accumulation,
    every microbatch uses the SAME total logical batch size; never average unequal
    microbatch means or treat none's one decision as three targets.
    """
    expected = validate_tokens(scores.tokens)
    if (scores.active.dtype != torch.bool or not torch.equal(scores.active, expected)
            or scores.log_probs.shape != expected.shape or not torch.isfinite(scores.log_probs).all()
            or (scores.log_probs > 1e-6).any()):
        raise ValueError("invalid structured log probabilities or active-decision mask")
    denominator = _logical_denominator(len(expected), logical_batch_size)
    per_example = -(scores.log_probs * expected).sum(-1) / expected.sum(-1)
    return per_example.sum() / denominator


def group_advantages(rewards: Tensor, *, epsilon: float = 1e-4) -> Tensor:
    """Detached population-standardized groups of exactly four; ties map to zero."""
    if (rewards.ndim != 2 or rewards.shape[1] != 4 or not len(rewards)
            or not rewards.is_floating_point() or not torch.isfinite(rewards).all()
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("advantages require finite prompt-by-four rewards and positive epsilon")
    rewards = rewards.detach().float()
    centered = rewards - rewards.mean(-1, keepdim=True)
    std = centered.square().mean(-1, keepdim=True).sqrt()
    return torch.where(std == 0, torch.zeros_like(centered), centered / (std + epsilon))


@dataclass(frozen=True)
class GRPOLoss:
    total: Tensor
    policy: Tensor
    kl: Tensor
    clipping_fraction: Tensor
    equal_reward_groups: int


def grpo_loss(current_log_probs: Tensor, old_log_probs: Tensor,
              reference_log_probs: Tensor, active: Tensor, rewards: Tensor, *,
              clip_epsilon: float = .2, kl_beta: float = .02,
              advantage_epsilon: float = 1e-4, logical_prompt_count: int | None = None) -> GRPOLoss:
    """Per-decision PPO clip and fixed-SFT sampled-action k3 reference KL.

    Inputs have [prompt,4,3] shape; complete groups must remain together even if
    their head/backbone scoring is microbatched. Old and reference probabilities
    and rewards are detached. The mean over active decisions gives one weight per
    completion, then mean four completions and logical prompts. Equal-reward
    groups have no policy gradient but retain the KL term. No reward clipping,
    advantage recomputation on partial groups, or silent overflow clamping occurs.
    """
    shape = current_log_probs.shape
    if (current_log_probs.ndim != 3 or shape[1:] != (4, 3) or shape[0] == 0
            or old_log_probs.shape != shape or reference_log_probs.shape != shape
            or active.shape != shape or active.dtype != torch.bool
            or rewards.shape != shape[:2] or not active[..., 0].all()
            or not torch.equal(active[..., 1], active[..., 2])):
        raise ValueError("GRPO requires complete prompt-by-four groups and causal decision masks")
    if (not 0 < clip_epsilon < 1 or not math.isfinite(kl_beta) or kl_beta < 0
            or any(not torch.isfinite(value).all() for value in
                   (current_log_probs, old_log_probs, reference_log_probs))
            or any((value > 1e-6).any() for value in
                   (current_log_probs, old_log_probs, reference_log_probs))):
        raise ValueError("invalid GRPO parameters or categorical log probabilities")
    denominator = _logical_denominator(shape[0], logical_prompt_count)
    advantages = group_advantages(rewards, epsilon=advantage_epsilon)
    current = current_log_probs.float()
    old = old_log_probs.detach().float()
    reference = reference_log_probs.detach().float()
    ratio = (current - old).exp()
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy_decisions = -torch.minimum(ratio * advantages[..., None],
                                     clipped * advantages[..., None])
    difference = reference - current
    # k3: exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1.
    kl_decisions = torch.expm1(difference) - difference
    if not torch.isfinite(ratio).all() or not torch.isfinite(kl_decisions).all():
        raise ValueError("GRPO ratio or reference KL overflow; stop the numerical gate")
    counts = active.sum(-1)

    def reduce(values):
        return ((values * active).sum(-1) / counts).mean(-1).sum() / denominator

    policy = reduce(policy_decisions)
    kl = reduce(kl_decisions)
    clipping_fraction = reduce(((ratio < 1 - clip_epsilon)
                                | (ratio > 1 + clip_epsilon)).float()).detach()
    equal = int((rewards.max(-1).values == rewards.min(-1).values).sum())
    return GRPOLoss(policy + kl_beta * kl, policy, kl, clipping_fraction, equal)
