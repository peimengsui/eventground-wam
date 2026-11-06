import math
from dataclasses import replace

import pytest
import torch

from eventground_wam.eventground_model import EventGroundHead
from eventground_wam.eventground_objectives import (
    CALIBRATION_REWARD,
    group_advantages,
    grpo_loss,
    structured_reward,
    structured_sft_loss,
    valid_prediction,
)


def event(kind="grasp", start=12, end=18):
    return {"next_event": kind, "interval": [start, end], "horizon": start}


NONE = {"next_event": "none", "interval": None, "horizon": None}


def reward(prediction, target=None, interval=(12, 18), **kwargs):
    return structured_reward(prediction, event() if target is None else target,
                             verified_interval=interval, evidence_digest="a" * 64,
                             complete_window=True, **kwargs)


def test_exact_positive_and_none_get_all_components():
    for result in (reward(event()), reward(NONE, NONE, None)):
        assert result.total == 1 and result.valid
        assert (result.event_iou, result.time_score, result.predicate, result.format) == (1, 1, 1, 1)


def test_wrong_type_and_none_mismatch_keep_only_constant_format():
    # If tied onsets selected grasp, matching contact's time does not earn grasp reward.
    for prediction, target, interval in ((event("contact"), event(), (12, 18)),
                                          (NONE, event(), (12, 18)),
                                          (event(), NONE, None)):
        result = reward(prediction, target, interval)
        assert result.total == .15 and result.format == 1
        assert result.event_iou == result.time_score == result.predicate == 0


def test_iou_timing_and_predicate_have_distinct_half_open_semantics():
    result = reward(event(start=15, end=21))
    assert result.event_iou == pytest.approx(3 / 9)
    assert result.time_score == pytest.approx(1 - 3 / 32)
    assert result.predicate == 1
    at_end = reward(event(start=18, end=20))
    assert at_end.event_iou == 0 and at_end.predicate == 0
    before = reward(event(start=10, end=20))
    assert before.event_iou == pytest.approx(6 / 10) and before.predicate == 0


def test_right_censored_event_uses_clipped_iou_and_full_verified_interval():
    result = reward(event(start=30, end=33), event(start=30, end=33), (30, 70))
    assert result.total == 1
    with pytest.raises(ValueError, match="disagree"):
        reward(event(), event(start=12, end=17), (12, 18))


@pytest.mark.parametrize("prediction", [
    {}, {**NONE, "confidence": .9}, event(start=True), event(start=0), event(start=18, end=18),
    event(end=34), {**event(), "horizon": 13}, {**event(), "next_event": "touch"},
    {**NONE, "horizon": 0}, {**event(), "interval": (12, 18)}, None,
])
def test_malformed_output_is_zero_total(prediction):
    assert not valid_prediction(prediction)
    result = reward(prediction)
    assert result.total == 0 and not result.valid


def test_reward_cannot_silently_use_missing_or_inconsistent_independent_evidence():
    for arguments in ({"verified_interval": None}, {"evidence_digest": "missing"},
                      {"complete_window": False}, {"weights": (.25,) * 4}):
        defaults = {"verified_interval": (12, 18), "evidence_digest": "a" * 64,
                    "complete_window": True}
        with pytest.raises(ValueError):
            structured_reward(event(), event(), **{**defaults, **arguments})


def test_single_calibration_changes_only_predeclared_weights():
    primary = reward(event(start=14, end=19))
    calibrated = reward(event(start=14, end=19), weights=CALIBRATION_REWARD)
    assert calibrated.total - primary.total == pytest.approx(
        .1 * (primary.time_score - primary.event_iou))


def test_sft_means_decisions_per_example_before_logical_batch():
    head = EventGroundHead(input_dim=4, hidden_dim=8)
    tokens = torch.tensor([[0, 0, 0], [3, 2, 5]])
    scores = head.teacher_forced(torch.zeros(2, 8, 4), tokens)
    logs = torch.tensor([[-2., 0., 0.], [-3., -6., -9.]], requires_grad=True)
    scores = replace(scores, log_probs=logs)
    loss = structured_sft_loss(scores)
    assert loss == 4  # (2 + (3+6+9)/3)/2, not token-weighted 5.
    loss.backward()
    torch.testing.assert_close(logs.grad, torch.tensor([[-.5, 0, 0], [-1/6] * 3]))


def test_group_population_normalization_and_constant_reward_cancellation():
    rewards = torch.tensor([[1., 2., 3., 4.], [.15, .15, .15, .15]], requires_grad=True)
    actual = group_advantages(rewards)
    expected = torch.tensor([[-1.5, -.5, .5, 1.5]]) / (math.sqrt(1.25) + 1e-4)
    torch.testing.assert_close(actual[:1], expected)
    assert torch.equal(actual[1], torch.zeros(4)) and not actual.requires_grad
    torch.testing.assert_close(actual, group_advantages(rewards + .15), atol=2e-7, rtol=2e-6)
    with pytest.raises(ValueError, match="four"):
        group_advantages(rewards[:, :3])


def group_inputs(prompt_count=3):
    gen = torch.Generator().manual_seed(19)
    old = -(torch.rand(prompt_count, 4, 3, generator=gen) + 1)
    current = (old + torch.randn(prompt_count, 4, 3, generator=gen) * .2).requires_grad_()
    reference = (old - .1).requires_grad_()
    mask = torch.ones(prompt_count, 4, 3, dtype=torch.bool)
    mask[:, 0, 1:] = False
    rewards = torch.rand(prompt_count, 4, generator=gen).requires_grad_()
    return current, old.requires_grad_(), reference, mask, rewards


def test_grpo_reference_old_and_rewards_are_detached():
    current, old, reference, mask, rewards = group_inputs()
    result = grpo_loss(current, old, reference, mask, rewards)
    assert torch.isfinite(result.total) and result.kl >= 0
    result.total.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert torch.equal(current.grad[:, 0, 1:], torch.zeros(3, 2))
    assert old.grad is reference.grad is rewards.grad is None


def test_equal_reward_group_has_zero_policy_but_nonzero_reference_regularization():
    current, old, reference, mask, rewards = group_inputs(1)
    rewards = torch.ones(1, 4)
    result = grpo_loss(current, old, reference, mask, rewards)
    assert result.policy == 0 and result.kl > 0 and result.equal_reward_groups == 1
    result.total.backward()
    assert current.grad.abs().sum() > 0
    zero_beta = grpo_loss(current.detach().clone().requires_grad_(), old, reference, mask, rewards,
                          kl_beta=0)
    assert zero_beta.total == 0


def test_group_and_unequal_microbatch_loss_gradient_equivalence():
    current, old, reference, mask, rewards = group_inputs()
    full = grpo_loss(current, old, reference, mask, rewards)
    full.total.backward()
    micro_current = current.detach().clone().requires_grad_()
    total = 0
    for start, end in ((0, 1), (1, 3)):
        piece = grpo_loss(micro_current[start:end], old[start:end], reference[start:end],
                          mask[start:end], rewards[start:end], logical_prompt_count=3)
        piece.total.backward()
        total += piece.total.detach()
    torch.testing.assert_close(full.total.detach(), total)
    torch.testing.assert_close(current.grad, micro_current.grad)


def test_ppo_clipping_uses_advantage_sign():
    old = torch.full((1, 4, 3), -2.)
    ratios = torch.tensor([.5, .5, 1.5, 1.5])[None, :, None].expand(1, 4, 3)
    current = old + ratios.log()
    active = torch.ones(1, 4, 3, dtype=torch.bool)
    rewards = torch.tensor([[0., 1., 2., 3.]])
    result = grpo_loss(current, old, current, active, rewards, kl_beta=0)
    advantage = group_advantages(rewards)
    # Negative advantages use the larger clipped surrogate magnitude: .8 not .5.
    expected = -(torch.tensor([[.8, .8, 1.2, 1.2]]) * advantage).mean()
    torch.testing.assert_close(result.policy, expected)
    assert result.clipping_fraction == 1


def test_kl_zero_at_reference_and_overflow_fails_closed():
    current, old, _, active, rewards = group_inputs(1)
    equal = grpo_loss(current, old, current.detach(), active, rewards)
    assert equal.kl == 0
    with pytest.raises(ValueError, match="overflow"):
        grpo_loss(torch.full_like(current, -1000), old, torch.zeros_like(current), active, rewards)


def test_reject_partial_group_and_illegal_masks():
    current, old, reference, active, rewards = group_inputs(1)
    with pytest.raises(ValueError, match="complete"):
        grpo_loss(current[:, :3], old[:, :3], reference[:, :3], active[:, :3], rewards[:, :3])
    active[0, 1, 1] = False
    with pytest.raises(ValueError, match="masks"):
        grpo_loss(current, old, reference, active, rewards)
