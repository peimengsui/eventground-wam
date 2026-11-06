import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eventground_wam.eventground_model import (
    ENCODING_KEYS,
    EventGroundHead,
    EventGroundModel,
    configure_eventground_trainables,
    serialize_tokens,
    validate_tokens,
)
from eventground_wam.eventground_objectives import structured_sft_loss


@pytest.fixture
def head():
    return EventGroundHead(input_dim=12, hidden_dim=16)


@pytest.fixture
def slots():
    return torch.randn(5, 8, 12, generator=torch.Generator().manual_seed(8))


def test_only_preceding_output_tokens_condition_decisions(head, slots):
    a = torch.tensor([[3, 5, 10]] * 5)
    b = a.clone()
    b[:, 2] = 33
    first, end_changed = head.teacher_forced(slots, a), head.teacher_forced(slots, b)
    for field in ("event_logits", "start_logits", "end_logits"):
        assert torch.equal(getattr(first, field), getattr(end_changed, field))
    b[:, 1] = 8
    start_changed = head.teacher_forced(slots, b)
    assert torch.equal(first.event_logits, start_changed.event_logits)
    assert torch.equal(first.start_logits, start_changed.start_logits)
    assert not torch.equal(first.end_logits, start_changed.end_logits)
    b[:, 0] = 1
    event_changed = head.teacher_forced(slots, b)
    assert torch.equal(first.event_logits, event_changed.event_logits)
    assert not torch.equal(first.start_logits, event_changed.start_logits)


def test_constrained_support_includes_final_frame_and_none(head, slots):
    tokens = torch.tensor([[0, 0, 0], [1, 1, 2], [2, 32, 33], [3, 12, 33], [3, 1, 33]])
    score = head.teacher_forced(slots, tokens)
    assert torch.equal(score.active.sum(-1), torch.tensor([1, 3, 3, 3, 3]))
    assert torch.equal(score.log_probs[0, 1:], torch.zeros(2))
    assert score.log_probs[2, 2] == 0  # Exactly one allowed end for start=32.
    assert torch.isneginf(score.end_logits[2, :-1]).all()
    assert serialize_tokens(tokens)[0] == {"next_event": "none", "interval": None, "horizon": None}
    assert serialize_tokens(tokens)[2] == {"next_event": "contact", "interval": [32, 33],
                                           "horizon": 32}


@pytest.mark.parametrize("tokens", [
    [[0, 1, 2]], [[3, 0, 4]], [[1, 4, 4]], [[2, 1, 34]], [[4, 1, 3]],
])
def test_invalid_support_rejected(tokens):
    with pytest.raises(ValueError, match="support"):
        validate_tokens(torch.tensor(tokens))


def test_sample_is_prompt_major_reproducible_and_scores_match(head, slots):
    first = head.sample(slots, generator=torch.Generator().manual_seed(23))
    second = head.sample(slots, generator=torch.Generator().manual_seed(23))
    assert first.tokens.shape == (20, 3)
    assert torch.equal(first.tokens, second.tokens)
    scored = head.teacher_forced(slots.repeat_interleave(4, dim=0), first.tokens)
    # Context projection batch shapes differ (B versus 4B); allow FP32 GEMM rounding.
    torch.testing.assert_close(first.log_probs, scored.log_probs, rtol=2e-6, atol=5e-7)
    assert not first.log_probs.requires_grad
    assert all(item["horizon"] is None or item["horizon"] == item["interval"][0]
               for item in serialize_tokens(first.tokens))


def test_slot_order_and_observation_gradients_retained(head, slots):
    slots.requires_grad_()
    tokens = torch.tensor([[3, 5, 9]] * 5)
    original = head.teacher_forced(slots, tokens)
    reversed_slots = head.teacher_forced(slots.flip(1), tokens)
    assert not torch.equal(original.event_logits, reversed_slots.event_logits)
    structured_sft_loss(original).backward()
    assert torch.isfinite(slots.grad).all() and (slots.grad.abs().sum((0, 2)) > 0).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())


def test_sampling_rng_assignment_is_independent_of_prompt_microbatch(head, slots):
    whole = head.sample(slots, generator=torch.Generator().manual_seed(91))
    generator = torch.Generator().manual_seed(91)
    pieces = [head.sample(slots[:2], generator=generator),
              head.sample(slots[2:], generator=generator)]
    assert torch.equal(whole.tokens, torch.cat([piece.tokens for piece in pieces]))
    torch.testing.assert_close(whole.log_probs, torch.cat([piece.log_probs for piece in pieces]),
                               atol=5e-7, rtol=2e-6)


def test_initialization_does_not_change_global_rng():
    torch.manual_seed(81)
    state = torch.get_rng_state().clone()
    first = EventGroundHead(input_dim=12, hidden_dim=16)
    assert torch.equal(state, torch.get_rng_state())
    second = EventGroundHead(input_dim=12, hidden_dim=16)
    assert all(torch.equal(value, second.state_dict()[key])
               for key, value in first.state_dict().items())


class FakeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.ones(12))

    def forward(self, inputs, state):
        self.calls += 1
        return SimpleNamespace(tokens=state[:, :, None] * self.scale,
                               key_padding_mask=torch.zeros_like(state, dtype=torch.bool))


def test_wrapper_cannot_accept_evidence_or_run_other_m4_branches(head):
    model = EventGroundModel(FakeAdapter(), head)
    inputs = {key: torch.zeros(2, 2) for key in ENCODING_KEYS}
    state = torch.ones(2, 8)
    slots = model(inputs, state)
    assert slots.shape == (2, 8, 12) and model.conditioning_adapter.calls == 1
    for extra in ("actions", "future_images", "event_label", "remaining_length", "past_key_values"):
        with pytest.raises(ValueError, match="five OFT"):
            model({**inputs, extra: torch.ones(2)}, state)
    assert model.conditioning_adapter.calls == 1
    with pytest.raises(TypeError):
        model(inputs, state, future_images=torch.zeros(1))
    assert not any(key in dict(model.named_modules()) for key in
                   ("future_core", "coupling", "action_decoder"))


class FakeLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(1, 1)
        self.lora_A = nn.ModuleDict({"b0_oft": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"b0_oft": nn.Linear(1, 1, bias=False)})


def lora_wrapper(head):
    adapter = FakeAdapter()
    adapter.backbone = nn.Module()
    adapter.backbone.language_model = nn.ModuleDict({"layer": FakeLoRA()})
    adapter.backbone.visual = nn.ModuleDict({"layer": FakeLoRA()})
    adapter.proprioception_projector = nn.Linear(8, 12)
    return EventGroundModel(adapter, head)


def test_exact_checkpoint_allowlist_freezes_visual_lora_and_all_base(head):
    model = lora_wrapper(head)
    names = tuple(name for name, _ in model.named_parameters() if ".lora_" in name)
    allowed = configure_eventground_trainables(model, names)
    assert len(allowed["language_lora"]) == len(allowed["frozen_vision_lora"]) == 2
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == (
        set(allowed["head"]) | set(allowed["language_lora"]))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    configure_eventground_trainables(model, names)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    for bad in (names[:-1], (*names, names[0]), (*names, "fake.lora_A.b0_oft.weight")):
        with pytest.raises(ValueError):
            configure_eventground_trainables(model, bad)


def test_logical_microbatch_sft_gradient_equivalence(head, slots):
    full, micro = copy.deepcopy(head), copy.deepcopy(head)
    tokens = torch.tensor([[0, 0, 0], [1, 2, 7], [2, 32, 33], [3, 6, 13], [0, 0, 0]])
    full_slots, micro_slots = slots.clone().requires_grad_(), slots.clone().requires_grad_()
    loss = structured_sft_loss(full.teacher_forced(full_slots, tokens))
    loss.backward()
    pieces = []
    for start, end in ((0, 2), (2, 5)):
        part = structured_sft_loss(micro.teacher_forced(micro_slots[start:end], tokens[start:end]),
                                   logical_batch_size=5)
        pieces.append(part.detach())
        part.backward()
    torch.testing.assert_close(loss.detach(), sum(pieces))
    torch.testing.assert_close(full_slots.grad, micro_slots.grad, atol=2e-8, rtol=2e-5)
    for a, b in zip(full.parameters(), micro.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=5e-5)


def test_reject_nonfinite_slots_and_nonfp32_head(head, slots):
    tokens = torch.tensor([[0, 0, 0]] * 5)
    slots[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        head.teacher_forced(slots, tokens)
    with pytest.raises(ValueError, match="FP32"):
        head.double().teacher_forced(torch.zeros_like(slots), tokens)
