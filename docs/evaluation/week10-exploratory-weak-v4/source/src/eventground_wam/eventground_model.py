"""Causal Week 10 event head; no future core, action decoder, or evidence inputs.

Instantiate the wrapper with a separately loaded child copy of M4's adapter.
The numerical recipe and exact parameter names must be sealed before optimization.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

EVENT_TYPES = ("none", "gripper_close", "contact", "grasp")
ENCODING_KEYS = frozenset({
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
})


def validate_tokens(tokens: Tensor) -> Tensor:
    """Return active decisions; none uses [0,0,0], positives use real offsets."""
    if (not isinstance(tokens, Tensor) or tokens.dtype != torch.int64
            or tokens.ndim != 2 or tokens.shape[1] != 3 or len(tokens) == 0):
        raise ValueError("event tokens must be nonempty int64 [batch,3]")
    event, start, end = tokens.unbind(-1)
    positive = event != 0
    valid = ((event >= 0) & (event < 4)
             & torch.where(positive, (start >= 1) & (start <= 32)
                           & (end > start) & (end <= 33), (start == 0) & (end == 0)))
    if not valid.all():
        raise ValueError("event tokens violate none or half-open interval support")
    return torch.stack((torch.ones_like(positive), positive, positive), -1)


def serialize_tokens(tokens: Tensor) -> list[dict]:
    """Only the deterministic serializer emits horizon; it cannot disagree."""
    validate_tokens(tokens)
    return [{"next_event": EVENT_TYPES[event],
             "interval": [start, end] if event else None,
             "horizon": start if event else None}
            for event, start, end in tokens.detach().cpu().tolist()]


@dataclass(frozen=True)
class DecisionScores:
    tokens: Tensor
    log_probs: Tensor
    entropies: Tensor
    active: Tensor
    event_logits: Tensor
    start_logits: Tensor
    end_logits: Tensor


class EventGroundHead(nn.Module):
    """Ordered slot projection followed by a three-decision GRU decoder.

    Event has no label input; start sees only event; end sees only event/start.
    None's start/end are inactive and have exactly zero log probability/entropy.
    The 256-wide default has 1,475,140 parameters for the fixed 2048-wide M4 slots.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256,
                 slots: int = 8, seed: int = 17):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (input_dim, hidden_dim)):
            raise ValueError("head dimensions must be positive integers")
        if slots != 8 or type(slots) is not int or type(seed) is not int:
            raise ValueError("Week 10 requires eight slots and an integer seed")
        self.input_dim, self.hidden_dim, self.slots, self.seed = input_dim, hidden_dim, slots, seed
        # Isolate initialization so creating a head cannot reset caller sampling.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.slot_norm = nn.LayerNorm(input_dim)
            self.slot_projection = nn.Linear(input_dim, hidden_dim)
            self.context = nn.Sequential(nn.Linear(slots * hidden_dim, hidden_dim), nn.Tanh())
            self.bos = nn.Parameter(torch.randn(hidden_dim) * .02)
            self.event_embedding = nn.Embedding(4, hidden_dim)
            self.start_embedding = nn.Embedding(33, hidden_dim)
            self.decoder = nn.GRUCell(hidden_dim, hidden_dim)
            self.event_output = nn.Linear(hidden_dim, 4)
            self.start_output = nn.Linear(hidden_dim, 32)
            self.end_output = nn.Linear(hidden_dim, 32)

    def _context(self, slots: Tensor) -> Tensor:
        if (not isinstance(slots, Tensor) or slots.ndim != 3
                or slots.shape[1:] != (8, self.input_dim) or len(slots) == 0
                or not slots.is_floating_point() or not torch.isfinite(slots).all()):
            raise ValueError("head input must be finite current-observation [batch,8,hidden]")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("EventGround head parameters must remain FP32")
        return self.context(torch.nn.functional.gelu(
            self.slot_projection(self.slot_norm(slots.float()))).flatten(1))

    @staticmethod
    def _temperature(temperature: float) -> float:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("categorical temperature must be finite and positive")
        return float(temperature)

    def _logits(self, context: Tensor, event: Tensor, start: Tensor,
                temperature: float) -> tuple[Tensor, Tensor, Tensor]:
        first = self.decoder(self.bos.expand(len(context), -1), context)
        second = self.decoder(self.event_embedding(event), first)
        third = self.decoder(self.start_embedding(start), second)
        event_logits = self.event_output(first) / temperature
        start_logits = self.start_output(second) / temperature
        end_logits = self.end_output(third) / temperature
        ends = torch.arange(2, 34, device=context.device)
        end_logits = end_logits.masked_fill(ends[None, :] <= start[:, None], -torch.inf)
        return event_logits, start_logits, end_logits

    @staticmethod
    def _scores(tokens: Tensor, logits: tuple[Tensor, Tensor, Tensor]) -> DecisionScores:
        active = validate_tokens(tokens)
        indices = (tokens[:, 0], (tokens[:, 1] - 1).clamp_min(0),
                   (tokens[:, 2] - 2).clamp_min(0))
        logs, entropies = [], []
        for stage, (values, index) in enumerate(zip(logits, indices, strict=True)):
            if (torch.isnan(values).any() or torch.isposinf(values).any()
                    or not torch.isfinite(values).any(-1).all()):
                raise ValueError("nonfinite categorical logits")
            normalized = values.log_softmax(-1)
            # Avoid 0 * -inf for masked categorical states.
            entropy = -(normalized.exp() * normalized.masked_fill(
                ~torch.isfinite(normalized), 0)).sum(-1)
            selected = normalized.gather(-1, index[:, None]).squeeze(-1)
            logs.append(torch.where(active[:, stage], selected, torch.zeros_like(selected)))
            entropies.append(torch.where(active[:, stage], entropy, torch.zeros_like(entropy)))
        return DecisionScores(tokens, torch.stack(logs, -1), torch.stack(entropies, -1),
                              active, *logits)

    def teacher_forced(self, slot_features: Tensor, tokens: Tensor, *,
                       temperature: float = 1.) -> DecisionScores:
        validate_tokens(tokens)
        if len(tokens) != len(slot_features) or tokens.device != slot_features.device:
            raise ValueError("tokens and observation slots must have matching batches/devices")
        temperature = self._temperature(temperature)
        with torch.autocast(device_type=slot_features.device.type, enabled=False):
            context = self._context(slot_features)
            logits = self._logits(context, tokens[:, 0], tokens[:, 1], temperature)
            return self._scores(tokens, logits)

    @torch.no_grad()
    def sample(self, slot_features: Tensor, *, group_size: int = 4, temperature: float = 1.,
               generator: torch.Generator | None = None, greedy: bool = False) -> DecisionScores:
        if type(group_size) is not int or group_size not in (1, 4):
            raise ValueError("sampling requires groups of four or one evaluation prediction")
        temperature = self._temperature(temperature)
        with torch.autocast(device_type=slot_features.device.type, enabled=False):
            contexts = self._context(slot_features)

            def choose(logits):
                if not torch.isfinite(logits).any(-1).all() or torch.isnan(logits).any():
                    raise ValueError("invalid generation logits")
                if greedy:
                    return logits.argmax(-1)
                return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)

            # Consume the RNG prompt by prompt, including all four candidates and
            # their three decisions. Hardware prompt microbatching must not change
            # which random draws belong to a logical prompt or split its group.
            sampled = []
            for one_context in contexts:
                context = one_context[None].expand(group_size, -1)
                first = self.decoder(self.bos.expand(group_size, -1), context)
                event_logits = self.event_output(first) / temperature
                event = choose(event_logits)
                positive = event != 0
                second = self.decoder(self.event_embedding(event), first)
                start_logits = self.start_output(second) / temperature
                start = torch.where(positive, choose(start_logits) + 1, 0)
                third = self.decoder(self.start_embedding(start), second)
                end_logits = self.end_output(third) / temperature
                ends = torch.arange(2, 34, device=context.device)
                end_logits = end_logits.masked_fill(ends[None, :] <= start[:, None], -torch.inf)
                end = torch.where(positive, choose(end_logits) + 2, 0)
                tokens = torch.stack((event, start, end), -1)
                sampled.append(self._scores(tokens, (event_logits, start_logits, end_logits)))
            return DecisionScores(**{
                name: torch.cat([getattr(row, name) for row in sampled], dim=0)
                for name in DecisionScores.__dataclass_fields__
            })


class EventGroundModel(nn.Module):
    """Own only the child OFT adapter and event head; the other M4 branches are absent."""

    def __init__(self, conditioning_adapter: nn.Module, head: EventGroundHead):
        super().__init__()
        self.conditioning_adapter = conditioning_adapter
        self.head = head

    def forward(self, backbone_inputs: Mapping[str, Tensor], proprioception: Tensor) -> Tensor:
        if not isinstance(backbone_inputs, Mapping) or set(backbone_inputs) != ENCODING_KEYS:
            raise ValueError("causal inputs must contain exactly the pinned five OFT encoding keys")
        if any(not isinstance(value, Tensor) for value in backbone_inputs.values()):
            raise ValueError("causal encodings must be tensors")
        if (proprioception.ndim != 2 or proprioception.shape[1] != 8
                or not torch.isfinite(proprioception).all()):
            raise ValueError("current proprioception must be finite [batch,8]")
        memory = self.conditioning_adapter(backbone_inputs, proprioception)
        if (memory.tokens.shape != (len(proprioception), 8, self.head.input_dim)
                or memory.key_padding_mask.shape != (len(proprioception), 8)
                or memory.key_padding_mask.any() or not torch.isfinite(memory.tokens).all()):
            raise ValueError("M4 must return eight unpadded pre-coupling action slots")
        return memory.tokens

    def score(self, backbone_inputs, proprioception, tokens, *, temperature=1.):
        return self.head.teacher_forced(self(backbone_inputs, proprioception), tokens,
                                        temperature=temperature)

    @torch.no_grad()
    def sample(self, backbone_inputs, proprioception, **sampling):
        return self.head.sample(self(backbone_inputs, proprioception), **sampling)


def configure_eventground_trainables(model: EventGroundModel,
                                    inherited_exact_lora_names: Sequence[str]) -> dict:
    """Freeze everything, then allow exact inherited language LoRA names and new head.

    Inherited visual LoRA is explicitly frozen with the entire vision encoder.
    The caller must read names from the verified M4 checkpoint, then seal the
    returned full allowlist (not just patterns) into the implementation contract.
    """
    names = tuple(inherited_exact_lora_names)
    parameters = dict(model.named_parameters())
    if not names or len(names) != len(set(names)):
        raise ValueError("the inherited checkpoint LoRA allowlist must be nonempty and unique")
    prefix = "conditioning_adapter.backbone."
    for name in names:
        if (name not in parameters or not name.startswith(prefix)
                or ".lora_" not in name
                or not name.endswith((".lora_A.b0_oft.weight", ".lora_B.b0_oft.weight"))):
            raise ValueError(f"invalid inherited exact LoRA parameter: {name}")
        if not name.startswith((prefix + "language_model.", prefix + "visual.")):
            raise ValueError(f"unknown inherited LoRA branch: {name}")
    actual = {name for name in parameters if ".lora_" in name}
    if actual != set(names):
        raise ValueError("checkpoint exact LoRA list does not cover all instantiated adapters")
    language = tuple(sorted(name for name in names if name.startswith(prefix + "language_model.")))
    vision = tuple(sorted(set(names) - set(language)))
    if not language:
        raise ValueError("no inherited language LoRA found")
    head = tuple(sorted(name for name in parameters if name.startswith("head.")))
    allowed = set(language) | set(head)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in allowed)
    return {"head": head, "language_lora": language, "frozen_vision_lora": vision}
