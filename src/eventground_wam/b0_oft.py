"""Qwen3.5 B0 revision: in-backbone state/action tokens and a residual MLP.

This is an OFT-inspired architecture, not a StarVLA checkpoint reproduction.
The original B0 objective, action normalization, cameras, and chunk length remain.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from eventground_wam.b0_model import (
    B0ActionPolicy,
    B0ModelError,
    B0ObjectiveConfig,
    ConditioningMemory,
    QwenB0InputBuilder,
    load_pinned_qwen_model,
    load_pinned_qwen_processor,
)

POLICY_FAMILY = "qwen35-oft-b0-v1"
ACTION_MARKER = "<|fim_prefix|>"
TRAINABLE_PREFIXES = ("conditioning_adapter.proprioception_projector.", "action_decoder.")


class OFTInputBuilder(QwenB0InputBuilder):
    """Keep native dual-image processing; append one state and eight action slots."""

    def __init__(self, processor: Any) -> None:
        super().__init__(processor)
        ids = processor.tokenizer(ACTION_MARKER, add_special_tokens=False)["input_ids"]
        repeated = processor.tokenizer(ACTION_MARKER * 8, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or repeated != ids * 8:
            raise B0ModelError("OFT marker must encode to exactly one token per action")
        self.marker_id = ids[0]

    def build_messages(self, instructions: Any, front_images: Any, wrist_images: Any) -> Any:
        if any(ACTION_MARKER in value for value in instructions):
            raise B0ModelError("instruction collides with reserved OFT marker")
        messages = super().build_messages(instructions, front_images, wrist_images)
        for conversation, instruction in zip(messages, instructions, strict=True):
            conversation[0]["content"][-1]["text"] = (
                f"Instruction: {instruction}\nRobot state: {ACTION_MARKER}\n"
                f"Predict the next 8 robot actions: {ACTION_MARKER * 8}"
            )
        return messages

    def _validate_encoding(self, encoding: Any, batch_size: int) -> None:
        super()._validate_encoding(encoding, batch_size)
        marker_positions(encoding["input_ids"], encoding["attention_mask"], self.marker_id)


def marker_positions(input_ids: Tensor, attention_mask: Tensor, marker_id: int) -> Tensor:
    valid = (input_ids == marker_id) & attention_mask.bool()
    if not torch.all(valid.sum(-1) == 9):
        raise B0ModelError("each OFT prompt requires one state and eight action slots")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    return positions.masked_fill(~valid, input_ids.shape[1]).sort(-1).values[:, :9]


def normalize_state(state: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Per-dimension train-only quantiles, with constant channels kept at zero."""
    span = upper - lower
    scaled = 2 * (state.float() - lower) / span.clamp_min(1e-6) - 1
    return torch.where(span > 1e-6, scaled.clamp(-1, 1), torch.zeros_like(scaled))


class OFTConditioningAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, marker_id: int, recipe: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        self.marker_id = marker_id
        self.config = SimpleNamespace(decoder_hidden_size=2048)
        stats = recipe["state_normalization"]
        self.register_buffer("state_lower", torch.tensor(stats["lower_q01"], dtype=torch.float32))
        self.register_buffer("state_upper", torch.tensor(stats["upper_q99"], dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(recipe["projector_initialization_seed"])
            self.proprioception_projector = nn.Sequential(
                nn.Linear(8, 2048), nn.GELU(), nn.Linear(2048, 2048),
            )

    def forward(self, backbone_inputs: Any, proprioception: Tensor, **kwargs: Any) -> Any:
        if any(value is not None for value in kwargs.values()):
            raise B0ModelError("future-token conditioning is not enabled for OFT B0")
        ids = backbone_inputs["input_ids"]
        mask = backbone_inputs["attention_mask"]
        if proprioception.shape != (ids.shape[0], 8) or not torch.isfinite(proprioception).all():
            raise B0ModelError("OFT state must be finite [batch,8]")
        positions = marker_positions(ids, mask, self.marker_id)
        embeds = self.backbone.get_input_embeddings()(ids)
        state = normalize_state(proprioception, self.state_lower, self.state_upper)
        state_embed = self.proprioception_projector(state).to(embeds.dtype)
        # Replace only the state placeholder. Actions never appear in model inputs.
        embeds = embeds.scatter(1, positions[:, :1, None].expand(-1, -1, embeds.shape[-1]),
                                state_embed[:, None, :])
        # inputs_embeds bypasses Qwen's automatic input-id-based M-RoPE computation.
        # Compute native image/text positions explicitly on every independent batch.
        position_ids, _ = self.backbone.get_rope_index(
            ids, image_grid_thw=backbone_inputs["image_grid_thw"],
            attention_mask=mask, mm_token_type_ids=backbone_inputs["mm_token_type_ids"],
        )
        inputs = {key: value for key, value in backbone_inputs.items()
                  if key not in {"input_ids", "position_ids"}}
        output = self.backbone(**inputs, inputs_embeds=embeds, position_ids=position_ids,
                               use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        actions = hidden.gather(1, positions[:, 1:, None].expand(-1, -1, hidden.shape[-1]))
        return ConditioningMemory(actions.float(), torch.zeros_like(positions[:, 1:],
                                                                    dtype=torch.bool))


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension),
                                     nn.ReLU())

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


class OFTActionHead(nn.Module):
    """StarVLA-style two-block residual MLP, one hidden vector per action step."""

    def __init__(self, seed: int, dimension: int = 2048, hidden: int = 4096) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_dimension=dimension)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.ReLU(),
                ResidualBlock(hidden), ResidualBlock(hidden), nn.LayerNorm(hidden),
                nn.Linear(hidden, 7),
            )

    def forward(self, memory: ConditioningMemory) -> Tensor:
        return self.network(memory.tokens)


def install_oft_lora(full_model: nn.Module, recipe: dict[str, Any]) -> tuple[str, ...]:
    from peft import LoraConfig, TaskType

    # Explicit full module names: excludes embeddings/LM output, includes all 24
    # language blocks (attention, DeltaNet projections, FFN) and vision linears.
    targets = [name for name, module in full_model.named_modules()
               if isinstance(module, nn.Linear) and name.startswith("model.")]
    if not targets:
        raise B0ModelError("no Qwen linear modules found for broad OFT adaptation")
    full_model.add_adapter(LoraConfig(
        r=recipe["lora_rank"], lora_alpha=recipe["lora_alpha"], lora_dropout=0.0,
        bias="none", target_modules=targets, task_type=TaskType.CAUSAL_LM,
    ), adapter_name="b0_oft")
    core = full_model.model
    names = tuple(name for name, parameter in core.named_parameters() if parameter.requires_grad)
    if any(".lora_" not in name for name in names):
        raise B0ModelError("OFT adaptation unexpectedly unfroze base weights")
    if not all(any(name.startswith(f"language_model.layers.{layer}.") for name in names)
               for layer in range(24)) or not any(name.startswith("visual.") for name in names):
        raise B0ModelError("OFT adapters must cover every language layer and the vision encoder")
    if recipe["gradient_checkpointing"]:
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return names


def build_oft_policy(recipe: dict[str, Any], cache_dir: str, device: Any) -> Any:
    model = load_pinned_qwen_model(cache_dir=cache_dir, local_files_only=True)
    names = install_oft_lora(model, recipe)
    builder = OFTInputBuilder(load_pinned_qwen_processor(cache_dir=cache_dir,
                                                       local_files_only=True))
    policy = B0ActionPolicy(
        OFTConditioningAdapter(model.model, builder.marker_id, recipe),
        OFTActionHead(recipe["decoder_initialization_seed"]),
        objective_config=B0ObjectiveConfig(recipe["objective"], recipe["gripper_bce_weight"]),
    ).to(device)
    policy.trainable_parameter_prefixes = TRAINABLE_PREFIXES
    return policy, tuple(f"conditioning_adapter.backbone.{name}" for name in names)
