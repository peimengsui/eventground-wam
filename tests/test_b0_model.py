from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from eventground_wam.b0_model import (
    QWEN_IMAGE_TOKEN_ID,
    ActionQueryTransformer,
    ActionQueryTransformerConfig,
    B0ActionPolicy,
    B0ModelError,
    ConditioningMemory,
    QwenB0InputBuilder,
    QwenConditioningAdapter,
    QwenConditioningConfig,
    b0_trainable_state_dict,
    load_b0_trainable_state_dict,
    normalized_action_l1_loss,
    validate_b0_trainable_parameters,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes, contract_digest


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.messages: list[list[dict[str, Any]]] | None = None
        self.arguments: dict[str, Any] | None = None
        self.bad_grid = False
        self.bad_token_count = False

    def apply_chat_template(
        self,
        messages: list[list[dict[str, Any]]],
        **arguments: Any,
    ) -> dict[str, Tensor]:
        self.messages = messages
        self.arguments = arguments
        batch_size = len(messages)
        input_ids = torch.zeros(batch_size, 140, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        token_types = torch.zeros_like(input_ids)
        for index in range(batch_size):
            active_tokens = 136 + index
            attention_mask[index, :active_tokens] = 1
            image_tokens = 127 if self.bad_token_count else 128
            input_ids[index, :image_tokens] = QWEN_IMAGE_TOKEN_ID
            token_types[index, :image_tokens] = 1
        grid = torch.tensor([[1, 16, 16]] * (batch_size * 2), dtype=torch.long)
        if self.bad_grid:
            grid[0, 1] = 8
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": token_types,
            "pixel_values": torch.zeros(batch_size * 2, 16, dtype=torch.float32),
            "image_grid_thw": grid,
        }


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, hidden_size))
        self.lora_gain = nn.Parameter(torch.tensor(0.25))
        self.last_use_cache: bool | None = None
        self.last_return_dict: bool | None = None
        self.last_grad_enabled: bool | None = None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        use_cache: bool,
        return_dict: bool,
        **_: Tensor,
    ) -> SimpleNamespace:
        self.last_use_cache = use_cache
        self.last_return_dict = return_dict
        self.last_grad_enabled = torch.is_grad_enabled()
        base = input_ids.to(dtype=self.scale.dtype).unsqueeze(-1)
        hidden = torch.sin(base / 10.0 + self.scale + self.lora_gain)
        assert hidden.shape[:2] == attention_mask.shape
        return SimpleNamespace(last_hidden_state=hidden)


def _small_policy() -> tuple[FakeBackbone, B0ActionPolicy]:
    backbone = FakeBackbone(hidden_size=32)
    adapter = QwenConditioningAdapter(
        backbone,
        config=QwenConditioningConfig(
            backbone_hidden_size=32,
            decoder_hidden_size=16,
            initialization_seed=3,
        ),
    )
    decoder = ActionQueryTransformer(
        config=ActionQueryTransformerConfig(
            model_dimension=16,
            layer_count=2,
            attention_head_count=4,
            feedforward_dimension=32,
            initialization_seed=5,
        )
    )
    return backbone, B0ActionPolicy(adapter, decoder)


def _backbone_inputs() -> dict[str, Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
        "mm_token_type_ids": torch.zeros(2, 4, dtype=torch.long),
        "pixel_values": torch.zeros(4, 8),
        "image_grid_thw": torch.tensor([[1, 16, 16]] * 4),
    }


def test_qwen_prompt_builder_locks_camera_order_and_processor_arguments() -> None:
    processor = FakeProcessor()
    builder = QwenB0InputBuilder(processor)
    front = [torch.full((3, 256, 256), 10, dtype=torch.uint8)]
    wrist = [torch.full((3, 256, 256), 20, dtype=torch.uint8)]

    encoding = builder(["pick up the red block"], front, wrist)

    assert processor.tokenizer.padding_side == "right"
    assert processor.arguments == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_dict": True,
        "return_tensors": "pt",
        "processor_kwargs": {"padding": True},
    }
    assert processor.messages is not None
    content = processor.messages[0][0]["content"]
    assert [item["type"] for item in content] == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == "Front camera image:"
    assert content[1]["image"] is front[0]
    assert content[2]["text"] == "Wrist camera image:"
    assert content[3]["image"] is wrist[0]
    assert content[4]["text"] == (
        "Instruction: pick up the red block\nPredict the next 8 robot actions."
    )
    assert encoding["image_grid_thw"].tolist() == [[1, 16, 16], [1, 16, 16]]
    assert int((encoding["input_ids"] == QWEN_IMAGE_TOKEN_ID).sum()) == 128


@pytest.mark.parametrize(("bad_grid", "bad_token_count"), [(True, False), (False, True)])
def test_qwen_prompt_builder_rejects_changed_visual_preprocessing(
    bad_grid: bool,
    bad_token_count: bool,
) -> None:
    processor = FakeProcessor()
    processor.bad_grid = bad_grid
    processor.bad_token_count = bad_token_count
    builder = QwenB0InputBuilder(processor)
    image = torch.zeros(3, 256, 256, dtype=torch.uint8)

    with pytest.raises(B0ModelError, match="grid|image tokens"):
        builder(["instruction"], [image], [image])


def test_conditioning_and_action_shapes_future_interface_and_l1() -> None:
    backbone, policy = _small_policy()
    inputs = _backbone_inputs()
    state = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    future = torch.ones(2, 3, 16)
    future_mask = torch.tensor([[False, False, True], [False, True, True]])
    targets = torch.zeros(2, 8, 7)

    output = policy(
        inputs,
        state,
        normalized_action_targets=targets,
        future_memory=future,
        future_key_padding_mask=future_mask,
    )

    assert output.normalized_actions.shape == (2, 8, 7)
    assert output.conditioning_memory.tokens.shape == (2, 8, 16)
    assert output.conditioning_memory.key_padding_mask.tolist() == [
        [False, False, False, True, False, False, False, True],
        [False, False, True, True, False, False, True, True],
    ]
    assert output.loss is not None and output.loss.ndim == 0
    assert torch.equal(output.loss, output.normalized_actions.abs().mean())
    assert backbone.last_use_cache is False
    assert backbone.last_return_dict is True


def test_gradient_allowlist_freezes_backbone_and_trains_only_declared_modules() -> None:
    backbone, policy = _small_policy()
    policy.train()
    assert backbone.training is False
    trainable_names = validate_b0_trainable_parameters(policy)
    assert trainable_names
    assert not any(name.startswith("conditioning_adapter.backbone.") for name in trainable_names)

    output = policy(
        _backbone_inputs(),
        torch.ones(2, 8),
        normalized_action_targets=torch.zeros(2, 8, 7),
    )
    assert output.loss is not None
    output.loss.backward()

    assert backbone.scale.requires_grad is False
    assert backbone.lora_gain.requires_grad is False
    assert backbone.scale.grad is None
    assert any(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if name.startswith("conditioning_adapter.backbone_projection.")
    )
    assert any(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if name.startswith("conditioning_adapter.proprioception_projector.")
    )
    assert any(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if name.startswith("action_decoder.")
    )


def test_exact_backbone_lora_allowlist_and_compact_checkpoint_round_trip() -> None:
    backbone = FakeBackbone(hidden_size=32)
    adapter = QwenConditioningAdapter(
        backbone,
        config=QwenConditioningConfig(backbone_hidden_size=32, decoder_hidden_size=16),
        trainable_backbone_parameter_names=("lora_gain",),
    )
    policy = B0ActionPolicy(
        adapter,
        ActionQueryTransformer(
            config=ActionQueryTransformerConfig(
                model_dimension=16,
                layer_count=2,
                attention_head_count=4,
                feedforward_dimension=32,
            )
        ),
    )
    exact_name = "conditioning_adapter.backbone.lora_gain"
    trainable_names = validate_b0_trainable_parameters(
        policy,
        allowed_exact_names=(exact_name,),
    )
    assert exact_name in trainable_names
    assert backbone.lora_gain.requires_grad is True
    assert backbone.scale.requires_grad is False
    output = policy(
        _backbone_inputs(),
        torch.ones(2, 8),
        normalized_action_targets=torch.zeros(2, 8, 7),
    )
    assert output.loss is not None
    output.loss.backward()
    assert backbone.lora_gain.grad is not None
    assert torch.isfinite(backbone.lora_gain.grad)

    checkpoint = b0_trainable_state_dict(
        policy,
        allowed_exact_names=(exact_name,),
    )
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name in trainable_names:
            parameters[name].add_(1.0)
    load_b0_trainable_state_dict(
        policy,
        checkpoint,
        allowed_exact_names=(exact_name,),
    )
    for name, expected in checkpoint.items():
        torch.testing.assert_close(parameters[name], expected, rtol=0.0, atol=0.0)
    with pytest.raises(B0ModelError, match="checkpoint keys differ"):
        load_b0_trainable_state_dict(
            policy,
            {name: value for name, value in checkpoint.items() if name != exact_name},
            allowed_exact_names=(exact_name,),
        )


def test_trainable_lora_honors_outer_no_grad_during_inference() -> None:
    backbone = FakeBackbone(hidden_size=32)
    adapter = QwenConditioningAdapter(
        backbone,
        config=QwenConditioningConfig(backbone_hidden_size=32, decoder_hidden_size=16),
        trainable_backbone_parameter_names=("lora_gain",),
    )
    adapter.eval()

    with torch.no_grad():
        memory = adapter(_backbone_inputs(), torch.ones(2, 8))

    assert backbone.last_grad_enabled is False
    assert memory.tokens.requires_grad is False


def test_action_decoder_initialization_and_checkpoint_reload_are_deterministic() -> None:
    config = ActionQueryTransformerConfig(
        model_dimension=16,
        layer_count=2,
        attention_head_count=4,
        feedforward_dimension=32,
        initialization_seed=91,
    )
    first = ActionQueryTransformer(config=config).eval()
    torch.manual_seed(999)
    _ = torch.rand(100)
    second = ActionQueryTransformer(config=config).eval()
    for first_value, second_value in zip(
        first.state_dict().values(),
        second.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)

    memory = ConditioningMemory(
        tokens=torch.randn(2, 5, 16),
        key_padding_mask=torch.tensor(
            [[False, False, False, True, True], [False, False, False, False, True]]
        ),
    )
    expected = first(memory)
    reloaded = ActionQueryTransformer(config=config).eval()
    reloaded.load_state_dict(first.state_dict())
    actual = reloaded(memory)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_masked_memory_values_do_not_change_actions() -> None:
    decoder = ActionQueryTransformer(
        config=ActionQueryTransformerConfig(
            model_dimension=16,
            layer_count=2,
            attention_head_count=4,
            feedforward_dimension=32,
        )
    ).eval()
    tokens = torch.randn(1, 5, 16)
    mask = torch.tensor([[False, False, False, True, True]])
    changed = tokens.clone()
    changed[:, 3:] = 100_000.0

    original_actions = decoder(ConditioningMemory(tokens, mask))
    changed_actions = decoder(ConditioningMemory(changed, mask))

    torch.testing.assert_close(changed_actions, original_actions, rtol=0.0, atol=0.0)


def test_normalized_l1_is_unweighted_mean_over_complete_chunk() -> None:
    predictions = torch.arange(112, dtype=torch.float32).reshape(2, 8, 7) / 10.0
    targets = torch.ones_like(predictions)

    loss = normalized_action_l1_loss(predictions, targets)

    assert torch.equal(loss, torch.abs(predictions - targets).sum() / predictions.numel())
    with pytest.raises(B0ModelError, match="shape"):
        normalized_action_l1_loss(predictions[:, :-1], targets[:, :-1])


def test_model_rejects_future_mask_without_future_memory() -> None:
    _, policy = _small_policy()
    with pytest.raises(B0ModelError, match="requires future_memory"):
        policy.conditioning_adapter(
            _backbone_inputs(),
            torch.ones(2, 8),
            future_key_padding_mask=torch.zeros(2, 1, dtype=torch.bool),
        )


def test_pinned_qwen_manifest_is_complete_and_sealed_before_weight_download() -> None:
    path = Path(__file__).parents[1] / "docs/model/Qwen3.5-2B-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["digest"] == contract_digest(manifest)
    assert manifest["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert manifest["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert manifest["file_count"] == len(manifest["files"]) == 13
    assert manifest["total_size_bytes"] == sum(item["size"] for item in manifest["files"])
    assert manifest["file_list_digest"] == hashlib.sha256(
        canonical_json_bytes({"files": manifest["files"]})
    ).hexdigest()
    assert manifest["provenance"]["model_files_downloaded_before_manifest"] is False
    weights = next(
        item for item in manifest["files"] if item["path"].startswith("model.safetensors-")
    )
    assert weights["lfs_sha256"] == (
        "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
    )
