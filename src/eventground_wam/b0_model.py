"""Qwen conditioning and deterministic action-query policy components for B0."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

QWEN_REPOSITORY_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
QWEN_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
QWEN_PROCESSOR_CLASS = "Qwen3VLProcessor"
QWEN_BACKBONE_HIDDEN_SIZE = 2_048
QWEN_IMAGE_TOKEN_ID = 248_056
QWEN_IMAGE_GRID_THW = (1, 16, 16)
QWEN_IMAGE_TOKENS_PER_CAMERA = 64
QWEN_LORA_ADAPTER_NAME = "b0_upper"
QWEN_LORA_LAYERS = (19, 23)
QWEN_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
QWEN_LORA_RANK = 16
QWEN_LORA_ALPHA = 32

B0_STATE_DIMENSION = 8
B0_ACTION_DIMENSION = 7
B0_ACTION_CHUNK_LENGTH = 8
B0_TRAINABLE_PARAMETER_PREFIXES = (
    "conditioning_adapter.backbone_projection.",
    "conditioning_adapter.proprioception_projector.",
    "action_decoder.",
)


class B0ModelError(ValueError):
    """Raised when a B0 prompt, tensor, or model violates the locked contract."""


@dataclass(frozen=True)
class QwenPromptConfig:
    """Locked prompt, image-processing, and token-selection semantics."""

    repository_id: str = QWEN_REPOSITORY_ID
    revision: str = QWEN_REVISION
    processor_class: str = QWEN_PROCESSOR_CLASS
    prompt_version: str = "front-wrist-instruction-action-request-v1"
    front_label: str = "Front camera image:"
    wrist_label: str = "Wrist camera image:"
    instruction_prefix: str = "Instruction: "
    action_request: str = "Predict the next 8 robot actions."
    add_generation_prompt: bool = True
    enable_thinking: bool = False
    padding_side: str = "right"
    image_token_id: int = QWEN_IMAGE_TOKEN_ID
    expected_image_grid_thw: tuple[int, int, int] = QWEN_IMAGE_GRID_THW
    expected_image_tokens_per_camera: int = QWEN_IMAGE_TOKENS_PER_CAMERA
    camera_order: tuple[str, str] = ("front", "wrist")
    token_selection: str = "all-non-padding-final-hidden-state"
    resize_policy: str = "processor-pinned-256x256-no-resize"

    def __post_init__(self) -> None:
        locked = {
            "repository_id": (self.repository_id, QWEN_REPOSITORY_ID),
            "revision": (self.revision, QWEN_REVISION),
            "processor_class": (self.processor_class, QWEN_PROCESSOR_CLASS),
            "prompt_version": (
                self.prompt_version,
                "front-wrist-instruction-action-request-v1",
            ),
            "padding_side": (self.padding_side, "right"),
            "camera_order": (self.camera_order, ("front", "wrist")),
            "token_selection": (
                self.token_selection,
                "all-non-padding-final-hidden-state",
            ),
            "resize_policy": (
                self.resize_policy,
                "processor-pinned-256x256-no-resize",
            ),
        }
        for name, (actual, expected) in locked.items():
            if actual != expected:
                raise B0ModelError(f"{name} must be locked to {expected!r}")
        if self.add_generation_prompt is not True or self.enable_thinking is not False:
            raise B0ModelError("Qwen B0 uses a generation prompt with thinking disabled")
        if self.image_token_id != QWEN_IMAGE_TOKEN_ID:
            raise B0ModelError(f"image_token_id must be {QWEN_IMAGE_TOKEN_ID}")
        if self.expected_image_grid_thw != QWEN_IMAGE_GRID_THW:
            raise B0ModelError(f"image grid must be {QWEN_IMAGE_GRID_THW}")
        if self.expected_image_tokens_per_camera != QWEN_IMAGE_TOKENS_PER_CAMERA:
            raise B0ModelError(
                f"each camera must contribute {QWEN_IMAGE_TOKENS_PER_CAMERA} image tokens"
            )
        text_fields = (
            self.front_label,
            self.wrist_label,
            self.instruction_prefix,
            self.action_request,
        )
        if any(not isinstance(value, str) or not value for value in text_fields):
            raise B0ModelError("prompt text fields must be non-empty strings")


class QwenB0InputBuilder:
    """Create and validate a batched Qwen processor encoding for two cameras."""

    REQUIRED_ENCODING_KEYS = frozenset(
        {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        }
    )

    def __init__(self, processor: Any, *, config: QwenPromptConfig | None = None) -> None:
        self.processor = processor
        self.config = config or QwenPromptConfig()
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "padding_side"):
            raise B0ModelError("Qwen processor must expose tokenizer.padding_side")
        tokenizer.padding_side = self.config.padding_side

    def build_messages(
        self,
        instructions: Sequence[str],
        front_images: Sequence[Any],
        wrist_images: Sequence[Any],
    ) -> list[list[dict[str, Any]]]:
        """Build the exact front/wrist/instruction/action-request prompt order."""

        batch_size = len(instructions)
        if batch_size == 0:
            raise B0ModelError("cannot build an empty Qwen batch")
        if len(front_images) != batch_size or len(wrist_images) != batch_size:
            raise B0ModelError("instructions and both camera batches must have equal lengths")
        conversations: list[list[dict[str, Any]]] = []
        for index, (instruction, front, wrist) in enumerate(
            zip(instructions, front_images, wrist_images, strict=True)
        ):
            if not isinstance(instruction, str) or not instruction:
                raise B0ModelError(f"instruction {index} must be a non-empty string")
            _validate_source_image(front, f"front image {index}")
            _validate_source_image(wrist, f"wrist image {index}")
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.front_label},
                            {"type": "image", "image": front},
                            {"type": "text", "text": self.config.wrist_label},
                            {"type": "image", "image": wrist},
                            {
                                "type": "text",
                                "text": (
                                    f"{self.config.instruction_prefix}{instruction}\n"
                                    f"{self.config.action_request}"
                                ),
                            },
                        ],
                    }
                ]
            )
        return conversations

    def __call__(
        self,
        instructions: Sequence[str],
        front_images: Sequence[Any],
        wrist_images: Sequence[Any],
    ) -> dict[str, Tensor]:
        conversations = self.build_messages(instructions, front_images, wrist_images)
        try:
            encoding = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=self.config.add_generation_prompt,
                enable_thinking=self.config.enable_thinking,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
        except Exception as error:
            raise B0ModelError("Qwen processor could not encode the locked B0 prompt") from error
        if not isinstance(encoding, Mapping):
            raise B0ModelError("Qwen processor must return a mapping")
        tensors = dict(encoding)
        self._validate_encoding(tensors, len(conversations))
        return tensors

    def _validate_encoding(self, encoding: Mapping[str, Any], batch_size: int) -> None:
        missing = self.REQUIRED_ENCODING_KEYS - set(encoding)
        if missing:
            raise B0ModelError(f"Qwen encoding is missing required fields: {sorted(missing)}")
        if any(not isinstance(encoding[name], Tensor) for name in self.REQUIRED_ENCODING_KEYS):
            raise B0ModelError("all required Qwen encoding fields must be tensors")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_types = encoding["mm_token_type_ids"]
        image_grid = encoding["image_grid_thw"]
        if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
            raise B0ModelError("input_ids must have shape [batch, sequence]")
        if attention_mask.shape != input_ids.shape or token_types.shape != input_ids.shape:
            raise B0ModelError("attention_mask and mm_token_type_ids must match input_ids")
        if image_grid.shape != (batch_size * 2, 3):
            raise B0ModelError("image_grid_thw must contain front and wrist rows per sample")
        expected_grid = torch.tensor(
            self.config.expected_image_grid_thw,
            device=image_grid.device,
            dtype=image_grid.dtype,
        )
        if not torch.equal(image_grid, expected_grid.expand_as(image_grid)):
            raise B0ModelError(
                f"each source image must retain grid {self.config.expected_image_grid_thw}"
            )
        expected_image_tokens = self.config.expected_image_tokens_per_camera * 2
        actual_image_tokens = (input_ids == self.config.image_token_id).sum(dim=1)
        if not torch.equal(
            actual_image_tokens,
            torch.full_like(actual_image_tokens, expected_image_tokens),
        ):
            raise B0ModelError(
                f"every prompt must contain {expected_image_tokens} merged image tokens"
            )
        bool_attention = attention_mask.to(dtype=torch.bool)
        if not bool_attention.any(dim=1).all():
            raise B0ModelError("every prompt must contain at least one non-padding token")
        # Right padding is locked so each attention row is a run of ones followed by zeros.
        if ((~bool_attention[:, :-1]) & bool_attention[:, 1:]).any():
            raise B0ModelError("Qwen attention_mask is not right padded")


def load_pinned_qwen_processor(
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load the pinned processor with an explicit torchvision image backend."""

    try:
        from transformers import AutoProcessor
    except ImportError as error:  # pragma: no cover - exercised in the B0 runtime
        raise RuntimeError("the pinned B0 runtime is required to load Qwen processing") from error
    try:
        processor = AutoProcessor.from_pretrained(
            QWEN_REPOSITORY_ID,
            revision=QWEN_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
            image_processor_kwargs={"backend": "torchvision"},
        )
    except Exception as error:
        raise B0ModelError("cannot load the pinned Qwen processor") from error
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    locked_processor_values = {
        "processor class": (type(processor).__name__, QWEN_PROCESSOR_CLASS),
        "image backend": (getattr(image_processor, "backend", None), "torchvision"),
        "patch size": (getattr(image_processor, "patch_size", None), 16),
        "spatial merge size": (getattr(image_processor, "merge_size", None), 2),
        "temporal patch size": (getattr(image_processor, "temporal_patch_size", None), 2),
        "shortest edge": (getattr(size, "shortest_edge", None), 65_536),
        "longest edge": (getattr(size, "longest_edge", None), 16_777_216),
        "image mean": (tuple(getattr(image_processor, "image_mean", ())), (0.5, 0.5, 0.5)),
        "image std": (tuple(getattr(image_processor, "image_std", ())), (0.5, 0.5, 0.5)),
    }
    for name, (actual, expected) in locked_processor_values.items():
        if actual != expected:
            raise B0ModelError(f"pinned Qwen {name} is {actual!r}, expected {expected!r}")
    return processor


def load_pinned_qwen_model(
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load the manifest-pinned BF16 Qwen model with PyTorch SDPA."""

    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as error:  # pragma: no cover - exercised in the B0 runtime
        raise RuntimeError("the pinned B0 runtime is required to load Qwen weights") from error
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            QWEN_REPOSITORY_ID,
            revision=QWEN_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
            use_safetensors=True,
            weights_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    except Exception as error:
        raise B0ModelError("cannot load the pinned Qwen3.5-2B BF16 model") from error
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    locked_model_values = {
        "model class": (type(model).__name__, QWEN_ARCHITECTURE),
        "model type": (getattr(config, "model_type", None), "qwen3_5"),
        "text hidden size": (
            getattr(text_config, "hidden_size", None),
            QWEN_BACKBONE_HIDDEN_SIZE,
        ),
        "text layer count": (getattr(text_config, "num_hidden_layers", None), 24),
        "vision patch size": (getattr(vision_config, "patch_size", None), 16),
        "vision merge size": (getattr(vision_config, "spatial_merge_size", None), 2),
        "vision output size": (
            getattr(vision_config, "out_hidden_size", None),
            QWEN_BACKBONE_HIDDEN_SIZE,
        ),
        "image token id": (getattr(config, "image_token_id", None), QWEN_IMAGE_TOKEN_ID),
    }
    for name, (actual, expected) in locked_model_values.items():
        if actual != expected:
            raise B0ModelError(f"pinned Qwen {name} is {actual!r}, expected {expected!r}")
    if not isinstance(getattr(model, "model", None), nn.Module):
        raise B0ModelError("Qwen conditional-generation model has no multimodal core")
    return model


def install_qwen_upper_lora(
    model: nn.Module,
    *,
    adapter_name: str = QWEN_LORA_ADAPTER_NAME,
) -> tuple[str, ...]:
    """Install the locked rank-16 LoRA on Qwen full-attention layers 19 and 23."""

    if adapter_name != QWEN_LORA_ADAPTER_NAME:
        raise B0ModelError(f"Qwen LoRA adapter name must be {QWEN_LORA_ADAPTER_NAME!r}")
    if type(model).__name__ != QWEN_ARCHITECTURE:
        raise B0ModelError(f"LoRA target must be a {QWEN_ARCHITECTURE}")
    try:
        from peft import LoraConfig, TaskType
    except ImportError as error:  # pragma: no cover - exercised in the B0 runtime
        raise RuntimeError("the pinned B0 runtime is required to install Qwen LoRA") from error
    try:
        model.add_adapter(
            LoraConfig(
                r=QWEN_LORA_RANK,
                lora_alpha=QWEN_LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=list(QWEN_LORA_TARGET_MODULES),
                layers_to_transform=list(QWEN_LORA_LAYERS),
                layers_pattern="layers",
                task_type=TaskType.CAUSAL_LM,
            ),
            adapter_name=adapter_name,
        )
    except Exception as error:
        raise B0ModelError("could not install the locked Qwen upper-layer LoRA") from error
    core = getattr(model, "model", None)
    if not isinstance(core, nn.Module):
        raise B0ModelError("Qwen model has no multimodal core after LoRA installation")
    expected_names = _expected_qwen_core_lora_names(adapter_name)
    actual_names = tuple(name for name, parameter in core.named_parameters() if parameter.requires_grad)
    if actual_names != expected_names:
        raise B0ModelError(
            "Qwen LoRA trainable names differ from the locked layer/module allowlist: "
            f"expected={expected_names}, actual={actual_names}"
        )
    return actual_names


@dataclass(frozen=True)
class QwenConditioningConfig:
    """Projection and proprioception-fusion dimensions for Qwen memory."""

    backbone_hidden_size: int = QWEN_BACKBONE_HIDDEN_SIZE
    decoder_hidden_size: int = 512
    state_dimension: int = B0_STATE_DIMENSION
    token_selection: str = "all-non-padding-final-hidden-state"
    memory_order: tuple[str, str, str] = ("qwen", "proprioception", "future")
    freeze_backbone: bool = True
    initialization_seed: int = 7

    def __post_init__(self) -> None:
        for name, value in (
            ("backbone_hidden_size", self.backbone_hidden_size),
            ("decoder_hidden_size", self.decoder_hidden_size),
            ("state_dimension", self.state_dimension),
        ):
            if type(value) is not int or value <= 0:
                raise B0ModelError(f"{name} must be a positive integer")
        if self.state_dimension != B0_STATE_DIMENSION:
            raise B0ModelError(f"state_dimension must be {B0_STATE_DIMENSION}")
        if self.token_selection != "all-non-padding-final-hidden-state":
            raise B0ModelError("unsupported Qwen token selection")
        if self.memory_order != ("qwen", "proprioception", "future"):
            raise B0ModelError("conditioning memory order must be qwen, proprioception, future")
        if self.freeze_backbone is not True:
            raise B0ModelError("initial B0 must freeze the Qwen backbone")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise B0ModelError("initialization_seed must be a nonnegative integer")


@dataclass(frozen=True)
class ConditioningMemory:
    """Projected tokens and a PyTorch key-padding mask (True means ignored)."""

    tokens: Tensor
    key_padding_mask: Tensor


class QwenConditioningAdapter(nn.Module):
    """Extract final Qwen tokens, project them, and append state/future memory."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        config: QwenConditioningConfig | None = None,
        trainable_backbone_parameter_names: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.config = config or QwenConditioningConfig()
        self.backbone = backbone
        named_backbone_parameters = dict(self.backbone.named_parameters())
        requested_trainable_names = tuple(trainable_backbone_parameter_names)
        if len(requested_trainable_names) != len(set(requested_trainable_names)):
            raise B0ModelError("trainable backbone parameter names must be unique")
        unknown_names = tuple(
            name for name in requested_trainable_names if name not in named_backbone_parameters
        )
        if unknown_names:
            raise B0ModelError(f"unknown trainable backbone parameters: {unknown_names}")
        requested_trainable_set = set(requested_trainable_names)
        for name, parameter in named_backbone_parameters.items():
            parameter.requires_grad_(name in requested_trainable_set)
        self.trainable_backbone_parameter_names = requested_trainable_names
        self.backbone.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.initialization_seed)
            self.backbone_projection = nn.Sequential(
                nn.LayerNorm(self.config.backbone_hidden_size),
                nn.Linear(
                    self.config.backbone_hidden_size,
                    self.config.decoder_hidden_size,
                    bias=False,
                ),
                nn.LayerNorm(self.config.decoder_hidden_size),
            )
            self.proprioception_projector = nn.Sequential(
                nn.LayerNorm(self.config.state_dimension),
                nn.Linear(self.config.state_dimension, self.config.decoder_hidden_size),
                nn.SiLU(),
                nn.Linear(self.config.decoder_hidden_size, self.config.decoder_hidden_size),
            )

    def train(self, mode: bool = True) -> QwenConditioningAdapter:
        super().train(mode)
        # The frozen Qwen core and its deterministic LoRA path remain in eval mode.
        self.backbone.eval()
        return self

    def forward(
        self,
        backbone_inputs: Mapping[str, Tensor],
        proprioception: Tensor,
        *,
        future_memory: Tensor | None = None,
        future_key_padding_mask: Tensor | None = None,
    ) -> ConditioningMemory:
        if "attention_mask" not in backbone_inputs:
            raise B0ModelError("backbone_inputs must contain attention_mask")
        attention_mask = backbone_inputs["attention_mask"]
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise B0ModelError("attention_mask must be a [batch, sequence] tensor")
        if proprioception.ndim != 2 or proprioception.shape != (
            attention_mask.shape[0],
            self.config.state_dimension,
        ):
            raise B0ModelError(
                f"proprioception must have shape [batch,{self.config.state_dimension}]"
            )
        if not torch.isfinite(proprioception).all():
            raise B0ModelError("proprioception contains non-finite values")
        backbone_parameters_trainable = bool(self.trainable_backbone_parameter_names)
        context = nullcontext() if backbone_parameters_trainable else torch.no_grad()
        with context:
            try:
                output = self.backbone(
                    **dict(backbone_inputs),
                    use_cache=False,
                    return_dict=True,
                )
            except Exception as error:
                raise B0ModelError("Qwen multimodal core forward failed") from error
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if not isinstance(last_hidden_state, Tensor):
            raise B0ModelError("Qwen multimodal core must return last_hidden_state")
        if last_hidden_state.shape != (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.config.backbone_hidden_size,
        ):
            raise B0ModelError(
                "Qwen last_hidden_state shape does not match attention mask and hidden size"
            )
        if not torch.isfinite(last_hidden_state).all():
            raise B0ModelError("Qwen final hidden state contains non-finite values")
        projection_parameter = next(self.backbone_projection.parameters())
        hidden = last_hidden_state.to(
            device=projection_parameter.device,
            dtype=projection_parameter.dtype,
        )
        qwen_memory = self.backbone_projection(hidden)
        padding_mask = ~attention_mask.to(device=qwen_memory.device, dtype=torch.bool)

        state = proprioception.to(
            device=projection_parameter.device,
            dtype=projection_parameter.dtype,
        )
        proprioception_memory = self.proprioception_projector(state).unsqueeze(1)
        memory_parts = [qwen_memory, proprioception_memory]
        mask_parts = [
            padding_mask,
            torch.zeros(
                (padding_mask.shape[0], 1),
                device=padding_mask.device,
                dtype=torch.bool,
            ),
        ]
        if future_memory is not None:
            if (
                future_memory.ndim != 3
                or future_memory.shape[0] != attention_mask.shape[0]
                or future_memory.shape[2] != self.config.decoder_hidden_size
            ):
                raise B0ModelError(
                    "future_memory must have shape [batch,future_tokens,decoder_hidden_size]"
                )
            future = future_memory.to(
                device=projection_parameter.device,
                dtype=projection_parameter.dtype,
            )
            if not torch.isfinite(future).all():
                raise B0ModelError("future_memory contains non-finite values")
            if future_key_padding_mask is None:
                future_mask = torch.zeros(
                    future.shape[:2],
                    device=future.device,
                    dtype=torch.bool,
                )
            else:
                if future_key_padding_mask.shape != future.shape[:2]:
                    raise B0ModelError(
                        "future_key_padding_mask must match future_memory's first two axes"
                    )
                future_mask = future_key_padding_mask.to(
                    device=future.device,
                    dtype=torch.bool,
                )
            memory_parts.append(future)
            mask_parts.append(future_mask)
        elif future_key_padding_mask is not None:
            raise B0ModelError("future_key_padding_mask requires future_memory")
        return ConditioningMemory(
            tokens=torch.cat(memory_parts, dim=1),
            key_padding_mask=torch.cat(mask_parts, dim=1),
        )


@dataclass(frozen=True)
class ActionQueryTransformerConfig:
    """Locked deterministic decoder architecture for every B0/M1-M4 action head."""

    model_dimension: int = 512
    action_query_count: int = B0_ACTION_CHUNK_LENGTH
    action_dimension: int = B0_ACTION_DIMENSION
    layer_count: int = 4
    attention_head_count: int = 8
    feedforward_dimension: int = 2_048
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True
    query_attention: str = "bidirectional"
    initialization_seed: int = 11

    def __post_init__(self) -> None:
        integer_fields = (
            ("model_dimension", self.model_dimension),
            ("action_query_count", self.action_query_count),
            ("action_dimension", self.action_dimension),
            ("layer_count", self.layer_count),
            ("attention_head_count", self.attention_head_count),
            ("feedforward_dimension", self.feedforward_dimension),
        )
        for name, value in integer_fields:
            if type(value) is not int or value <= 0:
                raise B0ModelError(f"{name} must be a positive integer")
        if self.action_query_count != B0_ACTION_CHUNK_LENGTH:
            raise B0ModelError(f"action_query_count must be {B0_ACTION_CHUNK_LENGTH}")
        if self.action_dimension != B0_ACTION_DIMENSION:
            raise B0ModelError(f"action_dimension must be {B0_ACTION_DIMENSION}")
        if self.model_dimension % self.attention_head_count != 0:
            raise B0ModelError("model_dimension must be divisible by attention_head_count")
        if self.dropout != 0.0 or not math.isfinite(self.dropout):
            raise B0ModelError("deterministic action decoder requires zero dropout")
        if self.activation != "gelu":
            raise B0ModelError("action decoder activation must be gelu")
        if self.norm_first is not True:
            raise B0ModelError("action decoder must use pre-normalization")
        if self.query_attention != "bidirectional":
            raise B0ModelError("action queries must use bidirectional self-attention")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise B0ModelError("initialization_seed must be a nonnegative integer")


class ActionQueryTransformer(nn.Module):
    """Decode all eight normalized actions from learned, bidirectional queries."""

    def __init__(self, *, config: ActionQueryTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionQueryTransformerConfig()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.initialization_seed)
            layer = nn.TransformerDecoderLayer(
                d_model=self.config.model_dimension,
                nhead=self.config.attention_head_count,
                dim_feedforward=self.config.feedforward_dimension,
                dropout=self.config.dropout,
                activation=self.config.activation,
                batch_first=True,
                norm_first=self.config.norm_first,
            )
            self.decoder = nn.TransformerDecoder(
                layer,
                num_layers=self.config.layer_count,
                norm=nn.LayerNorm(self.config.model_dimension),
            )
            self.action_queries = nn.Parameter(
                torch.empty(self.config.action_query_count, self.config.model_dimension)
            )
            self.action_projection = nn.Linear(
                self.config.model_dimension,
                self.config.action_dimension,
            )
            nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.action_projection.weight)
            nn.init.zeros_(self.action_projection.bias)

    def forward(self, memory: ConditioningMemory) -> Tensor:
        tokens = memory.tokens
        padding_mask = memory.key_padding_mask
        if tokens.ndim != 3 or tokens.shape[2] != self.config.model_dimension:
            raise B0ModelError(
                f"conditioning tokens must have shape [batch,memory,{self.config.model_dimension}]"
            )
        if padding_mask.shape != tokens.shape[:2]:
            raise B0ModelError("conditioning key-padding mask must match token axes")
        if not torch.isfinite(tokens).all():
            raise B0ModelError("conditioning memory contains non-finite values")
        bool_padding_mask = padding_mask.to(device=tokens.device, dtype=torch.bool)
        if bool_padding_mask.all(dim=1).any():
            raise B0ModelError("every sample needs at least one unmasked conditioning token")
        queries = self.action_queries.to(dtype=tokens.dtype).unsqueeze(0).expand(
            tokens.shape[0],
            -1,
            -1,
        )
        decoded = self.decoder(
            tgt=queries,
            memory=tokens,
            memory_key_padding_mask=bool_padding_mask,
        )
        actions = self.action_projection(decoded)
        expected_shape = (
            tokens.shape[0],
            self.config.action_query_count,
            self.config.action_dimension,
        )
        if actions.shape != expected_shape:
            raise B0ModelError(f"decoded actions have shape {actions.shape}, expected {expected_shape}")
        return actions


def normalized_action_l1_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return the unweighted mean L1 over a complete normalized ``[B,8,7]`` chunk."""

    if predictions.ndim != 3 or predictions.shape[1:] != (
        B0_ACTION_CHUNK_LENGTH,
        B0_ACTION_DIMENSION,
    ):
        raise B0ModelError("predictions must have shape [batch,8,7]")
    if targets.shape != predictions.shape:
        raise B0ModelError("normalized action targets must exactly match prediction shape")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise B0ModelError("normalized actions and targets must be finite")
    return functional.l1_loss(predictions, targets, reduction="mean")


@dataclass(frozen=True)
class B0ActionOutput:
    """B0 forward result with optional supervised normalized-action loss."""

    normalized_actions: Tensor
    conditioning_memory: ConditioningMemory
    loss: Tensor | None = None


class B0ActionPolicy(nn.Module):
    """Compose Qwen conditioning, proprioception fusion, and the shared action head."""

    def __init__(
        self,
        conditioning_adapter: QwenConditioningAdapter,
        action_decoder: ActionQueryTransformer,
    ) -> None:
        super().__init__()
        if (
            conditioning_adapter.config.decoder_hidden_size
            != action_decoder.config.model_dimension
        ):
            raise B0ModelError("conditioning and action decoder dimensions must match")
        self.conditioning_adapter = conditioning_adapter
        self.action_decoder = action_decoder

    def forward(
        self,
        backbone_inputs: Mapping[str, Tensor],
        proprioception: Tensor,
        *,
        normalized_action_targets: Tensor | None = None,
        future_memory: Tensor | None = None,
        future_key_padding_mask: Tensor | None = None,
    ) -> B0ActionOutput:
        memory = self.conditioning_adapter(
            backbone_inputs,
            proprioception,
            future_memory=future_memory,
            future_key_padding_mask=future_key_padding_mask,
        )
        predictions = self.action_decoder(memory)
        loss = (
            normalized_action_l1_loss(predictions, normalized_action_targets)
            if normalized_action_targets is not None
            else None
        )
        return B0ActionOutput(
            normalized_actions=predictions,
            conditioning_memory=memory,
            loss=loss,
        )


def validate_b0_trainable_parameters(
    model: nn.Module,
    *,
    allowed_prefixes: Sequence[str] = B0_TRAINABLE_PARAMETER_PREFIXES,
    allowed_exact_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return trainable names after enforcing the initial B0 allowlist exactly."""

    if not allowed_prefixes or any(not prefix for prefix in allowed_prefixes):
        raise B0ModelError("allowed trainable prefixes must be non-empty strings")
    if len(allowed_exact_names) != len(set(allowed_exact_names)):
        raise B0ModelError("allowed exact trainable names must be unique")
    allowed_exact_set = set(allowed_exact_names)
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    unexpected = tuple(
        name
        for name in trainable_names
        if name not in allowed_exact_set
        and not any(name.startswith(prefix) for prefix in allowed_prefixes)
    )
    if unexpected:
        raise B0ModelError(f"unexpected trainable parameters: {unexpected}")
    missing_prefixes = tuple(
        prefix
        for prefix in allowed_prefixes
        if not any(name.startswith(prefix) for name in trainable_names)
    )
    if missing_prefixes:
        raise B0ModelError(f"trainable parameter groups are empty: {missing_prefixes}")
    missing_exact_names = tuple(name for name in allowed_exact_names if name not in trainable_names)
    if missing_exact_names:
        raise B0ModelError(f"exact trainable parameters are missing: {missing_exact_names}")
    return trainable_names


def b0_trainable_state_dict(
    model: nn.Module,
    *,
    allowed_exact_names: Sequence[str] = (),
) -> dict[str, Tensor]:
    """Clone only allowlisted trainable tensors for a compact B0 checkpoint."""

    trainable_names = validate_b0_trainable_parameters(
        model,
        allowed_exact_names=allowed_exact_names,
    )
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().clone() for name in trainable_names}


def load_b0_trainable_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allowed_exact_names: Sequence[str] = (),
) -> None:
    """Strictly restore every and only allowlisted trainable B0 tensor."""

    trainable_names = validate_b0_trainable_parameters(
        model,
        allowed_exact_names=allowed_exact_names,
    )
    if set(state_dict) != set(trainable_names):
        missing = sorted(set(trainable_names) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(trainable_names))
        raise B0ModelError(
            f"trainable checkpoint keys differ: missing={missing}, unexpected={unexpected}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in trainable_names:
            value = state_dict[name]
            parameter = parameters[name]
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                raise B0ModelError(f"checkpoint tensor {name} has the wrong type or shape")
            if not torch.isfinite(value).all():
                raise B0ModelError(f"checkpoint tensor {name} contains non-finite values")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _expected_qwen_core_lora_names(adapter_name: str) -> tuple[str, ...]:
    return tuple(
        f"language_model.layers.{layer}.self_attn.{module}.lora_{matrix}."
        f"{adapter_name}.weight"
        for layer in QWEN_LORA_LAYERS
        for module in QWEN_LORA_TARGET_MODULES
        for matrix in ("A", "B")
    )


def _validate_source_image(image: Any, name: str) -> None:
    shape = getattr(image, "shape", None)
    if shape is None or tuple(shape) not in {(3, 256, 256), (256, 256, 3)}:
        raise B0ModelError(f"{name} must be a source RGB image with shape CHW or HWC 256x256")
    dtype = getattr(image, "dtype", None)
    if dtype is None or str(dtype) not in {"torch.uint8", "uint8"}:
        raise B0ModelError(f"{name} must use uint8 RGB values")
