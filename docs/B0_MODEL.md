# B0 Qwen conditioning and action model

The B0 model uses only `Qwen/Qwen3.5-2B` at immutable revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Its complete 13-file remote
manifest, including the 4.55 GB weight object's LFS SHA-256, was committed at
[`model/Qwen3.5-2B-manifest.json`](model/Qwen3.5-2B-manifest.json) before model
files or weights were downloaded.

## Processor and prompt contract

The processor is `Qwen3VLProcessor` from Transformers 5.17.0 with the explicit
`torchvision` image backend. The pinned image transform has mean/std 0.5,
patch size 16, temporal patch size 2, spatial merge size 2, and the original
256 by 256 source size. No processor resize occurs for B0 inputs.

Each user prompt has this exact causal order:

1. `Front camera image:` followed by the front image;
2. `Wrist camera image:` followed by the wrist image;
3. `Instruction: <source instruction>`;
4. `Predict the next 8 robot actions.`; and
5. the assistant generation prefix with thinking explicitly disabled.

Right padding is explicit. Every 256 by 256 camera must produce an image grid
of `[1,16,16]` and 64 merged image tokens, for 128 visual tokens per sample.
The input builder rejects a changed grid, token count, image order, shape, RGB
dtype, or missing Qwen field. All processor fields are forwarded, including
`mm_token_type_ids` and `image_grid_thw`.

## Conditioning memory

The adapter wraps the bare multimodal core at
`Qwen3_5ForConditionalGeneration.model`, so it consumes the final
`last_hidden_state [B,S,2048]` without allocating vocabulary logits or retaining
all layer outputs. Token selection is every final-layer token whose attention
mask is one. Padding remains in the tensor and is represented by a Transformer
key-padding mask.

Qwen memory is layer-normalized and projected to 512 dimensions. A two-layer
projector maps the 8D proprioceptive state to one additional memory token. The
locked memory order is Qwen tokens, proprioception token, then optional future
tokens. B0 supplies no future tokens, but the interface is already present for
M3/M4 without changing the action head.

The initial adapter freezes the Qwen base and keeps the core in eval mode. The
only trainable backbone tensors are rank-16, alpha-32, zero-dropout LoRA A/B
matrices on `q_proj`, `k_proj`, `v_proj`, and `o_proj` in the two uppermost
full-attention layers, 19 and 23. The Qwen hybrid stack's other full-attention
layers are 3, 7, 11, and 15; its intervening DeltaNet layers do not expose those
projection targets. The exact 16 LoRA tensor names join the conditioning
projection, proprioception projector, and action decoder in the enforced
trainable-name allowlist.

## Deterministic action decoder and objective

The shared action decoder has eight learned action queries, model dimension
512, four pre-normalized Transformer decoder layers, eight attention heads,
feed-forward dimension 2,048, GELU activation, bidirectional query
self-attention, and zero dropout. Queries cross-attend the complete conditioning
memory and emit one seven-dimensional normalized action each. Initialization is
isolated under a declared seed and does not mutate the caller's random state.

The original fixed-batch B0 objective was the unweighted arithmetic mean of
absolute error over the complete normalized `[B,8,7]` prediction and target
tensors. The accepted ten-demonstration revision retains L1 for the six
continuous dimensions and treats the binary gripper command as classification:
the seventh decoder value is a logit, binary cross entropy has unit weight, and
inference maps a nonnegative logit to `+1 = close` and a negative logit to
`-1 = open`. The default rank-16 LoRA is unchanged.

Both objectives are explicitly versioned in their checkpoint and report. The
closed-loop B0 adapter accepts only the reviewed
`continuous-l1-plus-gripper-bce-v1` unit-weight checkpoint. There is still no
flow matching, sampling process, padded action target, or external action
pretraining.

## Runtime and checks

`environments/b0/uv.lock` pins PyTorch 2.13.0, torchvision 0.28.0,
Transformers 5.17.0, PEFT 0.20.0, Accelerate 1.15.0, Pillow 12.3.0, and the
existing data dependencies. The initial attention backend is PyTorch SDPA; no
unpinned Hub kernel or remote code is enabled.

Run the processor-only integration without model weights:

```bash
./scripts/setup_b0.sh
./scripts/run_b0.sh python scripts/run_b0_model_smoke.py processor
```

Run the real BF16 forward/backward compatibility gate on `feifei` after the
manifest commit:

```bash
./scripts/run_b0.sh python scripts/run_b0_model_smoke.py model \
  --device cuda:0 \
  --output ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-model-v1/model-smoke.json
```

The model gate validates processor tensor shapes, conditioning/action shapes,
finite L1 forward/backward, exact trainable gradients, deterministic repeated
inference, trainable-state reload, latency, and peak CUDA memory. Outputs are
created exclusively and accompanied by a SHA-256 sidecar.
