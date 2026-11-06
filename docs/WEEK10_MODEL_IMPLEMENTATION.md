# Week 10 EventGround model implementation

The new implementation follows the immutable v3 study
`080cca5bd1068992556d8fba6c60215d5136f69d8a6106aee47692d209c704fa`.
It adds a separate event model and numerical objectives; no existing policy,
labeler, historical source, checkpoint or study file is modified. Implementation
tests do not admit labels or authorize a formal training run.

## Causal architecture

`EventGroundModel` owns a child copy of the verified M4 OFT conditioning adapter
and an `EventGroundHead`. It invokes only the adapter, which returns the eight
pre-coupling action-slot hidden states from the instruction, current front/wrist
RGB and current eight-dimensional proprioception. The wrapper rejects encoding
fields outside the exact five pinned OFT processor keys. It does not own or call
the action decoder, JEPA future core or coupling module. Targets and offline
physical evidence have separate APIs; neither enters the backbone.

The default head has **1,475,140 FP32 parameters**: per-slot LayerNorm(2048), a
shared 2048→256 projection and GELU, ordered concatenation of eight slots,
2048→256 context projection and tanh, then a 256-wide GRUCell with learned BOS,
event and start embeddings. Three linear readouts emit four event classes,
32 starts and 32 ends. No dropout is used. Initialization seed is 17 and preserves
the caller's CPU and available CUDA random streams.

Event class IDs are `none`, `gripper_close`, `contact`, `grasp` (0–3). Positive
tokens store real integer offsets `[event_id,start,end]`; starts range 1–32 and
ends 2–33, with end logits masked to enforce `end > start`. None is `[0,0,0]`
and has one active decision. Deterministic serialization derives horizon from
start and emits null interval/horizon for none. Teacher forcing reaches only
preceding head decisions: event cannot see a target, start sees event, and end
sees event/start. The labeler's earliest-onset and `grasp > contact > close`
tie rules remain authoritative; event ID order is not a target tie rule.

Only the new head and **exact inherited language-model LoRA parameters** are
trainable. Entire vision encoder, including its inherited visual LoRA, remains
frozen, implementing the explicit Week 10 vision-freeze requirement. The exact
checkpoint LoRA names must cover the instantiated adapter tensors; unknown,
omitted or duplicate names are rejected. The returned full names, including
frozen visual adapters, must be sealed in the child admission. This narrows M4's
original broad adaptation scope without modifying its stored endpoint.

## Objectives and reward

Supervised loss first averages negative log probability over the active one or
three decisions for each example, then averages over the logical prompt batch.
Unequal microbatches contribute sums divided by that same logical batch size.
Neither extra positive tokens nor a short last microbatch changes example weight.

The reward verifier retains the frozen primary weights `(0.40,0.25,0.20,0.15)`
for typed half-open interval IoU, normalized start accuracy, independently
verified interval membership and valid format. Only the predeclared calibration
`(0.30,0.35,0.20,0.15)` is accepted as an alternative. Correct none receives one
for every component; wrong type receives only the valid-format constant;
malformed output receives zero total. An independent evidence digest and a
complete window are required. The admitted target must equal the verified
interval clipped to end 33; the data loader remains responsible for verifying
the evidence digest against the independently admitted ledger. A syntactically
valid digest alone never establishes label quality.

GRPO uses complete groups of four, detached population-standardized reward
advantages and per-decision clipped ratios. It averages active decisions,
then the four completions, then logical prompts. It adds the fixed-SFT sampled
reference KL estimator `exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1`.
Equal-reward groups contribute zero policy advantage while retaining the KL
term. Old probabilities, rewards and reference probabilities are detached.
Overflow stops the numerical check; there is no silent ratio or reward clamp.
Sampling consumes random draws per prompt, across all four candidates and
their three decisions, to preserve assignment under prompt microbatch changes.

## Sealed numerical recipe and precision qualification

The active child recipe is
[numerical v2](../configs/training/week10_eventground_numerical_v2.json), digest
`1dfd49aee3e852ced7ee72c4a6c9a5fa23b0f387b8bb10386bbb8d01c57228ac`.
It is sealed before FP32 runtime qualification and before any optimization.
This document is not a label, resource or training admission.

| Setting | Sealed v2 value |
| --- | --- |
| Initialization / data seed | 17 |
| Logical / physical batch | 16 prompts / 1 prompt; final partial logical batch uses its actual count |
| SFT / continuation LR | head 3e-4; language LoRA 1e-5 |
| RLVR LR | head 1e-5; language LoRA 1e-6 |
| Optimizer | AdamW, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01 |
| Schedule / gradient norm | constant LR; global trainable-gradient norm ≤1 |
| Precision | FP32 child backbone and floating observation inputs; FP32 head, distributions and objectives |
| GRPO | group 4; temperature 1; clip 0.2; KL beta 0.02; advantage epsilon 1e-4 |
| Stage endpoints | inherited fixed passes and final endpoints, no best-checkpoint selection |

The one primary recipe and one allowed reward calibration are not a search.
SFT remains two passes over 2,000 admitted prompts; pilot remains one pass over
the nested 500; full RLVR/control remain one pass over the same 2,000, each
starting from the same fixed SFT endpoint. No action/world-model loss is added.

The preserved [numerical v1](../configs/training/week10_eventground_numerical_v1.json)
used the inherited BF16 backbone. Its
[zero-update qualification report](evaluation/week10-implementation-qualification-v1/run/report.json),
digest `8a42db6741e1099ff62a86761faaeeda9490e42f8e62a0c0210890cd883fc623`,
failed the predeclared global-gradient relative-L2 limit of `0.01`: full batch 2
versus accumulated microbatch 1 differed by `0.14419`. The loss difference
`0.0005348` passed the separate absolute limit of `0.001`; that did not override
the gradient failure. The failed recipe and evidence remain unchanged.

The subsequent [bounded batch-numerics diagnostic](evaluation/week10-batch-numerics-diagnostic-v2/run/report.json),
digest `997b7b9a598b712de6d9bdf582e1c381421bf0c5cfb89fdc57862f332a585aa1`,
used the same current observations and synthetic targets, with zero optimizer
updates and no admitted labels. Active inputs, image patches, state and native
position indices matched. Repeating an identical BF16 execution produced
bitwise-identical results, while retaining identical padding in the microbatch
comparison still left gradient relative L2 `0.14510`. Thus padding alone did
not resolve the observed dependence on batch shape. In the all-FP32 diagnostic,
both independent and shared-padding microbatches agreed with full batch 2 to
approximately `1.35e-5` global-gradient relative L2 and at most `4.77e-7` loss
absolute difference. Original parameters and buffers were restored exactly.
This supports an FP32 implementation remedy on these examples; it does not
identify a particular kernel or establish equivalence for every training input.

Numerical v2 changes child compute precision and pins the corresponding runtime,
input and validation code. It preserves the logical and physical batch sizes,
seed, exact trainable allowlist, learning rates, objectives, reward, stage
endpoints and both numerical tolerances. The verified M4 endpoint is loaded
before casting the child: stored BF16 values are exactly representable in FP32,
and the M4 checkpoint itself is untouched. FP32 arithmetic can change outputs
relative to BF16 arithmetic; exact value inheritance is not a claim of identical
inference. The diagnostic upcast its existing BF16-rounded floating inputs;
the v2 collator retains processor outputs in FP32. The fresh
[v2 gradient and resource probe](evaluation/week10-implementation-qualification-v2/run/report.json),
digest `5192467539c02c6cd0dfd95cc2b2531abd9255c9aeebf5061e030593bc56aab7`,
passed with that actual collator: gradient relative L2 `1.36044e-5` and loss
absolute difference `2.38419e-7`. Accumulating 16 synthetic-target prompts at
microbatch 1 took 2.663 seconds for forward/backward and peaked at 10.680 GiB
reserved GPU memory with resident Adam moment buffers. Both trainable and frozen
weights remained unchanged, with zero optimizer updates. This measures the
bounded SFT computation, not optimizer/checkpoint cost, admitted-label learning
or RLVR memory with a fixed reference model.

## Verification and remaining runtime qualification

Focused tests cover causal teacher forcing, rejected future input fields,
support boundaries and none, serialization, exact LoRA filtering, both reward
weight sets, tie-selected wrong types, interval boundaries, censored intervals,
constant-reward cancellation, masked gradients, group clipping/KL and unequal
microbatch loss/gradient equivalence. Sampling/teacher-scoring comparisons allow
FP32 GEMM roundoff (5e-7 absolute, 2e-6 relative), not scientific target changes.
The FP32 gradient-only check has passed; exact checkpoint recovery on the actual
host, admitted-label overfit, RLVR memory including the fixed reference, full
training throughput, admission-linked data and label QA remain separate gates.
The bounded diagnostics do not admit labels or authorize training. The frozen
vision and action weights do not guarantee unchanged action behavior after
language LoRA adaptation; Week 11's matched-exposure control is still required.
