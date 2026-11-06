# Week 7 shared auxiliary-training interface

This implementation is a child of the [frozen Week 7 experiment](WEEK7_EXPERIMENT.md).
It trains **M1 Pixel aux and M2 JEPA aux**. M3/M4 remain the later coupled arms.
The version is `oft-final-hidden-auxiliary-v1`; neither branch changes the original
OFT action-forward implementation or its inference signature.

## Shared policy route

`policy_with_features` runs the original OFT policy once and temporarily captures
its backbone's final hidden tensor `[B, S, 2048]`. The attention mask supplies the
padding mask. The tensor includes current image/text tokens, the injected current
state, and the eight action-query placeholders. It contains no demonstration
action values or future observations. The hook neither detaches nor modifies the
output and is removed on success or failure, before checkpoint recomputation.
This interface supports serial training calls, not concurrent forwards on one model.

The original action decoder still reads its eight OFT action slots. Both future
predictors read the same captured sequence through their existing feature
projection. No new trainable bridge, action-attention token, or future-derived
input is added. Maximum sequence length remains 512; longer sequences fail closed.

This is an explicit integration change from CPT: the future predictor now reads
trainable OFT hidden states, whose prompt includes state and action-query slots,
instead of frozen CPT prompt features. Loaded predictor weights and architectures
are unchanged. Training-only diagnostics must establish that this transition is
numerically valid and trainable; successful CPT alone does not establish that.

```text
current cameras + instruction + current state
                  |
          original Qwen/OFT forward
                  |
          shared final hidden states
             /                 \
   eight action slots       future predictor <--- demo actions [t,t+h), horizon
          |                       |
   original action head       predicted image / latent
          |                       |
   eight action outputs         future loss <--- detached future target
          |                       |
        action loss -------- joint backward
```

Future images/features enter only the loss. Demonstration future actions are
fields of a separate auxiliary input object and enter only the predictor. At
deployment, the predictor, teacher, future actions and targets are absent. The
export contains only the original policy's trainable state plus its pinned base
identity, in the existing B0 policy checkpoint format.

## Data and objectives

The original `fresh_plan` constructs both complete action-epoch plans. Their
known digests, 58,566 updates per epoch and 1,706,022 action presentations per
epoch must match before a pilot begins. Every logical batch, including short
batches, retains its original action order. No future-availability filter can
remove an action anchor.

For each action anchor `t`, attach every `h` in `{2,4,8,16,32}` for which `t+h`
is inside the same episode. Future rows are ordered by action slot, then horizon;
a Boolean `[B,5]` mask records availability. Source hashes, split membership,
pair-index records and current/future frame identities are checked by the inherited
future-pair loader. The action target remains the original normalized eight-action
chunk; auxiliary action sequences use `[t,t+h)` and pad to 32 actions.

Both methods receive identical current 128×128 box-averaged images, normalized
state, future-action sequences, horizons, valid future pairs and shared features.
M1 predicts both 128×128 RGB views with `L1 + 0.1 × (1 − SSIM)`.
M2 predicts 128 unit-normalized 2,048-dimensional visual tokens with cosine loss
plus `10 ×` the inherited teacher-relative variance penalty (floor 0.5,
epsilon 0.0001). Repeated horizons are averaged within an anchor before variance
is computed across anchors. A singleton anchor has zero variance penalty.

JEPA targets come from a separately loaded, frozen, unadapted pinned Qwen visual
tower using canonical single-image v2 execution. The teacher is not the policy's
LoRA vision tower. FP32 CPU target caching is bounded to 8 GiB, keyed by source
identity and target frame. Empty-cache recomputation must be bitwise identical.
Decoded episodes have an 8,192-frame LRU budget. The pilot uses synchronous data
loading so actual decoding and teacher costs are included in end-to-end timings.

## Exact logical-batch backward

The action backbone processes one whole original logical batch. Only future
predictor rows are split, with a prospective microbatch of ten pairs.

1. Keep the real shared hidden tensor connected to the original policy graph.
   Use an FP32 detached leaf copy for bounded predictor backward passes.
2. For Pixel, weight each microbatch mean by its number of valid pairs divided
   by the full logical batch's valid-pair count.
3. For JEPA, form a full predicted-token bank without a graph, differentiate
   the full grouped objective once, then deterministically replay each predictor
   chunk with its slice of the bank derivative. Exact replay equality is checked.
4. Accumulate the derivative with respect to the shared leaf. Backpropagate that
   derivative into the real OFT hidden tensor together with the original action
   loss, once. This preserves the full-batch variance term and shared gradients
   without retaining all predictor graphs alongside the policy graph.

All trainable shared and future parameters belong to one AdamW optimizer. The
first two groups retain the B0 head/state and LoRA learning rates. The future
group uses the same 0.0001 peak rate, betas (0.9, 0.95), epsilon 1e-8 and weight
decay 0.01. All groups use the inherited 117,132-update warmup/cosine schedule.
Clip the combined gradient norm to one after complete accumulation, then advance
the optimizer and scheduler once. BF16 rounding occurs at the boundary back into
the original policy; predictor parameters and leaf accumulation are FP32. Future convolutions disable
cuDNN TF32; the inherited SSIM implementation computes its image moments in FP64.
The inherited policy and canonical teacher remain BF16.

## Frozen diagnostics and numerical choices

The [prospective pilot rule](../configs/training/week7_auxiliary_pilot_v2.json)
fixes batch selection, trial count, tolerances, loss-weight calculation and limits.
Four metadata-selected complete training batches calibrate each outer weight,
without optimizer updates: clip the median of
`0.1 × ||dL_action/dH|| / ||dL_future/dH||` to `[0.0001, 10]`.
This matches the initial auxiliary gradient scale at the shared hidden interface;
it does not assert equal gradients at every layer or equal compute cost.
No validation scores, rollouts, or weight search enter this calculation.

Both arms must pass the same 64-update, four-anchor training-only memorization
check. This diagnostic uses constant 0.0001 learning rates, and its weights are
discarded. Resource measurements then restart from the pinned base, fresh shared
components and original CPT module, using the first 128 exact mixed batches with
cold caches and the production warmup schedule. Edge batches exercise the
shortest batch and longest episode/instruction cases.

The resource limit is seven projected days per arm, using 1.2 times the larger of
mean and p95 step time, plus measured checkpoint overhead. This is a prospective
auxiliary-arm feasibility ceiling, not a claim of equal wall time with B0 or an
authorization to launch production. The prospective production active-time cap is
eight days per arm. GPU reservation must stay below 90%, host peak RSS below
28 GiB, and the pilot service is limited to three hours. The final numerical
recipe must report actual calibrated weights and observed costs before step 5.

## Recovery and verification

A complete checkpoint contains shared trainables, the future module, all three
Adam groups and moments, scheduler, Python/NumPy/Torch/CUDA RNG, epoch/batch
cursor, action/future exposure counters and two identity chains. It uses the
existing atomic tensor-then-commit-record store with an auxiliary format/version
extension. A failed or partial update cannot be snapshotted or reused; recovery
must restore a complete commit. The teacher cache is never continuation state.

Verification covers input parity with B0, detached teachers, target isolation,
nonzero auxiliary gradients into the shared backbone/state, full-versus-chunked
future and joint gradients, action-only export, memorization, resources, and a
real SIGKILL during the third update after a checkpoint at update two. The resumed
four-update state must match the uninterrupted state bit for bit. A damaged latest
checkpoint must fall back to the preceding intact commit. All evidence is retained
under [week7-integration-v1](evaluation/week7-integration-v1/).

Local tests initially exposed a recovery fixture that recreated a different random
frozen toy embedding; the corrected test recreates the same frozen base and then
perturbs RNG before restore. Pixel batch-shape numerical comparisons also require
an explicit FP32 tolerance near the clamped RGB/L1 path's nonsmooth points:
relative gradient L2 below 0.001 and maximum absolute error below 0.00001.
JEPA's FP32 relative bound is 0.00001. These bounds were fixed before the GPU pilot;
the joint BF16 gradient comparison has a 0.01 relative bound. The observed errors
must accompany the pass/fail result rather than be described as bitwise equality.

The first GPU attempt stopped before optimizer updates because the diagnostic
setup had not put the pretrained backbone into training mode, leaving activation
checkpointing inactive. The second attempt exposed the default cuDNN TF32 setting
and failed the absolute Pixel gradient threshold. Both failed attempts and source
snapshots remain preserved. The prospective v2 rule requires IEEE FP32 future
convolution, retaining the original numerical thresholds and study choices.
