# Week 4: next experiments after the trajectory audit

**Updated:** September 21, 2026 PDT.  
**User decision:** use all-action L1 and defer the BCE-weight comparison for now.  
**Status:** user authorized **B first**, all-L1 with mixed-task batches. The
prospective [B launch specification](WEEK4_ALL_L1_B.md) supersedes the proposed
two-arm launch order below. A is deferred. Historical hybrid results and
continuation v1 remain unchanged.

## Objective for the next experiment

Keep **Qwen3.5-2B** and the current OFT-style architecture. Optimize mean absolute
error across all **8 steps × 7 action dimensions**, including gripper:
**loss = (6 × motion-L1 + gripper-L1) / 7**.

There is no BCE term or special gripper weight. Retain signed gripper targets:
**−1=open, +1=close**. Train on the raw regression output, with no sigmoid or
threshold inside the loss. For execution and binary gripper accuracy, threshold
the seventh output at zero (zero maps to close).

This L1 form matches
[OpenVLA-OFT's regression-head training](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py)
and [StarVLA's QwenOFT objective](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/model/framework/VLM4A/QwenOFT.py).
Original OpenVLA uses action-token prediction; StarVLA also includes other
action-head families. Matching the L1 form does not establish identical action
normalization, data, architecture or initialization to either reference.

The gradient audit motivates changing the objective but does not prove that
all-L1 will fix Goal failures. Its total loss is not directly comparable with
historical hybrid loss.

## Revised priority order

| Priority | Experiment | Controlled comparison and decision |
| --- | --- | --- |
| **1** | **All-L1 task-mixing screen** | Two continuations from identical update-72,000 parent state: **A, episode-blocked** versus **B, task-mixed**, both all-L1. Proposed endpoint: **15,000 additional updates each**, matching examples and batch sizes. Judge with the same **200 development rollouts per arm**. |
| **2** | **Fresh all-L1 B0 on v2** | Train the selected batching recipe from base Qwen3.5-2B, with fresh task-trained parameters and v2 train-only normalization. Keep the GPU budget comparable to the previous full run. |
| **3** | **Full rollout confirmation** | For an eligible candidate, run 2,000 identical clean trials with verified previous B0/OpenVLA result reuse, then dev-210 and failure review. Preserve the original acceptance rule. |

The BCE-weight **1.0 versus 0.2** comparison is **deferred**. The screen now has
**two arms / 30,000 updates**, rather than three / 45,000. This saves one
15,000-update run and its evaluation, roughly six hours at observed parent
throughput. Mixed-loader performance remains unprofiled.

Both arms change objective relative to the parent. Their matched comparison
isolates batching **under all-L1**, not the causal effect of replacing BCE.
Cached parent rollout outcomes remain an operational reference with that limit.

## Transition from the BCE-trained parent

The seventh output was trained as a logit. It becomes a signed regression value
under all-L1; its initial magnitude need not be near ±1 even if its sign is
correct. The revised freeze must declare identical parent restoration and
optimizer handling for both arms. Check raw gripper range, motion/gripper losses,
finite gradients and decoded actions before training. Any reset must be explicit
and common to both arms; do not silently replace one arm's gripper head.

The model has a generic normalized-action-l1 path, but production recipe
validation and validation/rollout contracts remain tied to the historical hybrid
objective. A config-field change alone is insufficient: the new version must
also handle binary execution/accuracy, checkpoint provenance and recovery.

## Measurements and decisions

- Keep the same **development-200** tasks, initial states and seeds. Record
  overall/suite/task success, paired changes and Goal failure examples.
- Report all-action L1, motion L1, raw gripper L1, binary gripper accuracy and
  decoded action error separately. Do not interpret lower all-L1 totals than
  historical hybrid totals as performance improvement.
- Gradient diagnostics should measure motion-L1 and gripper-L1 contributions
  weighted **6/7 and 1/7**, matching the optimized objective. The old BCE flag
  remains historical evidence, not a new acceptance criterion.
- Use the existing engineering screen as the proposed basis of the next freeze:
  at least **16/50 Goal successes**, net gains on at least three Goal tasks,
  at least **133/200 overall**, and existing Object/Long/Spatial regression limits.
  Record the new objective, initialization handling, measurements, tie-breaks and
  stop rules before training. If neither arm qualifies, stop the long continuation.
- These reused rollouts are development evidence, not an independent final
  significance claim.

Continuations retain **v1 training assignments** and descriptive seen-trajectory
validation. Do not score them on v2 test or claim unseen v2 validation. The
[new trajectory split](WEEK4_TRAJECTORY_SPLIT.md) is for **fresh training only**.

A fresh v2 candidate can be compared with historical policies on identical
rollouts, but changes to both recipe and split cannot isolate a recipe effect.
That claim requires an explicitly budgeted matched fresh v2 control; two full
fresh runs are not assumed here.

## Execution and budget

Record a new experiment version; preserve continuation v1. Set one shared
screen budget covering both arms, validation, diagnostics, rollouts, pilots and
retries, and profile mixed loading. The old v1 budget is not automatically
transferred. Do not shorten endpoints after seeing outcomes.

Implement and verify the all-L1 loss/decode/metric path, mixed sampler,
child-run identity and interruption/resume. Fresh v2 training additionally
needs version-aware loading and train-only normalization. Run on Feifei under
host-owned services with atomic recovery checkpoints, independent of SSH.

If still needed later, consider a controlled standard-clean versus visual-variant
training-data comparison. Public StarVLA clean/dev-210 evaluation can improve
reference context; its published clean score is not a dev-210 measurement.
