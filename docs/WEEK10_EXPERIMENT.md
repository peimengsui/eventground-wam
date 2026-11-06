# Week 10: EventGround experiment freeze

**Scientific scope frozen October 16, 2026.** The authority is
[week10_eventground_experiment_v1.json](../configs/training/week10_eventground_experiment_v1.json).
This completes the experiment-design step. Dataset construction, the executable
head/trainer, resource qualification and training remain outstanding.

The user explicitly selected **adding contact and grasp labels before training**.
Consequently, a gripper-close-only experiment cannot substitute for this version.
The current data do not yet pass the new event-admission gate. This freeze does
not assert that contact/grasp ground truth already exists or launch GPU work.

## Question and inherited evidence

Can short, structured event supervision teach the M4 representation to anticipate
the next robot event and its timing, and does verifiable-reward optimization
improve held-out temporal accuracy beyond supervised fine-tuning?

The completed M4 epoch-2 endpoint is the single parent. It already received
117,132 action updates, 3,412,044 action presentations and 16,259,770 future pairs.
Week 10 adds **zero action-training updates and zero world-model CPT updates**.
The original M4 bytes remain immutable. Separate child exports contain the
EventGround head and adapted backbone LoRA, with the parent hashes recorded.

A fresh [Feifei entry check](evaluation/week10-freeze-v1/feifei-entry.json)
rehashed the M4 policy and full predictor, verified completion links and checked
the sealed Week 9 closure. All five arms' 15,000 outcomes remain preserved.
Week 9 did not establish a coupling benefit; M4 is used because it is the
predeclared JEPA-WAM branch, not because it won the comparison. Original JEPA
quality-gate FAIL/research-use, rendering, single-seed and development-overlap
qualifications remain in force.

## Causal input and structured output

Each prediction receives the instruction, current front/wrist RGB and current
proprioception through M4's pinned Qwen3.5-2B/OFT preprocessing. The initial
version uses the **current observation only**. It receives no demonstrated
actions, future images, event labels, remaining episode length, simulator object
state or annotation confidence. Those may be used only by offline supervision
and the reward verifier. Current observation and target records have disjoint APIs.

The head reads the eight **pre-coupling** action-slot hidden states. It does not
run the future predictor, cross-attention refinement or continuous action head.
Its small autoregressive categorical decoder emits event type, start and end;
deterministic serialization produces:

```json
{"next_event": "grasp", "interval": [12, 18], "horizon": 12}
```

Offsets are integer control frames at 20 Hz, relative to the current anchor.
The search window is the next **32 frames (1.6 seconds)**. Intervals are
half-open `[start, end)`, with `1 <= start < end <= 33`.
`horizon = start` selects the first event frame, consistent with the existing
next-event target convention; it is derived, not an independently inconsistent
prediction. This makes explicit a convention left unspecified by the old
illustrative `[12,18], horizon=16` example.

An event continuing beyond the window has a clipped end of 33 and a recorded
right-censor flag. The model does not receive that flag. A negative example is:

```json
{"next_event": "none", "interval": null, "horizon": null}
```

`none` means no admitted event begins in the complete window. It does not mean
the robot will never interact. Ambiguous windows and incomplete episode tails
are excluded with explicit reasons, never converted to negatives. Anchors inside
an ongoing admitted event are excluded from this onset-anticipation study.

## Required event data and label gate

| Label | Meaning | Required evidence |
| --- | --- | --- |
| `gripper_close` | The existing command transition with observed aperture response. | Original extractor and accepted Week 2 QA; never relabel this as grasp. |
| `contact` | Onset of physical contact between a gripper/finger and a movable manipulation object. | Time-aligned contact/geometry evidence; table, fixture and self-contact excluded. |
| `grasp` | Onset of verified object retention during gripper motion. | Object/gripper relative-motion and retention evidence with contact support; mere closure or touching is insufficient. |
| `none` | No onset of any of the three types in the window. | Complete observable window with no unresolved candidate. |

Contact and grasp can concern any movable manipulation object; wrong-object
interactions are not silently discarded. Object identities remain label metadata.
Select the earliest admitted onset; exact-frame ties use the fixed priority
`grasp`, `contact`, `gripper_close`, then stable object ID. Preserve other events
in the annotation ledger even when they are not the single prediction target.

The pinned LeRobot snapshot lacks the required MuJoCo/object/contact state.
First audit whether original simulator states can be aligned to the **same
demonstration trajectories** and produce a versioned evidence sidecar. Source
identity, frame synchronization and action/proprioception agreement must be
verified. Approximate open-loop action replay does not establish ground truth.
Videos support independent QA; a VLM's guess cannot serve as its own verifier.
If trustworthy physical evidence cannot be recovered, preserve the finding and
resolve the data design before training; do not invent predicates or downgrade
to gripper-close-only training.

Before extracting the new labels or inspecting their QA outcomes, seal a child
labeler contract specifying simulator/source identity, object/geometry mappings,
contact persistence/debounce, grasp-retention/displacement thresholds, boundary
definitions, observability and ambiguity rules. These engineering thresholds
are **not measured or frozen in this Step 1 artifact**. They must implement the
semantics above, and cannot be tuned against Week 9 success or event validation.

Create a new immutable dataset; do not edit `week2-dataset-v1`. Preserve the
79 individually reviewed close labels separately from extractor-level promotion
of the unreviewed remainder. Do not claim every promoted label was human-reviewed.
For contact and grasp, require independent review of at least **200 training
episodes** across tasks/lengths, including negative cases and missed-event checks;
extend the queue only for coverage, to at most 300 episodes. Require at least
100 reviewed positive instances **per new event type**, observed precision and
recall >= 95% per type, and mean absolute start/end boundary error <= 2 frames
on fully observed matched intervals. Report counts, uncertainty intervals,
ambiguities and exclusions; these point-estimate gates are not a 95%-confidence
guarantee of population quality. Failure remains a failed gate.

## Splits and bounded data exposure

Inherit **trajectory-v2** groups, not the superseded Week 2 split. The 160 test
groups and their payloads stay locked. Derive two disjoint partitions of the
existing validation groups, stratified by canonical task and ordered by a stable
SHA-256 key with seed 10: alternating groups become `event-dev` and `event-report`.
The final report partition is opened only after all training decisions are sealed.
This is a new event-evaluation holdout within historically used validation data,
not an untouched task-generalization test.

The primary SFT set is **2,000 unique prompts**, 500 per target class including
none. The **500-prompt RL pilot** is a nested subset with 125 per class. Shortfall
in any class blocks production; no duplication or replacing grasp with close.
Select deterministically, round-robin across tasks and trajectory groups, with
at most four anchors per physical event and 16 prompts per group. Equivalent
trajectory variants cannot multiply supervision: use one hash-selected episode
per group and preserve all omitted-candidate counts. Lead-time coverage must
include 1-4, 5-8, 9-16 and 17-32 frames per positive event type.

Use a fixed balanced event-dev panel of at most 1,000 prompts (up to 250/class).
For event-report, select up to 3,000 prompts from a stride-eight eligible-anchor
grid, round-robin by task/group **without class balancing or outcome filtering**.
Record that sampling distribution and its class frequencies. Manifest IDs,
exact sample counts, quality/exclusion counts, order and hashes are child
artifacts sealed before training. No Week 9 rollout is used as training data.
Expansion to 5,000 training prompts would be a separately versioned extension.

## Model and training stages

Keep the vision encoder, base Qwen weights, state projector, action decoder,
future predictor and coupling weights frozen. Train only the new EventGround
head and the inherited backbone LoRA in a **copy** of M4. Pin the exact parameter
allowlist and head dimensions in the implementation contract before pilots.
No large language-model replacement, free-form chain-of-thought or additional
action loss is included. LoRA adaptation can still change action features;
frozen decoder weights alone do not establish unchanged control behavior.

1. **SFT:** cross-entropy for the short event/start/end sequence, averaged over
   active decisions per example and then over the logical batch. Teacher forcing
   is confined to the head's preceding output tokens. Use two fixed passes over
   the 2,000 prompts; select the final endpoint, not a validation-best checkpoint.
2. **RLVR pilot:** initialize from SFT, sample four structured candidates per
   prompt, one pass over the fixed 500 prompts, GRPO with a fixed SFT reference.
   No action/environment exploration is used to produce the group.
3. **Full RLVR, conditional on the gate:** restart from the same SFT endpoint,
   discard pilot-trained weights and make one pass over the fixed 2,000 prompts.
4. **SFT-continuation control:** also start from SFT and traverse the exact same
   main-run prompt batches once with supervised loss. This controls prompt and
   update exposure, not generated-token or total compute cost. Report all costs.

The numerical execution recipe must seal learning rates, exact head architecture,
GRPO clipping/KL/temperature, seeds, logical batches, optimizer/schedule, precision
and tolerances **before any optimization pilot**. These are next-step work, not
permission for an unrestricted search. Allow one primary recipe and one bounded
reward calibration only. Hardware microbatch changes must preserve the complete
logical batch and group-of-four advantages. Verify loss/gradient equivalence.

## Verifiable reward and bounded calibration

For a valid structured prediction:

`R = 0.40 * event_IoU + 0.25 * time_score + 0.20 * predicate + 0.15 * format`.

- `event_IoU`: half-open interval IoU when event types match; otherwise zero.
- `time_score`: `max(0, 1 - abs(predicted_start - true_start)/32)` for the correct
  positive type; otherwise zero.
- `predicate`: one only if the type matches and the proposed horizon falls in
  the independently evidence-verified event interval. It verifies the recorded
  trajectory, not a counterfactual future under newly proposed actions.
- `format`: one for schema/range-valid output. A deterministic serializer and
  constrained categorical support make this constant for well-formed samples;
  it then cancels from group-relative advantages. Do not claim format learning.
- Correct `none` receives one for each component; incorrect positive/negative
  classification receives zero for the three semantic components. Malformed
  output receives **zero total reward**. An all-equal-reward group contributes
  zero policy advantage, not NaN or an invented gradient; its KL is still logged.

Use event-dev only for the predetermined pilot gate. To expand, require mean
per-event-type positive IoU to improve by >= 0.01 over SFT, no > 0.02 absolute
increase in false positives on none examples, and no > 1-frame worsening in
start MAE on matched-type positives; coverage and type errors are reported too.
Format validity must be 100% and integrity/numerical checks must pass. These
are engineering promotion rules, not significance tests.

If the first pilot fails the improvement gate, allow **one** restart from SFT
on the same 500 prompts with weights `(0.30, 0.35, 0.20, 0.15)`, increasing
dense timing feedback. No other recipe changes or extra reward sweep. If it
still fails, retain SFT, package both failures, and stop RL expansion. A failed
integrity/data gate is not eligible for scientific-result acceptance.

## Evaluation, completion and Week 11 boundary

Primary comparison: final RLVR minus SFT on **macro event-type positive IoU**,
including wrong-type/none predictions on positive examples as zero. Compare
RLVR against SFT-continuation as an additional control. Report per-event IoU,
type confusion/precision/recall, no-event false positives, start/end errors,
matched-type coverage, joint correct-type/within-two-frames rate, horizon-bin
calibration and clipping/ambiguity counts. Timing conditional on correct type
must never hide missed or incorrectly classified events.

Use paired resampling of trajectory groups, 10,000 PCG64 draws, seed 10, with
common draws across arms and 2.5/97.5 percentile intervals. Recompute the macro
statistic per draw; explicitly count/mark draws missing a positive event type
and withhold that interval if more than 1% of draws are undefined. Report all
eligible report samples. Include simple always-none and training-only event/
timing-prior baselines, plus per-task and lead-time breakdowns. Success means
complete valid evidence, not a guaranteed positive RLVR effect.

Package SFT and completed RLVR/control endpoints, or SFT plus a documented
RLVR-stop decision, alongside label QA, sealed data/source/recipes, curves,
bootstrap outputs, latency, memory, all attempted-run costs and checksums.
Default host is Feifei, one substantial GPU job at a time. Prospective ceilings:
3 active GPU hours for correctness/resource qualification and 40 total active
GPU hours across qualification, SFT, both possible RL pilots, main RLVR and the
continuation control. Independent annotation/reconstruction costs are reported
separately. Reserve 50 GiB free disk; cap new outputs/sidecars at 100 GiB and GPU
use below 90%. These are ceilings, **not an ETA or a measured feasibility claim**.
Exceeding a limit requires a recorded revision; do not silently broaden it.

Week 10 measures event anticipation. **Week 11** must separately bind predicted
integer horizons to the future core, qualify off-grid horizons (M4 trained on
2/4/8/16/32 and deployed at 8), define none-event behavior and evaluate fixed
versus adaptive control. Include a fixed-horizon path with the same adapted
LoRA/head-training exposure before attributing a gain specifically to adaptive
horizons. Existing 2x2 results stay unchanged; full-10K/test remain locked.

## Remaining implementation artifacts

The new labeler/evidence contract, dataset manifests, head/optimizer contract,
correctness/recovery reports and actual-host resource admission must all reference
this study digest. All are currently pending. Any change to its scientific
scope, required ontology, parent endpoint, data exposure, reward alternatives,
evaluation or stopping rules requires a new version retaining the old freeze.

Read-only check: `.venv/bin/python scripts/verify_week10_experiment.py`.
