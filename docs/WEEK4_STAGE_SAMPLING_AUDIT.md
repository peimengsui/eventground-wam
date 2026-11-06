# Week 4: training-stage audit and controlled continuation design

**Later decision, September 24:** the user [closed Week 4](WEEK4_CLOSEOUT.md)
using the current best B0 and deferred the C/S continuation screen. The audit
and sealed prospective design below remain valid records; neither arm has
launched. The active next milestone is [Week 5 pixel CPT](WEEK5_STATUS.md).

**Completed:** September 24, 2026. **Scope:** train-only data/sampling inspection
and prospective experiment design. Zero optimizer updates and zero new policy
rollouts. Predict-eight/execute-eight remains fixed for B0 and OpenVLA-OFT+.

## Executive summary

- **No sampler omission was found.** All 11,435 training parquet hashes match the
  frozen source. Both reconstructed epoch plans match the actual training
  manifest; each of the 1,706,022 valid episode/anchor pairs occurs exactly once
  per epoch. Median batch diversity is **14 tasks**; about **99.29%** of batches
  contain multiple tasks. Mixed batching is already functioning.
- **The failing stages are present in the reviewed demonstrations.** The three
  target tasks contain **79 distinct numeric trajectories**, expanded to 530
  appearance variants. We reviewed 15 distinct training groups, not a random
  sample or a full success audit. The examples show drawer closing, mug insertion
  followed by door closure, and pickup/placement of the second mug.
- **Exposure is unequal, but stage starvation is not established.** These tasks
  receive 2.17–3.18% of sample presentations each; all ten Long tasks receive
  33.95%. Post-first-release anchors already occupy 32.93%, 42.92%, and 57.40%
  within drawer, microwave, and two-mug tasks. More exposure is a hypothesis to
  test, not a demonstrated repair.
- **The next design changes only within-episode sampling.** Two continuations
  start from the same epoch-2 checkpoint, with **15,000 updates / 436,311
  presentations each**. Control C uses the current sampler. Treatment S gives
  late anchors of the three target tasks twice the relative sampling weight.
  Batch sizes, episode/group/task counts, other-task samples, architecture,
  objective, normalization, and evaluation settings are matched.

The [design and exact sample plans](../configs/training/week4_b0_late_stage_screen_v1.json)
are frozen; a compatible child-run trainer and recovery pilot still need to be
implemented before launch. This audit did not start another training run.

## What training actually contains

Counts use **trajectory-v2 train only**, not the older episode split.

| Target task | Training variants | Numeric groups | Eight-action chunks per epoch | Share of all chunks | Chunks after first stable release |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bowl into drawer, then close | 185 | 27 | 44,030 | 2.58% | 14,500 (32.93%) |
| Mug into microwave, then close | 130 | 23 | 36,990 | 2.17% | 15,875 (42.92%) |
| Both mugs onto plates | 215 | 29 | 54,290 | 3.18% | 31,160 (57.40%) |

Each target trajectory has five or ten appearance variants. These are additional
visual examples, not five or ten independent manipulation trajectories. Across
all 40 tasks, training groups range from 22 to 40 and chunk shares from 1.42% to
5.66%. Within the target tasks, the most-exposed trajectory receives roughly
2.5–2.7 times the chunks of the least-exposed one because of episode length and
variant count. A separate task/group balancing experiment would be needed to
isolate those factors; they are deliberately unchanged in the proposed screen.

The trainer preserves short batches: **10,996/58,566** batches per epoch contain
fewer than 32 examples. We also summed the nominal per-example coefficient
`1 / batch_size`. Target task shares remain close to their presentation shares
(e.g. drawer 2.581% in epoch 1, 2.577% in epoch 2). This is a loss-coefficient
check, not a measurement of gradient influence or Adam updates.

### Episode boundary handling

An episode with N frames supplies N−7 observation anchors, each targeting eight
consecutive actions. The final seven observations cannot anchor a complete
chunk. Their actions are still targets, with multiplicities 7, 6, …, 1, versus
up to eight for interior actions. There is no dropped episode-tail batch.

The reviewed drawer closing and microwave door-closing motions span many frames
before that boundary. This boundary is therefore not an established explanation
for the large deficit. The proposed experiment keeps this contract unchanged.

## Visual review and stage-label caution

Selection was nine groups at the training-length quartiles (three per target
task), plus all six groups with unusual stable gripper-command sequences. We
inspected 15 contact sheets, each showing 12 times from both cameras: **360
sampled camera frames**. Source video hashes and selected frame indices are
retained in the [preview manifests](../outputs/week4-stage-sampling-20260924/README.md).
The review did not watch every video frame or infer unavailable success flags.

Representative findings:

- Drawer episodes **2578, 6070, 8790** show bowl placement followed by withdrawal
  and drawer closing. The late stage is visibly in the training source.
- Microwave episodes **1160, 3337, 6066** show mug transport/insertion, release,
  withdrawal, and door closing.
- Two-mug episodes **491, 1041, 6510** show white-mug placement followed by
  acquisition/transport of the yellow mug; the final placement area in 6510's
  shifted camera view is partly occluded.
- **First release is not a reliable completion label.** Drawer episode **862**
  releases the bowl near the table and regrasps it before drawer placement.
  Microwave **2305** and two-mug **991** also have early grip/release attempts
  before successful transport. Other examples contain second-object regrasping.
- Episode **2086** lacks a stable terminal release event while the sampled image
  shows the second mug at its plate with the gripper still adjacent. That alone
  does not establish a failed demonstration.

Thus command events are descriptive proxies only. We do not automatically label
first release as successful placement, add privileged object state to the policy,
or remove any training group based on this inspection. Corrective behavior does
exist in some source demonstrations; its coverage of B0's off-trajectory failures
has not been quantified.

### Existing validation supports judging with rollouts

Using the already completed epoch-2 validation report (no new validation payload
was read during this audit):

| Target task | Group-balanced all-action L1 | Gripper sign accuracy | Clean task success |
| --- | ---: | ---: | ---: |
| Drawer | 0.12425 | 97.78% | 13/50 |
| Microwave | 0.20140 | 96.09% | 36/50 |
| Two mugs | 0.13076 | 94.17% | 36/50 |

Drawer loss and gripper accuracy are not unusually poor despite weak closed-loop
success. This is compatible with a contact/control or distribution-shift problem;
it does not prove a particular cause. Loss improvements alone will not promote
another checkpoint.

## Frozen small experiment: C versus S

Both arms use parent epoch 2, SHA-256
`7a385fea68fe5d14f642b31e18f8e2e166ccd984d9b4f1ff5b68484191a9b720`.

- **C:** first 15,000 updates from the existing fresh sampler's next epoch
  (zero-based epoch index 2).
- **S:** identical per-batch episode slots. For the three target tasks only,
  redraw anchors with replacement within that same episode. Anchors
  `t >= ceil(0.6 * (N−7))` have weight 2; earlier valid anchors have weight 1.
  Every non-target token stays identical to C. The hash-based mapping and seed
  are fixed in the [planner](../scripts/prepare_b0_stage_screen.py).
- This is **late-trajectory emphasis**, not semantic phase annotation. The final
  40% is a transparent proxy spanning much of the reviewed closing and
  second-object manipulation. It can miss earlier insertion/acquisition errors.
- Each arm has exactly **436,311 presentations**, about **0.256 original epochs**,
  from the same 2,929 episodes and 1,188 numeric groups. Every batch size and
  episode/group/task exposure count matches. S has 423,527 unique episode/anchor
  pairs versus C's 436,311; repetition and reduced early coverage are explicit
  tradeoffs of reweighting.
- Fresh AdamW state for both arms, constant **1e-5** learning rate, no warmup,
  parent betas/decay/clipping. Same parent weights and seed. Optimizer reset is
  common to both, so this comparison tests the sampler under that continuation
  setup, not uninterrupted continuation of the old Adam state.
- Keep Qwen3.5-2B, broad rank-32 LoRA, OFT head, state input, both cameras,
  train-v2 normalizers, mean all-action L1, no augmentation, and predict-eight /
  execute-eight at 20 Hz. No task/group rebalance, language augmentation,
  gripper weighting, new data, or checkpoint selection by validation loss.

Actual prepared-plan exposures:

| Target task | Total presentations, each arm | Late anchors: C | Late anchors: S |
| --- | ---: | ---: | ---: |
| Drawer | 11,172 | 4,453 (39.86%) | 6,421 (57.47%) |
| Microwave | 7,950 | 3,169 (39.86%) | 4,560 (57.36%) |
| Two mugs | 13,767 | 5,480 (39.81%) | 7,859 (57.09%) |

Weight 2 doubles the relative odds of drawing a late versus early anchor; it
increases realized late presentations by about **43.8% overall**, not 100%.
There is no extra training compute hidden in S.

### Measurements and decision

1. Fixed 15,000-update endpoints, full v2 supervised validation with micro and
   hierarchical metrics, plus early/late temporal-bin diagnostics. Validation
   does not choose the checkpoint. The supervised test partition remains locked.
2. Both endpoints run the same **development-335** panel: all 50 existing states
   for each target task (150) plus the existing five development states for each
   other task (185). This retains the full development-200 and adds 135 target
   trials. It is an intentionally target-heavy engineering panel; its aggregate
   percentage is not a standard LIBERO score.
3. Cached parent outcomes on that panel are **85/150 target + 180/185 other**;
   reference **140/150 + 179/185**. Input/artifact verification is required for
   future reuse. These states were already inspected and the tasks selected from
   failure analysis: all screen results are development evidence, not independent
   confirmation on an untouched holdout.
4. S is eligible for a larger confirmation only with at least **eight additional
   target successes versus both C and parent**, gains on at least two target tasks
   versus C, at most two losses on any target task versus either comparator,
   at most three losses among the other 185 cases versus either comparator, and
   at most two losses in each other suite. Require complete valid artifacts and
   report per-task paired wins/losses, not only aggregate success. This is a
   prospective engineering screen, not a replacement Week 4 acceptance rule.
5. If S fails the screen, retain the parent and do not automatically extend
   training or search weights. If C improves equally, additional training is a
   simpler explanation than the sampler. If S improves loss without rollout
   gains, do not promote it. Any larger clean-2000/dev-210 confirmation remains a
   separate execution step under the original acceptance rules, including the
   unresolved strict rendering-input requirement.

Both arms together contain 872,622 presentations, about 0.511 original epochs.
At the completed run's observed 21.42 examples/s this is roughly **11.3 GPU-hours
of training**, plus validation, rollouts, setup and the recovery pilot. This is a
projection, not measured continuation throughput. The design caps training at
10 active hours per arm and combined validation/screening at 12 hours, with a
100 GiB output cap and 50 GiB free-space reserve. A future launch must use a
host-owned service, atomic full-state recovery every 300 seconds or 2,000 updates,
three retained recovery generations, and a verified interruption/resume pilot.

## Verification and artifacts

[Audit summary](evaluation/week4-stage-sampling-audit-v1.json) ·
[Independent local checks](evaluation/week4-stage-sampling-verification-v1.json) ·
[Visual annotations](evaluation/week4-stage-sampling-visual-review-v1.json) ·
[Development-335](../evaluation/splits/week4-late-stage-screen-v1/development-335.json).

The CPU audit completed on Feifei under a host-owned service, exit zero and zero
restarts. A first diagnostic attempt failed only when sealing integer dictionary
keys in its final JSON; that source and partial evidence are retained. The
corrected audit took 16.836 seconds. Local PyArrow is absent, so synthetic command
and terminal-boundary arithmetic checks ran successfully in the pinned host
runtime. Local lint passed. Both epoch plan digests, both proposed plan files,
all 15 preview hashes, train-only membership, valid anchor bounds, and exact
matched episode slots were checked. No existing training recipe, checkpoint,
rollout outcome, acceptance threshold, or deployed trainer was modified.
