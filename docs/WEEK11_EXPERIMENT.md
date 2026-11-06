# Week 11: adaptive-horizon integration experiment v1

**October 25, 2026: Step 1 is frozen.** The machine-readable authority is
[week11_integration_v1.json](../configs/evaluation/week11_integration_v1.json).
This defines the experiment, its inputs, panels, budget and decision rules.
The horizon-mapping contract, implementation, hardware qualification and rollouts
are subsequent steps. This request covers the design freeze only.

## Question and scope

Does the selected Week 10 event predictor improve the frozen M4 policy when its
predicted next-new-onset timing selects the JEPA future horizon?

The primary contrast is adaptive M4-EG minus original fixed-horizon M4. This is
an exploratory, single-training-seed deployment experiment. Week 9 results and
Week 10 train/dev/report findings were already visible before this freeze;
these panels are not untouched holdouts. No additional training, CPT, reward
calibration, RLVR arm, checkpoint selection or full-10K/test evaluation belongs
to v1. The revised-task RLVR no-go remains closed.

The selected endpoint is the SFT-only continuation at update 32, selected on dev
before its report evaluation. Its +1.15-point report IoU estimate has a 95%
interval spanning zero. Independent label QA remains unmet. Original JEPA
quality-gate failure/research-use, simulator rendering qualifications, historical
M4 source-alias exposure and demonstration-to-policy distribution shift carry
forward. None of these qualifications is resolved by this freeze.

## Pinned parents and loading boundary

| Artifact | SHA-256 |
| --- | --- |
| M4 fixed epoch-2 policy | `af5ba5c52601566bee80fd3212a5ed514cf09e3713ffa38c54493d693e933e0c` |
| M4 full predictor | `9a53396be341fa17ad11a76a4d56f90f72b04c13d2d7d1a6f806e3605842b9ad` |
| SFT-only continuation | `6c6ae33adf89c9e89fd821b3d2d024255edb9cb782372a3de0ecf3d450b1eaf5` |

The contract records exact Feifei paths, completion/identity/header digests,
normalizers, runtime references and the
[Week 10 handoff](evaluation/week10-closeout-v1/handoff.json). Local verification
checks those records and their cross-links. Actual remote bytes were last checked
by the October 24 closeout; they have **not** been freshly rehashed in Step 1.
Rehash them and validate the preservation completion before any model loading.

The child contains an event head and adapted language LoRA, not a complete robot
policy. Restore it through the pinned preservation loader after restoring the
exact M4 conditioning adapter. Retain the original M4 action-side adapter, action
head, future core and coupling unchanged. Event inference uses its existing FP32
recipe; action inference retains its existing BF16 backbone and FP32 future/core
coupling boundaries. Separate model instances are the reference design. Sharing
storage or switching adapters requires equivalent outputs and unchanged tensors
to be demonstrated before admission; it cannot change either numerical recipe.

## Fixed comparison arms

| ID | Event branch | Action branch and future horizon | Purpose |
| --- | --- | --- | --- |
| `M4-fixed` | Absent | Original M4, horizon 8 | Primary baseline and deployment cost |
| `M4-EG-fixed8` | Selected SFT, run every query; output logged but ignored | Original M4, horizon 8 | Control for wrapper overhead and branch isolation |
| `M4-EG-adaptive` | Same selected SFT, run every query | Original M4, mapped predicted onset | Candidate |

Every arm receives fresh Week 11 rollouts. Historical M4 results cannot fill
baseline cells. Primary: adaptive minus fixed. Secondary: adaptive minus
EG-fixed8, and EG-fixed8 minus fixed. The latter two separate horizon behavior
from wrapper effects; the adaptive treatment still includes added inference
cost, and no claim of equal wall time is made. All three arms are required for
the final decision. A missing control cannot be silently dropped.

Only current instruction, front/wrist RGB and proprioception enter either
branch. Predict the earliest new onset within 32 frames, including when another
event is already ongoing. `none` means no new onset in that window. The event
type selects its predicted start; v1 adds no learned event-type embedding to
JEPA. Only the horizon input changes on the action side. There is one initial
eight-action proposal, one future-core call, and one refinement; retain the
existing normalized proposal with zero padding at steps 8–31. Longer horizons
therefore rely on the inherited padding convention, not a newly planned
32-action trajectory. Execute eight actions at 20 Hz and replan every eight,
even when the selected horizon is shorter. Log this mismatch for diagnosis.

## Step 2 boundary: mapping still pending

The admissible horizon set is `{2,4,8,16,32}`. A separate child contract must
freeze one deterministic mapping, tie/clamping behavior, `none`/invalid handling,
and any confidence rule before new model inference or rollouts. The existing
diagnostic found 66 positive starts outside this set; neither nearest nor floor
nor ceiling is selected here. Do not sweep alternatives on Week 11 success or
the Week 10 report panel. The child must bind this parent digest and cannot
change arms, panels, checkpoints, budgets or decision thresholds. Any such
change needs a versioned amendment recording the evidence already seen.

## Exact panels

Panel manifests retain full original records, seeds, ordering and state hashes.
Selection reads metadata only and uses no per-task success outcomes.

| Panel | Membership per arm | Role |
| --- | --- | --- |
| Targeted-467 | All 155 Robot Initial States, 152 Objects Layout and 160 Camera Viewpoints tasks in the existing stratified-1K manifest | Primary success endpoint |
| Clean-200 | All 40 canonical tasks, original initial-state indices 0–4 | Clean-performance guardrail |
| Qualification-8 | First canonical task in each of four suites, original state indices 5 and 6 | Engineering smoke only |

Targeted-467 contains 90 dev-210 tasks and 377 other already evaluated Week 9
tasks. Report both slices without replacing the primary panel. Clean-200 is a
subset of the previously evaluated clean-2000. Qualification episodes have no
identical clean pair in the formal panel, but share canonical tasks and were
historically evaluated. Their success rates do not select a mapping or endpoint.

There are **2,001 formal episodes** across three arms and **24 qualification
episodes**. Run arms in table order, each targeted then clean, preserving panel
order. A fixed renderer check may repeat each qualification initial prefix once
without policy inference (eight repeats); it cannot select a favorable rendering.
No performance-dependent panel expansion or additional seeds are permitted.

For contact-focused analysis, freeze an explicit 98-task secondary subset of
Targeted-467 whose lowercase task name contains one of
`open_the_top_drawer`, `open_the_middle_drawer`, `close_it`,
`turn_on_the_stove`, or `push_the_plate`. This is an articulated-manipulation/
pushing **task-description proxy**, not measured contact density or independently
validated contact labels. Keep its exact task list even if later outcomes are poor.
Also report suite, category, difficulty (including Long/unlabeled) and dev-overlap
counts; all are descriptive secondary analyses.

## Qualification and recording

Before formal evaluation, bind frozen implementation and runtime manifests,
recheck endpoint bytes/completion chains, and verify:

- Original M4 tensors remain unchanged before/after event inference. On identical
  inputs, forcing horizon 8 in the new path reproduces original M4 normalized and
  decoded actions exactly on the same qualified device/runtime. Failure blocks
  admission; relaxed numerical tolerances require a documented amendment.
- Event predictions reproduce the pinned FP32 path; the chosen supported horizon
  reaches the future core. Controlled horizon/token perturbations demonstrate
  the intended dependency; no ground-truth future or simulator state enters it.
- Endpoint reload, deterministic greedy event decoding, fallback cases, finite
  actions, recording, interruption/resume and input audits work. Exceptions and
  nonfinite tensors are failures, not ordinary `none` predictions.
- The 24 smoke episodes complete with valid artifacts within budget. Success
  rate is recorded but is not a smoke gate or tuning objective. Renderer repeats
  and all arm pairs receive the inherited settling/first-query input audit.

Inherit Week 9 simulation/observation/seeding/outcome/recording rules, not old
Week 4 model acceptance thresholds. Record lossless dual-camera HDF5, all settling
frames, states, actions, outcomes, identity and timing. Add per-query event type,
raw interval/start, categorical probabilities, mapped horizon, fallback reason,
branch identities and component timings. No labels feed the deployed policy.

Audit every arm pair on exact scene/seed/instruction, settling and first-query
inputs. Keep every mismatched task; any mismatch makes the entire affected
contrast descriptive and ineligible for the positive progression gate. Corrupt
or missing artifacts additionally mark the study incomplete. Preserve partial
attempts and receipts. Permit at most one retry per episode for classified
infrastructure failures, charged to all resource caps. Policy exceptions,
nonfinite actions and policy-step exhaustion are scored failures, without retry.
Exhausted infrastructure failures remain failures in the fixed denominator.

## Analysis and prospective decision

Primary endpoint is successes / 467 and paired adaptive-minus-fixed difference
in percentage points. Clean uses successes / 200. Report counts, paired wins/
losses, and all three contrasts. Use 10,000 common-index PCG64 bootstrap draws,
linear 2.5/97.5 percentiles: targeted variant tasks with seed 7; clean canonical
task clusters retaining all five states with independent seed 8. Secondary
one-way subset intervals resample within the named subset with seed 7. These
conditional panel intervals do not estimate training-seed variance or unseen
canonical-task generalization; no multiplicity-adjusted superiority is claimed.

Measure synchronized end-to-end query latency including both branches,
preprocessing/transfers, future prediction, refinement, decoding and CPU return.
Retain first and failed queries; report median/p95, first/subsequent summaries,
equal-episode-weight summaries, peak memory and separate load/simulator/disk cost.
Event-only Week 10 forward latency is not an end-to-end comparison.

The following are new pragmatic Week 11 engineering criteria, not inherited
statistical guarantees. A **promising-for-Week-12** result requires all of:

1. All three arm/panel matrices and artifact/input audits pass.
2. Targeted adaptive-minus-fixed gain is at least **+2.0 points**, with its paired
   95% lower bound **strictly above zero**.
3. Clean adaptive-minus-fixed point difference is at least **−2.0 points**, and
   its paired 95% lower bound is at least **−5.0 points**.
4. On each formal panel, adaptive pooled p95 query latency is at most **2.5×**
   fixed M4 and at most **200 ms**; all resource caps hold.

These are exploration gates, not a powered noninferiority test or real-robot
deadline guarantee. A positive estimate with uncertainty spanning zero is
inconclusive. Any unmet criterion means no positive progression recommendation
under v1; retain whether it reflects harm, uncertainty, latency or audit failure.
Do not tune on interim outcomes or automatically enlarge the panel.

Report horizon/fallback frequencies per query and per episode, horizon versus
remaining action-chunk length, category/event breakdowns and selected successes
and failures. Assess raw/mapped timing sensitivity on the **already recorded**
396-prompt dev predictions after Step 2 is sealed, with type coverage, conditional
onset errors, lead bins and out-of-set counts. These are reused weak-label
diagnostics, not new policy-trajectory calibration or independent QA. Do not
reopen the Week 10 report for mapping selection. New on-policy event-label
accuracy claims require a separate validated measurement design.

## Bounded compute and closeout

Use Feifei's RTX 5090, serial jobs and the pinned Week 9 runtime versions. Before
launch, archive actual GPU UUID/driver/CUDA/backend flags and require unchanged
runtime across arms. No cloud provisioning or training is included.

Budget: **3 active hours** total qualification, **12 active hours per formal
arm/panel**, and **48 active hours aggregate** for qualification plus all formal
work, including failed attempts/retries. Output ceiling **200 GiB**, free-disk
reserve **50 GiB**, peak GPU allocation/reservation below **90%** of device
capacity, and aggregate process-tree host RSS at most **28 GiB**. These are hard
caps, not measured runtime estimates. Qualification must project total runtime
and storage with a 1.2× margin within the caps before formal launch. Resource
failure stops the study for a versioned decision; do not downcast the event
branch, remove a control or truncate a panel to fit.

The final bundle must contain parent and child contracts, deployment manifests,
all attempts/raw recordings, input audits, fixed statistics, event/horizon logs,
resource/latency measurements, qualitative examples and a reproducible composite
M4-EG specification. Original M4 and Week 10 exports remain immutable. Step 1
does not assert hardware admission, deployment readiness or robot benefit.
