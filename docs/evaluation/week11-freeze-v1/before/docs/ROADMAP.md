# Roadmap

This roadmap condenses the high-level 12-week execution plan into repository
milestones. Detailed experiment design remains in the source planning document.

**Current status, October 24, 2026:** [Week 9 closed](WEEK9_FINAL_RESULTS.md)
with all 15,000 outcomes. The separately approved
[Week 10 exploratory weak-label study is closed](WEEK10_CLOSEOUT.md), including
SFT/RLVR comparisons, report evaluation, diagnostics and checkpoint handoff.
The original independent-label-QA gate remains unmet. Week 11 integration and
robot rollouts have not started. Earlier dated phase entries below are historical.

## Phase 1: trustworthy foundation (weeks 1–4)

1. Reproduce LIBERO-Plus setup and official-checkpoint behavior.
2. Build a unified trajectory recorder and automatic event-labeling pipeline.
3. Implement and overfit the shared pure-VLA baseline (B0).
4. Train B0 and freeze the common protocol, seeds, and evaluation task IDs.

The project does not move to world modeling until the simulator and baseline are
reliable.

## Near-term execution

### Week 1: protocol and environment

**Status:** complete as of 2026-09-12.

The repository now contains the pinned LIBERO-Plus and reference-policy
environments, immutable evaluation protocol and split manifests, unified HDF5
episode recorder, pinned demonstration snapshot, and manifest-driven reference
policy evaluator. The local unit and lint checks pass.

The complete `sanity-35` reference-policy manifest ran from clean project
revision `5e7141183d19651d5bd78e9cbcc7a477fdf7082f`. The report covers all 35
tasks (`selection_is_complete_manifest: true` and `selected_task_count: 35`): 28
succeeded, seven reached their policy-step limit, and no exception or retry
occurred. All 35 HDF5 episodes independently pass the recorder contract and
match the hashes in the report. Compact evidence is committed at
[`evaluation/sanity-35-reference-policy-v1.json`](evaluation/sanity-35-reference-policy-v1.json).

### Week 2: dataset and automatic robot-event labeling

**Goal:** turn the pinned demonstrations into deterministic trajectory,
event-annotation, and multi-horizon sample indexes for later CPT and SFT work.

**Status:** complete as of 2026-09-10. Steps 1–7 are implemented and validated.
The deterministic boundary queue contains 200 episodes and 254 predictions
across 38 tasks with no selection shortfall. The reviewer completed a balanced
60-episode subset, inspected the full episodes for both proposed and missed
events, accepted all 79 `gripper_close` predictions, and authorized stopping the
review. Observed precision and recall are 100%, mean boundary error is zero, and
there are no corrections or ambiguous judgments. This explicitly revises the
original 200-episode target; it does not claim that the remaining 140 queue items
were reviewed. `week2-dataset-v1` freezes all 14,347 episodes and 10,300,666
samples; an independent regeneration matched all 11 deterministic artifacts
byte-for-byte, and all sample foreign keys plus split isolation passed. The
frozen v1 files remain immutable, so promotion of `gripper_close` to reliable
training supervision belongs in a subsequent dataset version.
See [`WEEK2_STATUS.md`](WEEK2_STATUS.md) for the step-by-step completion and
deliverable ledger.

High-level steps:

1. **Lock the data contracts.** Define versioned schemas for a trajectory record,
   event interval, split assignment, and multi-horizon sample. Every derived
   artifact should retain its source episode, frame range, task, source revision,
   annotation version, and digest.
2. **Build and validate the streaming reader.** Read the LeRobot Parquet and video
   shards without copying the 15 GiB source snapshot. Verify 20 Hz alignment,
   frame counts, state/action shapes, camera decoding, and deterministic episode
   ordering before deriving labels.
3. **Run an event-labeling feasibility pilot.** Start with a stratified sample and
   separate events directly supported by state/action signals (gripper transitions
   and motion phases) from events that need object-relative evidence (reach,
   contact, grasp, target entry, release, and success). The source snapshot has no
   MuJoCo state, so object-relative labels must be backed by a documented visual
   method, a simulator-derived sidecar, or omitted from the reliable ontology.
4. **Implement deterministic event extraction.** Add temporal smoothing,
   hysteresis, minimum-duration rules, confidence/evidence fields, and explicit
   handling for absent or ambiguous events. Preserve raw candidates separately
   from the final filtered annotations.
5. **Create leakage-resistant splits and sample indexes.** Group episodes by the
   strongest available initial-condition identity. If the source has no initial
   seed, define and record a conservative surrogate from task identity, initial
   proprioception, and initial-frame fingerprints. Generate train/validation
   manifests and horizons `2, 4, 8, 16, 32, next-event` without crossing episode
   boundaries.
6. **Add boundary visualization and QA.** Render compact front/wrist previews with
   event overlays and review 200 episodes stratified across tasks, event types,
   and trajectory lengths. Report per-event precision/recall, boundary error,
   coverage, ambiguity, and reviewer corrections rather than only aggregate
   accuracy.
7. **Scale and freeze the dataset version.** Process all 14,347 episodes, validate
   referential integrity and split isolation, publish dataset statistics and
   digests, and benchmark input throughput. Cache decoded images only if I/O is a
   measured bottleneck; defer JEPA-feature caching until its encoder is fixed.

Week 2 deliverables:

- versioned trajectory, event, split, and multi-horizon schemas;
- reproducible reader, labeler, index generator, validators, and visualization
  command;
- derived manifests and annotations stored outside Git, with compact provenance
  and statistics committed to the repository;
- a completed human-review report with event-specific quality metrics. The
  original 200-episode plan closed after a reviewer-authorized, balanced
  60-episode subset produced zero errors across 79 predictions.

**Gate:** the derived dataset must regenerate deterministically, contain no split
group overlap, and achieve approximately 95% quality on each event admitted to
the reliable ontology. Events that miss the gate are removed or marked weak; the
next phase must not silently train on them as ground truth.

### Week 3: implement and overfit the shared B0 policy

**Goal:** build the pure-VLA control baseline used as the common initialization
for every subsequent experiment.

**Status:** complete as of 2026-09-13; frozen at `week3-complete-v1` with one
documented acceptance revision.

**Backbone decision:** use only `Qwen/Qwen3.5-2B`. Qwen-VLA,
Qwen-RobotManip, and other pretrained action-policy models are excluded as
initializations, fallbacks, dependencies, and experiment-matrix entries. Pin an
immutable Qwen3.5-2B model revision and its complete manifest before training.

High-level steps:

1. Create an isolated B0 training environment without modifying the locked
   reference-policy runtime.
2. Implement train-split-only action normalization and the LIBERO gripper
   convention for eight-step, seven-dimensional action chunks.
3. Build a deterministic dual-camera, instruction, proprioception, and action
   training loader from the frozen Week 2 split.
4. Connect Qwen3.5-2B conditioning features to a trainable action-query
   Transformer that predicts the normalized action chunk with an L1 objective,
   while freezing the vision encoder and base transformer.
5. Overfit one fixed batch, then a fixed ten-demonstration cohort.
6. Integrate closed-loop inference with the existing evaluator and recorder;
   measure action latency, peak VRAM, and checkpoint reproducibility.

Week 3 delivers trainable B0 code, tests, pinned runtime/model provenance,
overfit curves, profiling evidence, and the first successful closed-loop
training-task rollout. Week 4 is blocked until the small cohort clearly overfits
and the closed-loop smoke gate passes. See
[`WEEK3_BASELINE_DESIGN.md`](WEEK3_BASELINE_DESIGN.md) for the locked interface
and exclusion rules. Flow matching is not required by the B0-to-M4-EG study and
is deferred beyond the core roadmap; all primary models retain the same
continuous-action L1 plus binary-gripper BCE objective.

All six roadmap steps and the detailed seven-step design sequence are complete.
The train-only data path exposes 1,715,954 valid chunks and passed deterministic
identity plus full normalization inversion. Qwen conditioning, the rank-16 LoRA
allowlist, deterministic action-query decoder, frozen-base gradients, BF16
execution, and exact checkpoint reload all passed. The fixed batch reached a
3.803% loss ratio and `0.048459` decoded MAE. The ten-demonstration checkpoint
reached a `0.126083` hybrid-objective ratio, `0.027970` decoded MAE, and
99.552957% gripper accuracy.

The original 100% gripper criterion remains failed. A reviewer-authorized design
revision accepted the residual after all 52 errors localized within four frames
of real command transitions and no label contradictions were found. This is a
Week 3 engineering exception, not a rewritten result or a new evaluation
threshold.

The predeclared cohort rollout sweep succeeded on one of ten base tasks. The
successful task repeated at step 118 through the shared HDF5 recorder and passed
deep artifact validation. Concurrent profiling measured 5.86 GB peak
process-attributed GPU memory with at least 28.07 GB globally free, 4.59/4.65 GB
Torch allocated/reserved peaks, and 32.94 ms median query latency. The closure
passes 191 tests and lint. See [`WEEK3_STATUS.md`](WEEK3_STATUS.md) and the sealed
[`evaluation/week3-completion-v1.json`](evaluation/week3-completion-v1.json).

**Week 4 handoff:** begin full B0 training from the pinned Qwen initialization
and frozen Week 3 code/data/protocol, not from the ten-demonstration diagnostic
checkpoint. The 1/10 cohort smoke rate and failed perturbed candidate are open
model-quality risks, not unfinished Week 3 work. Standard LIBERO, LIBERO-Plus
`dev-210`, and per-perturbation failure analysis remain mandatory Week 4 gates.

### Week 4: train, evaluate, and freeze the shared B0 baseline

**Goal:** establish the trained policy initialization and matched experimental
recipe inherited by later models.

**Subsequent training decision:** the [two-total-action-epoch protocol](PHASE2_TRAINING_PROTOCOL.md)
uses completed B0 as the evaluation control. Each M1–M4 policy starts from the
same base Qwen and fresh shared components, with identical effective batches
and sampling. This supersedes the closure-time B0-continuation proposal.

**Status: closed September 24 with an explicit user-authorized milestone
acceptance revision.** The selected fresh two-epoch all-L1 trajectory-v2 B0 is
the fixed reference for the remaining internal study: **93.35% clean / 77.14%
dev-210**, versus OpenVLA-OFT+ **96.75% / 80.95%**. Training, both rollout panels,
failure review, sampling audit and checksum-pinned handoff are complete.
[Closeout and fair-comparison rules](WEEK4_CLOSEOUT.md) ·
[sealed completion](evaluation/week4-completion-v1.json).

The original clean rule remains **INCONCLUSIVE** for this B0; strict v2 readiness
remains **NOT_ASSESSED_INPUT_MISMATCH** because of one reproduced rendering
discrepancy. Historical five-epoch B0 remains **FAIL**. Accepting this checkpoint
for project progression does not change those results or establish OpenVLA
parity. The [C/S sampling screen](WEEK4_STAGE_SAMPLING_AUDIT.md), replanning
changes and other B0 optimization are deferred. [Week 5](WEEK5_STATUS.md) is now
the active milestone; its training has not launched.

The [fresh v2 recipe](WEEK4_FRESH_ALL_L1_V2.md) supersedes the original recipe
below: OFT action head, rank-32 LoRA, all-action L1, trajectory-v2 groups and
predict-eight/execute-eight. Later arms retain these controls and matched
additional-training budgets.

#### Historical execution and original requirements

Started on 2026-09-14. The entry audit confirms that Weeks 1–3 are
closed under their documented acceptance decisions. As of 2026-09-16, the
five-epoch recipe, bounded-memory trainer, host-managed recovery, and full-bundle
evaluator are implemented; 209 tests pass locally and on Feifei. Clean and
`dev-210` task identities are frozen. An interrupted pilot resumed from its
checkpoint; the corrected deterministic pilot passed both reload checks and the
five-day budget guard. Five-epoch production training launched on Feifei at
**2026-09-15 07:43:40 PDT** under a persistent systemd service independent of SSH.
Training completed successfully on **2026-09-16 at 14:56 PDT** after five epochs
and 31 h 12 min, with epoch 5 selected and its checkpoint/reload validation
verified. The frozen clean-40 evaluation completed at **20:25:17 PDT** that day:
**26/40 successes (65%)**, 14 policy timeouts, zero exceptions/retries, and all
40 HDF5 artifacts independently validated. On **2026-09-17**, the installed public
OpenVLA reference completed the identical panel with **39/40 (97.5%)**; all 40
pairs' initial inputs and evaluation identities matched. The
[comparison audit](WEEK4_PUBLIC_REFERENCE.md) documents unmatched model/training
settings and the repaired report-writing failure (no episode reruns). The
[acceptance rule](WEEK4_CLEAN_ACCEPTANCE.md) was frozen on **2026-09-18** for
2,000 paired initial states, using the installed external reference for the
operational clean gate. The original undefined configuration-matching condition
is explicitly replaced for this gate; model/training differences remain visible.
The five-point margin is retained with a paired 95% interval requirement.
The completed larger comparison scored **B0 68.25% versus OpenVLA 96.75%**
on 2,000 identical task/state pairs and **failed** the frozen clean gate. On
2026-09-19, `dev-210` completed with **106/210 successes (50.48%)**, zero
exceptions/retries and all artifacts validated. The seven-category failure
analysis and 26 review previews are complete; see the
[results and proposed diagnostics](WEEK4_DEV210_FAILURE_ANALYSIS.md). Resolution
of the clean deficit and final Week 4 closure remained open at that point;
the September 24 decision above subsequently closed the milestone.
See [`WEEK4_STATUS.md`](WEEK4_STATUS.md),
[`B0_FULL_TRAINING.md`](B0_FULL_TRAINING.md), and the sealed
[`evaluation/week4-entry-v1.json`](evaluation/week4-entry-v1.json).

High-level steps:

1. **Declare the shared training and evaluation recipe.** Inherit the frozen
   Week 2 split, pinned Qwen3.5-2B initialization, rank-16 LoRA, eight-action
   decoder, and continuous L1 plus unit-weight gripper BCE. Declare sample
   ordering, optimizer/schedule, batch size, update budget, seeds, validation
   and checkpoint-selection rules. Pin the Standard LIBERO clean task/trial
   manifest and comparable public baseline before measuring the clean gate;
   preserve the existing LIBERO-Plus protocol and `dev-210` task IDs.
2. **Scale and validate the training/evaluation path.** Add a bounded-memory
   full-dataset trainer, complete resume state, validation metrics, and a
   general checkpoint/evaluator contract. Run a short throughput, memory, and
   save/resume pilot on `feifei` before the full run. The existing fixed-cohort
   trainer and cohort-restricted inference adapter are Week 3 diagnostics.
3. **Train full B0.** Start from the pinned Qwen weights with freshly initialized
   trainable components and use all 11,521 training episodes / 1,715,954 valid
   chunks. Select the checkpoint using the predeclared validation rule and
   preserve training curves, source/sample identities, and checkpoint hashes.
4. **Pass Standard LIBERO clean evaluation.** Apply the frozen
   [acceptance v1](WEEK4_CLEAN_ACCEPTANCE.md) to the same 2,000 task/state pairs
   for B0 and the pinned installed public reference. The rule resolves the
   original undefined configuration-matched comparator as an evaluation-matched
   external operational reference. Pass requires the lower endpoint of the
   paired 95% B0-minus-reference interval to be at least −5 percentage points;
   an upper endpoint below −5 fails, otherwise the result is inconclusive.
   Require complete valid evidence and preserve unmatched training/model
   qualifications. Prioritize policy fixes if clean control is inadequate.
5. **Evaluate `dev-210` and explain failures.** Run all 210 frozen tasks, report
   success and confidence intervals by suite, perturbation category, and
   difficulty, and retain failure videos plus validated HDF5 artifacts across
   all seven perturbation categories.
6. **Freeze the Week 4 handoff.** Publish `B0.ckpt`, the clean/perturbed baseline
   table, failure examples, and sealed provenance. Freeze the exact shared
   samples, optimizer, update budget, action steps, LoRA rank, seeds, task IDs,
   and checkpoint-selection rule for subsequent controlled experiments.

**Original gate (retained for provenance):** complete full training and both evaluation suites, establish the
declared clean-baseline comparison, and validate the reproducible checkpoint
bundle before beginning Week 5 world-model work. The Week 3 gripper acceptance
exception does not relax this gate. The 1/10 cohort smoke outcome remains a
model-quality risk to resolve through this week's training and evaluation.

**Current disposition:** the user accepted the completed v2 B0 and remaining
performance/validity qualifications as the internal reference. External-reference
near-parity is no longer a prerequisite for beginning Week 5. The checkpoint and
scientific evidence are frozen in the closure record; no prior gate is relabeled
as passed.

## Phase 2: controlled future-prediction study (weeks 5–9)

1. Train matched pixel and JEPA future predictors.
2. Compare auxiliary-only models (M1/M2) with inference-coupled models (M3/M4).
3. Evaluate the complete controlled matrix on shared LIBERO-Plus splits.

### Week 5: pixel temporal CPT — complete

All six [execution steps](WEEK5_STATUS.md) are complete as of September 26, 2026.
One full training pass completed 221,633 updates in 14.49 active hours. The
[final checkpoint and Step 6 validation](WEEK5_PIXEL_VALIDATION.md) are verified:
33.27% overall L1 and 45.04% medium/long-horizon motion L1 gains over copy,
positive SSIM gain and all 10 camera/horizon guardrails. All 480 fixed examples
were reviewed, and `C_pixel-v1` is frozen. Long-horizon ghosting and imperfect
motion remain documented; downstream policy improvement is untested.

Completed scope:

1. Regenerated future pairs under trajectory-v2 groups at horizons 2/4/8/16/32;
   kept test locked and weak next-event labels excluded.
2. Froze copy/background baselines, reconstruction metrics and success criteria.
3. Implemented the action/horizon predictor with 4.63M trainable parameters and
   frozen base Qwen features.
4. Froze the recipe, verified resources and recovery, then completed fresh
   training on Feifei under a persistent service.
5. Evaluated all held-out future pairs and froze C_pixel for the later study.

### Week 6: JEPA temporal CPT — closed, research use approved by exception

All six [steps](WEEK6_STATUS.md) and the [final results package](WEEK6_CLOSEOUT.md)
are complete as of September 28, 2026. Fresh production completed 221,633 updates
over the same 8,221,365 training pairs in 20.70 active hours. Full validation,
independent audits, 480 fixed visual examples and appearance comparison are complete.
Overall cosine error improves 28.61% and retrieval Recall@1 reaches 64.09%, with
both cameras passing collapse checks. The frozen gate nevertheless **fails**:
front-camera h=2 error regresses 7.22% against a 2% cap. The exact candidate is
retained; no gate-passing `C_jepa-v1` is produced. The separate training-only
diagnostic/residual follow-up is complete and does not replace this endpoint.

**October 2 progression revision:** at the user's request, further short-horizon
repair is deferred. The [recorded exception](evaluation/week6-progression-v1/decision.json)
approves the original endpoint as `C_jepa-study-v1` for M2/M4, while M1/M3 retain
`C_pixel-v1`. The [module registry](../configs/training/phase2_future_modules_v1.json)
pins both artifacts. This preserves a matched-budget comparison of these fixed
implementations, with the JEPA weakness disclosed; it cannot establish universal
Pixel/JEPA superiority. The original failed gate remains unchanged.

**Week 6 has no remaining work. Week 7 M1/M2 integration and pilots are complete; production training is next.**
Checkpoint dependencies are satisfied; method recipes, the OFT-compatible
feature route and resource/recovery checks still need implementation. No new
training launches as part of this revision. Week 9's complete controlled matrix
remains a future milestone. All policy arms
inherit the selected B0 recipe and exactly two action epochs, starting from the
same base-Qwen initialization. Reuse completed B0 as the control; no extra B0
continuation is needed. Disclose and separately match Pixel/JEPA CPT budgets.
The default is sequential training on Feifei, with resource/recovery pilots for
each architecture; cloud compute is optional acceleration.

### Week 7: auxiliary VLA policies M1 and M2

**Started October 3, 2026. Steps 1–4 complete.** The
[frozen experiment](WEEK7_EXPERIMENT.md) and [step ledger](WEEK7_STATUS.md) cover
both M1 Pixel and M2 JEPA. A fresh entry audit verifies Week 6 closure and the
explicit research-use exception for the unchanged JEPA endpoint.

The experiment fixes the original module hashes, fresh shared initialization,
two action epochs with identical batches, five horizons and paired dev-210
development evaluation against completed B0. Original JEPA and B0 validity
qualifications remain disclosed. Positive policy gains are not required for
completion. The [common OFT interface, numerical recipes and pilots](WEEK7_AUXILIARY_VERIFICATION.md)
are verified, including exact interrupted-process recovery. Conservative training
projections are 92.7 hours for M1 and 166.2 hours for M2 on Feifei. M1 production
started October 3 at 10:53 PDT under the [recorded launch](evaluation/week7-m1-launch-v2/launch.json).
M2 remains pending; diagnostic weights are excluded.

## Phase 3: event-adaptive prediction (weeks 10–11)

### Week 10: EventGround SFT and bounded temporal RLVR

**Closed under the separately approved exploratory weak-label scope.** The
revised next-new-onset task used 1,609 SFT prompts / two passes and bounded
500-prompt RLVR pilots. The matched SFT-only continuation is the provisional
candidate. Revised-task RLVR failed development promotion; no full/main RLVR
run was launched. The fixed report comparison and
[focused diagnostic](WEEK10_DIAGNOSTIC.md) are complete. Automatic labels remain
uncertified by independent QA, and no algorithm superiority or robot improvement
is established. See the [closeout](WEEK10_CLOSEOUT.md) and
[historical ledger](WEEK10_STATUS.md).

### Week 11: integration and targeted robot evaluation

**Not started.** Freeze and qualify how predicted event timing controls the
future-prediction horizon, then compare fixed and adaptive M4 on matched robot
tasks and perturbations. Checkpoint selection, horizon mapping, policy-weight
isolation, latency and rollout scope must be explicit. This work is separate
from the completed Week 10 event-prediction study.

## Phase 4: final evidence (week 12)

Run paired evaluation, confidence intervals, latency measurements, failure
analysis, and package the code, checkpoints, report, and demo.

## Scope guardrails

- Use `Qwen/Qwen3.5-2B` as the sole shared backbone and one action-head design.
- Do not use Qwen-VLA, Qwen-RobotManip, or another pretrained action policy.
- Keep LIBERO-Plus as the primary benchmark.
- Do not add world-model coupling before the baseline is validated.
- Do not add EventGround before the controlled M1–M4 comparison.
- Treat negative results as valid controlled evidence.
