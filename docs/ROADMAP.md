# Roadmap

This roadmap condenses the high-level 12-week execution plan into repository
milestones. Detailed experiment design remains in the source planning document.

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

**Status:** started on 2026-09-14. The entry audit confirms that Weeks 1–3 are
closed under their documented acceptance decisions. As of 2026-09-15, the
five-epoch recipe, bounded-memory trainer, host-managed recovery, and full-bundle
evaluator are implemented; 209 tests pass locally and on Feifei. Clean and
`dev-210` task identities are frozen. An actual interrupted pilot resumed from
its checkpoint; the corrected deterministic pilot's final result awaits retrieval
because Feifei is offline again. Production has not been launched by this task.
The matched clean comparator remains open. See [`WEEK4_STATUS.md`](WEEK4_STATUS.md),
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
4. **Pass Standard LIBERO clean evaluation.** Evaluate the declared clean
   manifest and compare with the configuration-matched public baseline. The
   original roadmap recommends a gap of at most approximately five percentage
   points; resolve the comparator and measurement protocol before treating this
   as a measured gate. Prioritize policy fixes if clean control is inadequate.
5. **Evaluate `dev-210` and explain failures.** Run all 210 frozen tasks, report
   success and confidence intervals by suite, perturbation category, and
   difficulty, and retain failure videos plus validated HDF5 artifacts across
   all seven perturbation categories.
6. **Freeze the Week 4 handoff.** Publish `B0.ckpt`, the clean/perturbed baseline
   table, failure examples, and sealed provenance. Freeze the exact shared
   samples, optimizer, update budget, action steps, LoRA rank, seeds, task IDs,
   and checkpoint-selection rule for subsequent controlled experiments.

**Gate:** complete full training and both evaluation suites, establish the
declared clean-baseline comparison, and validate the reproducible checkpoint
bundle before beginning Week 5 world-model work. The Week 3 gripper acceptance
exception does not relax this gate. The 1/10 cohort smoke outcome remains a
model-quality risk to resolve through this week's training and evaluation.

## Phase 2: controlled future-prediction study (weeks 5–9)

1. Train matched pixel and JEPA future predictors.
2. Compare auxiliary-only models (M1/M2) with inference-coupled models (M3/M4).
3. Evaluate the complete controlled matrix on shared LIBERO-Plus splits.

## Phase 3: event-adaptive prediction (weeks 10–11)

1. Train the EventGround head with structured SFT and narrowly scoped RLVR.
2. Integrate event-adaptive horizons into JEPA-WAM (M4-EG).
3. Compare fixed and adaptive horizons on targeted distribution shifts.

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
