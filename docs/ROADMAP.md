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
is deferred beyond the core roadmap; all primary models retain the same L1
action objective.

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
