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

**Status:** implementation complete; final gate evidence pending.

The repository now contains the pinned LIBERO-Plus and reference-policy
environments, immutable evaluation protocol and split manifests, unified HDF5
episode recorder, pinned demonstration snapshot, and manifest-driven reference
policy evaluator. The local unit and lint checks pass.

Before Week 1 is closed, run the complete `sanity-35` reference-policy manifest
from the current project revision. The resulting report must cover all 35 tasks
(`selection_is_complete_manifest: true` and `selected_task_count: 35`) and retain
the per-episode artifacts and environment provenance. A one-task smoke result is
useful setup evidence, but does not satisfy this gate.

### Week 2: dataset and automatic robot-event labeling

**Goal:** turn the pinned demonstrations into deterministic trajectory,
event-annotation, and multi-horizon sample indexes for later CPT and SFT work.

**Status:** steps 1–2 and the automated feasibility pilot are implemented;
human review of pilot candidates is next.

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
- a 200-episode human-review report with event-specific quality metrics.

**Gate:** the derived dataset must regenerate deterministically, contain no split
group overlap, and achieve approximately 95% quality on each event admitted to
the reliable ontology. Events that miss the gate are removed or marked weak; the
next phase must not silently train on them as ground truth.

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

- Use one shared 2B-class VLM backbone and one action-head design.
- Keep LIBERO-Plus as the primary benchmark.
- Do not add world-model coupling before the baseline is validated.
- Do not add EventGround before the controlled M1–M4 comparison.
- Treat negative results as valid controlled evidence.
