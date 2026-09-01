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
