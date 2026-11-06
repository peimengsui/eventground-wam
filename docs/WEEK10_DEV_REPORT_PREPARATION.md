# Week 10 event-dev and event-report preparation

The exact held-out **source metadata** is frozen in
[week10-evaluation-preparation-v1](evaluation/week10-evaluation-preparation-v1/).
This package reads only the already approved partition rows. It does not open
source payloads, stage actions or initial states, record demonstrations, extract
labels, run inference or authorize training. Both payload partitions remain
locked. No existing study, source partition or completed experiment is changed.

| Panel | Sources | Tasks | Source frames | Capture steps including ten settles/source |
| --- | ---: | ---: | ---: | ---: |
| event-dev | 103 | 40 | 17,123 | 18,153 |
| event-report | 77 | 40 | 13,070 | 13,840 |

Every source belongs to `yifengzhu-hf/LIBERO-datasets`, revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. Identity is the pinned repository,
revision, HDF5 path and original demo ID. The metadata files retain the exact
source IDs and inherited row order. All 480 train/dev/report IDs are unique;
the 103 and 77 panel identities cover the same 40 canonical tasks as training.
All generated variants remain in their physical source's partition.

## Opening gates and execution sequence

Event-dev execution waits for passing independent training-label QA **and** a
valid class/source-capped training prompt manifest. This avoids consuming held-out
evidence while training data remains unusable. Before any dev payload read, seal
a bounded execution child with exact input allowlists, code/runtime hashes,
resource accounting, prompt-selection code and the supporting train admissions.

Event-report has the additional frozen gate: **all training decisions must be
sealed first**, including the predetermined event-dev pilot go/no-go decision,
the resulting main-training/control scope, and fixed endpoint selection. A sealed
numerical recipe by itself does not satisfy that gate. The report partition must
not be used to tune predicates, prompts, head design, learning rates, checkpoints
or stopping decisions.

The future execution child must complete these operations in order:

1. Verify source IDs against the immutable panel rows and reject unknown,
   duplicate or cross-partition sources. Open only each admitted demonstration's
   full action trace, first full-state row and model XML; hash these alongside
   canonical scenes, assets and runtime. No original video or unrelated demo
   payload is needed. Exact array lengths must equal the metadata frame counts.
2. Qualify the staged source states against the existing 40 canonical schemas and
   visual-state predicates. The current initial-200 recorder is explicitly
   train-scoped; a separate tested evaluation recording child must enforce these
   panel identities rather than relabeling them as training sources.
3. Record every admitted source once, in the inherited order, with ten settling
   steps, 20 Hz, seed zero, Panda OSC_POSE, both lossless 256×256 cameras and the
   qualified fresh pre-action snapshot. Preserve all outcomes. A failed source
   stops execution and remains evidence; there is no silent retry, replacement,
   task removal or success-based selection.
4. Independently reconstruct physical state, contacts, poses and dynamic visual
   state for every frame. Verify all evidence-file hashes. Use the same fixed
   three-frame/two-view camera check: 618 dev and 462 report images. Preserve
   strict equality results and any rendering qualifications instead of silently
   accepting a new pixel tolerance.
5. Apply the exact frozen labeler only after capture verification, with source
   and implementation hashes. Preserve ambiguity, censoring, failures and all
   candidate exclusions. Use the training-validated rules without held-out
   retuning; evaluation-side label integrity/quality review remains a distinct
   documented gate, not a retroactive claim that every extracted label was
   individually reviewed.
6. Seal immutable prompt manifests with exact anchors, targets, exclusion counts,
   source grouping, order and media hashes. Input loaders expose only the current
   front/wrist images, current proprioception and instruction. Future physical
   evidence stays offline. Compare all models on the same admitted manifest.

## Frozen sampling and reporting rules

Event-dev is a balanced panel of **at most 1,000 prompts, at most 250 per class**.
Use deterministic task/source round-robin selection, without duplicates or class
substitution. If class coverage limits the panel, seal the common balanced count
and report the shortage before using it for the predetermined pilot gate.

Event-report uses the **stride-eight eligible-anchor grid**, at most **3,000
prompts**, deterministic task/source round-robin selection, **no class balancing
and no outcome filtering**. Record the actual selection distribution, class
frequencies and omissions. The ceiling is not a promised sample count.

The active JSON retains maximum four anchors per physical event and sixteen
prompts per source under its data section. This plan retains these exact global
maxima for both evaluation panels; it does not infer an exception from the prose
introducing them alongside the training sampler. The report's effective ceiling
is therefore min(3,000, 77×16) = **1,232**. All eligible stride-eight candidates
and capped omissions remain counted, and selection remains unbalanced. The future
execution manifest must bind these limits explicitly before payload inspection.

The study's evaluation remains unchanged: typed IoU, type precision/recall and
confusion, no-event false positives, start/end error with matched coverage,
joint type/two-frame accuracy, horizon-bin calibration, latency, memory and all
attempt costs. Bootstrap uses original physical source IDs, 10,000 common PCG64
draws with seed 10, linear 2.5%/97.5% quantiles and the existing undefined-draw
limit. The two baselines remain always-none and the train-only event/timing prior.
The primary comparison remains macro positive-event IoU for RLVR minus SFT,
with wrong types and misses contributing zero; the SFT-continuation control is
retained. Positive gain is not required to report a completed study honestly.

## Proposed child budgets, not execution authorization

The plan proposes the following ceilings for a later bounded child. They are
engineering estimates informed by the completed initial capture, not measured
performance for these held-out sources or authorization to consume resources.

| Operation | event-dev proposal | event-report proposal |
| --- | ---: | ---: |
| Staging/preflight | 600 s CPU; 1 GiB new artifacts | 600 s CPU; 1 GiB new artifacts |
| Capture + independent verification | 1,800 s active; 18,153 control steps | 1,500 s active; 13,840 control steps |
| Capture/verification output ceiling | 30 GiB | 24 GiB |
| Numeric extraction + manifest building | 600 s CPU; 1 GiB outputs | 600 s CPU; 1 GiB outputs |
| Minimum free space before execution | 150 GiB | 150 GiB |
| Attempts per source | One | One |

The observed train capture rate suggests roughly 24 GiB of dev evidence and
18 GiB of report evidence, but these are projections. The parent must verify a
cumulative storage/time ledger before activating any proposed ceiling. Child
caps do not increase an inherited aggregate cap, and unused training reserves
cannot be silently repurposed. If the remaining aggregate allowance cannot
accommodate the exact panel, the execution child stays blocked pending an explicit
resource revision; no source is dropped. Human annotation/review costs need their
own sealed, reported budget. Model evaluation cost must be accounted separately
inside the study's existing 40 GPU-hour limit and one-substantial-job rule.

## Claim limits

Source disjointness applies to the new canonical event dataset. Historical
cropped/resampled aliases remain unresolved, and M4's prior action-training
exposure to aliases is not ruled out. These panels do not establish unseen-task
generalization or clean reuse of historical Week 9 robot-rollout panels after
adaptation. The original rendering, single-seed, JEPA-quality and prior
development qualifications remain attached to all later results.

This preparation makes the **identity and remaining gates** concrete. It does
not mean event-dev/report recordings, labels or prompt manifests already exist.
