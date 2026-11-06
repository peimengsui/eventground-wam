# Week 10 exploratory RLVR pilot and held-out event panels

This child implements the user's approved steps 1–4 after the completed weak-label
SFT endpoint. The [prospective protocol](../configs/training/week10_weak_rlvr_evaluation_v1.json)
was sealed before any new held-out payload access. The original independently
reviewed-label route, M4, completed SFT and all earlier experiments are preserved.

## Scope and fixed decisions

The pilot uses 404 unique prompts selected deterministically from the existing
1,332-prompt training pool: 125 contact, 29 grasp, 125 gripper-close and 125 none.
There are 26 logical updates: 25 batches of 16 followed by one batch of four,
normalized by its actual size. Each prompt has four sampled completions.
Microbatch one, FP32, seed 17, optimizer settings and exact trainable parameters
remain the previously qualified numerical recipe.

Each attempt starts from the completed SFT parameters and a fresh optimizer. A
separate frozen copy of that SFT is the reference. The reward weights are 0.40
interval IoU, 0.25 onset timing, 0.20 onset inside the extracted event, and 0.15
valid format. These are scores against automatic physical predicates, not
independently verified physical ground truth. Correct weak negatives receive all
four components; invalid output receives zero. Constant format rewards cancel
within the four-completion advantage calculation.

The predefined dev gate requires all of:

- Macro positive-event IoU improves by at least 0.01; misses and wrong types score zero.
- No-event false positives increase by no more than 0.02.
- Matched-positive onset MAE worsens by no more than one frame.
- All output formats and integrity checks pass; no metric needed by the gate is undefined.

If the primary pilot fails this gate, exactly one calibration is permitted:
restart from the same SFT with weights 0.30/0.35/0.20/0.15 and repeat the identical
404 prompts. No further retries, checkpoint selection or hyperparameter search
are permitted. Both attempts and all outcomes remain evidence.

This task stops after the pilot decision. If a pilot passes, the predefined next
scope is a fresh-SFT, one-pass RLVR run over the 1,332 prompts and a matched
one-pass SFT continuation. Those full runs require a later user instruction. If
both pilots fail, that main/control branch is omitted and the negative result is
retained. Pilot weights never initialize the full run.

## Evaluation panels

The unchanged physical-source split provides 103 event-dev and 77 event-report
sources, both covering all 40 tasks. They are new canonical event panels; M4's
historical exposure to transformed aliases remains unresolved.

Event-dev applies the unchanged conservative eligibility rules, caps of four
anchors/event and sixteen/source, and the deterministic task/source sampler.
At most 250/class are selected, then every class is trimmed to the common minimum.
A shortage is reported without duplication or class substitution. Model results
cannot influence this selection.

Event-report uses eligible anchors on a stride-eight grid with no class balancing
or outcome filtering. The same caps impose an effective maximum of 1,232 prompts.
Its payload stays locked until the pilot decision and resulting training scope
are sealed. Preparing the report panel does not authorize model evaluation on it
or report-driven changes to training.

All source attempts, camera comparisons, uncertainties, censoring and exclusions
are retained. Every-frame state/contact/pose/visual reconstruction is independent
of the recorder. Strict camera equality and the inherited one-channel-level
rendering qualification are reported separately. A larger discrepancy stops
admission. Neither panel has independent human label QA; all metrics are explicitly
exploratory weak-label agreement.

## Qualification and resource limits

The real Feifei qualification passed in 42.06 seconds, with 18.15 GiB peak reserved
GPU memory. It performed two logical updates on 16 fixed prompts and replayed the
second update from a checkpoint. Parameters, optimizer, RNG and metrics reproduced
bitwise; frozen reference, original M4 and stored SFT stayed unchanged. The measured
projection for 26 updates plus initialization/checkpoints is about four minutes.
The hard bound is 30 minutes per pilot, not the projection.

Dev source staging completed all 103 sources in 234.56 seconds with 9,693,592 HTTP
bytes. Only actions, initial state and model XML were read; no original videos.
Both panels now have completed capture, independent verification and weak-label extraction.
The dev panel has 48 balanced prompts. The report has 423 prompts but zero grasp
examples after the frozen stride-eight grid. The three-class final-report primary
metric is therefore unestimable; this requires a separate coverage/scope decision
before full training. The primary pilot passed its dev gate (45.18% to 46.38% macro
positive IoU); the paired interval includes zero. No calibration or full run was
launched. See the [completed evidence and limitations](evaluation/week10-weak-rlvr-v1/README.md).

Training/inference remains within the inherited 40 GPU-hour and 100 GiB limits,
with one substantial GPU job at a time. Held-out capture has a separate explicitly
allocated budget: 30 GiB/30 minutes for dev and 24 GiB/25 minutes for report capture
plus verification, one attempt/source, and 150 GiB minimum free storage. Staging
and numeric extraction each have separate ten-minute/1 GiB ceilings per panel.
The historical train-300 reserve is not repurposed.

Evidence lives in [week10-weak-rlvr-v1](evaluation/week10-weak-rlvr-v1/).
