# Week 4: frozen B0 continuation experiment

**Frozen:** September 21, 2026 PDT.  
**ID:** `week4-oft-continuation-v1`.  
**Status:** experiment and measurements frozen; training has not launched.

## Purpose and scope

Test whether more training and mixing instructions within batches can recover
new B0's Goal performance while retaining its Object and LIBERO-10 gains.
Previous B0 remains the shared baseline. Keep Qwen3.5-2B, the new OFT architecture,
and continuous L1 plus unit-weight gripper BCE unchanged in this experiment.

The [Goal investigation](WEEK4_OFT_GOAL_DEV210_ANALYSIS.md) found wrong-goal
behavior and weak response to instruction changes. New B0 saw 1.222 epochs,
versus five for previous B0. Gripper BCE explained 97% of the diagnostic Goal
loss improvement, but its contribution to shared-parameter gradients is not yet
measured. These are hypotheses to test, not established causes.

This document freezes Step 1 of the next plan. The sampler, continuation runner
and measurement execution still need implementation and launch verification.
No training, new rollouts or GPU gradient diagnostics ran as part of the freeze.
The completed CPU audit read recovery tensors and dataset metadata only.

## Two arms with matched exposure

Both start from the evaluated new B0 **update 72,000**, policy SHA-256
`09ec7afad3f7435487334718cd32e310456d3fdea04c65bc888a097ceadaa584`.
The complete recovery checkpoint has SHA-256
`a8de8023abe6eb8996f919a9f3afcf6ad47bf4d34241f475aae84c05a5f844ef`.
Its model tensors exactly equal the selected policy. Adam state and RNG are
present; both arms must independently restore them without modifying the parent.

| Arm | Only experimental difference |
| --- | --- |
| A: control continuation | Preserve the original batches containing one episode each. |
| B: task-mixed continuation | Regroup the same examples across batches, mixing task instructions. |

Both use the **constant inherited learning-rate floor, 1e-5**, for the action/
state modules and LoRA. Preserve Adam moments, step counters, weight decay,
clipping and all model/data settings. There is no learning-rate restart or
objective change. This is a bounded continuation at the current learning rate,
not a test of every possible longer-training schedule.

Arm B uses windows of 128 source updates, truncated at each 15,000-update
measurement boundary. Sort `(epoch, episode, anchor)` presentations by the
declared SHA-256 key, then split into the original sequence of batch sizes.
No examples are added, removed or oversampled. The CPU audit verified matching
example multisets and batch sizes at all four boundaries. Arm B averages about
**13.2 different tasks per batch**; tiny batches can still contain one task.
Batch regrouping also changes within-window ordering; it does not isolate an
abstract language-learning mechanism from all other effects of regrouping.

## Budget and declared checkpoints

| Stage | Additional updates from parent | Additional presentations per arm | Additional epoch equivalent |
| --- | ---: | ---: | ---: |
| Screen A and B separately | 15,000 each | 435,945 each | 0.2541 |
| Selected arm only: monitor | 30,000 total | 872,968 | 0.5087 |
| Selected arm only: monitor | 45,000 total | 1,310,004 | 0.7634 |
| Selected arm only: endpoint | 60,000 total | 1,747,065 | 1.0181 |

The endpoint corresponds to cumulative update **132,000**, approximately **2.24
total epochs**, not five. The 30k/45k rows are monitoring checkpoints, not extra
branches or opportunities to select a favorable intermediate result.

- At most **75,000 updates / 2,183,010 presentations across both arms combined**.
- Shared hard budget: **30 active hours for training/validation + 6 for
  diagnostics, rollouts and audits; at most 36 active host-hours total**.
- All pilots, retries and occupied job time count; disconnected/stopped downtime
  does not. Counters persist through recovery. This is a budget, not an ETA.
- New outputs are capped at **100 GiB**, retaining at least 20 GiB free space.
  The existing parent and cached comparison artifacts are excluded and preserved.
- Stop at the earliest applicable cap after a recoverable atomic unit. An
  incomplete screen cannot win against a fully trained arm. An incomplete
  extension cannot nominate an earlier checkpoint as its endpoint.
- Later full clean-2000/dev-210 evaluation and loss/architecture experiments are
  outside this budget and need their own recorded execution plans.

## Fixed measurements

### Closed-loop development panel

The [development-200 panel](../evaluation/splits/week4-oft-continuation-v1/development-200.json)
contains **five states per task, all 40 tasks, 50 episodes per suite**. Select
the five lowest hashes of `week4-oft-continuation-v1|pair_id` per task, then use
numeric state order. Selection occurred before querying cached outcome scores.
Tasks, states, seeds, instructions, timeouts and success definitions remain
identical to the full clean comparison.

All selected states have previously been evaluated. This is disclosed
development evaluation, not an unseen holdout. The small panel can differ from
the full comparison, as the cached counts demonstrate:

| Policy | Spatial / 50 | Object / 50 | Goal / 50 | LIBERO-10 / 50 | Total / 200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent new B0, update 72k | 40 | 49 | 11 | 33 | 133 |
| Previous B0 | 41 | 45 | 29 | 15 | 130 |
| OpenVLA reference | 48 | 50 | 47 | 47 | 192 |

These are existing outcomes, not new measurements. Their sealed receipt
identities are saved in the [baseline ledger](evaluation/week4-oft-continuation-baselines-v1.json).
Reverify raw artifacts before execution. Report all 200 outcomes and paired
gains/losses, including failures; do not rerun policy failures or omit tasks.

### Loss, gripper behavior and gradients

At the parent and every declared checkpoint:

1. Use the existing fixed **2,560 validation examples**. Report continuous L1,
   gripper BCE, hybrid loss, decoded MAE and gripper accuracy separately, overall,
   by suite and by task. Full validation on all 212,471 examples and exact reload
   verification occur at the selected 60k endpoint.
2. Report gripper sign errors separately for open/close targets and near/far
   from source-command transitions. Near means within four frames of a sign
   change in the complete validation episode. Include denominators and horizon
   positions; overlapping chunks are repeated presentations, not independent
   source frames. Never substitute high overall gripper accuracy for correct
   grasp/release timing.
3. Use the [fixed gradient panel](../evaluation/splits/week4-oft-continuation-v1/gradient-640.json):
   40 batches of 16 examples, one existing validation episode per task. Compute
   continuous-L1 and unit-weight-BCE gradients separately, before clipping,
   without an optimizer step. Preserve RNG, model modes and training state.
   Report norms, ratios and cosine alignment by language LoRA, vision LoRA,
   state projector and action head, plus shared-backbone totals. Use FP64 sums
   of squares; explicitly mark zero-denominator ratios as undefined.

A BCE/motion gradient-norm ratio of at least two on at least 30/40 batches for
combined backbone LoRA is a **descriptive imbalance flag**. It does not establish
causation or automatically change this experiment's objective. L1 gripper and
BCE-weight ablations remain separate future experiments.

### Instruction following and failure review

Reuse [90 fixed Goal observations](../evaluation/splits/week4-oft-continuation-v1/prompt-90.json):
ten tasks, state indices 0/25/49, first/middle/last policy queries. Probe all ten
canonical instructions plus the same-goal “Please” paraphrase. Record action
changes and gripper flips. Middle/late inputs come from parent trajectories;
they are off-policy for other checkpoints. Sensitivity is descriptive and never
an acceptance or checkpoint-selection score.

Review all 50 Goal episodes at each declared checkpoint, including successes.
Record visible wrong-target/mechanism, wrong-destination, execution/placement
failure, and unclear cases with timestamps and source videos. Allow multiple
labels and unknowns; distinguish incidental contact from clear manipulation.
Report both all-episode and failure denominators. Saved-state contacts and task
predicates support the review but cannot prove intent. These annotations explain
behavior; the deterministic selection rule uses task outcomes.

## Predeclared decisions

### At 15,000 additional updates

Both arms must have complete, valid evidence at identical exposure. An arm is
eligible to extend only if **all** conditions hold relative to the cached parent:

- Goal gains at least **5/50 successes**: at least **16/50** in total.
- Those net Goal gains occur across at least **three different Goal tasks**.
- Overall success does not decline: at least **133/200**.
- Object loses at most two successes: at least **47/50**.
- LIBERO-10 loses at most two: at least **31/50**.
- Spatial loses at most three: at least **37/50**.

These are prospective engineering screening heuristics, not significance tests.
Among eligible arms select highest overall successes, then Goal, then LIBERO-10;
an exact tie selects A. If neither qualifies, stop this experiment and retain
the negative evidence. Loss and instruction-sensitivity values cannot override
the decision. No additional trials are added to break a tie.

### At 60,000 additional updates

Only the selected arm's fixed endpoint may advance to a larger comparison. It
must still satisfy the screen conditions and achieve at least **29/50 Goal
successes**, matching previous B0 on this development panel. Full-validation,
reload, provenance and artifact checks must pass. Otherwise retain previous B0
and document that this bounded continuation did not meet its target.

Advancing is not a Week 4 PASS. The original [acceptance rule](WEEK4_CLEAN_ACCEPTANCE.md)
remains: paired 95% lower bound at least **−5 percentage points** versus the
pinned OpenVLA reference on all 2,000 pairs, with validity checks, dev-210,
failure analysis and the final baseline/evidence freeze. Historical FAIL results
remain unchanged. Reused development states must remain disclosed.

## Execution and recovery requirements

Before future launch, implement the mixed-episode loader, child-run recovery
identities and shared budget accounting; bind the resulting source commit in
launch manifests. The current parent trainer cannot consume this config directly.
Verify both samplers against the frozen plan hashes, profile bounded-memory
loading, and demonstrate identical next updates after interruption/reload.
Verify that evaluation and gradient probes preserve training state.

Run on Feifei via a host-owned user systemd service with lingering, a single-writer
lock, automatic restart and atomic checksummed checkpoints every five minutes or
2,000 updates, whichever occurs first. Retain three rolling generations plus
immutable declared checkpoints. SSH must never own the job lifetime. Completed
services are disabled after audit. Parent runs remain immutable.

## Freeze verification

- [Machine-readable experiment](../configs/training/week4_b0_oft_continuation_v1.json).
- [Parent recovery and exposure audit](evaluation/week4-oft-continuation-parent-audit-v1.json):
  exact parent-policy tensors; verified original epoch-plan hashes; matched A/B
  sample multisets and batch sizes at every decision boundary.
- [Offline validator](../scripts/validate_continuation_freeze.py) checks 20 pinned
  dependencies, selection, cached counts and exposure identity. It runs no jobs.
- [Freeze verification receipt](evaluation/week4-oft-continuation-freeze-v1.json):
  26 targeted tests passed, lint passed, document links checked, and the original
  clean acceptance freeze validated unchanged.

```bash
.venv/bin/python scripts/validate_continuation_freeze.py
```

Any change to the scientific settings, budget, panel or decision rule requires a
new recorded version before its outcomes. This planning freeze is complete;
runtime readiness and empirical results are separate future milestones.
