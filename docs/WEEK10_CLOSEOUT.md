# Week 10 closeout

**October 24, 2026: the separately approved exploratory weak-label Week 10 study is complete.**
SFT, bounded RLVR pilots, matched supervised continuation, fixed report evaluation,
diagnostics and checkpoint handoff are complete. The original independent-label-QA
gate has not passed; this closure does not upgrade automatic labels to independent
ground truth. Full/main RLVR was not launched after the revised-task no-go.
Week 11 integration and robot rollouts have not started.

The study asked whether M4-derived features can predict the next event and timing,
and whether reward-based post-training adds value beyond SFT. The current evidence
supports carrying the SFT-only continuation as a **provisional event-predictor
candidate**. It does not establish RLVR superiority, SFT superiority, statistical
equivalence, or improved robot behavior.

## Completed evidence

The revised target is the earliest **new** onset in the next 32 frames, including
anchors where another event is ongoing. The classes are contact, grasp,
gripper_close and none. None means no new onset in that window; an event may
already be active. Predictions use only the current front/wrist RGB,
proprioception and instruction.

Fresh SFT used 1,609 prompts from 199 demonstration sources, two passes and
202 updates. Primary and calibrated RLVR each used 500 prompts / 32 updates from
the same completed SFT. The separately approved hybrid and SFT-only control also
started independently from that SFT, with identical 500 prompts, ordering,
32 updates, effective batch 16, seed and learning rates. Candidate-sampling
compute and wall time were not matched.

| Revised dev panel: 396 prompts / 103 sources | Macro positive IoU | None false positives | Development decision |
|---|---:|---:|---|
| Starting SFT | 41.76% | 2/99 | Baseline |
| RLVR primary | 42.31% | 4/99 | No-go |
| RLVR calibration | 42.14% | 4/99 | No-go |
| RLVR + supervised preservation | 42.52% | 4/99 | No-go |
| SFT-only continuation | 44.02% | 2/99 | Passed existing gate |

The hybrid-minus-control source-bootstrap interval is −3.32 to +0.15 percentage
points, including zero. The earliest RLVR pilot on the **different**, original
48-prompt development task passed its engineering gate; that earlier result is
preserved and cannot be compared directly with revised-task absolute scores.

The already selected continuation was then evaluated once on the separate
384-prompt / 80-source report panel. Macro IoU changed **35.20% → 36.35%**, with
none false positives unchanged at **3/96**. The +1.15-point change has a paired
95% source-bootstrap interval of **−1.17 to +3.42 points**. This is a positive
point estimate, not established reliable improvement. No RLVR report arm was
added after viewing that result. Report labels/counts and baseline results had
previously been inspected, so the report remains exploratory.

- [Full report comparison](evaluation/week10-sft-continuation-report-v1/analysis/RESULTS.md)
- [RLVR/SFT retrospective and hypotheses](evaluation/week10-sft-continuation-report-v1/analysis/RLVR_SFT_REVIEW.md)
- [Completed train/dev diagnostic](WEEK10_DIAGNOSTIC.md)
- [Interactive recorded examples](evaluation/week10-closeout-v1/examples.html)

## Checkpoint handoff

Selection remains the pre-report development decision; the diagnostic did not
select a new checkpoint or change any labels, gates, thresholds or predictions.

| Item | Identity |
|---|---|
| Candidate | SFT-only continuation, fixed final update 32 |
| Feifei endpoint | `/home/peimengsui/data/eventground-wam/libero-plus/results/week10-preservation-pilot-v1/runs/control/endpoint.pt` |
| SHA-256 | `6c6ae33adf89c9e89fd821b3d2d024255edb9cb782372a3de0ecf3d450b1eaf5` |
| Completion digest | `14b68c04372fa34d852306143bc8214b3ef6a2f4dc7c05158b5e6b8b4cc35fa9` |
| Parent SFT SHA-256 | `48bb85e5aa993dc3c617134a23f91319144859347dbad727a015c0377f15b6f9` |
| Task | `week10-future-onset-v1` |

The [machine-readable handoff](evaluation/week10-closeout-v1/handoff.json) binds
the M4 parent, SFT parent, training identity, completion, inference header, dev
metrics and existing report evidence. The
[fresh Feifei verification](evaluation/week10-closeout-v1/media-verification.json)
rechecks actual checkpoint bytes, M4 policy/predictor checksums and completion
links. The original stored M4 remains unchanged.

This is an event-head/language-LoRA child checkpoint, **not a replacement M4 robot
policy**. Restore the exact frozen M4 adapter and then the child endpoint using
the existing `build_eventground_from_m4` and `load_endpoint` path; validate the
preservation training completion first. The frozen source and recipe live in
`week10-preservation-pilot-v1/source`. The wrapper does not execute the action
decoder, JEPA future core or coupling module. No integration or rollout launch
command is part of this handoff.

## Interpretation and closure boundary

- Automatic labels and the earlier assisted review do not establish independent
  event precision or missed-event recall. Predicate consistency was rechecked;
  label correctness was not newly certified.
- These are single-training-seed experiments with adaptive dev reuse. Source
  bootstrap intervals quantify evaluation-source sampling, not training variance.
- Historical transformed-source alias exposure inherited from M4 remains
  unresolved. Existing JEPA research-use and simulator-input qualifications carry
  forward with the parent evidence.
- Event forecasting on demonstrations does not establish forecasting accuracy
  under the deployed policy's trajectories or downstream robot improvement.
- The diagnostic examples deliberately show errors and selected successes; they
  are not a representative sample and require no further user labeling to close
  this approved weak-label scope.

All five preceding evidence packages are retained byte-for-byte. The new
`week10-closeout-v1` bundle holds the diagnostic, example provenance, endpoint
handoff and verification records. The completed study's negative results remain
part of the final record. Further RLVR designs, independent label validation,
additional seeds, horizon integration and robot rollouts are separate future
experiments, not unfinished requirements for this closeout.
