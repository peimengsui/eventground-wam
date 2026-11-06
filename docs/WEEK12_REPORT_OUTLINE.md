# Week 12 technical report outline

**November 1, 2026: outline frozen for Step 1.** This is the structure for the
final report; drafting and figure production are later steps. The
[scope and research claims](WEEK12_SCOPE.md) govern its conclusions and evidence
boundaries.

Working title: **EventGround-WAM: Future Representations and Event-Adaptive
Control on LIBERO-Plus**.

Target length is **7.5 pages including references**, within the planned 6–8
pages. Detailed protocols, full tables and artifact inventories belong in the
linked supplement. Page allocations are editorial budgets, not rendered counts.

## Main report structure

| Section | Pages | Purpose and required content | Planned exhibit |
| --- | ---: | --- | --- |
| 1. Abstract and research questions | 0.75 | State RQ1–RQ4, the tested implementation scope, the main negative/qualified findings, and the engineering contribution | None |
| 2. Models and experimental controls | 1.25 | Define B0, pixel/JEPA auxiliary M1/M2, coupled M3/M4, and the separate event predictor/routing study; summarize data, matched action exposure, action generation and predictor acceptance history | Figure 1: architecture and event-routing diagram; Table 1: model/panel/evidence map |
| 3. Pixel versus latent targets and coupling | 1.25 | Answer RQ1/RQ2 using all five Week 9 arms, planned target/coupling/interaction effects, paired uncertainty, clean/perturbed results and input qualifications | Figure 2: success rates and paired effects; Table 2: inference and training cost with measurement definitions |
| 4. Event forecasting and post-training | 0.75 | Answer RQ4 using the revised future-onset task, matched SFT/RLVR comparison, development selection and report result; separate automatic-label agreement from on-policy accuracy | Table 3: compact dev/report comparison with uncertainty and evidence labels |
| 5. Adaptive deployment and targeted diagnosis | 1.50 | Answer RQ3 using all three formal Week 11 arms, failed gates and fixed-control checks; summarize closing/first-chunk interventions, matched continuation, adapter stopping decision and fresh H8 result as exploratory follow-ups | Figure 3: formal paired effects and latency; Figure 4: selected gain/loss examples with a compact follow-up disposition table |
| 6. Discussion, limitations and future work | 1.00 | Explain what the controls support, why case-specific motion changes do not establish a general mechanism, and what independent labels, preservation, on-policy validation and fresh seeds/panels could resolve | None |
| 7. Reproducibility and conclusion | 0.50 | Explain how to locate code, configs, checkpoints, outputs and environment pins; distinguish regeneration of analysis from GPU reruns; answer the research questions within their tested scope | Artifact index link |
| References | 0.50 | Cite the benchmark, backbone, methods and external claims using verified primary references during report drafting | None |
| Total | 7.50 | Includes the figures and tables above | Detailed material moves to the supplement if needed |

## Evidence assignments

Sections 1 and 7 use the four conclusions in the scope document. Sections 2–6
use the following sources; each numerical exhibit must later identify its
specific analysis file and panel, not just a weekly narrative.

| Report sections | Primary sources | Required distinction |
| --- | --- | --- |
| 2 | [B0 closeout](WEEK4_CLOSEOUT.md), [shared action protocol](PHASE2_TRAINING_PROTOCOL.md), [pixel validation](WEEK5_PIXEL_VALIDATION.md), [JEPA closeout](WEEK6_CLOSEOUT.md) | Matched training exposure does not imply equal computation, prediction quality or tuning effort; JEPA's failed gate remains visible |
| 3 | [Week 9 final results](WEEK9_FINAL_RESULTS.md), [evaluation protocol](WEEK9_EVALUATION_PROTOCOL.md) | Frozen planned comparisons with descriptive/input qualifications; perturbed and clean panels remain separate |
| 4 | [Week 10 closeout](WEEK10_CLOSEOUT.md), [SFT report comparison](evaluation/week10-sft-continuation-report-v1/analysis/RESULTS.md), [diagnostic](WEEK10_DIAGNOSTIC.md) | Revised and original onset tasks have different targets; dev selection precedes report scoring; report exposure and weak labels remain disclosed |
| 5 formal evaluation | [Week 11 experiment](WEEK11_EXPERIMENT.md), [formal results](evaluation/week11-closeout-v1/RESULTS.md) | Original fixed, event-running fixed8 and adaptive are the contemporaneous controls; “formal” does not mean untouched test evidence |
| 5 exploratory follow-ups | [Final Week 11 closeout](WEEK11_CLOSEOUT.md), [continuation screen](evaluation/week11-multihorizon-results-v1/RESULTS.md), [adapter screen](evaluation/week11-motion-screen-v1/RESULTS.md), [fresh H8 validation](evaluation/week11-h8-validation-v1/RESULTS.md) | Selected diagnostic panels, partial versus complete runs, and the later panel's repeated base tasks remain explicit |
| 6–7 | [Final handoff](evaluation/week11-final-closeout-v1/handoff.json), [evidence ledger](evaluation/week11-final-closeout-v1/evidence-ledger.json), [retained checkpoint manifest](evaluation/week11-final-closeout-v1/retained-checkpoints.json) | M4 is the retained Week 11 reference; continuation/adapter endpoints are research artifacts; further experiments remain deferred |

## Figure and example rules

Show uncertainty and the relevant denominator alongside every comparison.
Preserve each source's contrast direction and bootstrap unit. Use separate
panels for different cohorts and timing protocols; a common visual style does
not make their absolute values directly comparable. Include both gains and
regressions, and label exploratory screens and incomplete runs where they
appear. The adapter screen remains 27/81 episodes.

Step 3 will reuse the formal closeout's manifest-based paired-outcome example
selection where suitable and clearly label mechanism cases selected for
diagnosis. The [first-chunk interventions](evaluation/week11-first-chunk-v1/RESULTS.md)
and [phase/channel tests](evaluation/week11-phase-split-v1/RESULTS.md) can support
narrow causal statements for matched cases. A failure taxonomy should use
observed behavior and audited physical evidence. “Hallucinated future” or
“language ignored” are not established causes simply because a rollout fails.
No new example review or failure labeling is completed by this outline.

## Supplement and presentation derivatives

The supplement should hold complete contrasts and subgroup tables; panel reuse
and input audits; weak-label/task revisions; frozen decision gates and stopped
runs; detailed case IDs and selection rules; latency/cost definitions; and
reproduction commands with dependency and checkpoint manifests. The full
chronology remains in the weekly ledgers instead of consuming the main report.

The demo, blog, project summary and résumé bullets should derive from the final
report after its numbers are reconciled. They should explain the implemented
system and the controlled evidence, including the failed improvements. They
must not turn a selected recovery clip into an overall performance claim.

## Drafting handoff

Step 2 supplies verified tables and their source references. Step 3 supplies
the figures and example captions. Step 4 supplies the reproduction index.
Step 5 assembles those materials into this report, and Step 6 adapts them for
presentation and checks consistency. The report outline and scope are complete;
none of those later deliverables is marked complete by this freeze.
