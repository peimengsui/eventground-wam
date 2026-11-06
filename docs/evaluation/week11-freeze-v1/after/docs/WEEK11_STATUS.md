# Week 11 status

**October 25, 2026: Step 1, experiment design, is complete.**
The [frozen protocol](WEEK11_EXPERIMENT.md) pins the original M4 and selected
SFT-only event endpoint, three comparison arms, exact panels, resource caps and
prospective decision criteria. Local provenance and panel reconstruction are
checked by `scripts/verify_week11_protocol.py`; the verification record is in
`docs/evaluation/week11-freeze-v1/verification.json`.

The planned formal cohort contains 467 targeted perturbation tasks and 200 clean
episodes per arm, 2,001 episodes total. Eight separate qualification episodes per
arm are engineering checks. Original M4, an event-running fixed-horizon control,
and adaptive M4-EG isolate the event branch from the action-side LoRA.

| Step | State |
| --- | --- |
| 1. Experiment design and evidence pins | Complete; local metadata verified |
| 2. Exact horizon mapping and fallback contract | Pending |
| 3. Integration implementation | Pending |
| 4. Numerical/resource qualification and bounded pilot | Pending |
| 5. Targeted paired evaluation | Pending |
| 6. Analysis and reproducible handoff | Pending |

Actual remote checkpoint bytes were last verified in the October 24 Week 10
closeout. No fresh remote check, GPU inference, training or robot rollout was
performed for Step 1. Mapping and runtime qualification are prerequisites to
execution. The user's current request covers Step 1 only.

Week 10 remains closed in its approved exploratory weak-label scope. Independent
label QA remains unmet, and the revised-task RLVR no-go remains unchanged.
