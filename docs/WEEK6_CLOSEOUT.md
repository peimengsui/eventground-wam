# Week 6 closeout — closed; original JEPA approved for the controlled study

**Project progression revision, October 2, 2026:** the user requested moving on
without repairing the short-horizon weakness if the main comparisons remain
interpretable. The original, fully trained JEPA endpoint is now approved for
**downstream research use with a known limitation**, under the study identifier
`C_jepa-study-v1`. Both M2 and M4 must initialize their future modules from that
same byte-identical candidate. The original frozen gate remains **FAIL**; this
is an explicit project-progression exception, not a passed prediction gate.
The [progression decision](evaluation/week6-progression-v1/decision.json) and
[future-module registry](../configs/training/phase2_future_modules_v1.json)
govern the current handoff. Further residual, shrinkage and gating experiments
are deferred. **Week 6 has no remaining work; Week 7 integration is next.**

## Why progression is appropriate for this project

The primary research question concerns downstream policy success and robustness,
including whether inference-time coupling helps or harms. The original JEPA
module has substantial aggregate, motion and retrieval gains and passes
non-collapse checks. A front/h2 feature-error deficit does not establish that
its auxiliary training or future tokens are useless to a policy. Its effect on
M2/M4 remains an empirical question for the planned action-policy experiments.

The comparison remains interpretable as a study of **these fixed implementations
at matched training exposure**. It does not require equal prediction quality
across objectives. M1/M3 retain the accepted Pixel endpoint; M2/M4 share the
original absolute JEPA endpoint with the same CPT pair order, one pass, 221,633
updates and predictor core. All action policies retain the frozen two-epoch
controls. Decoder/head overhead, method losses, tuning effort, wall time and
inference cost remain disclosed differences. Pixel L1 and JEPA cosine errors are
not directly comparable; downstream paired policy outcomes are the main endpoint.

The weakness could disadvantage JEPA, especially M4's inference coupling, so a
negative result must be scoped to this recipe and cannot establish that JEPA in
general is inferior. M2 versus M4 remains especially useful because both start
from the same module. EventGround may later address known horizon weaknesses;
that development history must be disclosed and improvement assessed on
prospectively frozen confirmation panels. No policy benefit is asserted now.

The residual pilot is excluded from the main matrix: it used a different small
training budget, head initialization and an additional current-teacher input.
Its 9.05% low-change regression belongs to that pilot. The original checkpoint's
measured full-validation front/h2 regression is 7.22%. Neither result is erased.

## Original scientific acceptance decision — unchanged

**Closed September 28, 2026. Final decision: do not promote
`C_jepa-candidate.pt` to `C_jepa-v1`.** All planned training, validation, audits,
qualitative review and comparison work is complete. The frozen acceptance gate
remains **FAIL** because front-camera horizon-2 cosine error is **7.22% worse
than copying current features**, beyond the allowed **2%** regression.

This closes the Week 6 experiment and records its negative acceptance result.
The candidate is retained for reproducibility and diagnosis, and is now also
approved for the research use described above. There is no gate-passing JEPA
checkpoint from this run. The [decision record](evaluation/week6-final-v1/decision.json)
and [completion ledger](evaluation/week6-completion-v1.json) remain authoritative
for the original acceptance outcome; the separately recorded progression
revision changes only the downstream prerequisite. Historical reports and
failed attempts remain unchanged.

## Evidence behind the decision

| Item | Final result |
| --- | --- |
| Production training | Fresh initialization; one pass over 8,221,365 unique train pairs; 221,633 updates; 20.70 active hours; zero restarts |
| Validation coverage | 1,109,994 dual-camera pairs; 1,528 episodes; 180 groups; 40 tasks |
| Overall cosine error | Copy 0.247452 → JEPA 0.176657; 28.61% improvement |
| Medium/long-horizon motion error | 44.98% improvement over copy |
| Retrieval | 3,240 fixed queries; Recall@1 20.87% → 64.09%; MRR 0.461633 → 0.789365 |
| Representation collapse | Both cameras pass the frozen checks |
| Camera/horizon guardrail | Nine cells pass; front h=2 fails: 0.141754 → 0.151994, regression 7.22% [95% interval 6.22–8.25%] |
| Visual review | All 480 fixed examples plus 120 supplemental front h=2 examples inspected |
| Verification | Independent training, validation and appearance audits pass; 480 local tests pass |

The front-h2 deficit appears in 32/40 task means; 29/40 exceed 2%. Visual review
repeatedly finds copy advantages in nearly stationary front views, alongside
JEPA gains during larger wrist-view changes. This supports the measured weakness
but does not identify its causal training mechanism.

The appearance diagnostic finds lower JEPA error and drift in the shared teacher
output space, but lower Pixel native-token drift in all four perturbation point
estimates. The contrast difference is uncertain. Pixel's RGB decoder/resize
bridge and JEPA's direct teacher-space objective limit the comparison. Transform
strengths were fixed after main validation, so these results cannot replace the
prospective gate or establish general policy robustness.

The [detailed review](WEEK6_JEPA_CLOSEOUT_REVIEW.md) preserves all methods,
intervals and limitations. No threshold was relaxed, checkpoint selected after
scoring, extra optimizer update performed or test payload read during closeout.

## Results package

Download the [Week 6 results archive](../outputs/week6-final-v1/week6-results-v1.zip).
The [package receipt](evaluation/week6-final-v1/package-receipt.json) records its
SHA-256, inventory and verification. The archive contains:

- This decision and completion ledger, the full compact Week 6 reports and
  historical ledgers, numerical audits, source inventories and document snapshots.
- The [complete training loss curve](../outputs/week6-final-v1/training/loss-curve.png),
  PDF, full training event trace and plotting script.
- The [480-example review gallery](../outputs/week6-jepa-closeout-v1/review.html),
  supplemental front h=2 panels and per-task review notes.
- Frozen evaluation/training configurations, the exact validation source archive,
  diagnostic/audit scripts and relevant tests.
- The **unaccepted** candidate export, byte-for-byte verified against the
  original, with its disposition and SHA-256. It is a future-module export,
  not a standalone action policy or the frozen base/teacher weights.
- A file-by-file manifest and a standard-library verification script.

Raw demonstrations, frozen base-model weights, optimizer recovery state and
multi-gigabyte evaluation arrays remain on Feifei and are indexed by
[artifact locations](evaluation/week6-final-v1/artifact-locations.json). The
package is a portable evidence and candidate bundle; rerunning GPU evaluation
still requires the pinned data, base weights and runtime.

Candidate SHA-256:
`be2f863a2092668c3e5d50e62d0e41415115f7f582f4f23fa04bc74df8b38e2e`.
Original Feifei path:
`/home/peimengsui/data/eventground-wam/libero-plus/training/week6-jepa-cpt-v1/C_jepa-candidate.pt`.
Archive copy:
`/home/peimengsui/data/eventground-wam/analysis/week6-final-v1/week6-results-v1.zip`.

## Handoff

| Downstream work | Disposition |
| --- | --- |
| Shared B0 and accepted Pixel module | Retain the existing accepted artifacts and Phase 2 protocol |
| M1/M3 Pixel branches | Their Pixel checkpoint dependency is satisfied; integration design and pilots remain future work |
| M2/M4 JEPA branches | Original byte-identical endpoint approved as `C_jepa-study-v1` by the October 2 progression exception; same initialization for both branches; original gate remains FAIL |
| Follow-up JEPA experiment | Training-only diagnostic/residual pilot completed and archived; further shrinkage, gating, confirmation and retraining deferred at the user's request |
| Test split and action-policy training | Remain unopened/unlaunched by this closeout |

Week 7 can proceed to the versioned OFT-compatible M1/M2 integration design and
correctness/resource/recovery pilots. This record satisfies the JEPA checkpoint
dependency; it does not claim that the downstream trainer or production recipe
already exists. Future coupling requires an explicit candidate-action route;
demonstration future actions must not become rollout inputs. Further tuning, if
reopened, must disclose prior validation exposure. No additional training or
test-payload access occurred in this progression revision.

The [progression completion ledger](evaluation/week6-progression-v1/completion.json)
preserves the seven preceding document versions and verifies the unchanged
candidate, original decision, completed pilot, Phase 2 control contract and
sealed result packages. The original Week 6 and pilot archives remain intact;
the [current progression addendum](../outputs/week6-progression-v1/week6-progression-v1.zip)
is packaged separately with a [verification receipt](evaluation/week6-progression-v1/package-receipt.json).
