# Week 6 — validation audit and comparison review

**Requested closeout items 1–3 are complete.** The fixed candidate's measurements
are verified, all 480 predefined qualitative examples have been inspected, and
the Pixel/JEPA appearance comparison is complete. **The acceptance gate remains
FAIL:** front-camera horizon-2 error exceeds the frozen copy-baseline regression
limit. `C_jepa-candidate.pt` is not accepted as `C_jepa-v1`. The subsequent
[final decision and package](WEEK6_CLOSEOUT.md) close Week 6 on September 28.

The [completion ledger](evaluation/week6-closeout-review-v1.json) ties these
supplemental records to the immutable validation report. The original report's
pending-review fields describe its state at completion; the
[post-review gate](evaluation/week6-jepa-closeout-v1/reviewed-gate.json) now records
the completed review without rewriting that report or changing thresholds.

## 1. Independent validation audit

Full validation completed September 28, 2026 at **00:35:51 PDT**, covering
**1,109,994 dual-camera pairs, 1,528 episodes, 180 groups, 40 tasks and 3,240
fixed retrieval queries**. The [independent audit](evaluation/week6-jepa-closeout-v1/audit/verification.json)
passes all source/checkpoint seals, episode and array hashes, pair coverage,
saved qualitative identities, retrieval pools and unchanged-weight checks.

The auditor independently reconstructs task/group aggregation, paired
10,000-draw bootstrap intervals, retrieval tie handling and every quantitative
gate. Collapse diagnostics use a separate Torch SVD calculation. It also replays
two complete saved prediction episodes with scalar FP64 arithmetic, runs fresh
start/middle/tail prediction batches covering 186 pairs, re-encodes real images,
and checks two fresh retrieval panels. This is independent verification with
selected fresh GPU replay, not a second full GPU evaluation.

| Full validation metric | Copy | JEPA | Result |
| --- | ---: | ---: | --- |
| Overall cosine error | 0.247452 | 0.176657 | 28.61% lower; 95% interval 28.34–28.88% |
| Motion-weighted error, h=8/16/32 | 0.361010 | 0.198632 | 44.98% lower |
| Retrieval Recall@1 | 20.87% | 64.09% | +43.22 percentage points |
| Retrieval MRR | 0.461633 | 0.789365 | Higher |
| Front-camera h=2 cosine error | 0.141754 | 0.151994 | **7.22% worse; allowed maximum 2%** |

JEPA also beats the sampled train-only mean's overall error of 0.223306. Both
cameras pass the frozen variance, effective-rank and similarity checks. The
other nine camera/horizon cells pass.

The independently recomputed front-h2 regression interval is **6.22–8.25%**.
It occurs in **32/40 task means**, with **29/40 exceeding 2%**; these task counts
are descriptive breakdowns, not new acceptance criteria. Aggregate/bootstrap
error versus the saved report is at most 6.67e-16, retrieval rank error is zero,
collapse error is below 1.25e-14, and the 186 identically shaped fresh predictions
are bitwise exact. The evidence supports a real short-horizon weakness under
this protocol rather than an aggregation or retrieval-ranking error.

## 2. Completed qualitative review

The [visual review receipt](evaluation/week6-jepa-closeout-v1/visual-review.json)
records actual Codex inspection of **all 480 fixed camera examples**: 40 tasks,
one fixed episode per task, horizons 8 and 32, three anchor fractions, both
cameras. It includes exact identities, panel hashes and observations for every
task. The [review gallery](../outputs/week6-jepa-closeout-v1/review.html) displays
all panels and notes.

An additional **120 front-camera h=2 examples** were inspected to investigate the
failed cell. These use the same fixed episodes and anchor fractions. In total,
240 supplemental h=2 camera examples were generated; the 120 wrist examples
are not counted as visually reviewed. Supplemental examples were selected after
the main result and do not replace or enlarge the original acceptance panel.

Repeated observations:

- Larger wrist-view changes and longer-horizon front changes often show lower
  JEPA feature error, with localized residual hotspots.
- Nearly stationary front views repeatedly favor copying. Clear fixed-panel
  examples include `libero_10/task-03`, `libero_10/task-09`,
  `libero_goal/task-02` and `libero_object/task-07`. Supplemental h=2 panels show
  the same pattern. This is consistent with insufficient preservation of
  current features at small changes, but does not establish its training cause.
- Pixel RGB predictions often retain earlier geometry or show ghost contours
  around moving arms, plates, baskets and newly exposed objects. Some small-change
  cases favor Pixel or copy in the shared feature comparison.
- No obvious camera swap or current/target ordering error appears in the
  inspected panels. Blur, haze and unusual lighting already exist in some
  source image pairs.

JEPA has no RGB decoder: its maps show 8×8 token-wise cosine error with a fixed
0–0.50 scale, not generated future images. Low feature error does not prove exact
object placement, interaction state, physical dynamics or downstream policy
success. Visual observations do not substitute for full-validation statistics.

## 3. Pixel/JEPA appearance consistency

The plan named appearance comparison but did not freeze transformation strengths.
The [diagnostic contract](../configs/evaluation/week6_jepa_appearance_v1.json)
was therefore sealed before this diagnostic run **after main validation**. Its
results are descriptive and cannot become prospective acceptance criteria.

Both frozen models process the same 480 fixed camera examples with identity,
brightness ×0.7/×1.3, contrast ×0.7 around 127.5, and Gaussian blur radius 1.5.
The same transform is applied to current and future images and both cameras;
geometry, instructions, state, action intervals and horizons stay fixed.
Supplemental h=2 examples are excluded from appearance summaries.

The common output space is the canonical frozen Qwen teacher. JEPA predicts
these features directly. Pixel's decoded 128×128 RGB is clipped, quantized to
uint8, resized bilinearly to 256×256 and encoded by that teacher. Both are scored
against the transformed native future image's features. This bridge introduces
Pixel decoder, quantization and resize effects, so it is not a pure comparison
of the two predictor cores.

Cosine error and drift are lower-is-better. Values below average both cameras
and give each of the 40 tasks equal weight. Drift compares transformed output
with the same model's clean output.

| Transform | Pixel target error | JEPA target error | Pixel common-feature drift | JEPA common-feature drift |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.320273 | 0.182665 | ≈0 | ≈0 |
| Brightness ×0.7 | 0.317933 | 0.183657 | 0.118329 | 0.012116 |
| Brightness ×1.3 | 0.320929 | 0.188943 | 0.124459 | 0.014191 |
| Contrast ×0.7 | 0.318309 | 0.184167 | 0.126660 | 0.011628 |
| Blur radius 1.5 | 0.281474 | 0.199617 | 0.186807 | 0.054619 |

JEPA has lower error and drift in this shared teacher-output space. However,
the internal 128×256 future-token interface gives a different result. Drift is
computed within each model after per-token unit normalization; independently
trained token axes are never compared directly across models.

| Transform | Pixel native-token drift | JEPA native-token drift |
| --- | ---: | ---: |
| Brightness ×0.7 | 0.019111 | 0.024742 |
| Brightness ×1.3 | 0.017539 | 0.029474 |
| Contrast ×0.7 | 0.022404 | 0.024151 |
| Blur radius 1.5 | 0.006975 | 0.104357 |

Pixel's native-token drift is lower in all four point estimates. Paired task
bootstrap intervals support the differences for brightness and blur; the
contrast difference interval includes zero. This rules out a blanket claim that
JEPA's predictor representation is uniformly more stable.

JEPA's target error also increases from clean for every perturbation. Under blur
it increases by **0.016952**, while Pixel's decreases by **0.038798**. The target
teacher itself drifts by **0.239492** under blur: changing the target representation
can make the comparison easier for one model. Lower output drift alone is not
proof of robustness. These synthetic perturbations are layered over existing
source appearance variations in 40 episodes, not a full appearance-variant or
policy-success benchmark.

The [appearance report](evaluation/week6-jepa-closeout-v1/diagnostics/report.json)
includes camera breakdowns and paired 10,000-draw task-bootstrap intervals.
Its [independent audit](evaluation/week6-jepa-closeout-v1/diagnostics/verification.json)
checks all 7,200 metric rows, 40 receipts, exact panel coverage, independent
summary/bootstrap arithmetic and saved feature fields for the first fixed task.
Maximum summary error is 2.23e-16 and saved-field scalar replay error is below
2.70e-16. Fresh clean JEPA predictions match saved panel predictions within
6.26e-7, below the predeclared 1e-5 tolerance. The diagnostic run took 139 seconds
and used 4.50 GiB peak allocated GPU memory and 5.09 GiB peak host RSS.

## Verification and final decision

All **480 local tests** and lint for the new scripts/tests pass. Independent
validation and appearance audits pass. All weights remain unchanged; these
closeout activities perform zero optimizer updates and read zero test payloads.

Completing review changes only the procedural `qualitative_review` check to
true. The sole failed check is `camera_horizon`. The
[final disposition](evaluation/week6-final-v1/decision.json) rejects checkpoint
promotion and closes Week 6. The candidate and evidence are packaged for
reproducibility; follow-up experiments are deferred to a separately stated scope.
No threshold is relaxed and no additional training run is initiated.

Reproducible scripts are [validation audit](../scripts/verify_jepa_candidate.py),
[comparison and panel rendering](../scripts/review_jepa_candidate.py), and
[appearance audit](../scripts/verify_jepa_appearance.py). Full arrays and original
panels remain on Feifei at
`/home/peimengsui/data/eventground-wam/analysis/week6-jepa-closeout-v1`;
compact reports, receipts and review contact sheets are also retained locally.
