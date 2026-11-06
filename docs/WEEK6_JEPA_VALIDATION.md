# Week 6 — fixed-candidate validation

**Status: complete; acceptance FAIL.** Full validation ran on Feifei from
September 27, 2026, 21:40:10 PDT to **September 28, 00:35:51 PDT** with zero
restarts. The subsequent independent audit, 480-example review and appearance
comparison are complete in the [closeout review](WEEK6_JEPA_CLOSEOUT_REVIEW.md).
The front-camera h=2 gate fails; `C_jepa` remains unaccepted. The
[final decision and package](WEEK6_CLOSEOUT.md) close Week 6 on September 28.

## 1. Completed training audit

The [independent CPU audit](evaluation/week6-jepa-cpt-final-v1/verification.json)
passed all 8,221,365 unique training pairs across 11,435 episodes and all
221,633 optimizer updates. It reconstructed the complete sample chain, checked
optimizer/scheduler/RNG recovery state, verified fresh initialization and unchanged
frozen weights, and compared exported weights with the actual final checkpoint.
Training finished at 20:57:28 PDT after 20.70 active hours with no restarts.
See the [Step 5 completion ledger](evaluation/week6-step5-completion-v1.json).

The fixed candidate SHA-256 is
`be2f863a2092668c3e5d50e62d0e41415115f7f582f4f23fa04bc74df8b38e2e`.
There is no checkpoint search, additional training or threshold tuning.

## 2. Verified evaluator and train-only resource pilot

The [evaluator](../scripts/evaluate_jepa_cpt.py) uses the existing canonical
single-image teacher, current-observation input builder and trained predictor.
It processes complete episodes, checks every pair identity and accumulates the
frozen FP64 cosine/motion metrics. Prediction uses eight anchors per batch,
four current observations per feature microbatch and ten predictions per
predictor microbatch. Models remain frozen throughout evaluation.

The [metric implementation](../src/eventground_wam/jepa_evaluation.py) retains
the existing episode/group/task aggregation, paired 10,000-draw bootstrap,
camera/horizon breakdowns and exact eight-candidate retrieval pools. It checks
duplicate or missing pairs, split/source alignment, finite unit features,
truth-mask equality and retrieval-pool RGB hashes. Target group means must
exactly equal the previously saved baseline panels.

All **477 local tests** pass, including six evaluator tests, and new-code lint
passes. Tests cover independent scalar arithmetic, uneven batches, coverage and
source failures, retrieval ties/camera alignment, strict pilot-to-full identity
matching, and accounting for pilot-only work once in runtime projections.

The first pilot is preserved as [v1](evaluation/week6-jepa-validation-v1/pilot.json).
Its numerical, memory and disk checks passed, but its **6.2443-hour** projection
exceeded the six-hour limit. That estimate multiplied cold-start and pilot-only
retrieval-pool construction/numerical probes across the full validation set.

The separately frozen [v2 execution contract](../configs/evaluation/week6_jepa_candidate_v2.json)
measures those costs separately. It warms fixed shapes on 40 train pairs, charges
measured startup/warmup/probes once, and uses the slowest recurring episode
seconds per pair. It still applies a **1.5× margin**, adds **900 seconds** for
aggregation/audit writes, and requires a projection below **six hours**.
The scoring path and acceptance criteria are unchanged. Both immutable source
archives and pilot outcomes remain available on Feifei.

The [v2 pilot](evaluation/week6-jepa-validation-v2/pilot/pilot.json) passed:

| Check | Result |
| --- | --- |
| Episodes | Shortest representative train (75 frames), longest (505), longest instruction (148) |
| Exposure | 3,454 scored train pairs; 40 additional warmup presentations; zero optimizer updates |
| Scalar FP64 reference maximum error | 2.78e-16 |
| Single versus batch feature maximum error | 3.58e-7, below 1e-5 |
| Identical-shape prediction replay | Bitwise exact |
| Predictor, base and teacher weights | Unchanged |
| Peak GPU allocation / host RSS | 4.30 / 5.27 GiB |
| Projected new artifacts | 5.79 GiB, below 8 GiB |
| Runtime projection including margin | 4.904 hours |
| Validation/test payload reads | Zero |

## 3. Full validation completed

The [persistent service](../configs/systemd/eventground-jepa-validation-v2.service)
evaluated **1,109,994 dual-camera pairs / 1,528 validation episodes**, with
**3,240 fixed retrieval queries / 180 groups / 40 tasks**. It can start only
after a passed pilot matches the exact candidate, source, protocol and runtime.
The service has a 24 GiB host-memory limit, six CPU equivalents and an eight-hour
process backstop. Completed episode receipts and arrays support identity-checked
resumption; completed output cannot be overwritten by restarting the service.

Historical launch verification at **21:42:04 PDT** found **10,950 pairs / 15 episodes**
complete, including two retrieval-panel episodes and the first full replay
episode. Teacher panel equality and truth-mask agreement passed. The service had
zero restarts and an observed scoring projection of **2.85 hours**. This early
estimate excludes final aggregation and can change with episode lengths.
The conservative pilot estimate would place completion around **02:35 PDT on
September 28**; the initial observed pace suggests roughly **00:35–01:00 PDT**.

Outputs on **Feifei**:

- Source: `/home/peimengsui/data/eventground-wam/analysis/week6-jepa-validation-v2/source`
- Pilot: `/home/peimengsui/data/eventground-wam/analysis/week6-jepa-validation-v2/pilot`
- Full run: `/home/peimengsui/data/eventground-wam/analysis/week6-jepa-validation-v2/full`
- Service: `eventground-jepa-validation-v2.service`

The [final report](evaluation/week6-jepa-validation-v2/full/report.json) and
[completion record](evaluation/week6-jepa-validation-v2/full/completion.json)
pass full coverage and frozen-weight checks. Overall cosine error is 0.176657
versus 0.247452 for copy (28.61% lower); retrieval Recall@1 is 64.09% versus
20.87%. Both cameras pass collapse checks. Front-camera h=2 error is 0.151994
versus 0.141754 for copy, **7.22% worse against a 2% cap**. The remaining nine
camera/horizon cells pass. No test payloads were read.

All 480 saved qualitative examples have now been inspected, alongside 120
supplemental front h=2 examples. Independent aggregate, bootstrap, retrieval,
collapse and selected fresh GPU replays pass. The appearance comparison and its
independent audit are complete. See the [review](WEEK6_JEPA_CLOSEOUT_REVIEW.md)
for methods, nuanced comparison results and limitations.

The original final report remains immutable, including its historical
pending-review fields. The supplemental
[post-review gate](evaluation/week6-jepa-closeout-v1/reviewed-gate.json) marks
review complete and retains **FAIL** solely for `camera_horizon`. Earlier launch
documents are preserved in
[snapshots](evaluation/week6-closeout-review-v1-snapshots/WEEK6_JEPA_VALIDATION.md).

## Final disposition

Week 6 is closed with the candidate rejected for promotion to `C_jepa-v1`.
The [final decision](evaluation/week6-final-v1/decision.json) and
[results package](WEEK6_CLOSEOUT.md) retain the failed gate and all evidence.
No model or threshold changed and no additional training started. Any follow-up
experiment has separate scope and must disclose prior validation exposure.
