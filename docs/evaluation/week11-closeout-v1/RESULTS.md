# Week 11 formal closeout

**Decision: no positive progression recommendation for adaptive v1. Keep M4-fixed as the current baseline.**

All 2,001 formal episodes completed on October 25, 2026, at 16:25 PDT, after approximately 4 hours 34 minutes. The independent closeout revalidated every artifact and the frozen statistics. Targeted performance is lower with an entirely negative paired interval; clean performance is lower and its cluster interval reaches zero. Both original performance gates fail. Latency, resource and artifact/input gates pass. No training, new rollouts, mapping changes or automatic follow-up were performed.

## Outcomes and paired uncertainty

| Variant | Targeted successes | Clean successes | Targeted p95 | Clean p95 |
| --- | ---: | ---: | ---: | ---: |
| M4-fixed | 311/467 (66.60%) | 187/200 (93.50%) | 57.63 ms | 57.44 ms |
| M4-EG-fixed8 | 311/467 (66.60%) | 187/200 (93.50%) | 108.29 ms | 107.46 ms |
| M4-EG-adaptive | 279/467 (59.74%) | 177/200 (88.50%) | 108.62 ms | 107.84 ms |

| Panel | Contrast (left − right) | Difference, pp | Paired 95% CI, pp | Left-only / right-only wins | Both succeed / both fail |
| --- | --- | ---: | --- | ---: | ---: |
| clean-200 | M4-EG-adaptive − M4-EG-fixed8 | -5.00 | [-10.00, +0.00] | 7 / 17 | 170 / 6 |
| clean-200 | M4-EG-adaptive − M4-fixed | -5.00 | [-10.00, +0.00] | 7 / 17 | 170 / 6 |
| clean-200 | M4-EG-fixed8 − M4-fixed | +0.00 | [+0.00, +0.00] | 0 / 0 | 187 / 13 |
| targeted-467 | M4-EG-adaptive − M4-EG-fixed8 | -6.85 | [-10.28, -3.43] | 19 / 51 | 260 / 137 |
| targeted-467 | M4-EG-adaptive − M4-fixed | -6.85 | [-10.28, -3.43] | 19 / 51 | 260 / 137 |
| targeted-467 | M4-EG-fixed8 − M4-fixed | +0.00 | [+0.00, +0.00] | 0 / 0 | 311 / 156 |

![Paired effects and frozen confidence intervals](presentation-v2/figures/paired-effects.png)

Each panel uses 10,000 common-index PCG64 bootstrap draws across all three contrasts, with linear 2.5/97.5 percentiles. Targeted resamples the 467 variant tasks with seed 7. Clean resamples the 40 canonical tasks with seed 8, retaining all five states in every selected task. These are conditional descriptive panel intervals, not estimates of training-seed variability or unseen-task generalization. The exact zero control interval reflects identical observed outcomes; it is not a general equivalence guarantee. The clean interval includes zero.

The adaptive policy rescues 19 targeted cases but loses 51, for a net loss of 32. On clean it rescues 7 and loses 17, for a net loss of 10. Comparison with the event-running fixed8 control produces the same paired results.

## Original decision gates

| Criterion | Observed | Decision |
| --- | --- | --- |
| Targeted gain ≥ +2 pp | −6.85 pp | FAIL |
| Targeted paired lower bound > 0 | −10.28 pp | FAIL |
| Clean difference ≥ −2 pp | −5.00 pp | FAIL |
| Clean paired lower bound ≥ −5 pp | −10.00 pp | FAIL |
| clean-200: p95 ≤ 200 ms and ≤ 2.5× fixed | 107.84 ms; 1.877× | PASS |
| targeted-467: p95 ≤ 200 ms and ≤ 2.5× fixed | 108.62 ms; 1.885× | PASS |
| Complete artifacts and all required pairwise initial inputs | All 2,001 episodes validated; all comparisons exact | PASS |
| Original time, output, disk and memory limits | All within limits | PASS |

The result is an observed performance regression under this frozen integration. The measured latency is acceptable under the stated engineering limits; it does not compensate for the performance failures. No prospective criterion was changed after seeing outcomes.

## Fixed-control episode and trajectory audit

| Check | Targeted | Clean | Total |
| --- | ---: | ---: | ---: |
| Episode pairs checked | 467 | 200 | 667 |
| Fully bitwise-equal non-timing trajectories | 416 | 181 | 597 |
| Episodes with policy-query input differences | 21 | 3 | 24 |
| Episodes with action differences | 21 | 3 | 24 |
| Identical-input queries compared | 12334 | 4166 | 16500 |
| Different actions on identical inputs | 0 | 0 | 0 |
| Outcome disagreements | 0 | 0 | 0 |
| Episode-length disagreements | 6 | 2 | 8 |

All **667 success/failure pairs match**, and all **16,500 identical-input queries produce bitwise-identical decoded action chunks**. The query input comparison includes both recorded uint8 cameras, the float32 proprioception consumed by the action adapter, and verified identical instructions. Every non-timing dataset was compared; unequal-length tails were retained.

Of the 70 trajectories that are not entirely bitwise identical, 46 differ only in camera frames outside policy-query observations. The other 24 have policy-query camera differences and action differences. At each first differing query, simulator state, raw/effective proprioception and all prior executed actions still match. The first camera differences are at most **1 intensity level**, affecting at most **38 channel values**. This is consistent with small rendering variation preceding action divergence; it does not identify the rendering mechanism or prove a causal explanation for the adaptive regression.

Eight of these pairs have different successful episode lengths. Their total query counts differ by +5 on targeted and +4 on clean in event/fixed8. All eight preserve the same binary outcome. Thus the fixed8 control is outcome-equivalent on this panel, while full-trajectory equality must be qualified. There is no observed same-input action discrepancy from running the event branch.

| Panel / zero-based pair | Suite / runtime task | Fixed steps | Event/fixed8 steps | First differing query step | Changed camera channel values |
| --- | --- | ---: | ---: | ---: | ---: |
| targeted-467 / 102 | libero_spatial / 2024 | 87 | 86 | 18 | 3 |
| targeted-467 / 312 | libero_goal / 1094 | 148 | 142 | 98 | 1 |
| targeted-467 / 346 | libero_goal / 2180 | 127 | 129 | 74 | 4 |
| targeted-467 / 401 | libero_10 / 745 | 230 | 229 | 74 | 3 |
| targeted-467 / 406 | libero_10 / 799 | 218 | 226 | 106 | 8 |
| targeted-467 / 409 | libero_10 / 812 | 207 | 240 | 106 | 3 |
| clean-200 / 128 | libero_goal / 5 | 126 | 127 | 74 | 5 |
| clean-200 / 158 | libero_10 / 1 | 377 | 413 | 106 | 6 |

Full field-by-field differences and first-divergence contexts are preserved in [targeted control audit](remote/controls/targeted-467.json) and [clean control audit](remote/controls/clean-200.json).

## Predefined descriptive subsets

These one-way targeted subsets use the prescribed seed-7, 10,000-draw bootstrap within each subset. There is no multiplicity adjustment and no subset replaces the primary endpoint. The complete [machine-readable table](remote/secondary-statistics.json) includes all three contrasts. Since the fixed controls match every binary outcome, both adaptive-versus-control contrasts match in every subset. The contact subset is the frozen task-description proxy, not validated contact-density labels.

| Group | Subset | n | Fixed successes | Adaptive successes | Difference, pp | Paired 95% CI, pp |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| category | Camera Viewpoints | 160 | 147 | 135 | -7.50 | [-13.12, -2.50] |
| category | Objects Layout | 152 | 113 | 111 | -1.32 | [-7.24, +4.61] |
| category | Robot Initial States | 155 | 51 | 33 | -11.61 | [-18.71, -5.16] |
| contact_proxy | contact-proxy98 | 98 | 53 | 50 | -3.06 | [-10.20, +3.06] |
| contact_proxy | other369 | 369 | 258 | 229 | -7.86 | [-11.92, -3.79] |
| dev_overlap | dev-overlap90 | 90 | 61 | 56 | -5.56 | [-14.44, +3.33] |
| dev_overlap | other377 | 377 | 250 | 223 | -7.16 | [-10.88, -3.45] |
| difficulty | 1 | 80 | 63 | 56 | -8.75 | [-17.50, +0.00] |
| difficulty | 2 | 96 | 79 | 68 | -11.46 | [-18.75, -5.21] |
| difficulty | 3 | 101 | 65 | 67 | +1.98 | [-5.94, +9.90] |
| difficulty | 4 | 77 | 44 | 40 | -5.19 | [-12.99, +1.30] |
| difficulty | 5 | 113 | 60 | 48 | -10.62 | [-17.70, -4.42] |
| suite | libero_10 | 111 | 60 | 48 | -10.81 | [-18.92, -3.60] |
| suite | libero_goal | 125 | 84 | 78 | -4.80 | [-11.20, +1.60] |
| suite | libero_object | 120 | 83 | 81 | -1.67 | [-7.50, +4.17] |
| suite | libero_spatial | 111 | 84 | 72 | -10.81 | [-18.92, -2.70] |
| difficulty | Long/unlabeled | 0 | 0 | 0 | — | Not estimable |

All 467 selected targeted tasks have numeric difficulty labels 1–5. The predefined Long/unlabeled stratum is empty; no task was dropped and no empty-sample interval is estimated.

## Timing and horizon evidence

All first, subsequent and failed-query costs are retained; there were no query exceptions or retries. Per-arm reports retain pooled median/p95, first/subsequent timing, equal-episode-weight summaries, component timing, model load, simulator and recording costs. The event-running fixed8 control adds cost while preserving all binary outcomes.

| Adaptive panel | h=2 | h=4 | h=8 | h=16 | h=32 | Valid-none fallbacks to 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| targeted-467 | 880 | 4977 | 1781 | 1381 | 4433 | 970 |
| clean-200 | 259 | 1424 | 765 | 432 | 1643 | 508 |

These are deployment frequencies, not event-prediction accuracy or causal attribution. The previously frozen recorded-dev mapping diagnostic retains its weak-label scope; it was not used to change the mapping. On-policy label accuracy and independent label QA remain unverified.

## Preserved examples, resources and reproducibility

The [selected-case index](remote/selected-case-index.json) takes the first manifest ordinal in each paired outcome class on each panel. [Rendered examples](presentation-v2/presentation.json) show both cameras at the first policy query, midpoint and last recorded pre-action observation for all three arms. The selection does not favor visually persuasive examples and does not assert failure causes.

- [Targeted fixed-only success example](presentation-v2/figures/targeted-467-fixed_only_success.png)
- [Targeted adaptive-only success example](presentation-v2/figures/targeted-467-adaptive_only_success.png)
- [Clean fixed-only success example](presentation-v2/figures/clean-200-fixed_only_success.png)
- [Clean adaptive-only success example](presentation-v2/figures/clean-200-adaptive_only_success.png)

The accepted CPU analysis took 276.34 active seconds. The stopped first attempt's 50.09 seconds remain charged. Qualification, evaluation and analysis total **4.725 active hours** before the separately recorded presentation-rendering cost, well within the 48-hour limit. All panel totals remain below 12 hours. The overall highest GPU reservation is the preserved pilot peak of 23.78 GiB (75.9%); the formal run peaks at 15.19 GiB. Analysis makes zero GPU inference calls. Analysis process-tree RSS peaks at 1.59 GiB; free disk at close is 152.78 GiB, above the 50-GiB reserve.

All **10,048 raw evaluation files**, including **2,001 HDF5 recordings**, remain unchanged on Feifei under the complete [SHA-256 inventory](remote/raw-inventory.json). All **8,047 metadata files** (368,973,656 bytes) are mirrored in [formal-metadata.tar.gz](formal-metadata.tar.gz), and every decompressed file matches the remote inventory. The 76-GB raw recording directory was not downloaded or deleted. The local metadata-derived statistics exactly reproduce the remote analysis.

The accepted analysis source is [package-v3](package-v3/package.json), extending the executed formal archive without replacing an inherited byte. Its 27 focused tests pass both locally and on the pinned Feifei Python/NumPy runtime. Package v1 stopped on nested timing aggregation; package v2 remained local after a test-import issue; both sources and test/attempt records are retained. These issues affected postprocessing only. The original 79-test launch qualification and all prior freezes remain unchanged.

The [analysis method and rerun command](../../WEEK11_CLOSEOUT_METHOD.md), [paired matrices](remote/paired-matrices.json), [sealed analysis](remote/analysis.json), source archives, metadata archive, figures, tests, and before/after status snapshots form the closeout package. This closes the requested formal analysis and fixed-control audit. Adaptive failure-mechanism diagnosis and any new experiment remain separate work.

Analysis digest: `556b5509416c2203c4ba88e34f5b7689d545adac9f2114a953cc3fe315ac042f`.  
Accepted analysis source digest: `ef09bf55dcc11ea22778a55f2b086482f6d012fc29e603d676cf4bb8c222d6c6`.  
Formal source digest: `9330bb56d0c31ebaf4c9d90b953948fb45c923e5a1a8ad2e997e763fc7f73328`.  
Raw inventory digest: `81e2d4853f89a5d2bc1183279d0bffa1d33691a9e63e968441907331acad2ed3`.
