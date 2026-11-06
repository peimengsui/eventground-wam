# Week 9 final results

All five arms completed both frozen panels and the final analysis by **October 15, 2026 at 12:46 AM PDT**. All **15,000 outcomes** passed artifact audit. The evaluation services finished successfully with zero restarts. No checkpoint, task, outcome, retry allowance or scientific protocol was changed.

The study does **not establish an overall coupling benefit**. The coupling main effect is **+1.35 pp [−0.45, +3.15]** on perturbed tasks and **−0.15 pp [−1.85, +1.425]** on clean tasks. Both intervals include zero. Latent-versus-pixel and interaction intervals also include zero on both panels. These factorial results remain descriptive because their contributing input audits contain RGB mismatches.

![Five-arm success rates and factorial effects](evaluation/week9-final-v1/final-comparison.png)

## Outcomes and development overlap

| Panel | B0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| Perturbed-1K | 741/1,000 (74.10%) | 761/1,000 (76.10%) | 738/1,000 (73.80%) | 768/1,000 (76.80%) | 758/1,000 (75.80%) |
| Clean-2000 | 1,867/2,000 (93.35%) | 1,897/2,000 (94.85%) | 1,861/2,000 (93.05%) | 1,876/2,000 (93.80%) | 1,876/2,000 (93.80%) |
| Fresh dev-210 subset | 161/210 (76.67%) | 163/210 (77.62%) | 161/210 (76.67%) | 162/210 (77.14%) | 161/210 (76.67%) |
| Additional-790 subset | 580/790 (73.42%) | 598/790 (75.70%) | 577/790 (73.04%) | 606/790 (76.71%) | 597/790 (75.57%) |

The 1,000-task panel includes all 210 development tasks. The additional 790 tasks are a prespecified secondary slice, not a replacement primary panel. Clean uses 40 fixed canonical tasks × 50 initial states. Historical standalone dev runs remain separate. Full-10K and test remain locked.

## Planned comparisons

Intervals use the frozen 10,000 common PCG64 draws: variant tasks with seed 7 for perturbed; canonical-task clusters retaining all 50 states with seed 8 for clean. Intervals describe this single-seed study and are not multiplicity-adjusted confirmatory tests. Every completed outcome is retained.

| Panel | Contrast | Gap (pp) | Paired 95% interval (pp) | Exact prefixes | Input qualification |
|---|---|---|---|---|---|
| perturbed | M1 − B0 | +2.00 | [-0.70, +4.60] | 987/1000 | descriptive: RGB mismatches |
| perturbed | M2 − B0 | -0.30 | [-2.90, +2.20] | 988/1000 | descriptive: RGB mismatches |
| perturbed | M2 − M1 | -2.30 | [-4.80, +0.20] | 989/1000 | descriptive: RGB mismatches |
| perturbed | M3 − B0 | +2.70 | [+0.10, +5.30] | 989/1000 | descriptive: RGB mismatches |
| perturbed | M3 − M1 | +0.70 | [-1.90, +3.30] | 988/1000 | descriptive: RGB mismatches |
| perturbed | M4 − B0 | +1.70 | [-0.90, +4.30] | 990/1000 | descriptive: RGB mismatches |
| perturbed | M4 − M2 | +2.00 | [-0.50, +4.50] | 988/1000 | descriptive: RGB mismatches |
| perturbed | M4 − M3 | -1.00 | [-3.60, +1.60] | 988/1000 | descriptive: RGB mismatches |
| clean | M1 − B0 | +1.50 | [-1.90, +5.45] | 2000/2000 | exact inputs |
| clean | M2 − B0 | -0.30 | [-2.80, +2.20] | 1999/2000 | descriptive: RGB mismatches |
| clean | M2 − M1 | -1.80 | [-5.10, +0.95] | 1999/2000 | descriptive: RGB mismatches |
| clean | M3 − B0 | +0.45 | [-1.45, +2.35] | 1999/2000 | descriptive: RGB mismatches |
| clean | M3 − M1 | -1.05 | [-3.85, +1.40] | 1999/2000 | descriptive: RGB mismatches |
| clean | M4 − B0 | +0.45 | [-3.05, +4.35] | 1998/2000 | descriptive: RGB mismatches |
| clean | M4 − M2 | +0.75 | [-2.30, +3.90] | 1999/2000 | descriptive: RGB mismatches |
| clean | M4 − M3 | +0.00 | [-2.70, +2.90] | 1999/2000 | descriptive: RGB mismatches |

M3 − B0 on perturbed is the one planned pairwise interval that excludes zero, at +2.70 pp [0.10, 5.30]. Its 11 RGB mismatches, shared development tasks, one training seed and multiple comparisons prevent a confirmatory superiority claim. The treatment-specific coupling contrasts M3 − M1 and M4 − M2 both include zero on both panels. Paired win/loss counts and all prespecified subgroup tables are in the [analysis tables](evaluation/week9-final-v1/analysis-tables.md).

## Factorial effects

Target = ((M2 − M1) + (M4 − M3))/2; coupling = ((M3 − M1) + (M4 − M2))/2; interaction = (M4 − M2) − (M3 − M1). B0 is the external action-only reference and is not a factorial cell.

| Panel | Effect | Estimate (pp) | 95% interval (pp) | Qualification |
|---|---|---|---|---|
| perturbed | Latent − pixel | -1.65 | [-3.550, +0.250] | descriptive |
| perturbed | Coupling | +1.35 | [-0.450, +3.150] | descriptive |
| perturbed | Interaction | +1.30 | [-2.200, +4.900] | descriptive |
| clean | Latent − pixel | -0.90 | [-2.700, +0.700] | descriptive |
| clean | Coupling | -0.15 | [-1.850, +1.425] | descriptive |
| clean | Interaction | +1.80 | [-2.700, +6.750] | descriptive |

## Input audit and limitations

All ten panels passed artifact identity/hash/recording audits. This is distinct from input repeatability: every perturbed contrast has RGB mismatches; clean M1/B0 is the only listed contrast with all prefixes exact. No differing non-image fields or missing first-query inputs were found in these comparisons. Every mismatched row remains in the denominator. The whole affected contrast, including each factorial combination, stays descriptive.

Clean mismatches involve zero-based ordinals 1105 (`10:wrist_rgb`) and/or 1200 (`5:wrist_rgb`). Exact ordinals, fields and contrast membership for both panels are preserved in [input-mismatches.json](evaluation/week9-final-v1/input-mismatches.json); source artifact paths and hashes remain in the full analysis and raw backups. Historical B0 rendering qualifications and M1 standalone dev mismatches remain unresolved historical qualifications.

The original JEPA quality-gate FAIL and approved research-use exception remain in force. One training seed, development overlap, shared canonical tasks and additional coupled-model parameters/compute limit interpretation. M3/M4 retain future-core computation at deployment and add a 1,187,072-parameter coupling block; their contrasts estimate that whole treatment, not future semantics isolated from compute. Training losses of unlike pixel and latent objectives do not measure comparative rollout quality.

## Robustness ratios

| Arm | Perturbed / clean rate | 95% interval |
|---|---|---|
| B0 | 0.7938 | [0.7520, 0.8443] |
| M1 | 0.8023 | [0.7671, 0.8433] |
| M2 | 0.7931 | [0.7517, 0.8402] |
| M3 | 0.8188 | [0.7806, 0.8641] |
| M4 | 0.8081 | [0.7695, 0.8527] |

Ratios compare different task distributions; they are not matched perturbation causal effects. Independent panel streams are paired by draw index. No denominator draw was zero.

## Latency and resource cost

Latency is evaluator-side GPU-synchronized time for a complete eight-action query, including deployed preprocessing, backbone and action decoding. All first, subsequent and failed queries are retained, with no warm-up or outlier removal. It is not per-action latency or a robot deadline guarantee.

| Panel | Arm | Queries | Pooled median / p95 (ms) |
|---|---|---|---|
| perturbed | B0 | 26082 | 43.28 / 44.67 |
| perturbed | M1 | 25701 | 43.58 / 44.98 |
| perturbed | M2 | 26372 | 43.49 / 44.80 |
| perturbed | M3 | 24949 | 46.68 / 48.08 |
| perturbed | M4 | 25576 | 47.37 / 48.82 |
| clean | B0 | 42320 | 42.92 / 44.29 |
| clean | M1 | 41068 | 42.41 / 43.72 |
| clean | M2 | 42739 | 42.58 / 43.91 |
| clean | M3 | 41643 | 45.96 / 47.30 |
| clean | M4 | 42171 | 46.12 / 47.49 |

First/subsequent query summaries, equal-episode-weight summaries, episode wall time, load time, GPU peaks and deployed parameters are in the [analysis tables](evaluation/week9-final-v1/analysis-tables.md). All evaluations ran on the same frozen Feifei RTX 5090 runtime.

| Arm | Training GPU | Recorded active hours | Inherited original CPT hours |
|---|---|---|---|
| B0 | RTX 5090 | 45.91 | — |
| M1 | RTX 5090 | 66.47 | 14.49 |
| M2 | A10 24 GB | 244.36 | 20.70 |
| M3 | RTX 5090 | 66.86 | 14.49 |
| M4 | A100 SXM4 40 GB | 177.81 | 20.70 |

B0 ledger includes its historical pilot/training/validation budget; M1-M4 use production active-time ledgers. These are not equal-total-compute measurements.
Original CPT modules were each trained once and reused; do not double count shared CPT when summing study cost. Lambda training plus Feifei CPT sums mix hardware.

Every policy arm uses 117,132 updates and 3,412,044 action presentations. M1–M4 additionally use 16,259,770 valid future pairs. Logical batch size is 32 with fresh AdamW and matched shared optimizer settings. Microbatch implementation differs only under the documented equivalence checks. Hardware and wall time are not matched.

Historical pilot/tuning wall time is not fully consolidated and is not counted as zero. The [cost ledger](evaluation/week9-final-v1/cost-ledger.json) preserves measured final 128-batch probes, checkpoint/memory costs, historical failed timing gates, calibration/overfit/recovery exposure and source records. Continuous production GPU peaks and isolated optimizer time were not uniformly recorded. Original CPT, pilots, production training and evaluator cost are reported separately.

## Preservation and Week 10 handoff

The final package index links three independent raw-result packages: initial B0/M1/M3, M4, and M2. Together they preserve all 15,000 outcomes, partial attempts, receipts, timing records, qualification, endpoints, source and runtime/cohort identities. Previous packages and outcomes were not overwritten. Pinned Qwen base weights, benchmark assets and the installed runtime remain external dependencies.

Week 9 evaluation and analysis are complete. Week 10 can use these results for diagnosis and explicitly frozen ablations, while retaining the rendering and JEPA qualifications. These results do not justify choosing a winning mechanism or starting unapproved training. Any subsequent experiment requires its own documented scope; full-10K and test remain locked.

- [Full sealed analysis](evaluation/week9-final-v1/analysis.json) and [compact summary](evaluation/week9-final-v1/analysis-summary.json).
- [Raw package index and verification](evaluation/week9-final-v1/package-index.json).
- [Scientific protocol](WEEK9_EVALUATION_PROTOCOL.md) and [execution binding](WEEK9_EVALUATION_RUNNER.md).
- Final analysis digest: `8b74d1f7328c5fe56ac60b37d430482ab883a10858574b6d4685d0da4439cc31`.
