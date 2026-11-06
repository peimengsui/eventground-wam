# Week 9 M4 results

M4 completed both formal panels and the paired analysis by **October 14, 2026 at 4:35 AM PDT**. All **3,000 M4 outcome artifacts** passed audit, and its service exited successfully with zero restarts. Together with the preserved B0/M1/M3 results, 12,000 outcomes are complete. M2 and the final five-arm factorial analysis remain pending.

M4 reaches **75.80%** on perturbed-1K and **93.80%** on clean-2000. Compared with M3, its paired differences are **−1.00 pp [−3.60, +1.60]** and **0.00 pp [−2.70, +2.90]**. Both intervals include zero and both whole contrasts remain descriptive because of recorded input mismatches. The key JEPA coupling comparison, **M4 − M2**, is not yet available; these results do not establish a coupling benefit.

![Four completed arms and M4 paired differences](evaluation/week9-m4-v1/m4-comparison.png)

## Success, development overlap and latency

| Metric | B0 | M1 | M3 | M4 |
|---|---:|---:|---:|---:|
| Perturbed-1K | 741/1,000 (74.10%) | 761/1,000 (76.10%) | 768/1,000 (76.80%) | 758/1,000 (75.80%) |
| Clean-2000 | 1,867/2,000 (93.35%) | 1,897/2,000 (94.85%) | 1,876/2,000 (93.80%) | 1,876/2,000 (93.80%) |
| Fresh Week 9 dev-210 subset | 161/210 (76.67%) | 163/210 (77.62%) | 162/210 (77.14%) | 161/210 (76.67%) |
| Additional-790 subset | 580/790 (73.42%) | 598/790 (75.70%) | 606/790 (76.71%) | 597/790 (75.57%) |
| Perturbed median / p95 query latency | 43.28 / 44.67 ms | 43.58 / 44.98 ms | 46.68 / 48.08 ms | 47.37 / 48.82 ms |
| Clean median / p95 query latency | 42.92 / 44.29 ms | 42.41 / 43.72 ms | 45.96 / 47.30 ms | 46.12 / 47.49 ms |

Dev-210 is the overlapping subset of the fresh 1,000-task Week 9 panel. These outcomes do not replace historical standalone dev-210 runs. The additional-790 subset is also descriptive; full-10K and test remain locked. Latency is GPU-synchronized time per complete eight-action query, retaining every query, including the first and any failed query; no warm-up or outlier trimming is used.

## Planned paired comparisons and input qualifications

| Panel | Contrast | Difference | Paired 95% interval | Exact input prefixes |
|---|---|---:|---:|---:|
| Perturbed-1K | M4 − B0 | +1.70 pp | [-0.90, +4.30] pp | 990/1,000 |
| Perturbed-1K | M4 − M3 | -1.00 pp | [-3.60, +1.60] pp | 988/1,000 |
| Clean-2000 | M4 − B0 | +0.45 pp | [-3.05, +4.35] pp | 1,998/2,000 |
| Clean-2000 | M4 − M3 | +0.00 pp | [-2.70, +2.90] pp | 1,999/2,000 |

All four M4 contrasts above remain **whole-panel descriptive**. Perturbed M4/B0 and M4/M3 have 10 and 12 RGB-only mismatches. Clean M4/B0 has two wrist-image mismatches: zero-based ordinals 1105 (`10:wrist_rgb`) and 1200 (`5:wrist_rgb`). Clean M4/M3 has the latter mismatch only. No non-image fields differ, and every affected task remains included. These input-repeatability qualifications are distinct from missing/corrupt-artifact errors; all M4 artifacts passed audit. Existing B0/M1/M3 mismatches, estimates and intervals are unchanged.

Intervals use the frozen common 10,000 PCG64 draws: variant tasks with seed 7 for perturbed, and 40 canonical-task clusters with all 50 states retained per cluster with seed 8 for clean. No new contrast or outcome selection has been introduced.

## Perturbation categories

All category success rates below are descriptive; the full analysis retains paired subgroup intervals, suite, difficulty and suite-by-category tables. These are multiple development comparisons, not confirmatory tests.

| Category | Tasks | B0 | M1 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Background Textures | 108 | 90.74% | 88.89% | 87.04% | 90.74% |
| Camera Viewpoints | 160 | 91.25% | 92.50% | 90.62% | 91.25% |
| Language Instructions | 152 | 57.89% | 61.84% | 66.45% | 65.79% |
| Light Conditions | 113 | 89.38% | 89.38% | 88.50% | 95.58% |
| Objects Layout | 152 | 70.39% | 75.66% | 77.63% | 75.00% |
| Robot Initial States | 155 | 36.77% | 46.45% | 42.58% | 32.90% |
| Sensor Noise | 160 | 90.00% | 84.38% | 90.00% | 88.12% |

## Robustness ratios and measured cost

| Arm | Perturbed / clean success-rate ratio | Descriptive 95% interval |
|---|---:|---:|
| B0 | 0.7938 | [0.7520, 0.8443] |
| M1 | 0.8023 | [0.7671, 0.8433] |
| M3 | 0.8188 | [0.7806, 0.8641] |
| M4 | 0.8081 | [0.7695, 0.8527] |

Ratios compare distinct panel distributions. Their intervals pair independent panel bootstrap streams under the frozen rule; no denominator draw was zero.

M4 training used **177.81 active A100 hours**, one training seed, 117,132 updates, 3,412,044 action presentations and 16,259,770 future pairs. Coupled inference contains **2,311,574,855 parameters**; the loaded model marks **98,333,191 parameters trainable**. These counts are recorded separately from hardware time. Matched training exposure across arms does not imply matched wall time or GPU hardware.

| M4 panel | Evaluation wall time | Model load | Peak allocated / reserved GPU |
|---|---:|---:|---:|
| Perturbed-1K | 2.951 h | 3.69 s | 4.724 / 4.859 GiB |
| Clean-2000 | 3.364 h | 3.77 s | 4.724 / 4.859 GiB |

The two formal panel wall times total approximately 6.31 hours; qualification and cross-arm analysis are separate overhead. The original JEPA quality-gate failure and research-use exception remain disclosed. Other limitations include one training seed, development overlap/shared canonical tasks, historical renderer qualifications, and coupling’s additional parameters and compute. M4/M2 and the five-arm factorial effects remain unreported until M2 finishes the identical protocol.

## Evidence and preservation

- [Full sealed analysis](evaluation/week9-m4-v1/analysis.json), digest `748f9465ed8f03a38590da8254790288cdb657fce07a5a115277ebfdbdc82463`.
- [Completion snapshot](evaluation/week9-progress/20261014T1433Z/feifei.json), [compact comparisons](evaluation/week9-progress/20261014T1433Z/analysis-summary.json), and [timing/cost evidence](evaluation/week9-progress/20261014T1433Z/m4-cost-and-backup-progress.json).
- [Verified endpoint transfer](evaluation/week9-progress/20261014T0443Z/transfer-verification.json) and [immutable admission](evaluation/week9-progress/20261014T0443Z/M4-admission.json).
- [Scientific protocol](WEEK9_EVALUATION_PROTOCOL.md), [execution binding](WEEK9_EVALUATION_RUNNER.md), and [preserved initial results](WEEK9_INITIAL_RESULTS.md).
- [PNG figure](evaluation/week9-m4-v1/m4-comparison.png), [SVG figure](evaluation/week9-m4-v1/m4-comparison.svg), and [rendering script](evaluation/week9-m4-v1/render_comparison.py).

M4’s independent raw backup completed successfully: **12,602 files / 98.76 GB**, with all copied file checksums verified. It is stored at `/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/M4-v1`, sealed package digest `b9d13990fe2cf044c3125624c6ca5b0250569f232472400f7b1a50e6937af194`. It retains M4 outcomes, timings, attempts and receipts, qualification, endpoint exports and frozen source, plus the four-arm analysis and comparison metadata. The original B0/M1/M3 backup remains unchanged. All **584 downloaded metadata files** passed local checksum verification, along with all eight panel-report links, the cohort, source and completed endpoint chain. Raw rollouts and endpoint binaries remain on Feifei. Pinned Qwen weights, benchmark assets and the installed runtime remain external dependencies.

[Backup completion receipt](evaluation/week9-progress/20261014T1433Z/backup-completion.json) · [Local metadata verification](evaluation/week9-progress/20261014T1433Z/local-package-verification.json).
