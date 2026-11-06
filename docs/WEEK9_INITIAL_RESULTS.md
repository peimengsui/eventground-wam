# Week 9 initial results: B0, M1 and M3

All six planned panels are complete: **9,000 recorded outcomes**, with zero artifact-audit errors. This is an interim three-arm report; M2 and M4 are still training, and the final five-arm factorial analysis remains pending.

M3 improves the perturbed point estimate over M1 by 0.70 percentage points, but is 1.05 points lower on clean tasks. Both paired intervals include zero. These results do not establish a coupling benefit. M3 also has higher recorded query latency.

![Success rates and paired differences](evaluation/week9-initial-v1/initial-comparison.png)

## Success and latency

| Metric | B0: action-only | M1: pixel auxiliary | M3: pixel coupled |
|---|---:|---:|---:|
| Perturbed-1K successes | 741/1,000 | 761/1,000 | 768/1,000 |
| Perturbed success rate | 74.10% | 76.10% | 76.80% |
| Clean-2000 successes | 1,867/2,000 | 1,897/2,000 | 1,876/2,000 |
| Clean success rate | 93.35% | 94.85% | 93.80% |
| Perturbed median / p95 latency | 43.28 / 44.67 ms | 43.58 / 44.98 ms | 46.68 / 48.08 ms |
| Clean median / p95 latency | 42.92 / 44.29 ms | 42.41 / 43.72 ms | 45.96 / 47.30 ms |

Latency is per complete eight-action policy query, with GPU synchronization at both boundaries. Every recorded query is included, including the first query; no warm-up or outlier trimming is applied. The full analysis also separates first-query, subsequent-query and episode-median timings.

## Paired comparisons and input qualifications

| Panel | Contrast | Difference | Paired 95% interval | Exact input prefixes |
|---|---|---:|---:|---:|
| Perturbed-1K | M1 − B0 | +2.00 pp | [−0.70, +4.60] | 987/1,000 |
| Perturbed-1K | M3 − B0 | +2.70 pp | [+0.10, +5.30] | 989/1,000 |
| Perturbed-1K | M3 − M1 | +0.70 pp | [−1.90, +3.30] | 988/1,000 |
| Clean-2000 | M1 − B0 | +1.50 pp | [−1.90, +5.45] | 2,000/2,000 |
| Clean-2000 | M3 − B0 | +0.45 pp | [−1.45, +2.35] | 1,999/2,000 |
| Clean-2000 | M3 − M1 | −1.05 pp | [−3.85, +1.40] | 1,999/2,000 |

Whole contrasts with any input mismatch remain **descriptive** under the frozen protocol. On the perturbed panel, the differences are RGB-only: 13 M1/B0 tasks, 11 M3/B0 tasks and 12 M3/M1 tasks, including 3, 2 and 3 tasks with first-query image differences. On the clean panel, M3 differs from both B0 and M1 on one wrist-camera image at the first query: zero-based ordinal **1105**, field `10:wrist_rgb`. B0 and M1 match exactly on every clean input prefix. These are input-repeatability qualifications, not missing/corrupt-artifact audit errors. Every affected task remains in its panel.

Intervals use the frozen 10,000 PCG64 bootstrap draws with common indices across arms. Perturbed intervals resample the 1,000 variant tasks (seed 7). Clean intervals resample the **40 canonical tasks**, keeping all 50 states within each task (seed 8); 2,000 trials are not treated as 2,000 independent task clusters. Intervals are descriptive and carry no confirmatory multiple-comparison claim. In particular, the positive M3/B0 perturbed interval does not remove the input qualification.

## Development overlap

The perturbed-1K panel contains the existing dev-210 tasks plus 790 additional tasks. These rows use fresh Week 9 rollouts of those tasks; they do not replace the earlier standalone dev-210 results.

| Fresh Week 9 subset | B0 | M1 | M3 |
|---|---:|---:|---:|
| Overlapping dev-210 tasks | 161/210 (76.67%) | 163/210 (77.62%) | 162/210 (77.14%) |
| Additional 790 tasks | 580/790 (73.42%) | 598/790 (75.70%) | 606/790 (76.71%) |

The additional-790 figures are a disclosed descriptive subset. This panel is not an untouched held-out test. Suite, perturbation-category, difficulty, suite-by-category and termination-reason tables are retained in the full analysis. Full-10K and test remain locked.

## Protocol and evidence

All arms use the fixed epoch-2 endpoint and the same Feifei runtime, tasks, initial states, rollout rules and synchronized timing boundary. No outcome was selected or dropped. The initial service exited successfully, with zero restarts; all 9,000 completed outcome artifacts passed audit. Fixed endpoints and all input mismatch qualifications are retained.

Limitations remain: one training seed; development overlap and shared canonical tasks; the original JEPA quality-gate failure and research-use exception; historical renderer qualifications; coupling's additional parameters and compute; and differing training hardware with matched exposure rather than matched wall time. M2/M4 cells and final factorial effects are not imputed.

- [Full sealed analysis](evaluation/week9-initial-v1/analysis.json), digest `44240c3c46a95dacd1f6da950f35fd3455e40b12761814a57464c8f9dcb26d68`.
- [Completion snapshot](evaluation/week9-progress/20261010T2320Z/feifei.json) and [compact comparisons](evaluation/week9-progress/20261010T2320Z/analysis-summary.json).
- [Scientific protocol](WEEK9_EVALUATION_PROTOCOL.md) and [execution binding](WEEK9_EVALUATION_RUNNER.md).
- [PNG figure](evaluation/week9-initial-v1/initial-comparison.png), [SVG figure](evaluation/week9-initial-v1/initial-comparison.svg), and [rendering script](evaluation/week9-initial-v1/render_comparison.py).

The initial queue's last analysis was written **October 10, 2026 at 1:17 PM PDT**. The independent raw-results backup is now complete: **36,433 files, 292,906,431,405 bytes (292.9 GB)**, with each copied file checksum-verified. The bounded backup service exited successfully. Its directory is `/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/initial-B0-M1-M3-v1`, with sealed package digest `8336325c082041ab5a5b4767755bdcbe32ab44780e80722636ed7a879771cc3c`.

The backup includes all six panels' rollout HDF5, attempt records, timing sidecars and receipts, qualification evidence, completed endpoint bundles, immutable source and analysis. The original cohort stays in place. Pinned Qwen base weights, benchmark assets and the installed runtime remain external dependencies. A separate metadata archive was downloaded to `outputs/week9-initial-v1-metadata.tar.gz`; all 382 extracted metadata files were verified, together with the six panel, source, cohort and analysis links. Raw rollouts and endpoint binaries remain on Feifei.

[Backup completion receipt](evaluation/week9-progress/20261010T2320Z/backup-completion.json) · [Local package verification](evaluation/week9-progress/20261010T2320Z/local-package-verification.json).
