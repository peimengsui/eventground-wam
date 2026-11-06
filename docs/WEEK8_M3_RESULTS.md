# M3 fixed epoch-2 development results

M3 completed its fixed dev-210 evaluation on Feifei on **October 9, 2026 at 6:05 PM PDT**. It scored **162/210 (77.14%)**, versus **164/210 (78.10%)** for M1. The paired M3−M1 difference is **−0.95 percentage points**, with a 10,000-draw paired task bootstrap 95% interval of **[−6.67, +4.76] points** (PCG64 seed 7, linear quantiles).

This is a **descriptive development comparison**. Exact input auditing passed for 207/210 pairs. The three non-exact lighting tasks are ordinals 100, 203 and 204, the same locations as M1's earlier qualification. Differences are in settling RGB frames: ordinal 100, wrist frame 5; ordinal 203, agent frames 0/2/9; ordinal 204, agent frame 0. Scene, state, instruction, seed, settling actions and first-query inputs match. All tasks remain in the analysis; no tolerance was relaxed and no favorable rerun was selected. Both sides have all 210 valid artifacts and zero artifact-audit errors.

![M3 versus M1](evaluation/week8-m3-final-v1/M3-versus-M1-dev210.png)

| Result | M1 pixel auxiliary | M3 pixel coupled |
|---|---:|---:|
| Successes | 164/210 | 162/210 |
| Policy-step-limit failures | 46 | 48 |
| Policy or exhausted infrastructure failures | 0 | 0 |
| Recorded eight-action queries | 5,305 | 5,206 |
| All-query median | 43.33 ms | 46.34 ms |
| All-query p95 | 44.70 ms | 47.79 ms |
| First-query median / p95 | 44.49 / 45.05 ms | 47.52 / 48.27 ms |
| Subsequent-query median / p95 | 43.30 / 44.64 ms | 46.31 / 47.64 ms |
| Median of episode-median latencies | 43.31 ms | 46.24 ms |
| p95 of episode-median latencies | 44.49 ms | 47.39 ms |

M3 succeeds alone on 19 tasks; M1 succeeds alone on 21. Both succeed on 143 and both fail on 27. This panel does not establish an improvement from coupling. M3's pooled median query latency is about 3.01 ms (6.95%) higher. No first query, failed task, warm-up or outlier was removed. These development timings retain the existing M1-compatible evaluator boundary. The fresh Week 9 cohort explicitly synchronizes both GPU timing boundaries for every arm and should be used for the larger common-cohort timing comparison.

| Suite | M1 | M3 |
|---|---:|---:|
| Spatial | 47/52 | 46/52 |
| Object | 43/52 | 40/52 |
| Goal | 39/53 | 40/53 |
| Long horizon | 35/53 | 36/53 |

| Perturbation category | M1 | M3 |
|---|---:|---:|
| Background textures | 25/30 | 27/30 |
| Camera viewpoints | 27/30 | 25/30 |
| Language instructions | 19/30 | 21/30 |
| Light conditions | 29/30 | 27/30 |
| Objects layout | 20/30 | 21/30 |
| Robot initial states | 20/30 | 16/30 |
| Sensor noise | 24/30 | 25/30 |

These small subgroup counts are descriptive. No subgroup determines checkpoint selection, follow-up task membership or the primary estimand.

## Endpoint and cost

M3 completed all 117,132 optimizer updates, 3,412,044 action presentations and 16,259,770 future pairs. Its original training identity is `ecbfd7e5dc394d04c865f359e26440fd4b52d6d35354cb72bed593b1d19ed56d`. The fixed epoch-2 policy SHA-256 is `8d32417789ea8ea41a3f948cb1564bd3dc150b9d00e673be6d7df8f19562e0e2`. Both policy and full training-predictor exports, the completed exposure counts and the endpoint/completion linkage were verified before inference.

Training used **66.86 active hours**. The shared original pixel CPT used 14.49 hours, so production plus original CPT is approximately 81.35 hours, excluding pilots/calibration/recovery qualification. M1's corresponding production training was 66.47 hours. The coupling block adds 1,187,072 parameters and deployed computation; M3−M1 estimates that complete architectural treatment, not future semantics isolated from parameter count. All model selection, batch, training exposure and loss settings remain frozen.

The dev evaluation took **32 min 12.046 s wall time**, with a systemd-reported **10.4G host-memory peak**. The isolated inference qualification allocated 5,071,861,760 GPU bytes at peak; that pilot observation is not a continuous full-evaluation or training GPU maximum. Checksum verification, input analysis and backup copying are additional wall time. Historical qualification attempts are retained separately, including the initial checker label assertion and the backup path-handling fix; neither caused a formal rollout to be replaced.

## Result package

The independent, verified Feifei package is `/home/peimengsui/data/eventground-wam/libero-plus/results/week8-m3-v1`: **1,415 files, 18,575,181,079 bytes**, with a checksum inventory. It contains the complete M3 training directory, checkpoints and recovery artifacts; raw dev rollouts, attempt journals and receipts; M1's comparison rollouts; exact input audit and paired analysis; the immutable evaluator source; qualification evidence; and analysis/packaging helpers. Original absolute paths remain in sealed artifacts; the inventory maps them to copied files. The pinned base model and benchmark assets remain external dependencies.

Package digest: `e1052ab64a0eb41ad931a8ff4866e1ffe6a945fc6f0324e35e9b0439e10d6a43`.
Analysis digest: `8d23fced15549cb153a110de8433fdc4e999af9a4ef6bfc6a57901b60f317985`.

[Full paired analysis and input hashes](evaluation/week8-m3-final-v1/M3-versus-M1.json), [package inventory](evaluation/week8-m3-final-v1/package-manifest.json), [GPU qualification](evaluation/week8-m3-final-v1/qualification.json), [evaluation resource journal](evaluation/week8-m3-final-v1/evaluation-journal.txt), [evaluation time ledger](evaluation/week8-m3-final-v1/active-budget.json).

This single-seed development panel overlaps the Week 9 stratified-1K panel. Week 9 uses fresh B0/M1/M3 recordings and later M2/M4 under the [unchanged protocol](WEEK9_EVALUATION_PROTOCOL.md); the 790 additional tasks are a disclosed secondary slice, not a replacement primary endpoint. The prior B0/M1 rendering qualifications and JEPA quality-gate FAIL/research-use exception remain unchanged.
