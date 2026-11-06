# Week 4 final paired clean comparison

**Completed:** 2026-09-18 22:44:47 PDT (2026-09-19 05:44:47 UTC).  
**Verified:** 2026-09-19.  
**Frozen acceptance decision: FAIL.** Evaluation execution is complete; the
Week 4 clean-performance gate is not satisfied.

Both fixed policies evaluated the same 2,000 task/state pairs: four canonical
LIBERO suites, ten tasks per suite, and 50 distinct initial states per task.
All rollouts were fresh. The separate engineering pilot is excluded.

## Success rates

| Suite | B0 successes | B0 rate | OpenVLA successes | OpenVLA rate | B0 − OpenVLA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 412 / 500 | 82.4% | 490 / 500 | 98.0% | −15.6 pp |
| Object | 455 / 500 | 91.0% | 495 / 500 | 99.0% | −8.0 pp |
| Goal | 310 / 500 | 62.0% | 484 / 500 | 96.8% | −34.8 pp |
| LIBERO-10 | 188 / 500 | 37.6% | 466 / 500 | 93.2% | −55.6 pp |
| **Overall** | **1,365 / 2,000** | **68.25%** | **1,935 / 2,000** | **96.75%** | **−28.5 pp** |

Here pp means percentage points. The paired suite-stratified task-cluster
bootstrap 95% interval for B0 minus OpenVLA is **[−35.95, −21.25] pp**.
The frozen FAIL rule applies because its upper endpoint is below −5 pp.
The analysis uses all 2,000 pairs, 20,000 bootstrap replicates, the fixed seed,
and NumPy 1.26.4; the interval describes task-cluster variation within these
suites, not variation across training seeds or a guarantee on unseen tasks.

The paired outcomes were 1,340 successes for both policies, 25 B0-only successes,
595 OpenVLA-only successes, and 40 failures for both. The largest aggregate
shortfalls are LIBERO-10 and Goal. Task outcomes are retained in the full report;
these scores do not by themselves identify the cause of B0's failures.

## Execution and validity

- The host service exited successfully with status 0 after both phases and the
  final artifact analysis. B0 finished at 11:38 PDT; OpenVLA finished at 22:42 PDT.
- All 4,000 HDF5 artifacts passed the runner's final validation, and all 2,000
  paired initial raw-input checks matched exactly.
- The final audit reverified all sealed receipts and attempt journals, checked
  source/checkpoint/runtime identities against the frozen rule, recomputed all
  paired checks, and reproduced the exact frozen bootstrap statistics.
- B0 had 635 policy timeouts. OpenVLA had 63 timeouts and two invalid-action
  failures. There were zero episode infrastructure retries or missing outcomes.
- Three MuJoCo instability warnings map to LIBERO-10 task 9: B0 states 5 and 18,
  and OpenVLA state 18. All three were retained failures, not successful episodes.
  Their HDF5 hashes, schema and numeric arrays passed targeted revalidation;
  warnings and original artifacts remain preserved, with no rerun or exclusion.
  This does not establish a root cause for the simulator warnings.
- A generated diagnostic log blocked the transition to OpenVLA and exhausted
  five automatic service retries. The subsequent recovery preserved that log
  outside the source checkout and reused completed B0 artifacts. This history
  remains in the [recovery audit](evaluation/week4-clean-comparison-recovery-v1.json).
  No tracked source or frozen scientific input changed during recovery.

This is an **evaluation-matched external reference comparison**. Training data,
model size, pretraining, architecture, native preprocessing and normalization
are not matched. It establishes the declared operational gate outcome, not an
architecture-causal conclusion or a matched-training benchmark reproduction.

## Evidence

- [Complete sealed comparison report](evaluation/week4-clean-comparison-final-v1.json),
  digest `80ba48ef0eeb022d328529cfa360f2ddac212e3ef271b01b867695145763e472`.
- [Final verification audit](evaluation/week4-clean-comparison-verification-v1.json),
  digest `3324707c65062070075b64dd751ab2368a12ceb493293e8cb935f601c85a6f50`.
- [Frozen acceptance rule](WEEK4_CLEAN_ACCEPTANCE.md), unchanged digest
  `0e3404fbe44d16ad5d6742948d8aeaa8a388cfd04ac6527d21454f1dd0e82f03`.
- Run source: `7e40768d74fa37f856292d7a1b91e5fd1403122f`.
- Feifei output:
  `/home/peimengsui/data/eventground-wam/comparisons/week4-clean-comparison-v1`.
  Recorded disk usage is 128 GiB (`du -sh`), including both policies' recordings
  and diagnostics. No new model checkpoints were created by this comparison.

## Remaining Week 4 work

`dev-210`, the seven-category failure analysis, and the final baseline/protocol
freeze remain outstanding. The failed clean-performance gate also requires a
recorded diagnosis and any proposed remedy under the frozen change-control
rules. Do not relax the margin, select a replacement checkpoint on this panel,
or declare Week 4 closed based on completion of evaluation alone.
