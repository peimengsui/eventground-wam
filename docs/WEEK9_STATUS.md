# Week 9 evaluation and analysis

**October 25, 2026: the redundant B0 clean working copy was removed with user approval.**
All 8,005 files were freshly verified against their independent packaged copies
before removal, recovering 57.734 GiB. The complete results package remains
unchanged and preserves every outcome, recording, timing log and receipt.
Historical tools using `evaluation/week9-factorial-v1-cohort02/panels/B0/clean`
need restoration from the retained package or explicit path mapping. Other
working panels remain present. [Cleanup evidence and retained location](evaluation/week11-storage-readiness-v2/RESULTS.md).

**October 17, 2026: both Lambda training instances and filesystems are ready for retirement.**
M2 and M4 training services exited successfully with zero restarts. All 503
completed-endpoint/provenance files were reverified in both Feifei copies.
With the user's approval, the complete `eventground-wam` and
`eventground-wam-m4` NFS contents were also archived to Feifei: **3,318 files /
13.02 GB**, including training logs, pilot artifacts, epoch-1 weights and recovery
checkpoints. Every archived file and both complete archive hashes passed
verification; the epoch-2 model hashes match the existing admissions. Week 10
uses M4 on Feifei. No instance or Lambda filesystem was deleted by the agent.
[Retirement verification and backup locations](evaluation/lambda-retirement-v1/completion.json).

**October 15, 2026: Week 9 evaluation, final analysis and results packaging are complete.**
All five arms finished both frozen panels and the final analysis by **12:46 AM
PDT**. All **15,000 outcomes** passed artifact audit. Evaluation services exited
successfully with zero restarts. Both Lambda training endpoints remain complete
and unchanged. [Final report and figures](WEEK9_FINAL_RESULTS.md).

| Success rate | B0 | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Perturbed-1K | 74.10% | 76.10% | 73.80% | 76.80% | 75.80% |
| Clean-2000 | 93.35% | 94.85% | 93.05% | 93.80% | 93.80% |

Coupling's main effect is **+1.35 pp [−0.45, +3.15]** on perturbed and
**−0.15 pp [−1.85, +1.425]** on clean. All target, coupling and interaction
intervals include zero. The whole factorial comparisons remain descriptive:
every contributing RGB mismatch is retained, along with historical renderer,
JEPA research-use, single-seed and development-overlap qualifications. This
study does not establish an overall coupling benefit or a winning mechanism.

The three independent raw-result packages retain **61,638 files / 491.75 GB**,
covering all 15,000 outcomes and their timing, attempts, receipts, endpoints and
source. M2's final package contains **12,603 files / 100.08 GB**, with every copy
checksum verified. All 584 downloaded M2-package metadata files and all ten
panel-report links passed local verification. The fixed bootstrap and robustness
ratios also reproduced from the verified reports. Prior initial/M4 packages and
all completed outcomes are unchanged.

The **79-file final report bundle** is verified locally and on Feifei at
`/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/final-five-arm-v1`.
[Package index](evaluation/week9-final-v1/package-index.json) ·
[Numerical verification](evaluation/week9-final-v1/numerical-verification.json) ·
[Remote report-package verification](evaluation/week9-progress/20261015T1438Z/presentation-remote-verification.json) ·
[Cost ledger](evaluation/week9-final-v1/cost-ledger.json).
Historical pilot/tuning wall time was not fully consolidated and is explicitly
reported as unavailable, rather than zero; measured probes and preserved
historical exposure records are included.

The `complete-week-9-evaluations` follow-up has been **stopped (PAUSED)** after
verified completion, as requested. No training, protocol, task membership,
resource limit or retry allowance changed. Full-10K and test remain locked.
Week 10 diagnosis and proposed ablations can use the final package; this
completion does not launch additional experiments.

**October 14, 2026, 10:45 PM PDT: M2 perturbed-1K is complete; clean-2000 is running.**
M2 recorded **738/1,000 successes (73.80%)**, and all 1,000 artifacts passed
audit. The fresh dev-210 subset is **161/210 (76.67%)**; additional-790 is
**577/790 (73.04%)**. Median / p95 query latency is **43.49 / 44.80 ms**, retaining
all 26,372 queries. All five arms now have completed perturbed panels:

| Perturbed-1K | B0 | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Successes | 741 | 761 | **738** | 768 | 758 |
| Success rate | 74.10% | 76.10% | **73.80%** | 76.80% | 75.80% |

M2 − B0 is **−0.30 pp [−2.90, +2.20]**; M2 − M1 is **−2.30 pp
[−4.80, +0.20]**; M4 − M2 is **+2.00 pp [−0.50, +4.50]**. All intervals
include zero. These whole-panel contrasts remain **descriptive**, with 12,
11 and 12 RGB-only input mismatches, respectively. Every affected task remains
included. Prior four-arm estimates and comparison objects are unchanged.

M2 clean is at **965/2,000**, with its service active, zero restarts and no
recorded launch errors. Both Lambda training services remain successfully
completed and unchanged. The earlier **midnight–2 AM PDT, October 15** evaluation
finish estimate remains reasonable. Final two-panel analysis and the complete
M2 raw backup remain pending; the existing queue continues without intervention.

The nine completed panels and interim analysis have an independently verified
**50-file / 121.64 MB metadata snapshot** on Feifei and locally. Raw artifacts
remain in the cohort; this interim snapshot is not the final raw backup.
[Verified check](evaluation/week9-progress/20261015T0545Z/check.json) ·
[Interim analysis](evaluation/week9-progress/20261015T0545Z/analysis.json) ·
[Compact comparisons and mismatch records](evaluation/week9-progress/20261015T0545Z/analysis-summary.json) ·
[Local metadata verification](evaluation/week9-progress/20261015T0545Z/local-metadata-verification.json).

**October 14, 2026, 6:10 PM PDT: M2 training is complete and its Week 9 evaluation is running.**
M2 finished the fixed epoch-2 endpoint at **6:05 PM PDT**, with **117,132
updates**, **3,412,044 action presentations** and **16,259,770 future pairs**.
Its training service exited successfully with zero restarts. The frozen identity,
recipe, completion links, policy and predictor checksums, and all **233 training
source files** passed verification. Training used **244.36 active A10 hours**.

The completed bundle was transferred to Feifei at
`/home/peimengsui/data/eventground-wam/libero-plus/training/week7-m2-lambda-v1`:
**248 files / 313.91 MB**, with every destination checksum verified.
The unchanged runner verified the cohort runtime and created the immutable
[M2 admission](evaluation/week9-progress/20261015T0104Z/M2-admission.json).
[Endpoint verification](evaluation/week9-progress/20261015T0104Z/M2-endpoint-preflight.json) ·
[Transfer receipt](evaluation/week9-progress/20261015T0104Z/transfer-verification.json).

`eventground-week9-m2-v1.service` launched at **6:08 PM PDT**. Both qualification
panels passed: four artifacts, zero audit errors and exact repeated inputs.
The formal perturbed panel has started (**5/1,000** at this check), followed by
clean-2000 and the final five-arm analysis. The fresh dev-210 subset is included
in perturbed-1K. Allow approximately **6–8 hours** for the remaining evaluations
and analysis, based on the completed arms; packaging follows completion.
All eight completed B0/M1/M3/M4 panel reports, previous analysis and independent
backups remain unchanged. No training, protocol, retry or resource limits changed.

[Verified launch and qualification evidence](evaluation/week9-progress/20261015T0104Z/check.json).
Use the updated read-only collector at
`docs/evaluation/week9-progress/20261015T0104Z/collect_check.py`; it also captures
M2 service errors and later-arm backup receipts. The M2 service log is at
`/home/peimengsui/data/eventground-wam/libero-plus/evaluation/week9-m2-launch-v1/service.log`.
Continue the existing follow-up to complete M2, preserve all input qualifications,
package its results separately, and finish the five-arm report. Do not rerun
completed panels or duplicate the initial/M4 backups.

**October 14, 2026: M4 evaluation, analysis and packaging are complete.**
Both panels and their analysis finished by **4:35 AM PDT**. All **3,000 M4
outcome artifacts** passed audit; `eventground-week9-m4-v1.service` is inactive
with success and zero restarts. [M4 report and comparison figure](WEEK9_M4_RESULTS.md).

| Success rate | B0 | M1 | M3 | M4 |
|---|---:|---:|---:|---:|
| Perturbed-1K | 74.10% | 76.10% | 76.80% | **75.80% (758/1,000)** |
| Clean-2000 | 93.35% | 94.85% | 93.80% | **93.80% (1,876/2,000)** |
| Fresh Week 9 dev-210 subset | 76.67% | 77.62% | 77.14% | **76.67% (161/210)** |

M4 − M3 is **−1.00 pp [−3.60, +1.60]** on perturbed and **0.00 pp
[−2.70, +2.90]** on clean. Both intervals include zero. All M4 whole-panel
contrasts remain descriptive: perturbed M4/B0 and M4/M3 have 10 and 12 RGB
mismatches; clean has two and one, respectively. The clean wrist mismatch at
ordinal 1200 (`5:wrist_rgb`) is retained, as is the existing ordinal 1105
qualification. All prior B0/M1/M3 estimates, input audits, reports and backups
are unchanged. The key M4 − M2 coupling comparison is pending.

The separate M4 raw backup completed under `eventground-week9-package-m4-v1.service`:
**12,602 files / 98.76 GB**, each checksum-verified, at
`/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/M4-v1`.
Package digest: `b9d13990fe2cf044c3125624c6ca5b0250569f232472400f7b1a50e6937af194`.
All **584 downloaded metadata files** and the eight panel-report links passed
local verification. The 15-file report/figure companion is independently verified
under `M4-v1-presentation` at the same results root.
[Completion and backup evidence](evaluation/week9-progress/20261014T1433Z/check.json) ·
[Presentation verification](evaluation/week9-m4-v1/presentation-verification.json).

At **7:33 AM PDT**, M2 remains healthy at **112,067/117,132 updates (95.7%)**,
with matching identity, finite losses and zero restarts. It is the only pending
arm. Continue the existing follow-up to verify, transfer, admit and evaluate M2
after its fixed endpoint completes, then produce the final five-arm analysis and
package. Do not rerun completed M4 or initial B0/M1/M3 panels.

The completed M4 evaluation service log is at
`/home/peimengsui/data/eventground-wam/libero-plus/evaluation/week9-m4-launch-v1/service.log`.
Use the read-only collector at
`docs/evaluation/week9-progress/20261014T0459Z/collect_check.py`, which includes
later-arm services, qualifications, panel progress and M4 launch errors and handles
the completed training status schema. [Original launch verification](evaluation/week9-progress/20261014T0459Z/check.json)
retains all four audited qualification artifacts, exact repeated inputs and the
unchanged frozen source, runtime, protocol and retry/resource limits.

**M4 endpoint verification and admission:**
M4 finished its fixed epoch-2 endpoint at **9:08 PM PDT**, with all **117,132
updates**, **3,412,044 action presentations** and **16,259,770 future pairs**.
Its training service exited successfully with zero restarts. The policy and full
predictor checksums, completion chains, frozen training identity, execution recipe
and all 240 training-source files passed verification. The completed endpoint,
metadata and source are copied to Feifei at
`/home/peimengsui/data/eventground-wam/libero-plus/training/week8-m4-lambda-v1`.
The unchanged Week 9 runner verified the cohort runtime and created the immutable
[M4 admission](evaluation/week9-progress/20261014T0443Z/M4-admission.json).
[Verified transfer receipt](evaluation/week9-progress/20261014T0443Z/transfer-verification.json).

**Earlier GPU guard block resolved:** the user confirmed closing the desktop apps
and requested launch again. Text Editor had exited; remaining Ptyxis contained
only its agent and one idle shell, with no running jobs. That idle process was
closed with SIGTERM, and the unchanged frozen guard passed before launch.
[Closure evidence](evaluation/week9-progress/20261014T0459Z/ptyxis-close.json).
No further user action is required for this issue.

At **9:38 PM PDT**, M2 remains healthy at **107,300/117,132 updates (91.61%)**,
with approximately **20.5 hours** remaining, or **October 14 around 6 PM PDT**.
[Updated training curves and verification](evaluation/week8-training-progress/20261014T0439Z/charts/progress-summary.json).
All six initial panel reports, their analysis and independent backup remain
unchanged. M4's completed status schema omits running loss, identity and active-time fields; use the
sealed completion and endpoint reports for those checks instead of running-only
status assertions. M2 remains pending; the five-arm final report remains open.

**October 10, 2026: the initial B0/M1/M3 evaluation queue is complete.** All
**9,000 outcomes** passed artifact audit; `eventground-week9-initial-v3.service`
is inactive with success and zero restarts. Both panels' paired analysis was
complete at **1:17 PM PDT** for these three arms. The independent raw backup and
local metadata verification have also finished. [Initial results and figure](WEEK9_INITIAL_RESULTS.md).

| Success rate | B0 | M1 | M3 |
|---|---:|---:|---:|
| Perturbed-1K | 74.10% (741/1,000) | 76.10% (761/1,000) | 76.80% (768/1,000) |
| Clean-2000 | 93.35% (1,867/2,000) | 94.85% (1,897/2,000) | 93.80% (1,876/2,000) |

M3 − M1 is +0.70 pp on perturbed tasks and −1.05 pp on clean tasks, with paired
95% intervals [−1.90, +3.30] and [−3.85, +1.40] pp. The clean comparison adds
one RGB qualification: M3's first-query wrist image at ordinal 1105 differs
from B0/M1, so its whole clean contrasts remain descriptive. All prior
perturbed mismatches remain included. Clean B0/M1 inputs match on all trials.
[Sealed initial analysis](evaluation/week9-initial-v1/analysis.json) and
[completion evidence](evaluation/week9-progress/20261010T2320Z/feifei.json).

The independent raw-results backup finished successfully under
`eventground-week9-package-initial-v1.service`, copying and verifying
**36,433 files / 292.9 GB** to
`/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/initial-B0-M1-M3-v1`.
It includes all rollout, timing, attempt and receipt files, qualification
evidence, completed endpoint bundles, source and analysis. Its helper and
completion evidence live under `docs/evaluation/week9-progress/20261010T2320Z`.
[Backup completion receipt](evaluation/week9-progress/20261010T2320Z/backup-completion.json)
records sealed package digest `8336325c082041ab5a5b4767755bdcbe32ab44780e80722636ed7a879771cc3c`.
All 382 downloaded metadata files and the source/cohort/report/analysis links
passed [local verification](evaluation/week9-progress/20261010T2320Z/local-package-verification.json).
Raw files and endpoint binaries remain on Feifei; local metadata is under
`outputs/week9-initial-v1-metadata`. Do not repeat this initial backup when
adding M2/M4; preserve it and package the newly completed arms separately.
The report, PNG/SVG figure and supporting evidence also have a verified
12-file companion package at the same results root under
`initial-B0-M1-M3-v1-presentation`.
[Presentation verification](evaluation/week9-initial-v1/presentation-verification.json).

**October 12, 3:23 PM PDT: Feifei access recovered.** SSH now responds, and all
six completed panel reports, admissions, analysis and backup receipt match the
last successful verification. The initial service remains inactive with
success and zero restarts. The earlier request to restore SSH access is no
longer needed. M2 and M4 remain healthy and incomplete at **92,758/117,132**
and **97,518/117,132** updates; no endpoint was admitted and no run was changed.
[Recovery evidence](evaluation/week9-progress/20261012T2223Z/check.json).

**October 12 access issue (resolved above):** Feifei SSH timed out on three bounded read-only
checks, although Tailscale reports the host online. Its completed evaluation
records could not be freshly verified; the last successful check at 11:19 AM
PDT still showed all six initial panels, analysis and backup unchanged.
M2 and M4 remain active at **91,776/117,132** and **96,159/117,132** updates
(1:21 PM PDT), with matching frozen identities, zero restarts and finite logged
losses. Neither endpoint is complete. No training or evaluation was restarted
or changed. The user was notified to restore Feifei SSH access before later
endpoint evaluations. [Access-check evidence](evaluation/week9-progress/20261012T2021Z/check.json).

At the October 10 4:14 PM PDT training check, M2 and M4 remained active at
**70,137/117,132** and **66,515/117,132** updates, respectively, with their frozen
identities and zero restarts. No new endpoint was admitted. The existing
follow-up remains active to verify and add each completed endpoint independently
using the unchanged runner. The five-arm factorial report is still pending.

**October 10, 2026, 10:40 AM PDT automated check:** M3's perturbed-1K panel has
completed, with **768/1,000 successes (76.80%)** and zero artifact-audit errors.
The queue is active with zero restarts or recorded error files. At 10:39 AM,
M3 clean-2000 was at **505/2,000**, the final panel of the initial B0/M1/M3 queue.
The existing **1–3 PM PDT today** finish estimate remains reasonable. M2 and M4
continue their unchanged training at **67,452/117,132** and **62,838/117,132**
updates; both frozen identities match and neither is ready for admission.

| Completed perturbed-1K metric | B0 | M1 | M3 |
|---|---:|---:|---:|
| Successes | 741/1,000 | 761/1,000 | 768/1,000 |
| Success rate | 74.10% | 76.10% | 76.80% |
| Median query latency | 43.28 ms | 43.58 ms | 46.68 ms |
| p95 query latency | 44.67 ms | 44.98 ms | 48.08 ms |

All timing queries are retained; latency is per complete eight-action query.
M3 − M1 is **+0.70 pp**, paired 95% interval **[−1.90, +3.30] pp**. M3 − B0 is
**+2.70 pp**, interval **[+0.10, +5.30] pp**. Both whole-panel comparisons remain
**descriptive**: M3/M1 inputs match exactly on 988/1,000 tasks, and M3/B0 on
989/1,000. The 12 and 11 mismatches are RGB-only, including three and two tasks,
respectively, at the first query. All tasks remain included. These intervals
do not establish a confirmatory coupling benefit or remove the input
qualifications. The perturbed panel includes dev-210; the additional-790
breakdown remains in the full analysis. Final factorial estimates await all
five arms on both panels.

[Snapshot](evaluation/week9-progress/20261010T1739Z/feifei.json) ·
[Paired analysis summary](evaluation/week9-progress/20261010T1739Z/analysis-summary.json).
The five completed panels' reports, audits, identities and analysis are saved
in a verified **30-file, 64.6 MB metadata snapshot** on Feifei and locally.
[Copy and sealed-link verification](evaluation/week9-progress/20261010T1739Z/local-metadata-verification.json).
Feifei archive:
`/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/metadata/20261010T1739Z.tar.gz`.
Local directory: `outputs/week9-completed-metadata/20261010T1739Z`.
Raw rollout files remain in the cohort; this snapshot is not the complete raw
backup. The active runner and scientific protocol were not changed.

**October 10, 2026, 7:36 AM PDT continuation:** The queue remained active through
the interrupted task, with zero restarts and no recorded error files. B0 and M1
have completed both panels; all 6,000 recorded outcomes passed their artifact
audits. At 7:33 AM, M3 perturbed was at **287/1,000**, followed by clean-2000.
The remaining initial queue is estimated to finish **today around 1–3 PM PDT**,
using the completed B0/M1 panel durations with allowance for M3 and analysis;
this remains an estimate. M2 and M4 continue training at **65,958/117,132** and
**60,826/117,132** updates, respectively. Both frozen training identities match,
and neither completed endpoint is ready for admission. The existing two-hour
follow-up remains active.

| Completed panel | B0 | M1 | M1 − B0 |
|---|---:|---:|---:|
| Perturbed-1K | 741/1,000 (74.10%) | 761/1,000 (76.10%) | +2.00 pp |
| Clean paired-2000 | 1,867/2,000 (93.35%) | 1,897/2,000 (94.85%) | +1.50 pp |

The perturbed comparison is **descriptive**: 987/1,000 input prefixes match
exactly; 13 have RGB differences, including three at the first query. The clean
comparison has exact inputs on all 2,000 trials. The paired bootstrap 95%
intervals are [−0.70, +4.60] pp (perturbed) and [−1.90, +5.45] pp (clean, clustered
by the 40 canonical tasks). All tasks and timing queries remain included;
neither interval establishes a clear improvement. Final factorial estimates
still wait for all five arms.

[Current snapshot](evaluation/week9-progress/20261010T1433Z/feifei.json) ·
[Paired summary and qualifications](evaluation/week9-progress/20261010T1433Z/analysis-summary.json).
The completed reports, audit records, identities and analysis were copied into
an independent **24-file, 54.9 MB metadata package**, retrieved locally and
verified against their hashes and sealed identity links.
[Verification](evaluation/week9-progress/20261010T1433Z/local-metadata-verification.json).
Raw rollout files remain in the immutable Feifei cohort; this metadata snapshot
is not a complete raw-rollout backup. Its Feifei archive is
`/home/peimengsui/data/eventground-wam/libero-plus/results/week9-factorial-v1-cohort02/metadata/20261010T1433Z.tar.gz`.
Local copies are under `outputs/week9-completed-metadata/20261010T1433Z`.

**October 9, 2026, 8:12 PM PDT automated check:** B0 perturbed is at
**803/1,000**, with the queue active, zero restarts and no recorded error files.
No larger panel has completed yet. M2 is running at **60,500/117,132** updates;
M4 is running at **53,347/117,132**, both with their frozen training identities
and no restarts. Neither endpoint is ready for admission. No execution changes
were needed. [Check evidence](evaluation/week9-progress/20261010T0311Z/check.json).

**October 9, 2026, 7:31 PM PDT progress check:** B0's perturbed panel is at
**525/1,000 (52.5%)**, with zero restarts or recorded infrastructure errors.
B0 clean-2000 and both M1/M3 panels remain queued. The initial 9,000-rollout
queue is estimated to finish **October 10 around noon–4 PM PDT**, roughly
17–21 hours from this check. The estimate combines current suite/category
pacing with prior Feifei clean-2000 timings; clean-panel and analysis durations
remain uncertain. [Snapshot and ETA assumptions](evaluation/week9-progress/20261010T0230Z/eta-summary.json).

**October 9, 2026: the user authorized the fresh B0/M1/M3 cohort and later M2/M4 additions.** The scientific [protocol v1](WEEK9_EVALUATION_PROTOCOL.md) and [sealed contract](../configs/evaluation/week9_factorial_v1.json) remain unchanged. The [execution binding](WEEK9_EVALUATION_RUNNER.md) implements the fixed panels, common runtime, synchronized query timing, durable retries and exact input audits.

**The fresh larger evaluation is now running on Feifei.** M3's dev-210 comparison and independent backup finished first. B0, M1 and M3 then passed real GPU inference/recording checks on both panels, with exact repeated inputs on each selected smoke task. The durable queue `eventground-week9-initial-v3.service` has started B0's 1,000-task perturbed panel, followed by B0 clean-2000, M1's two panels and M3's two panels. [Launch verification](evaluation/week9-runner-v1/launch-verification.json) records the active service, admissions, qualification reports and artifact-backed progress. No pilot outcomes fill formal cells.

| Preparation | Status |
|---|---|
| Scientific protocol | Frozen and verified; no checkpoint, panel, seed or statistical changes. |
| Fixed B0/M1/M3 endpoints | All completed and CPU verified on Feifei, including M3 policy and full predictor checksums. |
| Perturbed panel | 1,000 existing tasks, with disclosed dev-210 overlap and additional-790 tables. |
| Clean panel | 40 canonical tasks × 50 fixed states; all 2,000 state and task bindings verified on Feifei. |
| Runner and analysis | Implemented; 83 focused tests passed locally and in the isolated Feifei package. |
| Runtime lock and GPU recording checks | **Passed for B0/M1/M3**; effective package/environment, driver/GPU UUID, CUDA/cuDNN and backend flags locked in [cohort 02](evaluation/week9-runner-v1/cohort02.json). |
| Initial order | B0 1K/clean-2000 → M1 1K/clean-2000 → M3 1K/clean-2000, with interim analysis after each panel. |
| M2/M4 | Continue their existing Lambda training. Add their completed epoch-2 endpoints through immutable admissions, then use the identical Feifei runner and panels. |

The production source is `/home/peimengsui/code/eventground-wam-week9-evaluation-v4`, digest `0047e136615e3234177b651e07bfad697fc662a151dd9ad199f1f69b77264304`. It preserves all 265 parent evaluator files unchanged. The cohort output is `/home/peimengsui/data/eventground-wam/libero-plus/evaluation/week9-factorial-v1-cohort02`; launch evidence is its sibling `week9-launch-v3`.

[Source package](evaluation/week9-runner-v1/package-04/package.json), [Feifei CPU preflight](evaluation/week9-runner-v1/feifei-preflight.json), [local tests](evaluation/week9-runner-v1/local-tests.txt), [Feifei tests](evaluation/week9-runner-v1/feifei-tests.txt), [launch script](evaluation/week9-runner-v1/launch_pipeline_v3.py).

Existing B0/M1 rendering qualifications and the original JEPA quality-gate FAIL remain visible. M3's preliminary repeatability audit also found two non-exact lighting prefixes; this does not justify task drops or favorable reruns. Every fresh Week 9 contrast receives its own full input audit. Partial-arm tables remain interim; final factorial estimates wait for all five fully audited panel matrices.

The first pipeline stopped before backup copying or Week 9 admission because the backup helper passed a string to a Path-only hash function. Package 03 fixes that helper and the analogous timing-sidecar check, adds a real artifact/sidecar audit regression test, and passes 83 checks on both hosts. No rollout was repeated for this fix; old package and failed service evidence remain preserved.

Package 04 initializes the pinned OpenCV loader before runtime locking; the original loader normally prepends its library directory. The first preparation stopped before any Week 9 episode or query. That unused cohort and failed evidence remain preserved; the replacement operational cohort is `week9-factorial-v1-cohort02`, with the same scientific protocol and endpoints. Model loading is checked against its initialized runtime again. [Staged import probe](evaluation/week9-runner-v1/import-runtime-probe.json).

A quiet two-hour follow-up, `complete-week-9-evaluations`, will finish analysis/packaging and admit M2/M4 independently when their fixed endpoints complete, then run the same Feifei panels. It reports meaningful completion, failures, new arm launches or required user input. It cannot select intermediate checkpoints or change training.
