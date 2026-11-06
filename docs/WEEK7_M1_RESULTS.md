# M1 fixed-endpoint results

**October 6, 2026: M1 training, dev-210 evaluation and review finished on Feifei.**
The result is **164/210 successes (78.10%)**, versus the frozen B0 reference's
**162/210 (77.14%)**. The observed difference is **+0.95 percentage points**.
This is a descriptive development result and does **not** establish that pixel
auxiliary training improves policy success. The final checksum inventory and
packaging status are recorded in [completion.json](evaluation/week7-m1-final-v1/completion.json).

## Outcomes and qualification

| Policy | Successes | Success rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| B0 | 162 / 210 | 77.14% | 71.01–82.30% |
| M1 | 164 / 210 | 78.10% | 72.02–83.16% |

The frozen common-task bootstrap gives **−4.29 to +6.67 percentage points** for
M1 minus B0 (10,000 draws, PCG64 seed 7, linear quantiles). There are 19 M1-only
successes, 17 B0-only successes, 145 shared successes and 29 shared failures.
All 210 tasks are retained. The interval is a descriptive summary, not a claim
of a fully input-matched treatment effect or an equivalence result.

The full artifact audit passed: all episode files, outcomes, identities and
hashes are valid. However, **207/210 complete canonical prefixes match exactly**.
The [three mismatches](evaluation/week7-m1-final-v1/input-mismatch-details.json)
are ordinals 100, 203 and 204, all Light Conditions tasks. They contain six
settling-camera frames with 4–15 differing RGB channel values per frame, each
different by one intensity level. First-policy-query observations, proprioception,
MuJoCo state, pre-query actions, seeds and scene identities match on all 210.
Both policies succeed on these three tasks. Nevertheless, the frozen rule
requires **the whole-panel contrast to remain descriptive**. No rows were
dropped, relabeled or rerun, and no tolerance was introduced after seeing results.
The source of the small rendering differences has not been established.

This qualification is separate from, and does not repair, the historical B0
versus OpenVLA qualification. The original JEPA gate also remains FAIL under
its recorded research-use exception. Single training seed, small development
cells and multiple descriptive breakdowns further limit inference. No test or
larger confirmation panel was accessed. A future claim requiring exact pairing
needs a separately recorded rendering/baseline-compatibility resolution before
confirmation; selective reruns are not part of this result.

![M1 and B0 evaluation summary](evaluation/week7-m1-final-v1/evaluation-summary.png)

## Descriptive breakdowns

| Perturbation | B0 successes / 30 | M1 successes / 30 | Net tasks |
| --- | ---: | ---: | ---: |
| Background Textures | 27 | 25 | −2 |
| Camera Viewpoints | 29 | 27 | −2 |
| Language Instructions | 17 | 19 | +2 |
| Light Conditions | 28 | 29 | +1 |
| Objects Layout | 22 | 20 | −2 |
| Robot Initial States | 13 | 20 | +7 |
| Sensor Noise | 26 | 24 | −2 |

| Suite | B0 | M1 |
| --- | ---: | ---: |
| Spatial | 39/52 | 47/52 |
| Object | 44/52 | 43/52 |
| Goal | 42/53 | 39/53 |
| Long (`libero_10`) | 37/53 | 35/53 |

Difficulty levels 1–5 respectively give M1 **33, 38, 35, 31, 27** successes out
of 42 each, versus B0 **34, 37, 30, 34, 27**. The largest observed gains are in
Spatial and Robot Initial States, with losses elsewhere. These are exploratory
patterns rather than evidence of a robust category-specific effect. Full
counts and intervals are in the [analysis](evaluation/week7-m1-final-v1/analysis.json).

All 46 M1 failures exhausted the policy step limit. There were **zero
infrastructure retries, zero policy exceptions, and zero automatic service
restarts** during the successful evaluation. Eight failures show late low
end-effector motion under the fixed diagnostic definition; seven also retain
translation commands. These flags describe trajectories and do not diagnose causes.

The [visual review](evaluation/week7-m1-final-v1/visual-review.json) covers eight
deterministically selected pairs: the first three M1-only failures, first three
M1-only successes, first shared failure and first shared success. Both cameras
were inspected at five evenly spaced policy frames per arm. In two pudding-box
failures, M1 reaches the basket and remains in a similar late pose without
completing the goal. Some bowl successes show successful transport where B0
times out; the shared failure leaves the target bowl on the stove while the
arms interact near the plate. These contact sheets are illustrative, not a
full-motion review, predicate reconstruction or failure-prevalence estimate.
Their immutable HDF5 source records remain the authority for scoring.

## Completed training and cost

M1 completed on **October 6 at 05:21:52 PDT**, at the prescribed epoch-2
endpoint: **117,132 optimizer updates**, **3,412,044 action presentations** and
**16,259,770 valid future pairs**. Training took **66.47 active hours** on Feifei's
RTX 5090 with no automatic restarts. No early stopping or checkpoint selection
was introduced. Checkpoint reload and exact endpoint/provenance verification
passed before evaluation.

The action policy retains B0's OFT-compatible architecture. During training,
the original `C_pixel-v1` predictor adds pixel supervision at horizons 2/4/8/16/32
for both cameras. The frozen joint objective is action L1 plus
**4.836645309900765 × (pixel L1 + 0.1 × SSIM penalty)**. It starts from the same
pinned base model and fresh shared initialization as B0, with the original
pixel CPT module; it never loads trained B0 or pilot weights. Logical action
batches, action exposure and seeds remain matched. The adapted predictor is
exported separately and removed at inference.

| Resource | Recorded value |
| --- | ---: |
| Shared trainable policy parameters | 92,907,527 |
| Training-only predictor parameters | 4,631,427 |
| Total trainable parameters | 97,538,954 |
| M1 production training | 66.47 active hours |
| Original pixel CPT | 14.49 active hours; 221,633 updates / 8,221,365 pairs |
| Production plus CPT, excluding pilots/tuning | 80.96 active hours |
| Training service host-memory peak | 17,734,381,568 bytes (16.52 GiB) |
| Qualification-pilot whole-device GPU memory | 11,099 MiB |
| Successful dev-210 evaluation | 34 min 35.848 s wall time |
| Evaluation host-memory peak | 10.6G, rounded systemd journal value |
| B0 / M1 median policy-query latency | 43.06 / 43.33 ms |
| B0 / M1 p95 policy-query latency | 44.34 / 44.70 ms |

Continuous production GPU peak was not recorded; pilot GPU memory must not be
reported as a full-run maximum. The extra CPT, auxiliary parameters and compute
mean this is a matched-action-exposure study, not an equal-total-compute claim.
Historical pilot/tuning attempts, calibration, overfit and recovery evidence are
preserved in the package. Their full wall time is not consolidated into the
80.96-hour production+CPT sum. The [training summary](evaluation/week7-m1-final-v1/training-summary.json)
records these distinctions and exposure details.

The last 500-update window of sparse training logs averages **0.07883 action**,
**0.05647 pixel auxiliary**, and **0.35196 joint loss**. The final logged update
is 117,129; the endpoint completion record verifies 117,132. These curves are
training diagnostics. No held-out trajectory-v2 endpoint validation was run,
and no such score was used for checkpoint selection.

![Final M1 training curves](evaluation/week7-m1-final-v1/training-losses.png)

## Execution and retained evidence

The first evaluation startup rejected the known small Chrome GPU process;
the second lacked the LIBERO import environment. Both stopped before model
loading or any episode. A separate wrapper reused the qualified desktop-process
classifier; a recorded service drop-in restored the original B0 environment.
All 227 prior evaluation-package files remain byte-identical. No scientific
contract, simulator vendor source, numerical recipe or package version changed.
The successful run was **07:36:27–08:11:03 PDT**. The
[startup disposition](evaluation/week7-m1-final-v1/evaluation-startup-disposition.json)
retains both failures and their fixes. Focused runtime, evaluation and backup
checks pass 22 tests; the new helpers pass lint.

The independent Feifei package is:

```text
/home/peimengsui/data/eventground-wam/libero-plus/results/week7-m1-v1
```

It contains the full training output and three recovery states; action endpoint
and adapted predictor; all 210 M1 and 210 B0 HDF5 records; receipts, logs,
contracts and frozen source/runtime identities; costs, pilot evidence,
analysis, figures, visual review and checksum inventories. The initial verified
backup contains 1,055 files / 17,984,726,819 bytes (16.75 GiB); the final closeout
adds the review supplement and re-verifies the entire inventory. Files are
independent copies. Original reports retain their historical pending-status
fields; `completion.json` records the subsequent completed review and packaging.

The fixed endpoint and adapted predictor also have a checksum-verified
[off-host backup](evaluation/week7-m1-final-v1/off-host-backup.json) under local
`outputs/week7-m1-final-v1/training/epochs/epoch-02`. Evaluation metadata and
reports are local under `outputs/week7-m1-final-v1/evaluation`; the full rollout
backup is on Feifei. Original report absolute paths are mapped by the inventory
and `baseline-path-map.json`. Frozen base weights, dataset and simulator assets
remain separately pinned external dependencies.

Policy checkpoint ID: `eventground-wam/week7-m1-auxiliary-v2`.
Checkpoint SHA256:
`c674f6977bdcf12d5ddc52c7a89bc89b9e14f6819b46bc0baecc4acb39b781d3`.
Adapted predictor SHA256:
`130eec27c3c4e89f325d27e18816dc2d7feaa627e1bfd43470d696287c3a467e`.

Feifei's GPU has been released by M1. The M3/M4 design remains unchanged and
predates these outcomes. M3 implementation and qualification are still required
before production launch. M2 continues separately on Lambda; Week 7 remains
open until M2 is also evaluated and packaged.
