# Week 5 — Pixel temporal CPT

**Started:** September 24, 2026. **Updated:** September 26, 2026. **Status:** all six
steps are complete and `C_pixel-v1` is accepted. [Production CPT](WEEK5_PIXEL_PRODUCTION.md) finished at 10:54 PDT:
one full pass through 8,221,365 training pairs in 221,633 updates, independently
verified. [Step 6 validation](WEEK5_PIXEL_VALIDATION.md) passed all frozen gates
on 1,109,994 held-out pairs, with independent verification and all 480 fixed
examples reviewed. Visible long-horizon limitations remain. Action-policy training has not started. CPT here means
continued training for future prediction.

Week 4 is [closed with an explicit milestone acceptance revision](WEEK4_CLOSEOUT.md).
The selected epoch-2 B0 is the fixed evaluation control for the remaining study. Its incomplete external
performance gates remain recorded; no further B0 optimization blocks this phase.

**Updated training decision:** [two total action-training epochs](PHASE2_TRAINING_PROTOCOL.md)
for each B0/M1–M4 policy, with identical effective batch boundaries and sampling.
M1–M4 start from the pinned base Qwen plus fresh shared components; the completed
B0 supplies the recipe and reference score. The earlier B0-continuation proposal
is superseded. Pixel/JEPA CPT and later event training have separate budgets.

## Goal and scope

Follow the [original Week 5 plan](EventGround-WAM_12-Week_Execution_Plan.md#week-5--pixel-temporal-cpt):
learn a lightweight deterministic predictor of future images from the current
observation, instruction, intervening actions and requested horizon:

`(observation_t, instruction, actions[t:t+h], h) → observation[t+h]`.

The deliverable is **C_pixel.ckpt**, reconstruction metrics by horizon/camera,
and qualitative future predictions. Week 5 establishes future-prediction ability;
action-policy improvement is evaluated in Weeks 7–9. Use pinned base features
for the initial CPT design and train separate future modules. Importing
B0-trained features/weights or adapting the shared backbone in CPT requires an
explicit design/control revision.

## Execution order

| Step | Work | Completion evidence |
| --- | --- | --- |
| 1 | **Complete:** prepare a versioned future-pair index using trajectory-v2 assignments and fixed horizons 2, 4, 8, 16, 32 at 20 Hz. | [Verified freeze](WEEK5_FUTURE_PAIRS.md): 8,221,365 train and 1,109,994 validation pairs; source hashes, same-episode bounds, group isolation and byte-identical regeneration passed. No test payload use. |
| 2 | **Complete:** freeze measurements and simple baselines on validation. | [Verified controls and criteria](WEEK5_PIXEL_BASELINES.md): all 1,109,994 validation pairs, both cameras; copy L1/SSIM 0.058562/0.720547 and train-only constant 0.150935/0.582252; group uncertainty, motion diagnostics and 480 fixed examples. |
| 3 | **Complete:** implement the pixel predictor and its action/horizon conditioning. | [Verified implementation](WEEK5_PIXEL_PREDICTOR.md): 4.63M trainable parameters, frozen base Qwen, 128×256 future tokens; 98.48% controlled and 28.24% real-train L1 reduction; action/time checks, target isolation and exact weight reload. |
| 4 | **Complete:** freeze the numerical recipe and stopping rule, then run a recoverable pilot. | [Verified recipe and recovery](WEEK5_PIXEL_TRAINING.md): one pass / 8,221,365 pairs / 221,633 updates; ~22.9-hour projection with 1.5× margin; 8.50 GiB full-device peak; bitwise process resume and matched accumulated gradients. |
| 5 | **Complete:** fresh production training and final checkpoint verification. | [Verified endpoint](WEEK5_PIXEL_PRODUCTION.md): 221,633 updates / 8,221,365 pairs / 14.49 active hours; zero restarts; exact exposure-chain replay; export equals final recovery weights; frozen base unchanged. |
| 6 | **Complete:** validate and freeze the fixed final candidate against copy/background. | [Accepted `C_pixel-v1`](WEEK5_PIXEL_VALIDATION.md): all 1,109,994 pairs; 33.27% overall L1 and 45.04% motion L1 gains over copy, positive SSIM gain, all 10 camera/horizon checks, independent replay and 480 reviewed examples. |

### Data and measurement decisions

Step 1 is frozen as `trajectory-v2-future-pairs-v1` for both Pixel and JEPA.
The [completion record](evaluation/week5-step1-completion-v1.json) pins the
manifest, production checks and artifact locations. It also preserves this
document's earlier Phase 2 snapshot; the accepted action-training contract and
historical closure records remain unchanged.

Step 2 is frozen as `week5-pixel-measurement-v2`. Its
[completion ledger](evaluation/week5-step2-completion-v1.json) preserves the exact
step 1 status snapshot, independent CPU replay and bootstrap recomputation,
and the superseded v1 measurement history. All 180 validation groups / 40 tasks
provide motion support. The learned-predictor gate subsequently passed in Step 6.

Step 3 is recorded in its [completion ledger](evaluation/week5-step3-completion-v1.json),
which preserves the exact Step 2 status snapshot and the two rejected diagnostic
recipes. Architecture v2 and diagnostic recipe v3 pass the implementation checks.
The diagnostic weights are explicitly **not C_pixel** and are not selected on
held-out validation.

Step 4 freezes `week5-pixel-cpt-v1` and the common Pixel/JEPA
`trajectory-v2-cpt-order-v1`. Its [completion ledger](evaluation/week5-step4-completion-v1.json)
preserves the prior status/README snapshots and pins the pilot evidence. The
pilot starts fresh, runs 128 representative logical updates, and verifies
actual SIGKILL/restart, complete-state equality, corrupted-checkpoint fallback
and full/microbatch gradients. Step 5 completed from an independently
verified copy of that source with fresh predictor weights and an empty optimizer.
The [launch ledger](evaluation/week5-step5-launch-v1.json) preserves the Step 4
document snapshots and records startup/recovery verification. Its
[completion ledger](evaluation/week5-step5-completion-v1.json) pins the full
endpoint audit. Step 6 assessed the fixed final candidate and accepted the
byte-identical `C_pixel-v1` export. Its [completion ledger](evaluation/week5-step6-completion-v1.json)
preserves the Step 5 document snapshots and pins the full validation, independent
audit, qualitative review and [checkpoint acceptance](evaluation/week5-pixel-validation-v1/acceptance.json).

- Reuse the source demonstrations, **not the old Week 2 v1 split assignments**.
  Rebuild or rejoin future samples under the exact Week 4 trajectory-v2 group
  split. Fixed-horizon targets cannot cross an episode boundary. Store the
  source episode, anchor, target frame, horizon, split and digest for every pair.
- Week 2 frozen v1 admitted **zero next-event samples**. Start with fixed
  horizons; event-conditioned sampling requires a separately versioned, validated
  label set. Do not convert weak event labels into ground truth implicitly.
- Freeze the future-pair manifest once for both Pixel (Week 5) and JEPA (Week 6).
  Keep horizons and source exposure identical. Any augmentation must preserve
  temporal and camera correspondence and be shared between compared methods.
- Step 2 fixes deterministic reconstruction measurements at **128×128**, L1
  and SSIM. Step 3 declares 4,631,427 trainable future-module parameters and
  zero trainable base parameters. Step 4 freezes L1 + 0.1 × (1 − SSIM);
  LPIPS is optional and is not currently a launch dependency.
- The [numerical exit gate](WEEK5_PIXEL_BASELINES.md#prospective-success-criteria)
  requires at least 5% overall L1 and 10% medium/long-horizon motion L1 gains
  over copy, with positive paired 95% lower bounds, plus SSIM, background and
  camera/horizon guardrails. Bootstrap numeric groups within task; correlated
  frames/appearance variants are not independent trials. Fixed qualitative
  examples and motion coverage address stationary-background dominance.

Step 3 fixes the initial predictor architecture and frozen-base feature route.
Step 4 fixes the one-pass order, logical batches, L1/SSIM objective, AdamW
schedule, final-update selection, resource limits and full recovery. The
221,633-update endpoint uses no validation-based stopping or checkpoint search.
The measurement resolution and numerical exit criteria were frozen in Step 2.

### Integration constraints carried forward

The selected OFT B0 currently rejects future-token conditioning. Its earlier
Week 3 action-query interface is not the selected action head. Week 5 can train
a separate future module; Weeks 7–8 must define a versioned integration that
retains the selected OFT action path and verifies equality with future coupling
disabled. Pixel/JEPA and auxiliary/coupled arms must share that integration.

Ground-truth future actions are available for CPT only. Before inference coupling,
define how the model obtains candidate actions at deployment without using
demonstration future actions; match that choice for Pixel-WAM and JEPA-WAM and
report any extra inference passes. Predicted futures must never read true future
images at rollout time.

## Roadmap after Week 5

| Week | Main experiment | Comparison |
| --- | --- | --- |
| 6 | JEPA temporal CPT, C_jepa | Identical future pairs/horizons; copy-latent and collapse diagnostics. |
| 7 | M1 pixel-aux and M2 JEPA-aux | Same fresh base-Qwen/shared-component initialization and two total action epochs; identical B0 batches; future modules removed at inference. |
| 8 | M3 Pixel-WAM and M4 JEPA-WAM | Independent fresh policy runs of two total action epochs; matched coupling interface; M1/M3 share C_pixel and M2/M4 share C_jepa; record inference cost. |
| 9 | Core 2×2 comparison | B0 and M1–M4 on the same scheduled stratified-1K panel, paired uncertainty and clean control. |
| 10–11 | EventGround and event-adaptive study | Add validated event supervision and compare against fixed-horizon M4. |
| 12 | Final evaluation and write-up | Frozen selections, full-10K protocol and complete reporting, including negative results. |

OpenVLA scores remain external context. The main scientific question is whether
future representations improve the shared Qwen3.5-2B policy over matched
controls. Sampling/replanning optimization of B0 is deferred by the Week 4
closure decision. See the [shared comparison rules](WEEK4_CLOSEOUT.md#fair-comparisons-after-closure).
