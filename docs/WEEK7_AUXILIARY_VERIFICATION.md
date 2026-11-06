# Week 7: shared auxiliary training and feasibility

**October 3, 2026 — steps 2–4 complete.** M1 Pixel aux and M2 JEPA aux have an
implemented shared OFT interface, verified data/loss/training paths, sealed
numerical recipes, and passed GPU correctness, resource and process-recovery
pilots. Production policy training and rollout evaluation have not started.

- [Interface and gradient boundaries](WEEK7_AUXILIARY_INTERFACE.md)
- [M1 numerical recipe](../configs/training/week7_m1_auxiliary_v1.json)
- [M2 numerical recipe](../configs/training/week7_m2_auxiliary_v1.json)
- [Independent evidence verification](evaluation/week7-integration-v1/verification.json)
- [Steps 2–4 completion ledger](evaluation/week7-steps2-4-completion-v1.json)

## Implementation

The original OFT action forward remains intact. Both future predictors read its
final hidden states; their loss gradients reach the shared backbone and state
projector. Future targets and demonstration future actions stay outside the
action path. Deployment uses the original policy architecture with the predictor
and teacher removed. A standard action-only checkpoint export was reloaded and
verified against the policy state.

`auxiliary_policy.py` implements the feature route; `auxiliary_data.py` attaches
future targets; `auxiliary_losses.py` implements bounded full-logical-batch
backward; `auxiliary_training.py` owns joint Adam/scheduler updates and complete
recovery state. Existing B0 and CPT source files were preserved.

The exact original action plans were regenerated and matched for both epochs:
117,132 updates and 3,412,044 action presentations. All valid horizons attach
8,129,885 dual-camera future pairs per epoch (16,259,770 over two epochs), without
dropping an action example. The independent old/new loader comparison was bitwise
identical for images/text encodings, state, action targets and identities on full,
one-sample, longest-episode and longest-instruction batches.

The shared 92,907,527 trainable policy parameters initialize bitwise identically
to the recorded fresh B0 tensors. Additional future parameters are 4,631,427 for
Pixel and 4,764,928 for JEPA, preserving the common 4,238,592-parameter future core.
All 40 training instructions were checked; the maximum prompt is 198 tokens,
within the 512-token interface. The maximum distinct episode frames in any mixing
window are 4,441 and 4,365 for epochs 1 and 2, within both cache bounds.
See the [initialization and metadata audit](evaluation/week7-integration-v1/initialization-audit.json).

## Numerical verification

The [prospective v2 pilot rule](../configs/training/week7_auxiliary_pilot_v2.json)
retains the original thresholds and fixes future convolutions to IEEE FP32.
Shared policy/teacher BF16 execution and the inherited FP64 SSIM moments remain
unchanged. Four fixed training batches calibrate each outer loss weight from the
median initial shared-hidden gradient ratio; no optimizer updates, validation
scores, rollouts or parameter search enter that calculation.

| Check | M1 Pixel aux | M2 JEPA aux |
| --- | ---: | ---: |
| Frozen outer future-loss weight | 4.836645309900765 | 1.5140534185764003 |
| FP32 future-gradient relative L2 error | 0.000028929 | 0.000001961 |
| FP32 future-gradient maximum absolute error | 0.0000015013 | 0.0000000182 |
| Joint BF16 gradient relative L2 error | 0.006860 | 0.003787 |
| 64-update training-only action-loss reduction | 79.1% | 85.4% |
| 64-update training-only future-loss reduction | 36.4% | 55.3% |
| Target isolation / predictor removal | Pass | Pass |
| Actual SIGKILL and exact resumed state | Pass | Pass |
| Corrupted latest-checkpoint fallback | Pass | Pass |

Gradient comparisons use four real action anchors with every valid horizon,
comparing the whole diagnostic loss against predictor microbatches. The main
128-update resource runs use the unchanged original logical action batches,
including their short batches. JEPA computes variance across the full logical
batch's anchors, never by averaging microbatch variance penalties. Targets are
detached, and the canonical teacher remains frozen and bitwise unchanged.

The inference action output is bitwise unchanged by the temporary feature route,
changed supervision targets, or removal of the auxiliary path. Auxiliary-only
backward reaches shared backbone/state parameters and leaves action-head
parameters without auxiliary gradients. FP32 numerical tolerances and the 1%
joint BF16 tolerance are explicit; gradient equivalence is not described as
bitwise equality.

Recovery checkpoints cover both models, all Adam groups/moments, scheduler,
RNG, cursors and action/future identity chains. Each process was killed during
the first predictor backward chunk of update 3, after committing update 2.
Restarting with an empty teacher cache produced exactly the same four-update
state as uninterrupted execution. Both arms saw the same 128 action examples
and 609 future pairs in this check. The separate device-memory replay reproduced
those state digests again.

There are **108 passing local tests**, including 19 new auxiliary tests, plus
successful lint and the read-only evidence audit. The [JUnit record](evaluation/week7-integration-v1/local-tests.xml)
is retained. Memorization is an implementation sanity check; these losses are
not policy-quality or cross-method objective-superiority evidence.

## Feasibility on Feifei

One NVIDIA RTX 5090 runs one substantial GPU job at a time. Each primary resource
pilot began from original initialization and cold caches, then completed 128
updates: 3,696 action examples and 17,575 future pairs per arm. Edge batches also
completed backward. No pilot-trained weights are permitted in production.

| Measurement | M1 Pixel aux | M2 JEPA aux |
| --- | ---: | ---: |
| Peak allocated GPU memory, primary pilot | 8.82 GiB | 9.53 GiB |
| Peak reserved GPU memory, primary pilot | 9.85 GiB | 10.83 GiB |
| Sampled whole-device memory, four-update supplement | 10.84 GiB | 11.59 GiB |
| Peak host RSS | 11.55 GiB | 11.52 GiB |
| Mean end-to-end update | 2.03 s | 2.43 s |
| p95 end-to-end update | 2.37 s | 4.26 s |
| Conservative two-epoch projection | 92.7 h | 166.2 h |
| Three recovery checkpoints plus two action exports | 3.01 GiB | 3.01 GiB |

The projection uses 1.2 times the larger of mean and p95 update time, plus measured
checkpoint overhead. Mean-only extrapolations are approximately 66 and 79 hours;
the accepted resource plan uses the conservative estimates above. The memory
supplement samples NVML every 100 ms through four full forward/backward/Adam
updates per arm; the allocator peaks cover the longer primary pilots and edge
probes. [Device traces and storage rehashes](evaluation/week7-integration-v1/device-memory-v1/report.json)
and the [runtime identity](evaluation/week7-integration-v1/runtime.json) are retained.

Both arms fit the host. **JEPA's 166.2-hour projection is close to the frozen
168-hour feasibility ceiling.** Sequential conservative projections total about
10.8 days, so this research phase need not fit one calendar week. Full-run timing
remains an estimate. Each production recipe retains a 192-active-hour cap,
50-GiB free-space reserve, 100-GiB output cap and three recovery checkpoints.
The caches are memory-only. Measured primary-pilot storage is about 3.84 GiB;
checkpoint/export files were rehashed remotely. The output cap also reserves
space for logs and later rollout artifacts, whose final size is not yet measured.

## Retained corrections and interpretation

All source snapshots, failure logs and dispositions are preserved:

1. [Attempt 1](evaluation/week7-integration-v1/attempt-01/disposition.json) stopped
   before optimizer updates: the diagnostic loader left the backbone in evaluation
   mode, disabling activation checkpointing. Explicit training mode fixed it.
2. [Attempt 2](evaluation/week7-integration-v1/attempt-02/disposition.json) stopped
   before optimizer updates when the Pixel absolute gradient error exceeded the
   frozen bound. The prospective v2 rule disables cuDNN TF32 for future convolutions;
   thresholds, examples, loss definitions and model weights were unchanged.
3. [Attempt 3](evaluation/week7-integration-v1/attempt-03/disposition.json) completed
   Pixel memorization and resource updates, then failed strict JSON export on a
   NumPy scalar. Using a native Python scalar fixed the report writer.
4. [Attempt 4](evaluation/week7-integration-v1/attempt-04/run/completion.json) completed
   the matched two-arm suite. The additional discarded Pixel engineering retry
   is disclosed; it did not alter the prospective tuning rule, choose a model,
   or contribute weights/exposure to either production policy.

Week 6's original JEPA gate remains **FAIL**, with the unchanged original endpoint
used as `C_jepa-study-v1` under the recorded research-use exception. These checks
establish correctness and feasibility of auxiliary training, not that the original
prediction gate has been repaired. The original B0 rendering-validity qualification
and development-only interpretation also remain in force.

## Reproduce the evidence checks

```bash
.venv/bin/python scripts/verify_week7_experiment.py
.venv/bin/python scripts/verify_auxiliary_integration.py \
  --attempt docs/evaluation/week7-integration-v1/attempt-04
.venv/bin/python scripts/finalize_auxiliary_recipes.py \
  --attempt docs/evaluation/week7-integration-v1/attempt-04
```

The final command verifies existing recipes for exact equality and refuses to
rewrite a changed recipe. The accepted executable/input source archive contains
206 verified files. GPU execution used that sealed source snapshot and the pinned
B0 runtime. Production launch remains step 5: recheck host availability and hashes,
enforce the declared budgets, and initialize fresh from the pinned base and
original CPT modules. Step 6 remains paired development evaluation.
