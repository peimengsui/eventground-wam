# Week 3 entry status

**Snapshot date:** 2026-09-13

**Entry outcome:** pass. Data, compute, local engineering, canonical-code, and
Week 1 reference-policy prerequisites are complete. B0 implementation can start
from the clean `week3-entry-v1` Git tag.

| Prerequisite | Status | Evidence |
| --- | --- | --- |
| Frozen Week 2 dataset | Pass | `week2-dataset-v1` is present on `feifei`; its manifest SHA-256 is `53d80156d9f6eb0451572b29ab7fcf38add08405ffa6ab26c7400899aad47ae2` and matches the committed freeze evidence. |
| Source snapshot | Pass | The `current` link resolves to immutable dataset revision `22c57433fef692b5b9ecc0795344daac7fa867a5`. |
| Training split | Pass | 11,521 train, 1,396 validation, and 1,430 test episodes with zero cross-split group overlap. The assignment file hashes to `71b8fc2a3ce04c84cd41e3c9f052f1a677a0fa2d546201f8540f57eeb3c46758`. |
| Action-sample index | Pass | The 10,300,666-record multi-horizon file hashes to `6e317de0db00792ed9af7b4a500fb2f4cdb90a7d0b4d9a8a9ca2eacb56e69d5f`. B0 uses its frozen split identity, not event targets. |
| Event-label dependency | Not applicable to B0 | Frozen v1 has no admitted `next-event` samples. B0 contains no event conditioning or future-prediction target. |
| Action objective | Locked, revised for binary gripper | B0 uses one deterministic action-query Transformer. The accepted ten-demonstration checkpoint uses normalized continuous-action L1 plus unit-weight gripper BCE; inference decodes the gripper logit to exact `-1/+1`. Its conditioning-memory interface accepts future tokens for M3/M4. Generative action objectives remain outside the study. |
| Local tests | Pass | `uv run pytest`: 130 passed after adding the reference-evidence contract test. |
| Local lint | Pass | `uv run ruff check .`: all checks passed. |
| Compute capacity | Pass for entry | RTX 5090 with 32,607 MiB total and 31,849 MiB free at inspection; 1.6 TiB storage free. Concurrent model/simulator memory remains a Week 3 runtime gate. |
| Complete `sanity-35` | Pass | All 35 tasks ran from clean project revision `5e7141183d19651d5bd78e9cbcc7a477fdf7082f`: 28 successes, seven policy timeouts, zero exceptions, and zero retries. Every HDF5 artifact validates and matches its report hash. |
| Week 2 human-QA completion decision | Pass | The reviewer reconfirmed that the balanced 60-episode review is the completed gate and that no further review is required. The digest-sealed completion evidence records 79 accepted predictions, zero errors, and the review/report hashes. The remote 200-row blank template is a historical input template, not the completed review artifact. |
| Canonical code revision | Pass | Local `main`, origin, the `feifei` Git remote, and the clean compute-host checkout were synchronized at `5e7141183d19651d5bd78e9cbcc7a477fdf7082f` before the reference run. The final evidence-bearing entry revision is tagged `week3-entry-v1`. |

## Integration and reference evidence

The verified Week 2 completion and L1-only Week 3 design were fast-forwarded to
`main`, pushed to both Git remotes, and pulled into the clean compute checkout.
The final source state, including compact reference evidence, is identified by
the `week3-entry-v1` tag; B0 work starts from that tag or a direct descendant.
The complete reference report and episode artifacts live outside Git under the
revision-specific path recorded in
[`evaluation/sanity-35-reference-policy-v1.json`](evaluation/sanity-35-reference-policy-v1.json).

Week 3 acceptance thresholds are locked in
[`WEEK3_BASELINE_DESIGN.md`](WEEK3_BASELINE_DESIGN.md). At the entry snapshot,
no Qwen checkpoint had yet been downloaded or pinned.

## B0 data-layer implementation

The data layer was implemented from `week3-entry-v1` on the
`codex/week3-b0-dataloader` branch. The loader and train-only action-statistics
contract are documented in [`B0_DATALOADER.md`](B0_DATALOADER.md). B0 derives
its `8 x 7` chunks from frozen episode assignments rather than the semantically
different Week 2 future-frame horizon index.

The real-data gate ran on `feifei` from clean revision
`b62f0ec5a13f3315157b265561017d5877232784`. Statistics covered all 1,796,601
actions in the 11,521 training episodes and expose 1,715,954 valid chunks. Two
complete identity passes matched at
`c507ff3f6fe7c9f6ccf7866fa179b82c30aac9fd34d0730ebfa2ae9be2653963`.
Episode 0 produced the expected 270 samples, direct anchors matched the source
Parquet rows, real dual-camera frames met the RGB CHW contract, and all
2,238,036 train/validation/test actions inverted within `1.12e-16` absolute
error with exact gripper polarity. Compact evidence is committed at
[`evaluation/week3-b0-data-v1.json`](evaluation/week3-b0-data-v1.json); the two
full artifacts remain outside Git at the path recorded there.

## B0 model-layer implementation

The immutable Qwen source manifest and the conditioning/action-model contract
are documented in [`B0_MODEL.md`](B0_MODEL.md). The implementation uses the
bare Qwen multimodal core, all non-padding final hidden tokens, explicit 8D
proprioception fusion, an optional future-memory interface, eight deterministic
learned action queries, and a normalized-action mean L1 baseline. The later
overfit-only gripper classification experiment is separately versioned and
documented below.

The real BF16 forward/backward gate passed on the RTX 5090 from clean revision
`ac349d07a4c4703d59d13ba23ca02e2461a6272f`. It produced conditioning memory
`[1,170,512]` and actions `[1,8,7]`; all 18,637,335 allowlisted parameters had
finite gradients with nonzero signal in every required group and no frozen-base
gradient. Repeat and post-reload maximum errors were both zero after an explicit
pre-load perturbation changed predictions by 2.921875. Peak allocated CUDA
memory was 4,748,491,776 bytes, with 0.673-second forward and 0.034-second
backward smoke timings. Compact evidence is committed at
[`evaluation/week3-b0-model-v1.json`](evaluation/week3-b0-model-v1.json); the
full processor and model smoke reports remain outside Git at the path recorded
there.

## B0 one-fixed-batch overfit

The predeclared four-sample canonical training batch passed both locked overfit
thresholds on the RTX 5090 from clean revision
`795f152b71e62bbbf61f1cf5f330e5be8847f0d1`. Evaluation-mode normalized-action
L1/MAE fell from `1.2741760015` to `0.0484590121` at step 880, or 3.803% of its
initial value. Gripper sign accuracy was 100%; every allowlisted parameter had a
finite gradient, every required trainable group had nonzero signal, and the
frozen Qwen base received no gradient. Peak allocated/reserved CUDA memory was
5,508,346,368/5,639,241,728 bytes and training took 56.46 seconds. A deliberate
checkpoint perturbation changed actions by 1.0865 before strict reload restored
them with zero maximum difference. Compact evidence is committed at
[`evaluation/week3-b0-overfit-1batch-v1.json`](evaluation/week3-b0-overfit-1batch-v1.json);
the manifest, complete 881-record curve, 73.6 MB checkpoint, report, and
SHA-256 sidecars remain outside Git at the path recorded there.

## B0 ten-demonstration overfit

The fixed 1,454-sample, ten-task cohort completed with the versioned hybrid
continuous-L1 plus gripper-BCE objective. At the final reload-verified checkpoint,
decoded action MAE was `0.02796977`, the objective ratio was `0.12608315`, and
gripper accuracy was 11,580/11,632 (`99.552957%`). The best diagnostic observed
11,581 correct (`99.561554%`). The model retains the default rank-16 LoRA; no
rank-increase experiment was used.

The original exact-100% memorization gate is transparently preserved as failed.
After the reviewer accepted 99.55% as sufficient, a deterministic residual audit
found that the 52 final errors represent 32 unique source action frames and all
fall within four frames of a gripper-command transition. There are zero
non-binary targets, zero overlapping-chunk label contradictions, and zero exact
conditioning duplicates with conflicting targets. This supports a low-risk,
post-hoc engineering acceptance for advancing beyond the overfit diagnostic; it
does not claim rollout or generalization quality.

Further BCE reweighting was stopped on reviewer direction. Its incomplete
weight-10 directory is preserved but is not acceptance evidence. The sealed
unit-weight result, error audit, hashes, and disposition are summarized in
[`evaluation/week3-b0-overfit-10demo-bce-v1.json`](evaluation/week3-b0-overfit-10demo-bce-v1.json).

## B0 evaluator integration

The accepted checkpoint now has a strict inference adapter and a single-task
entry point documented in [`B0_CLOSED_LOOP.md`](B0_CLOSED_LOOP.md). It reuses
the established LIBERO reset, settling, eight-action open-loop execution,
retry, latency, and HDF5 recorder path. Before loading tensors it verifies the
accepted review, training report, ten-demonstration cohort, normalizer, and
checkpoint hash chain. It then restores the pinned Qwen plus default rank-16
LoRA model, inverts the train-only continuous-action normalization, clips only
continuous values to the simulator range, and preserves exact binary gripper
commands.

The combined B0/LIBERO/HDF5 runtime imports passed on `feifei`. A load-only gate
verified the complete accepted hash chain, reconstructed Qwen plus rank-16 LoRA,
strictly restored the checkpoint, and used 4,508,463,616 allocated CUDA bytes.
A real synthetic-input forward/decode query returned a finite `8 x 7` chunk with
exact binary gripper output, in-range continuous actions, and zero clipping. The
compact result is committed at
[`evaluation/week3-b0-evaluator-integration-v1.json`](evaluation/week3-b0-evaluator-integration-v1.json).

The closed-loop training-task gate is now complete. A complete predeclared sweep
of the sealed ten-task cohort found one unperturbed base-task success: cohort
episode 9, `put the bowl on the plate`, completed at policy step 118. The same
task was then rerun from clean revision `f544944` through the shared evaluator
and HDF5 recorder under the locked `week3-b0-training-rollout-v1` protocol. Its
128-step artifact (10 settling plus 118 policy steps) independently validates,
used no retry, raised no exception, and had no continuous-action clipping.

The compact sealed evidence is committed at
[`evaluation/week3-b0-closed-loop-v1.json`](evaluation/week3-b0-closed-loop-v1.json).
It also preserves the first failed perturbed rollout and all ten diagnostic
sweep outcomes: one success and nine failures. Accordingly, this satisfies the
Week 3 one-training-task smoke criterion but does not establish cohort success
rate, robustness, or generalization.

The accepted rollout bundle subsequently passed a deeper cross-file audit of
its report digest and sidecar, embedded protocol, checkpoint/task/seed identity,
source BDDL and initial-state hashes, action-chunk alignment, binary gripper
commands, latency rows, and terminal success signal. A fresh successful rollout
from clean revision `8858f60` was concurrently sampled every 100 ms. The sampled
process peaked at 5,858,394,112 GPU bytes while Torch reported 4,591,992,832
allocated and 4,649,385,984 reserved bytes; at least 28,073,525,248 GPU bytes
remained globally free. Median policy-query latency was 32.94 ms across all 15
queries and 32.89 ms after excluding the first cold query. No latency target was
imposed by the Week 3 design.

The validation and profile gate is sealed at
[`evaluation/week3-b0-rollout-validation-profile-v1.json`](evaluation/week3-b0-rollout-validation-profile-v1.json).
