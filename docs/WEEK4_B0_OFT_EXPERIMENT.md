# Week 4: Qwen3.5 OFT-style B0 experiment

Authorized by the user on 2026-09-19 as a new experiment following the original
B0's failed clean gate. This prospectively revises the Week 3 architecture and
adapter restrictions for this run. The original checkpoint, reports, and frozen
acceptance result remain historical evidence; this run is not a paper reproduction.

## Fixed choices

- Keep `Qwen/Qwen3.5-2B` revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`. This is the official post-trained
  model, not `Qwen3.5-2B-Base`. The official Qwen model catalog was checked on
  2026-09-19; no newer official vision-language model near 2B justified a switch.
- Preserve continuous six-dimensional action L1 plus unit-weight gripper BCE,
  binary gripper decoding, eight-step action chunks, native 256px dual-camera
  Qwen preprocessing, and the training-only action normalization.
- Preserve the Week 2 episode split and dataset revision. Start from the pinned
  Qwen weights and fresh action/state modules, not the previous B0 checkpoint.
- Keep the completed clean paired-2000 and dev-210 definitions for later evaluation.
  No test episodes or rollout outcomes enter this training or checkpoint selection.

## Architecture

1. Append one state placeholder and eight action placeholders to the instruction.
   Each placeholder is the existing single-token `<|fim_prefix|>` marker (248060).
   StarVLA's literal `🔍` splits into three Qwen3.5 tokens, as verified on the
   pinned tokenizer. Repeated markers and their encoded positions are checked
   explicitly; no vocabulary resizing or trainable token embeddings are added.
2. Replace only the state embedding with an MLP projection of the robot state,
   using per-dimension q01/q99 statistics computed from all 1,796,601 training
   frames. Clip to [-1,1]; constant channels map to zero. No LayerNorm across
   heterogeneous raw state dimensions.
3. Run the multimodal Qwen core once. Explicitly preserve native image/text
   M-RoPE positions when supplying custom input embeddings.
4. Read the eight final action-token states and apply a two-block residual MLP
   (2048 input, 4096 hidden, seven outputs) following the StarVLA OFT pattern.
5. Train rank-32, alpha-16 LoRA on all linear layers in the multimodal core:
   all 24 language blocks, including DeltaNet/attention/FFN, plus vision linears.
   Original weights and token embeddings remain frozen. Use non-reentrant
   gradient checkpointing when enabled by the profiled recipe.

The in-backbone state projection follows OpenVLA-OFT; the one-token-per-action
MLP follows StarVLA. Qwen's native causal/hybrid attention remains. This is a
combined architecture/adaptation experiment, not an isolated action-head ablation.
The new action path currently has no future-token interface; its integration
must be designed explicitly before adopting it for later M3/M4 experiments.

## Resource and recovery contract

The previous run took about 31.2 hours on Feifei's RTX 5090. Target a similar
single-GPU budget: 30 active hours including intermediate validation, with up
to 36 hours budgeted for final full validation/reload and overhead. Fix the update
budget after a hardware pilot; stop earlier at the active-time threshold.
Report actual sample exposures and epoch-equivalent exposure, not "five epochs"
when compute limits result in fewer passes. No augmentation is added in this run.

Validation uses every frozen validation chunk at declared update intervals and
at the endpoint. Select the lowest hybrid validation objective, with earliest
step breaking ties. Record checkpoints by update number.

Host-owned user systemd service, lingering, restart-on-failure, single-writer
lock, atomic checksummed checkpoints every five minutes, and three retained
recovery generations. Resume includes optimizer, scheduler, RNG, sample cursor,
validation history, and accumulated active time. A GPU pilot must verify finite
gradients, frozen base weights, reload predictions, and an identical next update.

## Dataset and evaluation clarification

Our source is `Sylvest/libero_plus_lerobot`, containing 14,347 demonstrations
and 40 task instruction labels; 11,521 episodes are in our training split.
The repository name does not establish per-episode perturbation composition.
It does not mean we trained on the 10,030-task LIBERO-Plus evaluation inventory.

StarVLA's published simple baseline uses the standard four-suite LIBERO
demonstration mixture. Original OpenVLA-OFT also uses LIBERO demonstrations,
but the exact reference we evaluated is the LIBERO-Plus mix-SFT release
`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`. Its exact membership and
overlap with our source remain unverified.

Our B0 68.25% versus reference 96.75% comparison used canonical clean LIBERO
tasks, with 40 tasks and 50 identical initial states per task. The B0-only
dev-210 score of 50.48% is a separate LIBERO-Plus robustness measurement.

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-2B),
[Qwen catalog](https://huggingface.co/Qwen/models),
[StarVLA paper](https://arxiv.org/html/2604.05014#S5.SS1),
[reference matching audit](WEEK4_PUBLIC_REFERENCE.md).

## Execution status

All 260 CPU tests pass locally and in Feifei's pinned runtime. The final batch-32
GPU pilot measured 23.11 training samples/second and 80.18 validation samples/second,
with 9.16 GB peak allocated memory. Prediction reload and next-update recovery
both had maximum absolute error 0.0. Native Qwen3.5 DeltaNet backward needs
gradient checkpointing at the tested batch sizes; uncheckpointed batches 16/32
exceeded memory and were stopped without launching production.

Final configuration: batch 32, gradient checkpointing, rank-32 broad LoRA,
LoRA and head/projector learning rates 1e-4, maximum 72,000 updates, full
validation every 20,000 updates and at epoch/end boundaries. The pilot projected
approximately 29.1 hours, or 34.9 hours including a 20% margin. Runtime also stops
training at 30 accumulated active hours before final validation/reload. The
[passing final pilot](evaluation/week4-b0-oft-pilot-v6.json) and
[formal training manifest](evaluation/week4-b0-oft-training-manifest-v1.json)
pin source revision `1ee18a31ff37c0de5c104a9b46f89a8c568da276`.

The formal run launched on Feifei on **2026-09-19 at 20:09 PDT**, as the enabled
host-owned service `eventground-b0-oft-v1.service`, with user lingering enabled.
Artifacts are written to
`/home/peimengsui/data/eventground-wam/libero-plus/derived/b0/week4-b0-oft-v1`.
Training completed successfully at **2026-09-21 01:17 PDT**, after 29.11 active
hours (29.13 hours from the first training event through completion). It reached
all 72,000 updates and 2,096,581 example presentations within the resource budget.
The service exited normally; its sole restart was the launch-time recovery test.

The [sealed launch audit](evaluation/week4-b0-oft-launch-v1.json) verifies a
controlled SIGTERM saved completed update **84**, exited with code 75, and caused
one automatic restart after 60 seconds. The new process restored update/cursor
84 and advanced to **100** with finite loss **0.773334**. The 835,580,261-byte
recovery checkpoint passed its SHA-256 check and contains model parameters,
optimizer/scheduler, RNG, sample cursor/hash chain, validation history, and active
time. Service enablement, lingering, matching source/recipe, and progress were
checked through a fresh SSH connection. SSH disconnects do not own job lifetime.
The host checkout remains pinned to the training revision; these later evidence
and documentation commits are local only.

### Completed training comparison

The [sealed final report](evaluation/week4-b0-oft-training-report-v1.json) selects
update **72,000**, the lowest full-validation objective among the five candidates.
The [completion verification](evaluation/week4-b0-oft-completion-v1.json) confirms
the manifest identity, planned update/sample totals, all validation event records,
and exact agreement between selected-checkpoint validation and full disk-reload
validation. Host SHA-256 checks matched both the selected policy and `B0.ckpt`:
`09ec7afad3f7435487334718cd32e310456d3fdea04c65bc888a097ceadaa584`.

Both runs use the same frozen 212,471 validation examples, action normalization,
and continuous L1 plus gripper BCE objective; validation batch grouping differs.

| Metric | Previous final B0 | New selected B0 | Change |
| --- | ---: | ---: | ---: |
| Validation objective | 0.273990 | 0.234067 | 14.57% lower |
| Continuous-action MAE | 0.141890 | 0.135359 | 4.60% lower |
| Decoded all-action MAE | 0.136448 | 0.127403 | 6.63% lower |
| Gripper accuracy | 94.8102% | 96.0168% | +1.2066 percentage points |

Validation objectives at updates 20k / 40k / 58,937 / 60k / 72k were
0.31967 / 0.27301 / 0.24163 / 0.24495 / 0.23407. The final result improves on the
brief 60k increase. The new run used 24.44% of the previous run's example
presentations, with a similar single-GPU time budget (29.11 versus about 31.2
hours). This establishes improved supervised validation performance and sample
efficiency for this run. Multiple model/optimizer choices changed together, so
the result does not isolate their individual effects.

The subsequent [clean LIBERO evaluation](WEEK4_OFT_CLEAN_COMPARISON.md) completed
on September 21: **65.25%** success versus previous B0 **68.25%** and reused
OpenVLA **96.75%** on the same 2,000 task/state/seed pairs. New B0 improves
Object and LIBERO-10, but Goal falls from 62.0% to 20.4%. The overall new-minus-old
paired interval is [−9.85, +4.25] pp; there is no demonstrated overall rollout
gain despite lower supervised validation loss. The clean acceptance gate remains
FAIL. The subsequent [dev-210 evaluation and Goal investigation](WEEK4_OFT_GOAL_DEV210_ANALYSIS.md)
are complete: 114/210 (54.29%) dev successes, up from 106/210, while the Goal
subset falls to 2/53. The investigation supports weaker instruction-to-goal
selection and separates gripper-loss improvements from continuous-control error.
No follow-up training or checkpoint reselection has been performed.
The next [controlled continuation experiment](WEEK4_CONTINUATION_EXPERIMENT.md)
is now frozen, including the resource cap, matched examples, measurement panels
and deterministic extension rule. This is a planning freeze, not a training launch.

### Exact exposure budget

One full training pass contains **1,715,954 examples** in **58,937 optimizer
updates** at batch size up to 32. Episode boundaries retain smaller batches.
The 72,000-update plan contains **2,096,581 example presentations**, equivalent
to **1.22181655 epochs**; it can stop earlier at the active-time limit. An example
is one observation with an eight-action target chunk, and repeated/overlapping
chunks are not new unique demonstrations. The recipe's `epochs: 5` provides up
to five deterministic data-order plans; the update/time limits govern this run.

The previous B0 trained for five complete passes, or 8,579,770 example
presentations. This experiment matches its GPU-time budget, while adapting many
more backbone layers and training 92,907,527 parameters. Its smaller exposure
must be considered when interpreting the eventual comparison.

### Public recipe exposure context

Optimizer steps are comparable only after accounting for global batch size and
dataset membership. These counts refer to example presentations, including
repeats, rather than unique demonstrations:

| Run | Optimizer updates | Global batch | Example presentations | Epochs |
| --- | ---: | ---: | ---: | ---: |
| Previous B0 | 563,555 | Up to 16 | 8,579,770 | 5 |
| New OFT-style B0 cap | 72,000 | Up to 32 | 2,096,581 | 1.222 |
| StarVLA published Qwen3-VL-4B OFT result | 30,000 | 128 | About 3,840,000 | 9.54 reported |
| OpenVLA-OFT Spatial/Object/Long, each suite separately | 150,000 | 64 | About 9,600,000 per policy | Depends on each filtered suite |
| OpenVLA-OFT Goal selected checkpoint | 50,000 | 64 | About 3,200,000 | Depends on the filtered suite |

[StarVLA's paper](https://arxiv.org/html/2604.05014#S5.SS1) reports the 30k-step,
9.54-epoch result using eight A100s and batch 16 per GPU. Its separate
[released 4-in-1 checkpoint](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1)
is at 50k steps (configured maximum 80k); the saved card does not establish the
historical GPU count, so its exact exposure is not inferred here. The paper's
default maximum of 100k is also different from the selected 30k result.

The [OpenVLA-OFT authors' LIBERO recipe](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)
specifies eight GPUs at batch eight, normally selecting 150k steps and 50k for
Goal. Example counts above are derived as updates times global batch. Its base
model has additional robot pretraining. These official recipe figures do not
establish the complete training history of our installed Sylvest mix-SFT reference.
