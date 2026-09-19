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

Implementation and CPU checks in progress. Hardware pilot and production launch
receipts will record the final recipe, source revision, measurements, and service.
