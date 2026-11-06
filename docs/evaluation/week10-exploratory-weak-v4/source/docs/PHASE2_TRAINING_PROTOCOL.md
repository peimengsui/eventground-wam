# Phase 2: two total action-training epochs

**Accepted:** September 24, 2026. The user approved two total action-training
epochs with unchanged effective batches and sampling. This supersedes the
closure-time proposal to initialize M1–M4 from trained B0 and add a matched
B0 continuation. Week 4 remains closed; all measured results remain unchanged.

**Future-module handoff, October 2:** the user authorized closing Week 6 and
moving on without short-horizon JEPA repair. The
[sealed module registry](../configs/training/phase2_future_modules_v1.json)
assigns accepted `C_pixel-v1` to M1/M3 and the original absolute JEPA endpoint,
`C_jepa-study-v1`, to M2/M4 by an explicit research-use exception. The original
JEPA gate remains FAIL. The endpoint weights, CPT exposure and all action
controls below remain unchanged. No residual-pilot weights, copy gate or extra
JEPA-only CPT are included. See the [closeout and interpretation limits](WEEK6_CLOSEOUT.md).

## Fixed comparison

| Policy | Shared policy initialization | Action-training budget |
| --- | --- | --- |
| B0 | Pinned base Qwen3.5-2B, fresh OFT head/projector/LoRA | Existing completed two-epoch result. |
| M1 / M2 | Same pinned base and fresh shared components; respective CPT future module | Two total epochs each. |
| M3 / M4 | Same pinned base and fresh shared components; same respective CPT module as M1 / M2 | Two total epochs each, independently trained; no continuation from trained M1/M2. |

The trained B0 is the evaluation control, not the starting weights of M1–M4.
Reuse its recipe and initialization seeds. Verify the shared components' initial
tensors before attaching independently seeded future modules. No additional
B0 continuation is required for this action-exposure comparison.

The sealed [control contract](../configs/training/phase2_two_epoch_v1.json)
pins the actual B0 recipe and epoch-plan digests:

- **1,706,022 action chunks/epoch**, **3,412,044 presentations total**.
- **58,566 optimizer updates/epoch**, **117,132 total**; fixed epoch-2 endpoint.
- Exactly the same ordered `(epoch, episode, anchor)` batches as B0, including
  128-update mixing windows, 15k stage fences, seed 17 and retained short batches.
- Effective batch is **up to 32**, with the exact existing short-batch sequence;
  mean 29.1299 examples/update. Do not pad, drop, merge or rebalance those batches.
- Same trajectory-v2 split, train-only normalizers, dual cameras/state,
  all-action L1, predict-eight/execute-eight, shared-parameter AdamW schedule
  and rank-32 LoRA. Method-specific losses, predictor parameters and coupling
  are the declared experimental differences.
- Future targets attach to those action anchors. Missing long-horizon targets
  near episode ends must not remove or replace action samples. Define common
  valid-target masks and future-loss normalization before training.

### Microbatches may differ; optimizer batches must match

If a model cannot fit a full logical batch in GPU memory, split that same batch
into microbatches and accumulate gradients. For a 32-example batch, four groups
of eight still produce **one optimizer update**. Weight each microbatch's mean
action loss by its sample count divided by the logical batch count. Clip gradients
and step the optimizer/scheduler only after the complete logical batch. Preserve
the exact short batches rather than assuming every update contains 32 examples.

Normalize masked future losses over the full logical batch's valid targets.
Batch-coupled terms such as JEPA variance regularization also require logical-
batch statistics; independent microbatch losses are not equivalent. Verify
loss/gradient agreement within declared numerical tolerance and process resume
in the pilot. Checkpoint at complete optimizer boundaries or save all partial
accumulation state. These are required implementation checks; the current B0
trainer has not been extended with accumulation in this update.

[PyTorch's accumulation guidance](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)
describes effective-batch updates; [activation checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
can trade additional computation for lower activation memory.

## Separate pretraining and EventGround budgets

Pixel/JEPA CPT has its own common future-pair manifest, effective batches and
exposure budget, to be frozen before CPT production. Those updates are reported
in addition to the two action epochs. Start from pinned base representations;
do not import B0-trained weights or a B0 teacher silently into CPT. The initial
resource plan is frozen feature extraction plus small trainable predictors;
shared-backbone CPT adaptation would need an explicit design/control revision.

For M4-EG, count all inherited action training: its M4 parent already consumed
the two epochs, so do not add another two action epochs. Event-label SFT/RLVR
are separately budgeted stages. The present main-matrix contract authorizes no
extra action updates. Effects of extra event training or backbone adaptation
need matched controls or explicitly limited claims. Equal action exposure is
not equal total compute or equal pretraining.

## Feifei feasibility assessment

Read-only host inspection on September 24 found **RTX 5090, 32,607 MiB VRAM**,
**122 GiB RAM (117 GiB available)**, **24 logical CPUs** and **1.3 TiB free disk**.
The completed production B0 logs record a maximum **8.53 GiB PyTorch-allocated
GPU memory**. That excludes driver/context and allocator-reserved overhead; it
is not a full-device peak measurement for the proposed models.

| Work | Assessment before architecture pilots |
| --- | --- |
| Pixel / JEPA CPT | Expected to fit with frozen feature extraction, bounded caches, small predictors and deterministic 128–160 pixel targets. |
| M1 / M2 auxiliary policies | Expected to fit with existing BF16/LoRA/checkpointing and, if needed, accumulation preserving logical batches. |
| M3 / M4 coupled policies | Plausible on 32 GB with bounded future-token counts and small coupling modules; profile full forward/backward and rollout latency first. |
| EventGround SFT | Plausible with short structured outputs and limited trainable components. |
| EventGround RLVR | Most uncertain; generation and policy/reference storage require a pilot. Sequential generation, adapter sharing or offloading are possible design choices, not measured implementations. |

Use **one substantial GPU job at a time**, persistent host-owned services and
the existing recovery discipline. Frozen teacher features can be cached only
after measuring storage demand. Multi-model caches, recovery checkpoints and
full-10K recordings need a projected disk budget; current free space is not a
guarantee that all future artifacts fit indefinitely.

The default plan is Feifei; H100 rental is optional acceleration, not an assumed
requirement. No future-model runtime or memory guarantee is established yet.
The major expected cost is sequential wall time: new modules add work beyond
B0, and CPT plus rollout evaluation add separate stages. Profile each model
before production; do not change logical batches or sampling to meet a resource
limit silently.

## Provenance of this revision

The original [Week 4 completion record](evaluation/week4-completion-v1.json)
remains byte-for-byte unchanged. Its closure-time document versions are retained
under `evaluation/week4-completion-v1-snapshots/`; their SHA-256 values still
match that record. The new control contract records the snapshot mapping and
supersedes only the forward training/initialization plan. No Week 5 training
or new policy rollout was launched for this update.

The October 2 module-handoff revision preserves this protocol's earlier document
bytes in `evaluation/week6-progression-v1/before/docs/PHASE2_TRAINING_PROTOCOL.md`.
The original `phase2_two_epoch_v1.json` remains byte-for-byte unchanged. The new
module registry supplies a scoped prerequisite exception, not a new training
budget. The [progression ledger](evaluation/week6-progression-v1/completion.json)
records source hashes and preservation checks. No policy training was launched
as part of this handoff.
