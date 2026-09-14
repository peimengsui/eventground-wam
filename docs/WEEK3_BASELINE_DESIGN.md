# Week 3 shared B0 design

## Decision

The shared visual-language backbone for B0 and every later M1–M4/M4-EG model is
`Qwen/Qwen3.5-2B`. This choice is locked at the model-family level. Before the
first download, implementation must record an immutable Hugging Face revision
and a complete file manifest so all experiments resolve the same weights.

Qwen-VLA, Qwen-RobotManip, and other pretrained action-policy checkpoints are
outside the project scope. They must not be used as initialization, fallback
models, action heads, implementation dependencies, or members of the primary
experiment matrix. A compatibility failure with Qwen3.5-2B is a documented
blocker requiring an explicit design decision; it does not permit silently
substituting a different backbone.

## B0 contract

```text
instruction + front RGB + wrist RGB + 8D proprioception
                              |
                     Qwen3.5-2B backbone
                              |
                 proprioception/projector fusion
                              |
             deterministic action-query Transformer
                         (L1 objective)
                              |
                       8 x 7 action chunk
```

- Inputs are the synchronized 20 Hz front and wrist streams, task instruction,
  and eight-dimensional proprioceptive state from the frozen Week 2 dataset.
- Outputs are eight steps of seven-dimensional LIBERO delta control. Action
  statistics come from the training split only, and the gripper convention is
  tested explicitly in both normalization directions.
- The initial B0 freezes the vision encoder and base transformer weights. The
  trainable path is limited to the proprioception/projector fusion, upper-layer
  LoRA selected during implementation, and the action decoder.
- B0 has no future-prediction target, world-model path, event conditioning,
  event-adaptive horizon, generative action objective, external action
  pretraining, or simulator state input.
- Camera orientation, resize, token selection, and normalization are locked by
  tests before training; no preprocessing transform may remain implicit.

The canonical action objective for B0 and the M1–M4/M4-EG matrix is L1
regression over the normalized action chunk. The decoder uses learned action
queries and exposes a conditioning-memory interface so M3/M4 can later
cross-attend predicted future tokens without changing the action contract.
Flow matching and other generative action objectives are deferred beyond the
required 12-week study and do not gate any planned experiment.

The accepted implementation revision keeps L1 as the sole continuous-action
objective but treats the discrete gripper sign as unit-weight binary
classification. Later matched policies inherit continuous L1 plus gripper BCE;
this is not a generative action objective and does not reopen flow matching.

## Runtime isolation and provenance

Qwen3.5 support belongs in a dedicated `environments/b0` runtime. It must not
upgrade or otherwise change the pinned Week 1 reference-policy environment.
The runtime lock records Python, PyTorch/CUDA, Transformers, PEFT, attention
backend, and Qwen processing dependencies. Model provenance records the repo ID,
immutable revision, manifest hash, license, and processor configuration.

Before model work proceeds, a compatibility smoke test must demonstrate:

1. a dual-image plus instruction forward pass;
2. stable extraction of the hidden representation consumed by the action head;
3. gradients only through the declared projector, LoRA, and action-decoder
   parameters;
4. BF16 forward/backward and checkpoint reload on the RTX 5090; and
5. enough memory headroom for a closed-loop simulator process.

## Implementation sequence

1. Pin the Qwen3.5-2B model revision and create the isolated runtime lock.
2. Implement and test train-split-only action normalization and inversion.
3. Stream deterministic dual-camera/action chunks from the frozen Week 2 split.
4. Implement the Qwen conditioning adapter, learned action-query decoder, and
   normalized-action L1 objective.
5. Overfit one fixed batch, then a fixed ten-demonstration cohort.
6. Integrate the existing evaluator/recorder and run a closed-loop training-task
   rollout.
7. Record loss curves, decoded-action error, latency, peak VRAM, configuration,
   seeds, source revision, and checkpoint hashes.

## Locked acceptance criteria

All overfit metrics use a predeclared, fixed diagnostic set of samples.
Training loss by itself is not an acceptance metric.

### Data and preprocessing

- Action statistics are computed only from the 11,521 episodes assigned to the
  frozen training split. Their artifact records the dataset-freeze manifest and
  split-assignment hashes.
- Normalize-then-invert reproduces every finite test action within an absolute
  tolerance of `1e-6`. The gripper tests cover both `-1 = open` and `+1 = close`
  without changing polarity.
- Each example has front RGB and wrist RGB from the same anchor frame, one 8D
  proprioceptive state, and an in-episode `8 x 7` action chunk. Incomplete tail
  chunks are excluded rather than padded.
- Camera orientation, RGB conversion, resize/crop, prompt construction, and
  Qwen token selection are explicit configuration fields covered by tests.
- Two loader runs with the same configuration and seed produce the same ordered
  sample-identity digest.

### Model and runtime

- Model weights are not downloaded until the immutable Qwen revision and
  complete remote manifest have been recorded. The runtime lock and model
  manifest identify their own hashes.
- A dual-image plus instruction BF16 forward/backward pass succeeds on the RTX
  5090, and the conditioning tensor consumed by the action head has the locked
  shape and token-selection semantics.
- The trainable-parameter set exactly matches an allowlist containing only the
  proprioception/projector fusion, selected upper-layer LoRA adapters, and
  action decoder. The vision encoder and base transformer receive no gradients.
- Given a fixed input in deterministic evaluation mode, checkpoint reload
  changes no decoded normalized action by more than `1e-3` absolute.
- The model and one closed-loop LIBERO simulator process run concurrently
  without out-of-memory failure. Peak allocated and reserved VRAM are recorded;
  Week 3 records latency rather than imposing a latency target.

### Overfit and rollout gates

- On one fixed batch, diagnostic action L1 loss falls to at most 5% of its
  initial value and decoded normalized-action MAE is at most `0.05`.
- On the fixed ten-demonstration cohort, diagnostic action L1 loss falls to at
  most 20% of its initial value, decoded normalized-action MAE is at most
  `0.10`, and gripper sign accuracy is 100% for targets whose absolute gripper
  command is at least `0.5`.
- At least one predeclared training task represented in the ten-demonstration
  cohort completes successfully in closed loop. Its HDF5 episode validates
  against the shared recorder contract.
- The complete local unit suite and lint check pass from a clean source revision.

Failure of any criterion blocks Week 4. Changing a threshold after observing a
run requires a documented design revision rather than editing the original run
record.

## Deliverables and gate

Week 3 delivers trainable B0 code, unit and integration tests, a pinned runtime
and model manifest, single-batch and ten-demonstration overfit curves, profiling
evidence, and the first closed-loop rollout artifacts.

Week 4 training is blocked until the fixed small cohort clearly overfits, action
decode/normalization checks pass, save/reload reproduces outputs within the
declared tolerance, and the policy completes at least one successful closed-loop
training-task rollout.

## Completion record and accepted revision

**Status:** Week 3 is complete as of 2026-09-13 and is frozen by the
`week3-complete-v1` tag. The detailed implementation and artifact ledger is in
[`WEEK3_STATUS.md`](WEEK3_STATUS.md); the aggregate sealed decision is
[`evaluation/week3-completion-v1.json`](evaluation/week3-completion-v1.json).

All seven implementation steps completed. The final system uses the pinned
Qwen3.5-2B multimodal core, default rank-16 upper-layer LoRA, explicit
proprioception fusion, four-layer deterministic action-query decoder, and the
future-memory interface required by later M3/M4 models. Data comes only from the
frozen Week 2 split. Continuous action dimensions use train-normalized L1;
following the versioned overfit experiment, the binary gripper dimension uses
unit-weight BCE and decodes to exact `-1/+1` commands.

The following post-observation acceptance revision is deliberately additive;
it does not edit the locked criterion above:

- The original ten-demonstration 100% gripper sign gate failed. The final
  reload-verified checkpoint reached 11,580/11,632 correct signs
  (99.552957%); the best diagnostic reached 11,581 (99.561554%).
- The reviewer accepted the final checkpoint for Week 3 after an audit reduced
  52 wrong slots to 32 unique source frames and placed every error within four
  frames of a true gripper-command transition. There were no non-binary labels,
  overlapping-chunk contradictions, or duplicate inputs with conflicting
  targets.
- This exception applies only to the Week 3 memorization diagnostic. It neither
  changes future evaluation metrics nor establishes rollout or generalization
  quality. The stopped weight-10 experiment is excluded from evidence.

Every other gate passed without a threshold revision. The one-batch loss ratio
was 3.803% with `0.048459` decoded normalized-action MAE. The final cohort
hybrid-objective ratio was `0.126083` with `0.027970` decoded MAE. A predeclared
ten-task closed-loop sweep found one success; that task repeated at 118 policy
steps through the shared recorder, and both the accepted and profiled HDF5
bundles passed deep provenance/action validation.

The concurrent RTX 5090 run peaked at 5,858,394,112 process-attributed GPU bytes
and retained at least 28,073,525,248 globally free bytes. Torch reported
4,591,992,832 allocated and 4,649,385,984 reserved bytes. Median policy-query
latency was 32.94 ms over all 15 queries and 32.89 ms excluding the cold first
query; the design imposed no latency target.

The original rollout gate is therefore satisfied, and Week 4 is unblocked.
The 1/10 cohort sweep and failed perturbed candidate remain explicit risk
signals: Week 4 owns full B0 training, clean and `dev-210` evaluation, and
seven-category failure analysis before the shared policy can be treated as a
reliable initialization.
