# Week 8 M3/M4 design freeze

Frozen October 5, 2026 (Pacific), before any M1 or M2 dev rollout results.
The machine-readable authority is
[week8_coupled_design_v1.json](../configs/training/week8_coupled_design_v1.json).
**Scientific design frozen; implementation and hardware qualification pending.**
This does not authorize production training or provision an A100.

## Question and matched arms

Does making predicted futures available to the action head improve the policy,
beyond learning from the same future-prediction loss during training?

| Arm | Starting future module | Training supervision | Deployed future context |
| --- | --- | --- | --- |
| M1 | Original C_pixel-v1 | Pixel L1 + 0.1 SSIM | None |
| M2 | Original C_jepa-study-v1 | JEPA cosine + teacher-relative variance | None |
| M3 | Original C_pixel-v1 | Same as M1 | Pixel-supervised core tokens |
| M4 | Original C_jepa-study-v1 | Same as M2 | JEPA-supervised core tokens |

M3/M4 start fresh from the pinned Qwen base, fresh OFT head/state projector/LoRA,
and original CPT modules. They do not start from trained M1/M2 or B0. Phase 2's
two-total-epoch controls supersede the older Week 4 continuation wording.
M3−M1 and M4−M2 measure the complete coupling treatment, including its new
parameters and compute. M4−M3 compares the two supervised representations
under the same coupling. This is not a parameter-matched proof that future
semantics alone caused any gain. Original JEPA's gate remains **FAIL**; its
research-use exception and lack of latent-quality certification remain visible.

## One proposal, one future, one refinement

1. Run the existing OFT backbone once on instruction, current front/wrist RGB,
   current state and eight action placeholders. Capture the final prompt features
   and the eight 2,048-dimensional action-slot hidden states, as in Week 7.
2. Apply the existing OFT head to produce an initial eight-action proposal.
   Decode its gripper at zero to exactly −1/+1; invert the frozen action affine
   normalizer, clip continuous simulator actions to [−1,1], then re-normalize.
   This matches the executable proposal; simply clipping normalized values would
   not. Detach this proposal before future conditioning. Pad steps 8–31 with zeros.
3. Predict **horizon 8** from the current features, state, two current images and
   proposal. Use all **128 × 256** future-core tokens (64 per camera). Do this for
   every action anchor, even where a demonstrated horizon-8 target is unavailable.
   Target availability must never enter the deployed action path.
4. Refine the action hidden states with one residual cross-attention block. The
   eight hidden states are queries; the 128 future tokens are keys/values.
   Reuse the same OFT head on the refined hidden states to produce the final
   eight actions. Execute all eight at 20 Hz under the unchanged action decoder.

There is one backbone call, one candidate-conditioned future-core call and two
calls to the **same** action head per inference query. No candidate search,
recursion, future RGB re-encoding or horizon selection is included. M3's image
decoder and M4's 2,048-dimensional JEPA target projection are training-only;
both action paths consume their supervised core's 256-dimensional hidden tokens.
Token-only inference must be numerically equal to the corresponding existing
predictor's `future_tokens`; skipping the output decoder must not change them.

### Exact coupling block

FP32 LayerNorm(2048, eps=1e-5) on queries and LayerNorm(256, eps=1e-5) on future
tokens, each with affine parameters. Independent biased linear projections:
Q: 2048→256, K: 256→256, V: 256→256. Eight heads of width 32; scaled dot-product
attention, no dropout, all future tokens visible. Concatenate heads and apply
a biased 256→2048 projection, then add to the original action hidden states.
No extra feed-forward block or learned gate. Total new trainable parameters:
**1,187,072** in either arm. Initialize LayerNorm scales to one/biases to zero;
Q/K/V use PyTorch Linear defaults in that order within a forked RNG seed
**20261005**; initialize the final projection's weight and bias to zero.
Adding this module must not change any shared initial tensor or training RNG.

Zero initialization makes the initial final action output equal to the original
uncoupled output. On the first backward pass, action gradients reach the output
projection but not its upstream K/V/core through that zero projection. Test
upstream action-gradient reachability after a controlled nonzero projection
or an optimizer step; do not mistake the expected initial zeros for a failure.

## Training and information boundaries

Optimize final-action L1 plus the inherited future loss. M3 keeps weight
**4.836645309900765**; M4 keeps **1.5140534185764003**, including cosine + 10×
teacher-relative variance, epsilon 1e-4 and variance floor ratio 0.5. The JEPA
teacher remains frozen. No additional loss on the initial proposal is introduced.

The auxiliary branch still uses actual intervening demonstrated actions and
available targets at horizons 2/4/8/16/32, exactly as M1/M2. It shares the future
predictor and current backbone features with the candidate branch. It is a
separate teacher-forced call: demonstrated actions, future images, teacher
features and target masks cannot affect the candidate or refined action output.
Do not supervise a counterfactual candidate-conditioned prediction against a
future generated by different demonstrated actions.

Action gradients reach the final head, attention, future core and current
backbone/state features. The numerical candidate-action route is detached;
the current-feature route remains differentiable. Auxiliary gradients retain
the Week 7 route. Ground-truth targets and frozen base/teacher have no gradients.

Preserve both exact ordered batch plans, short batches, training seed 17, fresh
head seed 11/state seed 7, rank-32 alpha-16 broad LoRA, two epochs, **117,132**
optimizer updates and **3,412,044** action presentations. Preserve all **16,259,770**
valid future pairs. Missing future targets never remove action anchors. AdamW
remains lr 1e-4, betas (0.9,0.95), epsilon 1e-8, weight decay 0.01, global clip 1,
2% warmup and cosine floor 0.1. Add a coupling parameter group with the same
settings; clip and step once per original logical batch. Policy precision remains
inherited BF16, future/coupling computations and loss accumulation FP32.

## Implementation and admission before training

Implement in new modules/wrappers; preserve the accepted Week 7 source. Seal
numerical recipes, parameter inventories, source packages and runtime identities
only after these symmetric checks pass for both arms:

- Exact shared initialization, original CPT checksums, zero-coupling action
  equality, token-only predictor parity, train/export/reload inference equality.
- Counterfactual target/action/mask perturbations leave policy outputs unchanged;
  perturbing predicted future tokens changes the refined output once coupling is
  nonzero. Verify the intended gradients, fixed teacher and detached proposal.
- Preserve whole-logical-batch losses and gradients, including JEPA variance;
  inherit Week 7 numerical tolerances. Test missing horizons and short batches.
- Checkpoint/resume must match uninterrupted model, optimizer, scheduler, RNG,
  sample/future chains and counters, including damaged-latest fallback.
- Train-only timing on the first 128 exact mixed batches plus inherited edge
  batches, with cold caches; no validation/rollout-based hyperparameter choice.
  Retain failed attempts. No pilot weights enter production.

Use the inherited pilot ceiling of 3 active hours, conservative projection
1.2×max(mean, p95) plus checkpoint allowance against 168 hours, production cap
192 active hours, host memory 28 GiB, GPU memory below 90%, output 100 GiB and
free-space reserve 50 GiB. Qualify the actual Feifei and prospective A100 runtimes
separately. Predictor microbatch begins at the inherited 10; any smaller value
requires whole-batch equivalence verification and a recorded execution recipe.
If a limit fails, preserve the failure and propose a separately versioned budget
or design revision; do not silently borrow M2's Lambda 480-hour exception.

Target placement: M3 on Feifei after M1 evaluation releases it; M4 on a separately
qualified A100. GPU type, region and launch time are pending availability. Keep
the existing Lambda A10 dedicated to M2. Cross-hardware numerical equality is
not assumed; verify tolerances and same-device recovery. Training cost, memory,
parameters and inference latency must be reported alongside policy results.

## Evaluation and freeze discipline

Fixed epoch 2 and the same paired dev-210, 20 Hz, predict-eight/execute-eight,
normalizers and canonical input audit as Week 7. Evaluate all arms on the same
Feifei runtime if feasible, independent of training host. If any input pair
mismatches, retain all 210 tasks and label that entire contrast descriptive.
Do not discard mismatches or retroactively pass B0's prior rendering decision.
Use the inherited 10,000-draw PCG64 seed-7 paired task bootstrap and common
resampled task indices across arms, plus descriptive Wilson intervals and
suite/category/difficulty breakdowns. Test and larger confirmation panels stay
locked. Optional later ablations require their own prospective plan.

This version cannot be retuned using M1/M2 success, failures or comparisons.
Bug fixes preserving this contract may be documented and reverified. A scientific
change requires a new version with its reason and the outcomes already observed;
the old freeze and failed qualifications remain part of the evidence.
