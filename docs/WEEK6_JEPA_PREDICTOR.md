# Week 6 Step 3 — matched JEPA predictor

**Status: complete.** The matched predictor, training-only v3 diagnostic and
independent audit pass, together with 458 local tests and new-code lint.
The accepted implementation check uses 1,600 real-batch updates; earlier failures
are retained. Production CPT and held-out predictor acceptance remain later
steps. See the [completion ledger](evaluation/week6-step3-completion-v1.json).

## Architecture and matching

The [predictor](../src/eventground_wam/jepa_predictor.py) executes the accepted
Pixel future core without modifying its frozen implementation. It removes the
RGB decoder and uses the parent's decode hook as an identity to obtain future
tokens. Its public output exposes normalized visual features and future tokens.
No RGB reconstruction is performed by JEPA.

Current conditioning is identical to Pixel: frozen base-Qwen current-image and
instruction hidden states, current 128x128 box-reduced dual-camera images,
train-only normalized state, ordered actions `[t,t+h)`, and h=2/4/8/16/32.
The inherited masks exclude padded features and action suffixes. The predictor
receives no future RGB, target features, future state or sample identifiers.

| Component | Trainable parameters |
| --- | ---: |
| Exact Pixel future core and input projections | 4,238,592 |
| Shared affine 256-to-2048 prediction projection | 526,336 |
| Total JEPA module | 4,764,928 |
| Frozen base and target teacher | 0 |

The common core uses Pixel's seed 20260925. Its initial tensors and future-token
outputs must match Pixel exactly. The separate projection uses seed 20260926.
No trained Pixel/B0 weights are used. JEPA starts with a fresh projection rather
than Pixel's exact image-copy initialization; this declared difference is tested
against both its own initial error and the measured current-feature copy error.

The interface stays `[B,128,256]`: 64 front tokens followed by 64 wrist tokens,
both 8x8 row-major. The new head maps each token to 2048 channels, then applies
differentiable FP32 non-affine LayerNorm (epsilon 1e-5) and unit L2 (epsilon 1e-6).
The geometry matches [the accepted target contract](WEEK6_JEPA_TARGETS.md), while
prediction gradients reach the core and, if enabled later, shared input features.
The teacher's normalization remains detached and its single-image execution
path is unchanged.

The [data wrapper](../src/eventground_wam/jepa_data.py) reuses the verified Pixel
pair reader, source checks and normalizers. It carries native future RGB in a
separate target field and retains the original conditioning interface.

## Diagnostic loss and variance rule

The [sealed v3 recipe](../configs/training/week6_jepa_predictor_v3.json) declares
`loss = mean cosine error + 10 * variance penalty` for this bounded check.
These coefficients do not freeze the production recipe.

For variance, first average feature fields over horizons sharing one
`(episode, anchor)`. Then compute population standard deviation across distinct
anchors at each fixed camera/spatial/channel coordinate, scaling unit vectors by
sqrt(2048). The penalty averages `relu(0.5 * teacher_std - prediction_std)`;
both standard deviations use epsilon 1e-4 inside the square root. The teacher
side is detached. Spatial texture and repeated horizons cannot supply extra
independent observations. A single distinct anchor contributes zero penalty and
reports its support count explicitly.

The complete logical batch must supply these statistics. Averaging separate
microbatch variance penalties is not equivalent. Step 4 must verify the
accumulation strategy and freeze the production short-batch policy and weights.

## Bounded verification

The [diagnostic runner](../scripts/check_jepa_predictor.py) pins all inputs and
uses the locked Feifei CUDA runtime. Selection, update counts and thresholds
are declared before the real run:

- Four controlled examples have identical current images, instruction-feature
  placeholders and state. At a given horizon their action multisets match;
  order and horizon determine the target latent fields. After 1,600 AdamW updates
  at 1e-4, require 80% initial cosine-error reduction, reversal worsening cosine
  error by more than 0.02, and explicit-horizon removal changing predictions by
  more than 1e-4 mean absolute feature value. A failure stops before real data.
- Use exactly Pixel Step 3's four training tasks/episodes and common anchors,
  each with all five horizons: 20 pairs and four distinct real anchors. Train a
  fresh JEPA model for 1,600 updates with the same optimizer settings. Require at
  least 20% cosine-error reduction against both initialization and copy-latent,
  finite active gradients in every conditioning branch, action/order/horizon/
  Qwen-feature sensitivity above 1e-5, and exact GPU checkpoint reload.
- Verify unchanged frozen base and vision tensor hashes, accepted teacher
  implementation hashes, exact repeated extraction and native/box target-camera
  alignment. Save the feature bank, source identities and diagnostic checkpoint.
- The [independent verifier](../scripts/verify_jepa_predictor.py) reconstructs
  all 20 source samples, checks exact Pixel sample identities, replays the model
  on CPU, independently computes cosine and grouped variance in NumPy float64,
  and freshly encodes all 20 target image pairs and four current pairs with
  the frozen teacher. No validation or test payloads are used.

Unit tests additionally check exact matching of the core, gradients through
normalization, target detachment, padded-value poisoning, grouped variance,
invalid predictions, target isolation and corrupt checkpoint rejection.

## Verified results

The [v3 report](evaluation/week6-jepa-predictor-v3/report.json) passes every
prospective implementation criterion on the exact same 20 pairs from four
training episodes used for Pixel Step 3. Every run starts with fresh weights.

| Real training-batch measurement | Cosine error |
| --- | ---: |
| Fresh JEPA initialization | 0.983709 |
| Copy current teacher features | 0.231251 |
| JEPA after 1,600 updates | **0.132110** |
| Reverse the valid action prefix | 0.162740 |
| Zero actions | 0.197836 |
| Zero explicit horizon | 0.201857 |
| Zero current Qwen features | 0.750132 |

Final cosine error is 86.57% below initialization and 42.87% below copy.
Action, action-order, horizon and feature ablations all change predictions above
the declared threshold. Every conditioning branch receives finite active
gradients. Initial Pixel/JEPA future tokens, GPU checkpoint reload, repeated
feature extraction and frozen-encoder before/after hashes match exactly.

The controlled task finishes at cosine error 0.00001689; reversing action order
raises it to 0.588840. Removing the explicit horizon changes predicted features
by 0.003373 mean absolute value. All three controlled criteria pass.

The [independent audit](evaluation/week6-jepa-predictor-v3/independent-verification.json)
reconstructs all source conditions and confirms exact Pixel sample identities.
Fresh teacher encoding of all 20 target pairs and four current pairs is exact.
CPU replay differs from CUDA by at most 1.91e-6 in output features and 2.46e-5
in future tokens, within the predeclared limits; repeated CPU output is exact.
Independent NumPy float64 loss and ablation arithmetic differs by at most
1.96e-7. The bank and checkpoint hashes are independently verified on Feifei.

All four [diagnostic panels](evaluation/week6-jepa-predictor-v3/visual-review.json)
were inspected: current/future camera alignment and horizon labels are consistent,
and the common-scale latent-error maps show lower error with remaining residuals.
These panels display source images and feature errors; JEPA does not generate RGB.

The v3 diagnostic takes 44.42 seconds internally (46.23 seconds service wall time)
and reports 4.2G peak host memory. Predictor-phase Torch allocation peaks at
0.587 GiB (1.492 GiB reserved); this excludes the earlier base-model extraction
peak and is not a production resource estimate. The separate audit takes
6.98 seconds service wall time with 1.7G peak host memory.

## Scope and handoff

The original v1 controlled run failed action-order learning after 400 updates
with variance weight 0.1. Two fixed 1,600-update investigations at learning rates
1e-4 and 3e-4 also failed. A second declared investigation compared sparse targets
with weights 1 and 10, and a dense rotation with weight 0.1. Only weight 10 passed
the original controlled task: final cosine error 0.0000169 and reversed-action
error 0.588840. The v2 recipe retains the original sparse targets, architecture,
initialization, learning rate, real sample selection and every pass threshold;
it changes the diagnostic variance weight and controlled update count. All
failures and exploratory runs are retained under
`docs/evaluation/week6-jepa-predictor-v1/`. These investigations used synthetic
data exclusively. Production coefficients still require Step 4.

The retained [v2 real report](evaluation/week6-jepa-predictor-v2/report.json)
measures final cosine error 0.343687 versus 0.231251 for copy-latent, despite a
65.06% improvement from initialization. The core matches Pixel exactly, frozen
encoders remain unchanged, all conditioning branches receive finite active
gradients, action/order/horizon/features affect predictions, and GPU reload is
exact. Local CPU replay and NumPy arithmetic independently reproduce the failed
result. The [v3 recipe](../configs/training/week6_jepa_predictor_v3.json) fixed
1,600 real updates before execution; all samples, model settings and pass
thresholds remained unchanged. It passed without further revision.

Across v1, both synthetic investigations, v2 and v3, diagnostic exposure totals
11,600 synthetic updates / 46,400 synthetic presentations and 1,900 real updates /
38,000 real pair presentations. The real runs use the same 20 unique pairs from
four training episodes (300 v2 updates plus 1,600 fresh v3 updates). No validation
or test payloads were read. These diagnostic budgets differ from Pixel's; only
the core and sample identities are matched here. Production will still use the
shared fixed pair exposure and update order, with separate compute accounting.

The checkpoint is named `diagnostic-jepa.pt` and marked
`step3-diagnostic-only-not-C_jepa`. It must be discarded as an initializer for
production. Its weight snapshot does not include optimizer/RNG/cursor recovery.
The [Step 2 held-out gate](WEEK6_JEPA_BASELINES.md) remains unassessed.

Successful training-batch overfit establishes implementation feasibility only.
It does not establish held-out temporal prediction, causal dynamics, instruction
generalization or policy success. Step 4 will freeze the production loss,
schedule, logical-batch implementation, target caching decision and recovery
checks before a fresh production run.
