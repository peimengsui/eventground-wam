# Week 4: inexpensive diagnostics before further training

**Completed:** September 21, 2026 PDT, on Feifei.  
**Scope:** training-source audit and frozen parent gradient panel. Zero optimizer
updates; no training launched or recipe changes made.

## Executive summary

1. **Validation is not independent at the demonstration-trajectory level.**
   1,393 of 1,396 validation episodes share their complete state/action sequence
   and instruction with training. Direct checks confirm different camera images
   can accompany these identical numeric sequences. Lower validation loss does
   not establish better generalization to unseen demonstrations. Clean LIBERO
   and dev-210 rollout outcomes remain separate, unchanged observations.
2. **Gripper BCE has larger backbone gradients than motion L1 at the new B0
   endpoint.** The median norm ratio is **4.65×**, exceeding 2× on **34/40** fixed
   batches and triggering the predeclared imbalance flag. This supports testing
   BCE weighting, but does not establish that BCE caused the Goal regression.
3. **Zero idle actions** were found under the criterion below. Missing no-op
   filtering is an unlikely explanation. All 40 tasks are present, with unequal
   training exposure.

The [verification receipt](evaluation/week4-inexpensive-diagnostics-verification-v1.json)
binds all three reports, direct duplicate checks, executed source hashes,
runtime versions and independently recomputed counts.

## Separate motion and gripper gradients

Used new B0 update **72,000**, policy SHA-256
`09ec7afad3f7435487334718cd32e310456d3fdea04c65bc888a097ceadaa584`,
with the unchanged [gradient-640 panel](../evaluation/splits/week4-oft-continuation-v1/gradient-640.json):
40 batches of 16 examples, one episode per task. Separately differentiated mean
six-dimensional motion L1 and unit-weight mean gripper BCE, in the parent's
training mode with native gradient checkpointing. Norms and dot products use
FP64 accumulation before clipping.

| Trainable component | Median BCE/motion gradient norm | Batches with ratio ≥2 |
| --- | ---: | ---: |
| Backbone LoRA, combined | **4.650** | **34/40** |
| Language LoRA | 4.644 | 34/40 |
| Vision LoRA | 4.167 | 34/40 |
| State projector | 3.729 | 35/40 |
| Action head | 0.158 | 0/40 |
| All trainable parameters | 1.601 | 14/40 |

The frozen flag requires a backbone ratio ≥2 on at least 30/40 batches: **met**.
Suite median backbone ratios: Spatial **4.55**, Object **7.37**, Goal **3.45**,
Long **5.11**. The effect is not confined to Goal. Median backbone gradient
cosine is **0.079**; 10/40 cosines are negative. The objectives are generally
weakly aligned rather than consistently opposed.

These are endpoint Euclidean gradient norms, not accumulated training gradients
or Adam-preconditioned update sizes. A whole-model norm obscures the different
balance in the shared backbone and action head. Also, **39/40** panel episodes
share their numeric trajectory with training: valid gradient diagnostics, but
not evidence of held-out generalization.

No optimizer was constructed. Full parameters and persistent buffers, checkpoint
files, RNG and module modes passed preservation checks; `.grad` fields remained
empty. A first-batch direct combined backward check had **2.932% relative L2
difference** from adding the separate gradients, retained explicitly as BF16
backward rounding rather than exact equality. A synthetic FP32 test also checks
gradient accounting and superposition.

[Full gradient results](evaluation/week4-oft-parent-gradient-diagnostic-v1.json).

## Underlying trajectories cross train and validation

Scanned all training and validation action/state arrays from the pinned Sylvest
snapshot. **Test numeric arrays were not read**; test metadata counts alone were
included. Identity requires equal canonical instruction, array shapes and all
float32 state/action values over the entire episode. The action-only duplicate
check produces the same counts.

| Quantity | Result |
| --- | ---: |
| Training episodes | 11,521 |
| Distinct training numeric trajectories | **1,680** |
| Excess repeats of those trajectories | 9,841 |
| Maximum training multiplicity | 10 |
| Training frames / valid eight-action chunks | 1,796,601 / 1,715,954 |
| Frames / chunks counting each distinct trajectory once | 270,306 / 258,546 |
| Validation episodes matching training | **1,393/1,396 (99.79%)** |
| Validation chunks on matching trajectories | **212,000/212,471 (99.78%)** |
| Numeric-trajectory groups crossing the split | 673 |

Independently reloaded five deterministic train/validation pairs from original
parquet files. Every state/action value matched exactly, with maximum absolute
error zero. First front-camera images differed in all five pairs; first wrist
images differed in four. This establishes **trajectory-level overlap**, not
identical complete multimodal examples or useless visual variants.

The previous split groups used task, initial state and initial RGB from both
cameras. Visual changes can place variants of the same numeric demonstration in
different groups. The historical zero-overlap check was true for that surrogate;
it was insufficient to establish independent underlying demonstrations. This
qualifies the Week 2 split claim and both B0 runs' validation interpretation.
It does not establish overlap with simulator rollout initial states, which was
not tested here.

Regrouping validation now cannot make the already-trained parent unsee these
trajectories. A future fresh training run needs a newly versioned split grouped
by original demonstration identity, or full numeric trajectory when provenance
is unavailable, with near-duplicate checks as well.

## Data composition, filtering and balance

All **40** source instructions exactly match the canonical clean task strings.
Training exposure follows available demonstrations and episode length, rather
than equal task sampling:

| Suite | Training episodes | Frames | Valid chunks | Chunk share |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 3,012 | 370,564 | 349,480 | 20.37% |
| Object | 3,242 | 479,711 | 457,017 | 26.63% |
| Goal | 3,020 | 352,067 | 330,927 | 19.29% |
| Long | 2,247 | 594,259 | 578,530 | 33.71% |

Per-task episode counts range from **118 to 416**. Goal is not omitted.

The OpenVLA-style idle criterion is `norm(action[:6]) < 1e-4` with unchanged
gripper command versus the previous source frame; the first frame tests motion
only. It identifies **zero** training or validation frames for removal. The 28
stationary-motion training frames and seven validation frames are gripper
transitions and should be retained. All numeric values are finite, gripper
commands are exactly ±1, and no raw motion component exceeds [-1, 1]. No data
was filtered or rewritten.

### Perturbation categories remain provisional

Original metadata lacks category/severity, original demo IDs, simulator
initial-state IDs and success flags. A
[community category supplement](https://huggingface.co/datasets/katsukiono/libero-plus-episode-categories/tree/64d4c3244865528f8842e4a2ccc3e7a12fc51530)
labels the episodes as follows:

| Supplement label | All episodes | Training episodes |
| --- | ---: | ---: |
| `env` | 2,876 | 2,270 |
| `lighting` | 2,876 | 2,330 |
| `language` | 2,872 | 2,296 |
| `camera_pose` | 2,871 | 2,326 |
| `sensor_noise` | 2,852 | 2,299 |

All 14,347 episode IDs, tasks, lengths and suites match our pinned snapshot.
This corroborates row alignment, not the labels' authenticity or semantics.
The supplement's RLDS path recovery was reviewed but not rerun; no third-party
recovery code was executed. **`env` must not be relabeled clean.** All source
instructions, including rows labeled `language`, remain the 40 canonical
strings; paraphrase augmentation is not established.

## Bounded comparison with public standard LIBERO data

Pinned current public metadata for the four IPEC clean repositories referenced
by the StarVLA recipe:

| Suite | Published episodes | Published frames |
| --- | ---: | ---: |
| Spatial | 432 | 52,970 |
| Object | 454 | 66,984 |
| Goal | 428 | 52,042 |
| Long | 379 | 101,469 |
| **Total** | **1,693** | **273,465** |

These snapshots contain 40 tasks, two 256-pixel camera views, 20 Hz recording,
8D state and 7D actions. Revisions and metadata hashes are in the
[source audit](evaluation/week4-b0-training-source-audit-v1.json).
They are not authenticated paper-era training files; these counts do not revise
the paper's reported exposure or establish matched training data.

A bounded probe downloaded the first published episode per task, selected before
inspecting overlap. After converting only the public gripper encoding
(`1 - 2 * gripper_open`), **0/40** exactly matched a full training state/action
sequence. This strict, nonrandom sample does not establish that the datasets
are disjoint. Near-duplicates, all public episodes and camera matching were
outside scope. [Probe results](evaluation/week4-standard-libero-overlap-probe-v1.json).

## Implications before the next training run

1. **Correct measurement interpretation first.** Preserve the historical split
   and results; identify current validation as visual variants of mostly seen
   trajectories. Design an independently grouped split for future fresh
   training. Current validation loss alone cannot establish unseen-demo sample
   efficiency or justify longer training.
2. **Retain rollout-based decisions.** The frozen episode-blocked/task-mixed
   continuation remains interpretable using its declared rollout screen; current
   validation is descriptive. This audit does not replace its arm definitions,
   exposure, selection or acceptance rule.
3. **Prioritize a separately declared BCE-weight ablation.** Reduced BCE weighting
   is now a better-supported candidate while retaining the requested L1+BCE
   objective. It needs a prospective versioned recipe and matched rollouts.
   Neither switching to gripper L1 nor training longer is established as a fix.

Previous/new B0 clean scores remain **68.25%/65.25%**, versus OpenVLA **96.75%**;
dev-210 remains **106/210 and 114/210**. Week 4's clean gate remains **FAIL**.
No training or new rollout was launched by this diagnostic task.

## Execution and reproducibility

All three jobs ran as host-owned user systemd services on Feifei, independent of
SSH, and exited successfully without restarts. Script elapsed times were
**62.09 s** for gradients, **15.16 s** for the data audit and **3.44 s** for the
clean probe, excluding preparation and subsequent verification.

- [Gradient runner](../scripts/diagnose_b0_gradients.py)
- [Source audit](../scripts/audit_b0_training_source.py)
- [Clean numeric probe](../scripts/audit_b0_clean_overlap.py)
- [Diagnostic calculations](../src/eventground_wam/b0_diagnostics.py)
- [Verification receipt](evaluation/week4-inexpensive-diagnostics-verification-v1.json):
  98 executed source/config/panel files match local hashes; all 10 nonmetadata
  base-model files match their pinned manifest. Includes direct pair checks and
  runtime versions.

Host artifacts remain under
`/home/peimengsui/data/eventground-wam/analysis/week4-inexpensive-diagnostics-v1`,
including numeric episode rows, pinned public metadata, per-batch gradients,
logs and isolated source. Local detailed copies are under
`outputs/week4-inexpensive-diagnostics`.

**Checks:** 16 targeted tests passed; diagnostic code lint passed; sealed report
digests, duplicate counts and gradient medians/flag counts verified; all 20
continuation-freeze source pins passed unchanged.
