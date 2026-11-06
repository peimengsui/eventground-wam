# Implementation and code

The policy uses a Qwen3.5-2B backbone, front/wrist images, robot state and a
continuous action head. One query produces eight actions. Auxiliary policies
remove the future branch at deployment; coupled policies use predicted future
tokens for one action refinement. Pixel and latent name the training targets:
images and visual features. The deployed coupled action head consumes internal
tokens rather than rendered future images.

## Models and training entry points

| Component | Implementation | Training entry point |
| --- | --- | --- |
| Action-only baseline B0 | [Backbone/action head](../src/eventground_wam/b0_model.py), [training](../src/eventground_wam/b0_training.py) | [Fresh two-epoch baseline](../scripts/train_b0_fresh.py) |
| Pixel future predictor | [Predictor](../src/eventground_wam/pixel_predictor.py), [training](../src/eventground_wam/pixel_training.py) | [Pixel pretraining](../scripts/train_pixel_cpt.py) |
| Latent future predictor | [JEPA predictor](../src/eventground_wam/jepa_predictor.py), [features](../src/eventground_wam/jepa_features.py) | [Latent pretraining](../scripts/train_jepa_cpt.py) |
| Auxiliary policies M1/M2 | [Policy](../src/eventground_wam/auxiliary_policy.py), [training](../src/eventground_wam/auxiliary_training.py) | [M1](../scripts/train_auxiliary_policy.py), [M2](../scripts/train_auxiliary_lambda.py) |
| Coupled policies M3/M4 | [Policy](../src/eventground_wam/coupled_policy.py), [exports](../src/eventground_wam/coupled_checkpoint.py) | [M3](../scripts/train_coupled_policy.py), [M4](../scripts/train_coupled_lambda.py) |
| Event prediction | [Model](../src/eventground_wam/eventground_model.py), [future-onset objective](../src/eventground_wam/eventground_future_onset.py) | [Event training](../scripts/train_week10_future_onset.py) |

The [shared protocol](PHASE2_TRAINING_PROTOCOL.md) describes initialization,
trainable/frozen components and matched action exposure. Authored recipes remain
in `configs/`. Historical run catalogs, checkpoint manifests and evidence are
external research artifacts; see [their usage guide](RESEARCH_ARTIFACTS.md).

## Inference and evaluation

- [Baseline inference](../src/eventground_wam/b0_inference.py) and
  [coupled runtime](../scripts/coupled_runtime.py) expose the action path.
- [Five-policy runner](../scripts/evaluate_week9.py),
  [execution](../src/eventground_wam/week9_execution.py) and
  [protocol](../configs/evaluation/week9_factorial_v1.json) define the main study.
- [Event-extension runner](../scripts/run_week11_formal.py),
  [execution](../src/eventground_wam/week11_execution.py) and
  [horizon mapping](../configs/evaluation/week11_horizon_mapping_v1.json)
  implement fixed, event-running fixed, and adaptive prediction horizons.
- [Environment locks](../environments/) pin the simulation/training dependencies.
  Historical hardware inventories and numerical-setting records are kept with
  the external evidence packages.

The internal numbered filenames preserve experiment provenance. They are not
independent installable packages or a promise that completed launch commands can
be rerun unchanged. Original host/output/source guards are retained.

## Statistics, figures and examples

[Build results](../scripts/build_week12_results.py),
[render research figures](../scripts/render_week12_figures.py) and
[verify results](../scripts/verify_week12_results.py) operate on retained evidence.
The [artifact guide](RESEARCH_ARTIFACTS.md) explains the separate inputs and
analysis-reproduction commands. Generated tables and the portable archive are
not tracked in this code-only checkout.

The [blog figure renderer](blog/eventground-wam/source/render_blog_assets.py)
uses consolidated plotting data and retained panels. The
[media renderer](blog/eventground-wam/source/render_main_study_media.py)
exports matched recorded rollouts and a saved prediction example; it does not
run a model. Rendered figures/videos and their provenance live with the separate
website/evidence artifacts. Regenerating videos requires retained HDF5/NPZ
inputs, numpy, Pillow, h5py with HDF5 2.0 support, imageio-ffmpeg and the recorded
macOS Arial fonts. Those recordings and exports are not in this repository.

## Scope of reproduction

The default CPU tests exercise code with synthetic fixtures. Retained-evidence
audits require the `--research-artifacts` option and the corresponding local
files. This repository does not claim a fresh replication of training or robot
behavior. The source, authored configs and dependency locks are versioned;
full datasets, checkpoints, generated panels and reported outputs are external.
