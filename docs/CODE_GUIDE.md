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
trainable/frozen components and matched action exposure. The
[run catalog](evaluation/week12-repro-v1/run-catalog.json) maps each historical
entry point to its exact config, checkpoint lineage and evidence.

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
- [Environment locks](../environments/) and
  [recorded runtime inventories](evaluation/week12-repro-v1/environments.json)
  identify the simulation/training dependencies and numerical settings.

The internal numbered filenames preserve experiment provenance. They are not
independent installable packages or a promise that completed launch commands can
be rerun unchanged. Original host/output/source guards are retained.

## Statistics, figures and examples

[Build results](../scripts/build_week12_results.py),
[render research figures](../scripts/render_week12_figures.py) and
[verify results](../scripts/verify_week12_results.py) operate on retained evidence.
The [portable reproduction guide](evaluation/week12-repro-v1/GUIDE.md) is the
supported route from a fresh checkout: verify/extract the committed archive,
then run its analysis wrapper. This preserves the sealed historical inputs.

The [blog figure renderer](blog/eventground-wam/source/render_blog_assets.py)
uses the consolidated plotting data and retained panels. The
[media renderer](blog/eventground-wam/source/render_main_study_media.py)
exports matched recorded rollouts and a saved prediction example; it does not
run a model. Committed exports can be viewed directly. Regenerating the videos
requires the external retained HDF5/NPZ inputs identified in
[media provenance](blog/eventground-wam/media-provenance.json), plus numpy,
Pillow, h5py with HDF5 2.0 support, imageio-ffmpeg and the recorded macOS Arial
fonts. Those raw recordings are not included in the source release.

## Scope of reproduction

CPU tests and the portable statistical/figure rebuild are checked separately.
This release does not assert a fresh replication of training or robot behavior.
The source and analysis archive are versioned; full datasets, backbone/checkpoint
weights and raw simulation recordings remain external. The manifests preserve
identities without claiming the original host paths are public downloads.
