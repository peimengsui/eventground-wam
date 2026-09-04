# Unified episode recording

The unified recorder writes one HDF5 artifact per LIBERO-Plus episode. It is the
source of record for later training, evaluation, and event labeling; rendered
PNG or MP4 files are previews only.

## Artifact contract

Each file embeds the complete locked evaluation protocol and canonical episode
metadata. The `steps` group contains row-aligned datasets for every control
transition, including the ten stabilization transitions:

- both 256-by-256 RGB observations after the locked 180-degree transform;
- 8-D end-effector, orientation, and gripper proprioception;
- the full predicted 8-by-7 action chunk and the action actually executed;
- the full flattened MuJoCo state, reward, success signal, and policy-query latency.

Floating-point values are stored as float64 and images as uint8. HDF5 gzip
compression is lossless. A step row pairs the observation and MuJoCo state from
immediately before the executed action with the reward and `done` returned by
that transition.

## Crash safety

Recording starts at `<episode>.hdf5.partial`. Normal finalization writes the
outcome, closes the file, validates every required field and invariant, and then
atomically renames it to `<episode>.hdf5`. Exceptions and interrupted contexts
leave the partial artifact in place with an `aborted` status. Existing complete
or partial files are never overwritten automatically.

## Commands on `feifei`

Record a short real-simulator episode with the built-in zero-action smoke policy:

```bash
./scripts/run_libero.sh python scripts/check_libero.py \
  --suite libero_spatial \
  --task-id 0 \
  --steps 5
```

Validate a completed artifact independently:

```bash
./scripts/run_libero.sh python scripts/validate_episode.py \
  ~/data/eventground-wam/libero-plus/smoke/libero_spatial-task-0.hdf5
```

The Python integration surface is `EpisodeRecorder`, `EpisodeMetadata`, and
`StepRecord` in `eventground_wam.episode_recorder`. A production policy supplies
its real action chunk and measured inference latency in place of the smoke
policy's zero chunk; the artifact format remains identical.
