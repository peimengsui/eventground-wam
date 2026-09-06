# LIBERO-Plus demonstrations

The training demonstrations use the official
[`Sylvest/libero_plus_lerobot`](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot)
LeRobot v2.1 snapshot. The setup never follows a mutable branch:

- Revision: `22c57433fef692b5b9ecc0795344daac7fa867a5`
- Files: 43,048
- Payload size: 15,995,527,260 bytes (14.897 GiB)
- Source-manifest SHA-256:
  `e673a1d3e58a1756879b4f30ce38474aa3407f97232ee5184fd7e1e6ed901510`

This representation was selected over the official RLDS archive because it is
native to the PyTorch-oriented LeRobot data model and is much smaller: 14.897
GiB versus 70.359 GiB. It provides 14,347 episodes and 2,238,036 frames across
the 40 standard LIBERO tasks. Each step has an 8D state and 7D action. Front
and wrist observations are 256 x 256 AV1 video at 20 Hz.

The training videos are lossy source data supplied by the benchmark authors.
They do not include MuJoCo state. They are suitable for Week 2 event labeling
and model training, but they do not replace the lossless HDF5 evaluation
artifacts written by the unified episode recorder.

## Download on `feifei`

From the project checkout:

```bash
uv sync --project environments/demonstrations --locked
./scripts/setup_demonstrations.sh
```

The script creates a separate locked `uv` environment, verifies the immutable
Hugging Face repository revision and complete remote manifest, downloads with
`snapshot_download`, and runs the integrity check before publishing the
`current` symlink. A process-local cache reuses each repository-wide Xet
connection token until shortly before expiry, avoiding one identical Hub API
request per LFS file. Interrupted downloads are resumable by running the same
command again. The public snapshot supports
anonymous access, but its 43,048 individual files can exhaust anonymous request
limits. If Hugging Face reports HTTP 429, authenticate once with a read-only
token and resume; do not paste the token into logs:

```bash
environments/demonstrations/.venv/bin/hf auth login
./scripts/setup_demonstrations.sh
```

`HF_TOKEN` may be used instead of the stored login in unattended sessions.
The setup defaults to four workers so its sustained resolver traffic remains
below the free authenticated limit. `EVENTGROUND_DOWNLOAD_WORKERS` can override
that value for accounts with a higher limit, and `EVENTGROUND_HF_ETAG_TIMEOUT`
controls the 30-second metadata-request timeout.

The versioned files live at:

```text
~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/
├── current -> snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/repo
└── snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/
    ├── repo/
    ├── source-manifest.tsv
    └── source.json
```

The existing read-only SSH mount exposes this directory locally under:

```text
~/feifei-eventground/datasets/libero-plus-lerobot/current
```

## Validation

The default check verifies all 43,048 paths and byte sizes, the pinned metadata
and representative payload SHA-256 values, the complete task/episode metadata,
sample Parquet schemas, and decodability of both camera streams:

```bash
environments/demonstrations/.venv/bin/python scripts/check_demonstrations.py
```

For an expensive full pass over every LFS payload checksum:

```bash
environments/demonstrations/.venv/bin/python \
  scripts/check_demonstrations.py --full-sha256
```

## Week 2 streaming interface

The versioned derived-data contracts are documented in
[`DATA_CONTRACTS.md`](DATA_CONTRACTS.md). The Python interfaces are
`eventground_wam.demonstration_contracts` and
`eventground_wam.demonstration_reader`.

Build the complete trajectory-contract index without reading media payloads:

```bash
./scripts/run_demonstrations.sh python scripts/generate_trajectory_index.py
```

`DemonstrationReader.iter_episode_steps(episode)` is the synchronized payload
interface. It yields one validated telemetry row and its matching decoded front
and wrist frames, checks both video timestamps against the 20 Hz telemetry clock,
and rejects count or image-property mismatches. Each frame payload is a raw
PyAV `VideoFrame`, so consumers can convert only the current frame to the tensor
or array representation they need without materializing an episode.

The conservative event-labeling feasibility study and its initial results are
documented in [`EVENT_LABELING_PILOT.md`](EVENT_LABELING_PILOT.md). Run it with:

```bash
./scripts/run_demonstrations.sh python scripts/run_event_labeling_pilot.py
```

Run the routine streaming check from the repository root. It exhausts all
episode metadata in canonical order and fully validates three evenly spaced
Parquet episodes, including their referenced source files:

```bash
./scripts/run_demonstrations.sh python scripts/validate_demonstration_stream.py
```

Also decode both camera videos for the selected episodes:

```bash
./scripts/run_demonstrations.sh python scripts/validate_demonstration_stream.py \
  --decode-videos
```

Stream all 2,238,036 Parquet rows through the schema, identity, timestamp,
shape, finite-value, and source-file checks:

```bash
./scripts/run_demonstrations.sh python scripts/validate_demonstration_stream.py \
  --all-episodes
```

Adding `--decode-videos` to `--all-episodes` decodes all 28,694 videos and is
intentionally an expensive audit. PyArrow reads bounded batches and PyAV decodes
one video at a time; neither path materializes the complete dataset.

The pinned snapshot has no initial seed, initial-condition identifier, MuJoCo
state, or object state. Its final action dimension is binary `-1/+1`, but the
metadata does not declare which polarity means open or closed. Event labeling
must establish that convention independently before naming gripper events.
`norm_stats.json` is not part of the locked Week 2 contract until its apparent
action-scale mismatch with the stored Parquet rows is reconciled.
