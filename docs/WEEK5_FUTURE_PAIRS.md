# Week 5 step 1: shared future-image pairs

**Version:** `trajectory-v2-future-pairs-v1`.  
**Completed:** September 25, 2026. Full source verification, generation,
every-pair validation and byte-identical regeneration passed on Feifei. No CPT
or policy training launched. The [preparation receipt](evaluation/week5-future-pairs-preparation-v1.json)
and [frozen manifest](../configs/datasets/trajectory-v2-future-pairs-v1/manifest.json)
retain the evidence; the [step completion record](evaluation/week5-step1-completion-v1.json)
pins the handoff.

This index serves Pixel CPT in Week 5 and JEPA CPT in Week 6. It reuses the
exact frozen Week 4 numeric-trajectory-v2 assignments, not the old Week 2 split.
The [data rule](../configs/datasets/trajectory_v2_future_pairs_v1.json) pins the
split manifest, compact groups, expanded assignments, metadata and task mapping.

## Coverage

These production counts match the frozen split metadata and the full pair scan.

| Partition | Episodes | Numeric groups | Tasks | Dual-camera pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 11,435 | 1,341 | 40 | 8,221,365 |
| Validation | 1,528 | 180 | 40 | 1,109,994 |
| Total emitted | 12,963 | 1,521 | 40 | 9,331,359 |

| Horizon in frames | Seconds at 20 Hz | Train pairs | Validation pairs |
| --- | ---: | ---: | ---: |
| 2 | 0.10 | 1,763,197 | 237,890 |
| 4 | 0.20 | 1,740,327 | 234,834 |
| 8 | 0.40 | 1,694,587 | 228,722 |
| 16 | 0.80 | 1,603,107 | 216,498 |
| 32 | 1.60 | 1,420,147 | 192,050 |

For episode length `L` and horizon `h`, anchors are exactly `0 <= t < L-h`.
The target is frame `t+h` in both cameras. Intervening actions are the raw
Parquet rows `[t, t+h)`, exactly `h` actions, excluding the target-frame action.
Instruction, state and images used as conditioning come from the current frame.
Future images are targets only. No pair crosses an episode boundary.

The 1,384 test episodes are excluded before any payload file is opened, hashed,
probed or decoded. Test assignment metadata participates only in isolation and
coverage checks. There are no next-event samples. No target resizing,
augmentation, feature extraction, sampling weights or CPT budget is decided by
this preparation step. The index does not change B0's action samples; future
policy trainers must use masks for unavailable targets while retaining every
action-training anchor.

## Format and provenance

Each bundle contains:

- `sources.jsonl`: one sealed record per selected source episode, containing its
  instruction, task, split, numeric group, assignment digest, pinned dataset and
  split identities, and the Parquet/front-video/wrist-video paths, sizes and
  verified SHA-256 values.
- `train.bin` and `validation.bin`: fixed-width, 78-byte records in ascending
  episode, horizon, anchor order. Each record covers both camera views, so the
  pair count is not multiplied by two.
- `manifest.json`: sealed rule, source/code provenance, output hashes, counts by
  split/horizon/task/camera, and all excluded tail-anchor counts.

The record layout is `struct.Struct('<IIIBB32s32s')`: unsigned little-endian
episode, anchor, target; unsigned byte horizon and split code (`0=train`,
`1=validation`); 32 raw bytes of source digest; 32 raw bytes of pair digest.
The pair digest is SHA-256 of the domain
`eventground_wam.future_image_pair.v1` followed by a NUL byte, the raw source
digest, and the packed `<IIIBB` identity fields. This binds every pair to both
camera files and the exact v2 assignment without duplicating media.

The two binary indexes occupy 727,846,002 bytes (about 694 MiB), plus the source
registry and manifests. The source snapshot remains the image/action store.
Python's `struct` can read the files without NumPy; later training loaders can
memory-map them and cache the source registry. The `read_pair` helper is a
diagnostic reference resolver, not a complete training loader.

## Required production verification

The preparation job performs these checks before atomically publishing a new
output directory:

1. Verify the frozen split/source metadata and reproduce the exact expanded
   assignment hash. Check whole-group isolation, episode membership and task
   coverage without resplitting or rerolling.
2. Rehash all 38,889 selected payload files against the pinned source inventory.
   Validate every selected Parquet row, schema, frame index and timestamp. Check
   both videos' frame-count, frame-rate, resolution and codec headers.
3. Decode every synchronized frame in the first episode of each task in each
   selected split: 80 episodes, both cameras. Verify timestamps and packed RGB24
   conversion and retain decoded sequence hashes. This is a decode smoke test;
   it is not represented as a complete decode of all videos.
4. Generate the primary bundle and validate every stored pair's digest, source
   foreign key, boundary, order and exact coverage.
5. Reconstruct source records independently from metadata/inventory in reverse
   episode order, regenerate the full index, and compare all four artifact
   files byte-for-byte. This second pass does not repeat all media decoding.

`preparation.json` records the completed checks and runtime; `primary/` and
`replica/` retain the matching artifacts. Existing results are never overwritten.
Incomplete jobs remain under a `.partial` directory and are not frozen results.
The runtime lock and executed code files are hashed and retained in a dedicated
source snapshot, including code not yet committed to Git.

## Reproduce and inspect

Use the existing locked demonstration runtime, with the source snapshot mounted
on the compute host. The job is CPU-only and should run under a host-owned
service for independence from the SSH session.

```bash
./scripts/run_demonstrations.sh python scripts/prepare_future_pairs.py \
  --data-root /home/peimengsui/data/eventground-wam/libero-plus \
  --output /home/peimengsui/data/eventground-wam/libero-plus/derived/future-pairs/trajectory-v2-future-pairs-v1 \
  --workers 4

./scripts/run_demonstrations.sh python scripts/validate_future_pairs.py \
  --bundle /home/peimengsui/data/eventground-wam/libero-plus/derived/future-pairs/trajectory-v2-future-pairs-v1/primary
```

The validator can run without source images: it checks every index row against
the frozen split and sealed source registry. It does not rehash source media;
that evidence belongs to the preparation receipt. Any copied or subsequently
modified source payload must be rechecked against its recorded hash before use.

```python
from pathlib import Path
from eventground_wam.future_pairs import read_pair

sample = read_pair(Path('/path/to/primary'), 'validation', 0)
# sample['anchor'], sample['target'], sample['action_stop_exclusive']
# sample['source']['files'] resolves both views and the action source.
```

Local validation currently passes all 328 tests (21 new pair-index tests) and
Ruff. The tests cover exact tail handling, action intervals, deterministic order,
camera/source binding, test exclusion, group leakage, digest corruption,
duplicate/missing/trailing pairs and existing-output protection.

## Completed host run

The CPU-only `eventground-future-pairs-v1.service` finished with exit zero and
zero restarts in 42.43 seconds of script runtime. It checked 38,889 payload hashes,
2,027,013 telemetry rows and 25,926 video headers. The 80-episode decode smoke
covered 13,880 synchronized steps (27,760 camera frames). All 9,331,359 pair
digests, boundaries and source relationships passed. The separately invoked
validator also passed. No test payload was read and no training launched.

Frozen artifacts on Feifei:

```text
/home/peimengsui/data/eventground-wam/libero-plus/derived/future-pairs/trajectory-v2-future-pairs-v1/
  primary/{sources.jsonl,train.bin,validation.bin,manifest.json}
  replica/{sources.jsonl,train.bin,validation.bin,manifest.json}
  preparation.json
```

The identical manifests have digest
`a597ee9c65796d4f4bf415287f4c452f42b9a0e8a1edd25696d7ada281b5eb18`.
Executed source, its archive and file hashes, the service log and standalone
verification are retained at
`/home/peimengsui/data/eventground-wam/analysis/week5-future-pairs-v1/`.
Large indexes and raw images remain outside Git; compact manifests and reports
are stored in the project.

With step 1 complete, [Week 5 step 2](WEEK5_STATUS.md) can establish copy and
constant-background reconstruction baselines on the identical validation pairs.
