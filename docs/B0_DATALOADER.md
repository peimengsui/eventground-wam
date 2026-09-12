# B0 dataloader contract

The Week 3 B0 data layer streams the immutable `week2-dataset-v1` source without
copying frames or changing the frozen artifacts. It joins source episodes to
`primary/splits/split-assignments.jsonl`; the source LeRobot declaration that
all episodes are `train` is not a model split.

## Sample contract

For an episode of length `L`, valid anchors are
`range(L - 8 + 1)`. At anchor `t`, a sample contains:

- the unchanged task instruction;
- front and wrist images decoded to owned, contiguous RGB `uint8 [3,256,256]`;
- the eight-dimensional `float32` state at `t`; and
- normalized `float32 [8,7]` actions from rows `t` through `t+7`.

The loader never pads a tail or crosses an episode boundary. This deliberately
does not consume Week 2 `horizon=8` sample records. Those records target frame
`t+8`, require nine frames, and expose `L-8` anchors; an eight-action B0 chunk
requires eight rows and exposes `L-7` anchors. The frozen train split therefore
has 1,715,954 B0 samples, not the 1,704,433 train `horizon=8` samples.

Source videos are already in canonical upright orientation, so the locked
default camera transform is `identity`. Conversion is PyAV `rgb24`; layout is
CHW; resize and crop are both absent. Qwen-specific resize, prompt construction,
and token selection remain part of the later conditioning adapter and must be
locked there rather than hidden in this loader.

Episode order is canonical when shuffling is disabled. With shuffling enabled,
episodes are sorted by a stable SHA-256 key over seed, epoch, episode identity,
and split-assignment digest. Explicit episode sharding gives workers disjoint
sets without omissions. `ordered_identity_digest()` does not open Parquet or
video files and seals the complete configured order.

## Action normalization

The normalization artifact is fitted from all 1,796,601 raw actions in the
11,521 training episodes. Each action is counted once; overlapping chunks are
not used to compute statistics, and no video is decoded during fitting.

For continuous dimensions 0–5, exact linearly interpolated training q01/q99
bounds define an affine transform centered at their midpoint and scaled by half
their range. Constant dimensions use unit scale. Normalization does not clip,
so inversion remains exact outside the fitted quantile range. Dimension 6 is
an identity transform with `-1 = open` and `+1 = close`; execution-time clipping
or gripper thresholding is not part of the normalizer.

The artifact records the dataset-freeze manifest, split manifest, and split
assignment hashes, source and generator Git identities, action and episode
counts, raw moments and extrema, fitted bounds, and a canonical artifact digest.
The loader rejects an artifact from a different source identity.

## Reproducible commands

Run these commands from a clean checkout using the isolated B0 runtime. The
commands refuse to overwrite their outputs. This first runtime lock contains
only the data dependencies; Qwen, PyTorch/CUDA, PEFT, and attention packages
are added only after the immutable model revision has been selected.

```bash
./scripts/setup_b0.sh

./scripts/run_b0.sh python scripts/run_b0_data.py statistics \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --output ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json

./scripts/run_b0.sh python scripts/run_b0_data.py smoke \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --normalization ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --output ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/dataloader-smoke.json
```

The smoke gate re-hashes the frozen source files, recomputes the ordered sample
digest twice, directly compares episode 0's first/middle/last chunks with its
Parquet rows, validates RGB/state shapes, and checks normalization inversion for
every train, validation, and test action at absolute tolerance `1e-6`.
