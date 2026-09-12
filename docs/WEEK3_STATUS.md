# Week 3 entry status

**Snapshot date:** 2026-09-12

**Entry outcome:** pass. Data, compute, local engineering, canonical-code, and
Week 1 reference-policy prerequisites are complete. B0 implementation can start
from the clean `week3-entry-v1` Git tag.

| Prerequisite | Status | Evidence |
| --- | --- | --- |
| Frozen Week 2 dataset | Pass | `week2-dataset-v1` is present on `feifei`; its manifest SHA-256 is `53d80156d9f6eb0451572b29ab7fcf38add08405ffa6ab26c7400899aad47ae2` and matches the committed freeze evidence. |
| Source snapshot | Pass | The `current` link resolves to immutable dataset revision `22c57433fef692b5b9ecc0795344daac7fa867a5`. |
| Training split | Pass | 11,521 train, 1,396 validation, and 1,430 test episodes with zero cross-split group overlap. The assignment file hashes to `71b8fc2a3ce04c84cd41e3c9f052f1a677a0fa2d546201f8540f57eeb3c46758`. |
| Action-sample index | Pass | The 10,300,666-record multi-horizon file hashes to `6e317de0db00792ed9af7b4a500fb2f4cdb90a7d0b4d9a8a9ca2eacb56e69d5f`. B0 uses its frozen split identity, not event targets. |
| Event-label dependency | Not applicable to B0 | Frozen v1 has no admitted `next-event` samples. B0 contains no event conditioning or future-prediction target. |
| Action objective | Locked | B0 and M1–M4/M4-EG use one deterministic action-query Transformer with normalized action-chunk L1. Its conditioning-memory interface accepts future tokens for M3/M4. Generative action objectives are outside the required 12-week study. |
| Local tests | Pass | `uv run pytest`: 130 passed after adding the reference-evidence contract test. |
| Local lint | Pass | `uv run ruff check .`: all checks passed. |
| Compute capacity | Pass for entry | RTX 5090 with 32,607 MiB total and 31,849 MiB free at inspection; 1.6 TiB storage free. Concurrent model/simulator memory remains a Week 3 runtime gate. |
| Complete `sanity-35` | Pass | All 35 tasks ran from clean project revision `5e7141183d19651d5bd78e9cbcc7a477fdf7082f`: 28 successes, seven policy timeouts, zero exceptions, and zero retries. Every HDF5 artifact validates and matches its report hash. |
| Week 2 human-QA completion decision | Pass | The reviewer reconfirmed that the balanced 60-episode review is the completed gate and that no further review is required. The digest-sealed completion evidence records 79 accepted predictions, zero errors, and the review/report hashes. The remote 200-row blank template is a historical input template, not the completed review artifact. |
| Canonical code revision | Pass | Local `main`, origin, the `feifei` Git remote, and the clean compute-host checkout were synchronized at `5e7141183d19651d5bd78e9cbcc7a477fdf7082f` before the reference run. The final evidence-bearing entry revision is tagged `week3-entry-v1`. |

## Integration and reference evidence

The verified Week 2 completion and L1-only Week 3 design were fast-forwarded to
`main`, pushed to both Git remotes, and pulled into the clean compute checkout.
The final source state, including compact reference evidence, is identified by
the `week3-entry-v1` tag; B0 work starts from that tag or a direct descendant.
The complete reference report and episode artifacts live outside Git under the
revision-specific path recorded in
[`evaluation/sanity-35-reference-policy-v1.json`](evaluation/sanity-35-reference-policy-v1.json).

Week 3 acceptance thresholds are locked in
[`WEEK3_BASELINE_DESIGN.md`](WEEK3_BASELINE_DESIGN.md). No Qwen checkpoint has
yet been downloaded or pinned as part of this status snapshot.

## B0 data-layer implementation

The data layer was implemented from `week3-entry-v1` on the
`codex/week3-b0-dataloader` branch. The loader and train-only action-statistics
contract are documented in [`B0_DATALOADER.md`](B0_DATALOADER.md). B0 derives
its `8 x 7` chunks from frozen episode assignments rather than the semantically
different Week 2 future-frame horizon index.

The real-data gate ran on `feifei` from clean revision
`b62f0ec5a13f3315157b265561017d5877232784`. Statistics covered all 1,796,601
actions in the 11,521 training episodes and expose 1,715,954 valid chunks. Two
complete identity passes matched at
`c507ff3f6fe7c9f6ccf7866fa179b82c30aac9fd34d0730ebfa2ae9be2653963`.
Episode 0 produced the expected 270 samples, direct anchors matched the source
Parquet rows, real dual-camera frames met the RGB CHW contract, and all
2,238,036 train/validation/test actions inverted within `1.12e-16` absolute
error with exact gripper polarity. Compact evidence is committed at
[`evaluation/week3-b0-data-v1.json`](evaluation/week3-b0-data-v1.json); the two
full artifacts remain outside Git at the path recorded there.
