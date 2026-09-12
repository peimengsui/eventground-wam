# Week 3 entry status

**Snapshot date:** 2026-09-12

**Entry outcome:** data, compute, and local engineering prerequisites pass. Code
integration is still required before starting the reproducible B0 implementation
from a clean canonical revision. The complete Week 1 `sanity-35` reference run
remains a parallel project-level gate.

| Prerequisite | Status | Evidence |
| --- | --- | --- |
| Frozen Week 2 dataset | Pass | `week2-dataset-v1` is present on `feifei`; its manifest SHA-256 is `53d80156d9f6eb0451572b29ab7fcf38add08405ffa6ab26c7400899aad47ae2` and matches the committed freeze evidence. |
| Source snapshot | Pass | The `current` link resolves to immutable dataset revision `22c57433fef692b5b9ecc0795344daac7fa867a5`. |
| Training split | Pass | 11,521 train, 1,396 validation, and 1,430 test episodes with zero cross-split group overlap. The assignment file hashes to `71b8fc2a3ce04c84cd41e3c9f052f1a677a0fa2d546201f8540f57eeb3c46758`. |
| Action-sample index | Pass | The 10,300,666-record multi-horizon file hashes to `6e317de0db00792ed9af7b4a500fb2f4cdb90a7d0b4d9a8a9ca2eacb56e69d5f`. B0 uses its frozen split identity, not event targets. |
| Event-label dependency | Not applicable to B0 | Frozen v1 has no admitted `next-event` samples. B0 contains no event conditioning or future-prediction target. |
| Action objective | Locked | B0 and M1–M4/M4-EG use one deterministic action-query Transformer with normalized action-chunk L1. Its conditioning-memory interface accepts future tokens for M3/M4. Generative action objectives are outside the required 12-week study. |
| Local tests | Pass | `uv run pytest`: 128 passed. |
| Local lint | Pass | `uv run ruff check .`: all checks passed. |
| Compute capacity | Pass for entry | RTX 5090 with 32,607 MiB total and 31,849 MiB free at inspection; 1.6 TiB storage free. Concurrent model/simulator memory remains a Week 3 runtime gate. |
| Week 1 reference smoke | Partial pass | One of 35 tasks completed successfully with no exception or retry. |
| Complete `sanity-35` | Pending | Existing report has `selected_task_count: 1` and `selection_is_complete_manifest: false`; the required full 35-task report does not yet exist. |
| Week 2 human-QA completion decision | Pass | The reviewer reconfirmed that the balanced 60-episode review is the completed gate and that no further review is required. The digest-sealed completion evidence records 79 accepted predictions, zero errors, and the review/report hashes. The remote 200-row blank template is a historical input template, not the completed review artifact. |
| Canonical code revision | Pending | `main`, the compute-host checkout, the completed Week 2 branch, and uncommitted Week 2/Week 3 files are not yet synchronized to one clean revision. |

## Start rule

Before creating the B0 runtime or downloading model weights:

1. Review and commit the confirmed Week 2 human-QA completion record and the
   Week 3 design documents.
2. Fast-forward the canonical `main` branch to that reviewed commit and
   synchronize the origin and compute-host checkouts.
3. Confirm all checkouts are clean and record that Git revision as the Week 3
   implementation base.

Run the full `sanity-35` evaluation independently. It does not block local
runtime, data-loader, or model construction, but it must pass before a Week 3
closed-loop result is accepted and before Week 4 begins.

Week 3 acceptance thresholds are locked in
[`WEEK3_BASELINE_DESIGN.md`](WEEK3_BASELINE_DESIGN.md). No Qwen checkpoint has
yet been downloaded or pinned as part of this status snapshot.
