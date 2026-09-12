# Boundary visualization and human QA

The boundary-QA workflow keeps cohort selection, video decoding, and human
scoring separate. This makes the 200-episode review queue reproducible without
requiring all 400 camera videos to be decoded in the same run.

## Deterministic selection

`greedy-task-event-length-coverage-v1` considers unique episodes with at least
one supplied event candidate or filtered interval. It first favors episodes
that cover an unseen task, event type, or within-task length stratum, then
balances the current task, event, and length loads. Source episode index is the
final tie-breaker. The three length strata are rank-based within each task;
one-episode tasks use `middle`, and two-episode tasks use `short` and `long`.

The default request is 200 unique episodes. If fewer than 200 event-bearing
episodes exist, selection succeeds with every eligible episode and records both
`selection_shortfall` and `target_satisfied: false`. It never silently pads the
queue with duplicates.

Prepare the queue from derived contract streams stored outside Git:

```bash
./scripts/run_demonstrations.sh python scripts/run_boundary_qa.py prepare \
  --trajectory-index /path/to/trajectory-contracts.jsonl \
  --events /path/to/event-intervals.jsonl \
  --output-dir /path/to/week2-boundary-qa
```

This writes `selection-manifest.json`, `review-items.jsonl`, and
`review-output-template.jsonl`. Each item retains repository-relative front and
wrist source paths, every proposed event interval, bounded context, deterministic
sampled frame indices, and its expected preview path.

## Compact previews

The renderer decodes each camera only once per selected episode and produces one
compact PNG contact sheet per proposed event. The front camera is the top row
and the wrist camera is the bottom row. A green border denotes a frame inside
the predicted interval, a red left bar marks the start frame, and a blue right
bar marks the last frame of the half-open interval.

Render a five-episode smoke subset without changing the selection:

```bash
./scripts/run_demonstrations.sh python scripts/run_boundary_qa.py render \
  --repository-root /path/to/libero-plus-lerobot/current \
  --review-items /path/to/week2-boundary-qa/review-items.jsonl \
  --output-dir /path/to/week2-boundary-qa \
  --limit 5
```

Use `--start` plus `--limit` for later batches and give each batch a distinct
`--manifest-output`. `--limit 0` renders all remaining items. Existing previews
and manifests are never overwritten.

## Single-file review workbook

Build one self-contained HTML review workbook after rendering the previews:

```bash
./scripts/run_demonstrations.sh python scripts/run_boundary_qa.py workbook \
  --review-items /path/to/week2-boundary-qa/review-items.jsonl \
  --preview-root /path/to/week2-boundary-qa \
  --repository-root /path/to/libero-plus-lerobot/current \
  --events /path/to/event-intervals.jsonl \
  --output /path/to/week2-boundary-qa/boundary-review.html
```

The HTML embeds every PNG, queue record, extractor-evidence string, validation
rule, and review control. It shows one episode at a time, provides synchronized
front/wrist video controls, records verdicts and missed events, and saves review
progress in browser-local storage. The original MP4 files remain external to
avoid duplicating hundreds of full videos; `--repository-root` supplies their
local references. Omitting `--repository-root` produces a preview-only workbook,
and omitting `--events` hides extractor evidence without changing the review
schema.

For a network-mounted QA directory or a browser that struggles with the large
self-contained file, add `--reference-previews` and use an output file inside
`--preview-root`. This creates a fast-loading page that references the existing
`previews/` tree while retaining the same single review interface and JSONL
export:

```bash
./scripts/run_demonstrations.sh python scripts/run_boundary_qa.py workbook \
  --review-items /path/to/week2-boundary-qa/review-items.jsonl \
  --preview-root /path/to/week2-boundary-qa \
  --output /path/to/week2-boundary-qa/boundary-review-lite.html \
  --reference-previews
```

An episode cannot be marked complete until it has a verdict for every
prediction, valid corrected or missed-event ranges, a reviewer ID, and explicit
confirmation that the full episode was checked for misses. Use **Download
completed-reviews.jsonl** to export records accepted by the existing
`summarize` command. **Load reviews JSONL** resumes from an earlier export; edits
also persist automatically in the same browser.

## Review-output schema

A review record is episode-level. A completed record must contain:

- a nonempty `reviewer_id` and `status: completed`;
- a sorted list of event types that the reviewer exhaustively checked in the
  episode;
- exactly one judgment for every prediction of each reviewed type;
- one of `accepted`, `corrected`, `rejected`, or `ambiguous` per prediction;
- a corrected half-open frame range only for `corrected` judgments;
- every confirmed event missed by the detector in `missed_events`.

“Exhaustively checked” matters: the missed-event list provides the false
negatives required for recall. Acceptance means the proposed start and end were
both confirmed exactly. Ambiguous predictions are reported separately and are
excluded from precision and recall denominators. Pending records must remain
identical to their generated template, preventing partially edited templates
from being mistaken for completed reviews.

Aggregate a review JSONL file with:

```bash
./scripts/run_demonstrations.sh python scripts/run_boundary_qa.py summarize \
  --review-items /path/to/week2-boundary-qa/review-items.jsonl \
  --reviews /path/to/completed-reviews.jsonl \
  --output /path/to/week2-boundary-qa/qa-report.json
```

For every ontology event type, the report includes review coverage, TP/FP/FN
counts, precision, recall, mean absolute start/end boundary error in frames,
ambiguity rate, and reviewer-correction count. With no completed reviews, all
human quality fields are `null`, `human_quality_metrics_available` is `false`,
and the report explicitly states that quality is unavailable. During a partial
review, metrics include completed records only and are labeled non-final.

## Integrated queue result

An expanded 240-episode, six-per-task source cohort produced 300 raw
`gripper_close` candidates and 228 event-bearing episodes. The deterministic
selector filled the requested queue with 200 unique episodes, 261 predictions,
38 represented tasks, and zero selection shortfall. Two source tasks had no
candidate-bearing episode and therefore could not appear in an event-boundary
review queue.

All 200 review items and pending templates pass strict schema, foreign-key, and
digest validation. A real dual-camera preview was rendered as a decoder and
overlay smoke test. The pending report records 200 pending and zero completed
reviews; all human quality metrics remain unavailable until reviewers complete
the templates. This prepares the artifact and tooling portion of roadmap step 6
without claiming that the human quality gate has passed.

## Frozen-version review package

The final review package was regenerated from the immutable
`week2-dataset-v1` trajectory and interval streams. Of 13,691 event-bearing
episodes, the selector chose 200 unique episodes with zero shortfall and 38
represented tasks. It rendered all 254 proposed `gripper_close` boundaries as
dual-camera PNG previews; the complete package occupies 42 MiB on `feifei` at
`~/data/eventground-wam/libero-plus/derived/boundary-qa/week2-dataset-v1/`.

The generated pending report contains 200 pending and zero completed reviews,
so its precision, recall, boundary error, ambiguity, and correction fields are
correctly `null`. File hashes and selection/report digests are recorded in
[`freeze/week2-dataset-v1-qa.json`](freeze/week2-dataset-v1-qa.json). The
review package completes the automated preparation for step 6, not the required
human judgments. A generated `boundary-review.html` combines the full frozen
queue and its 254 previews into the review workflow described above.

## Completed Week 2 review

On 2026-09-10, reviewer `psui` completed a digest-matched subset of 60 episodes
and confirmed that the full episodes were checked for both incorrect proposals
and missed `gripper_close` events. The subset was balanced across trajectory
length (20 short, 20 middle, and 20 long), covered 34 of the 38 eligible tasks in
the full queue, and contained 79 predictions. All 79 were accepted with no
corrections, rejections, missed events, or ambiguous judgments. The completed
subset report therefore records 100% precision, 100% recall, zero-frame mean
absolute start/end boundary error, and status `human_review_complete`.

The reviewer authorized stopping after these 60 zero-error episodes. This is an
explicit change from the original 200-episode target, and the remaining 140
queue episodes are not represented as reviewed. Completion evidence, including
SHA-256 hashes of the review export, digest-matched subset, and completed report,
is recorded in
[`freeze/week2-dataset-v1-human-qa.json`](freeze/week2-dataset-v1-human-qa.json).
The earlier pending package evidence remains unchanged as a historical record of
the state at freeze time.

Human QA qualifies the deterministic `gripper_close` extractor, but does not
rewrite the immutable `week2-dataset-v1` interval stream. Promotion from weak to
reliable labels and generation of `next-event` supervision require a new frozen
dataset version.
