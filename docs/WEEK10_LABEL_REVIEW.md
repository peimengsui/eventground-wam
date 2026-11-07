# Week 10 independent event-label review

The initial 200 recordings are physically verified, but no production reference
annotations or label-quality result exist yet. The frozen extractor's output is
a set of candidates. Running the extractor twice, reproducing its predicates or
asking its author to sign those candidates does not establish precision/recall.

## What the inherited contract requires

V3 (local archive: `WEEK10_EXPERIMENT_V3.md`) preserves the existing independent QA gates and
requires independent full-episode QA before training. V2 (local archive: `WEEK10_EXPERIMENT_V2.md`)
requires at least 200 training episodes, all 40 tasks, at most 300 for coverage,
at least 100 reviewed positives each for contact and grasp, precision and recall
at least 95%, and mean start/end error at most two frames. Missed events and
negative cases must be reviewed. The earlier close QA does not transfer to these
new recordings.

The original experiment (local archive: `WEEK10_EXPERIMENT.md`) says: “a VLM's guess cannot serve
as its own verifier.” Its boundary-error population is fully observed matched
intervals. Counts, uncertainty intervals, ambiguities and exclusions must be
reported; passing point estimates does not guarantee population quality at 95%
confidence. The [labeler guide](WEEK10_EVENT_LABELER.md) explicitly distinguishes
numeric implementation verification from independent human QA.

The active JSON contract says *independent review*, rather than explicitly
specifying a human reviewer. This tooling follows the established guide's human
review interpretation: a separately identified person supplies the reference
annotations, without seeing extractor or model predictions first. That is a
stated operational interpretation, not a newly discovered prohibition on other
independent methods. Changing the verifier design requires a separate review;
this tool cannot turn model-generated annotations into human evidence.

## Blind evidence package

[`eventground_qa.py`](../src/eventground_wam/eventground_qa.py) builds a manifest
from the completed capture and exact initial-source queue. It retains all 200
sources, five from each task, with original HDF5/contact/geometry checksums.
Neither task-success outcomes nor extractor outputs are included. No source is
selected by success, event prevalence or model prediction.

[`prepare_week10_label_review.py`](../scripts/prepare_week10_label_review.py)
creates a source index and an **unfinished** annotation form. It does not fill
events, negatives, review-completion checkboxes or reviewer identities. The
read-only viewer can display the saved full-resolution front/wrist frames and
raw physical evidence directly from the HDF5/JSON files. It does not import a
simulator, integrate physics, infer events, expose candidate labels or copy the
46.99 GiB recording archive. Its localhost server has a bounded session and
serves only enumerated evidence routes, never arbitrary paths.

Preparation requires a sealed child with the exact capture completion digest
and `review_preparation_limits` containing `wall_seconds` no greater than 1,800
and `output_bytes` no greater than 1 GiB. That bound covers material preparation;
human annotation time and any viewer sessions must also be recorded against the
separately sealed reconstruction/annotation budget. No additional budget is
inferred from the training GPU-hour allowance. Existing outputs are not replaced.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_week10_label_review.py prepare \
  --capture-base docs/evaluation/week10-production-capture-visual-v2 \
  --preparation-contract /absolute/path/to/sealed-preparation-contract.json \
  --output /absolute/path/to/fresh-review-materials
```

When full HDF5/contact evidence is available locally, the reviewer can start a
read-only session using the same prepared materials:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_week10_label_review.py serve \
  --capture-base /absolute/path/to/week10-production-capture-visual-v2 \
  --materials /absolute/path/to/fresh-review-materials --port 8765 --seconds 1800
```

Open `http://127.0.0.1:8765`. Input hashes are verified before serving any frame.
On Feifei, the same loopback viewer can be accessed with an explicitly arranged
SSH port forward. The local metadata-only backup cannot display missing HDF5s.
The viewer never marks an episode as reviewed merely because it was loaded or
played; the reviewer must complete the separate reference form.

## Reference schema and import

A reviewer records every event in a full pass, including events the extractor
may have missed. Each reference has a unique `reference_id`, `event_type`,
`object_id`, integer `start`/`end`, and explicit `left_censored`/`right_censored`.
Intervals are zero-based, half-open at 20 Hz. Contact/grasp use the canonical
geometry-map object name; close has `object_id: null`. An interval beginning at
zero or ending at the last exclusive frame is marked censored.

Each episode also needs:

- Complete, non-overlapping `reviewed_frame_ranges` covering `[0, frames)`.
- True confirmations for both-camera and physical-evidence review, negative
  cases and a search for missed events.
- All ambiguities with event type, object, frame range and a specific reason.
- Explicit negative ranges for each event type, exactly the complement of its
  reference-positive and ambiguous spans. Uncertain frames are never negatives.

Keep task failures, initially active events, ambiguous periods and tails. Do not
fill missing events by copying candidates. Censored events can count toward
precision/recall but not fully observed boundary accuracy. Uncertain reference
spans are reported separately, with every excluded prediction counted.

At the top level the reviewer identifies themself, records start/completion
times with timezones, uses `reviewer_kind: "human"`,
`annotation_origin: "human_blind_full_episode"`, and `created_by` equal to the
reviewer identity. They attest independence and no candidate exposure before the
reference is sealed, and sign the exact statement already present in the form.
After completion, the reviewer or curator seals the unchanged form using the
repository's `seal_contract`, and preserves the original submitted file and its
SHA-256. A SHA seal is an integrity checksum; it is **not identity authentication**.
The curator must separately authorize the reviewer identity and retain provenance
of that authorization. No script should manufacture the attestation on behalf
of a person who has not done the review.

Only after the reference is sealed, evaluate it against the immutable candidates:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_week10_label_review.py assess \
  --materials /absolute/path/to/fresh-review-materials \
  --review /absolute/path/to/sealed-human-reference.json \
  --candidates /absolute/path/to/production-candidate-package \
  --trusted-reviewer actual-independent-reviewer-id \
  --labeler-actor codex-labeler-author \
  --output /absolute/path/to/new-qa-assessment.json
```

`--trusted-reviewer` is a curator-supplied trust set, not an identity inferred from
the review file. Include every known extractor/model author identity in
`--labeler-actor`. Unknown, generated, self-reviewed, incomplete or unbound inputs
raise an error; a genuine complete review with weak results produces a sealed
`fail` assessment. No failed result is overwritten.

## Matching and reporting

The review manifest records the matching rule before reference review:
deterministic maximum-cardinality one-to-one matching between same-type,
same-object intervals with positive overlap and equal censor flags. Preference
within that matching uses overlap and stable input order; a local greedy choice
cannot reduce the number of matched events. Start and end absolute errors are
computed separately on fully observed matches. Censor disagreement is an
unmatched prediction/reference, not silently ignored.

For contact and grasp, admission gates require at least 100 reference positives,
precision and recall at least 0.95, and **both** start and end MAE at most two
frames, with at least one fully observed match. Full source/task coverage and
negative/missed-event review are mandatory. The report includes all counts,
Wilson 95% intervals for precision/recall, ambiguity exclusions, censored-match
counts, negative frames and a match ledger. Wilson intervals are reported, not
substituted for the frozen point-estimate thresholds.

Close precision/recall and boundary errors are also reported and its full-episode
review is mandatory. The inherited new-event gates explicitly name contact and
grasp; this implementation does not silently add a new close threshold or claim
that old close QA transfers. Any additional close admission criterion should be
explicitly recorded by the curator before declaring the overall dataset ready.

The receipt's `status: "pass"` means the declared independent QA gates passed.
It always has `labels_admitted: false` and `training_authorized: false`. A separate
dataset admission must verify exact candidate/source/capture links and capped
prompt coverage. No independent reference has yet been supplied, so no production
QA receipt or admission can honestly be issued.

## Other remaining pre-training gates

Even a passing review does not guarantee enough unique future-event prompts.
The frozen 2,000-prompt set needs 500 per class, at most four anchors per physical
event and 16 per source, complete lead-time coverage, no duplication, and a
nested 500-prompt pilot. Unknown/ambiguous/ongoing/tail windows stay excluded.
Coverage reserves require a documented shortfall and remain within the original
all-300 recording limits; they cannot replace failures or tune label rules.

Event-dev has 103 approved source identities, but its payload remains unopened
and its recording requires a separate bounded child plan. Event-report's 77
sources stay locked until all training decisions are sealed. Train/dev/report
payload staging, capture, labels and prompt manifests need separate provenance;
the initial train-side capture does not authorize held-out reads. Model/recipe
implementation and numerical/resource qualification can proceed independently,
but optimization must wait for dataset and launch admission.

The new tests use explicitly synthetic references only. They verify falsified
hashes, untrusted/self-generated attestations, missing episodes/frames/negative
checks, source mismatches, one-to-one matching, false positives/negatives, positive
coverage, censoring and boundary errors. Their passing results are software
verification, not evidence that these 200 recordings have valid labels.
