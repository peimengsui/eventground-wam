# Demonstration data contracts

Week 2 derived data uses schema version `1` from
`eventground_wam.demonstration_contracts`. Records are JSON objects suitable for
newline-delimited JSON manifests. Large arrays and images are not embedded: a
trajectory points at the existing Parquet and video shards so later readers can
stream the 14.897 GiB source snapshot without making a second copy.

## Locked source

All v1 records identify this exact input:

- repository: `Sylvest/libero_plus_lerobot`;
- revision: `22c57433fef692b5b9ecc0795344daac7fa867a5`;
- source-manifest SHA-256:
  `e673a1d3e58a1756879b4f30ce38474aa3407f97232ee5184fd7e1e6ed901510`;
- 14,347 episodes and 40 tasks, with contiguous zero-based indexes;
- 20 Hz samples, 8D `observation.state`, and 7D `action`;
- `observation.images.front` and `observation.images.wrist`, both 256 × 256 RGB;
- no source MuJoCo state and no source initial seed.

The absence of MuJoCo state and initial seed is represented explicitly as
`source_has_mujoco_state: false` and `source_initial_seed: null`. A validator
rejects records that claim otherwise. An initial-condition surrogate may be
derived later, but it must not be described as the original simulator seed.

## Common envelope and provenance

Every serialized record has exactly four envelope fields in addition to its
type-specific fields:

| Field | Meaning |
| --- | --- |
| `artifact_type` | One of the five names below. |
| `schema_version` | Integer `1`; new incompatible shapes require a new version. |
| `provenance` | Pinned source, source episode/task/length, project revision, and annotation version. |
| `digest` | Lowercase SHA-256 of every other field in canonical JSON form. |

Canonical JSON uses sorted object keys, compact separators, ASCII escaping, and
rejects non-finite floats or non-JSON values. The top-level `digest` field is
excluded from its own hash. Digests are stable content identifiers and foreign
keys; changing an annotation, frame boundary, source reference, or provenance
changes its digest.

`project_git_revision` is a complete lowercase 40-hex commit SHA.
`annotation_version` is a lowercase identifier such as `week2-pilot-v1`. All
frame ranges are nonempty, half-open `[frame_start, frame_end)`, and must remain
within the named episode.

## Record types

### `eventground_wam.trajectory`

One complete source episode. It contains the task instruction, complete episode
frame range, source rate and shapes, and three repository-relative paths:

- one `data/chunk-NNN/episode_NNNNNN.parquet` shard;
- front-camera video;
- wrist-camera video.

`TrajectoryRecord.from_episode(...)` derives the exact LeRobot paths. The v1
validator requires the full episode range and the pinned layout, which gives a
streaming reader deterministic ordering by `source_episode_index` without
decoding or copying media while it builds the index.

### `eventground_wam.event_candidate`

A raw detector proposal before temporal filtering or ontology admission. It
contains a trajectory foreign key, proposed event type and interval, confidence,
detector name/version, and nonempty evidence descriptions. Candidate records are
retained even when a downstream policy does not admit them into the reliable
ontology. Absence is represented by no candidate for that detector/event pair,
not by an empty or invented interval.

Evidence strings begin with a modality prefix:
`state:`, `action:`, `front_rgb:`, `wrist_rgb:`,
`simulator_sidecar:`, or `human_review:`. State/action evidence alone may support
`gripper_close`. Because this snapshot lacks object and MuJoCo state, the other
object-relative event types require RGB, simulator-sidecar, or human-review
evidence. This prevents an action command from being silently relabeled as
physical contact, grasp, placement, or success.

The locked v1 event names are `reach`, `contact`, `gripper_close`,
`grasp_established`, `lift`, `transport`, `target_region_entry`, `release`, and
`success`.

### `eventground_wam.event_interval`

A filtered event admitted as a training annotation. It retains one or more raw
candidate digests, merged evidence, confidence, and one of two ontology tiers:
`reliable` or `weak`. Review status is `unreviewed`, `accepted`, or `corrected`.
Rejected proposals remain candidates and do not become final intervals.

### `eventground_wam.split_assignment`

An episode-level assignment to `train`, `validation`, or `test`. Its frame range
must cover the complete episode. V1 uses
`task-state-rgb-surrogate-v1`: a digest derived from task identity, initial
proprioception, and initial-frame fingerprints. Every episode with the same
`initial_condition_digest` must be assigned to the same split; the dataset-level
split generator is responsible for checking that invariant across records.

### `eventground_wam.multi_horizon_sample`

A future target identified by an anchor frame, target frame, trajectory digest,
and split-assignment digest. Fixed horizons are exactly `2`, `4`, `8`, `16`, or
`32` frames. Their target must equal `anchor + horizon` and cannot name an event.
A `next-event` sample instead references a final event interval and targets that
event's first frame. The sample range is exactly `[anchor, target + 1)`, making
episode-boundary checks local and unambiguous.

At 20 Hz, the fixed horizons correspond to 0.1, 0.2, 0.4, 0.8, and 1.6 seconds.

## Validation and versioning

Use `validate_contract` for a deserialized object, `validate_contract_sequence`
for a JSONL stream, `validate_trajectory_reference` for trajectory foreign keys,
and `validate_sample_references` for split and next-event foreign keys. Validation
is strict: missing and unknown fields, wrong scalar types, invalid digests,
unrecognized enums, unsupported evidence, and cross-episode frame ranges fail.

V1 records are immutable. Corrections produce a new record with updated
`annotation_version`, provenance, and digest. Adding an optional field is also a
schema change because v1 rejects unknown keys; this keeps regeneration and
training inputs identical across implementations.
