# Week 2 completion status

**Overall status as of 2026-09-10:** complete. Engineering, the complete dataset
freeze, and the human boundary-quality gate are finished. The reviewer ended the
original 200-episode queue after a balanced 60-episode subset produced 79 of 79
accepted `gripper_close` boundaries, and confirmed that the full episodes were
also checked for missed events. This is a documented, reviewer-authorized change
from the original 200-episode plan rather than a claim that all 200 were reviewed.

| Roadmap step | Status | Evidence |
| --- | --- | --- |
| 1. Lock data contracts | Complete | Versioned trajectory, event, split, and multi-horizon contracts and validators. |
| 2. Validate streaming reader | Complete | Complete metadata index plus synchronized Parquet/front/wrist validation. |
| 3. Event-labeling pilot | Complete | Telemetry feasibility established `gripper_close` as the only supported candidate type. |
| 4. Deterministic extraction | Complete | Raw candidates, weak intervals, and explicit admitted/absent/ambiguous decisions. |
| 5. Leakage-resistant splits/indexes | Complete | Group-surrogate splits and fixed horizons `2, 4, 8, 16, 32`; weak events are excluded from `next-event`. |
| 6. Boundary visualization and QA | Complete | 60 full episodes and 79 predictions reviewed: 100% precision/recall, zero boundary corrections, and zero ambiguous judgments. |
| 7. Scale and freeze | Complete | `week2-dataset-v1` covers all 14,347 episodes; primary and replica match byte-for-byte. |

## Deliverable ledger

- **Complete:** versioned schemas and contract validators.
- **Complete:** reproducible reader, labeler, extractor, split/sample generator,
  freeze validator, and preview renderer.
- **Complete:** complete-source derived artifacts outside Git, plus committed
  compact provenance, statistics, hashes, and throughput evidence.
- **Complete with documented protocol change:** a 60-episode completed
  human-review report with event-specific precision, recall, boundary error,
  coverage, ambiguity, and corrections. The original target was 200 episodes.

The deterministic-regeneration and split-isolation portions of the Week 2 gate
pass. Human review measured 100% precision and recall on the accepted subset,
with zero-frame mean absolute boundary error and no corrections or ambiguous
judgments, so `gripper_close` passes the approximately 95% reliable-event quality
gate. The immutable `week2-dataset-v1` artifacts are not rewritten in place: its
16,742 intervals remain encoded as weak/unreviewed and excluded from `next-event`
supervision. Promoting the validated extractor output requires a new dataset
version.

## Evidence locations

- Frozen artifacts on `feifei`:
  `~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1/`
- Frozen-data human-review package on `feifei`:
  `~/data/eventground-wam/libero-plus/derived/boundary-qa/week2-dataset-v1/`
- Committed freeze evidence:
  [`freeze/week2-dataset-v1.json`](freeze/week2-dataset-v1.json)
- Committed QA-package evidence:
  [`freeze/week2-dataset-v1-qa.json`](freeze/week2-dataset-v1-qa.json)
- Committed human-QA completion evidence:
  [`freeze/week2-dataset-v1-human-qa.json`](freeze/week2-dataset-v1-human-qa.json)

The completed subset contains 20 short, 20 middle, and 20 long episodes spanning
34 of the 38 tasks represented by the full queue. It contains 79 true positives,
zero false positives, zero false negatives, zero corrections, and zero ambiguous
judgments. The completed-review export and digest-matched subset both pass the
strict review schemas; the resulting report has status `human_review_complete`.
Any promotion into training artifacts must create a new immutable dataset version
rather than changing `week2-dataset-v1` in place.
