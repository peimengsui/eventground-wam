# Week 2 completion status

**Overall status as of 2026-09-07:** engineering and the complete dataset freeze
are finished. Week 2 is not yet closed because the required 200 human reviews
and event-specific quality metrics are still pending.

| Roadmap step | Status | Evidence |
| --- | --- | --- |
| 1. Lock data contracts | Complete | Versioned trajectory, event, split, and multi-horizon contracts and validators. |
| 2. Validate streaming reader | Complete | Complete metadata index plus synchronized Parquet/front/wrist validation. |
| 3. Event-labeling pilot | Complete | Telemetry feasibility established `gripper_close` as the only supported candidate type. |
| 4. Deterministic extraction | Complete | Raw candidates, weak intervals, and explicit admitted/absent/ambiguous decisions. |
| 5. Leakage-resistant splits/indexes | Complete | Group-surrogate splits and fixed horizons `2, 4, 8, 16, 32`; weak events are excluded from `next-event`. |
| 6. Boundary visualization and QA | Human review pending | The frozen-data package contains 200 review episodes and 254 rendered previews, but zero completed judgments. |
| 7. Scale and freeze | Complete | `week2-dataset-v1` covers all 14,347 episodes; primary and replica match byte-for-byte. |

## Deliverable ledger

- **Complete:** versioned schemas and contract validators.
- **Complete:** reproducible reader, labeler, extractor, split/sample generator,
  freeze validator, and preview renderer.
- **Complete:** complete-source derived artifacts outside Git, plus committed
  compact provenance, statistics, hashes, and throughput evidence.
- **Pending:** a completed 200-episode human-review report with event-specific
  precision, recall, boundary error, coverage, ambiguity, and corrections.

The deterministic-regeneration and split-isolation portions of the Week 2 gate
pass. The approximately 95% reliable-event quality requirement has not been
measured. Consequently, all 16,742 frozen intervals remain weak/unreviewed and
all are excluded from `next-event` supervision. This prevents the pending human
gate from silently leaking into training.

## Evidence locations

- Frozen artifacts on `feifei`:
  `~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1/`
- Frozen-data human-review package on `feifei`:
  `~/data/eventground-wam/libero-plus/derived/boundary-qa/week2-dataset-v1/`
- Committed freeze evidence:
  [`freeze/week2-dataset-v1.json`](freeze/week2-dataset-v1.json)
- Committed QA-package evidence:
  [`freeze/week2-dataset-v1-qa.json`](freeze/week2-dataset-v1-qa.json)

To close Week 2, reviewers must replace the 200 pending templates with completed
judgments and run `scripts/run_boundary_qa.py summarize`. Only an event type
meeting the quality gate may be promoted to the reliable ontology; any promoted
or boundary-corrected annotations require a new immutable dataset version rather
than changing `week2-dataset-v1` in place.
