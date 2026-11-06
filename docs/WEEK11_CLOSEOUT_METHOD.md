# Week 11 formal closeout and fixed-control audit

This analysis child fulfills the user's request to complete the frozen formal
analysis and compare the two fixed controls episode by episode. It reads the
completed 2,001-episode evaluation and starts no training or new rollouts.
The original experiment, mapping, launch package and raw evidence remain intact.

The child archive extends the exact formal source archive without replacing any
inherited byte. A separate `analysis-source-manifest.json` hashes the additions
and binds the formal source digest. Both the original execution directory and
the child's imported dependencies are rehashed against that formal manifest.

The analyzer first hashes every raw output file and then independently verifies
the completed cohort, report digests, exact task manifests, receipts, attempt
journals, HDF5 schema/outcomes, timing sidecars, event mappings, all-query latency,
and settling/first-query input audits. The input/output directory inventory,
sizes and modification times must remain unchanged throughout the read-only
audit. A shared process lock prevents simultaneous evaluation execution.

Statistics follow the frozen protocol: all three paired contrasts share 10,000
PCG64 bootstrap draws, linear 2.5/97.5 percentiles. Targeted tasks use seed 7.
Clean uses seed 8 and resamples the 40 canonical tasks, retaining all five
states in each selected task. Every predefined targeted one-way subset uses
seed 7 and the same draw count; all three contrasts share indices within that
subset. Suite, category, difficulty including Long/unlabeled, dev-overlap90
versus other377, and contact-proxy98 are reported descriptively. The remaining
369 tasks are included as the contact-proxy complement. There is no multiplicity
adjustment or claim about training-seed variance or unseen-task generalization.

The control audit compares all 667 fixed/fixed8 episode pairs, not merely their
aggregate totals. It streams every non-timing HDF5 step field, records bitwise
equality, first differing step, differing row count and numeric magnitude, and
retains unequal-length tails. At matched policy queries it compares both uint8
camera arrays, float32 proprioception (the action adapter's actual input cast),
and the decoded predicted action chunk. Instructions are verified identical.
Any different action on identical inputs is flagged separately. First-input
divergence records include camera differences, simulator state, raw/effective
proprioception, and equality of all preceding executed actions. These observations
can establish the order of recorded differences; they do not establish a causal
mechanism for the adaptive policy's failures.

The existing full-query and component timings, per-episode horizon/event/fallback
counts and simulator/recording costs are preserved and summarized. Representative
case IDs select the first manifest ordinal in each paired outcome class, without
choosing favorable examples. Existing weak-label mapping diagnostics retain
their earlier scope; no new event-accuracy claim is made.

Analysis time is recorded in a separate crash-persistent ledger and added to
the completed qualification/evaluation total. Per-panel re-audits are charged
alongside the original panel ledgers; full control comparison is additionally
charged to the fixed8 panel. Worker process-tree RSS and disk reserve are checked
between records/pairs. This CPU analysis performs zero GPU inference calls.
Raw recordings remain on Feifei under their complete SHA-256 inventory; the
analysis, source and metadata are mirrored locally. The original raw files are
not deleted or rewritten.

The first analysis attempt stopped while summarizing a nested component-timing
dictionary, after inventory and first-panel validation. Its source and ledgers
are preserved under v1. Package v2 was not executed remotely; its local test
collection exposed an eager import of the Feifei-only RSS dependency. The
corrected v3 loads that dependency only for resource monitoring and adds the
stopped v1 attempt's aggregate
and panel charges through `--prior-analysis`; no rollout is retried or replaced.

Example command from the extracted analysis source:

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/peimengsui/code/eventground-wam-week4-run-v2/environments/b0/.venv/bin/python \
  scripts/analyze_week11_formal.py \
  --source-root /home/peimengsui/code/eventground-wam-week11-formal-v1 \
  --recordings /home/peimengsui/data/eventground-wam/libero-plus/results/week11-formal-v1 \
  --output /home/peimengsui/data/eventground-wam/libero-plus/evaluation/week11-closeout-v3 \
  --prior-analysis /home/peimengsui/data/eventground-wam/libero-plus/evaluation/week11-closeout-v1 \
  --workers 2
```

The final decision applies every original performance, interval, latency,
artifact/input and resource criterion. Failure remains a failure under v1;
there is no automatic follow-up launch, mapping revision or panel extension.
