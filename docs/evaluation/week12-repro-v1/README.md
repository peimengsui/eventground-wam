# Week 12 Step 4: reproducibility package

This package captures reproducibility references for the completed study and
provides a portable rebuild of the consolidated numerical results and figures.
It adds no training, inference, rollouts, label collection or checkpoint selection.

- [Reproduction guide](GUIDE.md): commands, prerequisites and historical rerun boundaries.
- [Portable source and analysis inputs](analysis-reproduction.tar.gz), with a
  [per-file manifest](bundle-manifest.json) and [standalone verifier](verify.py).
- [Checkpoint manifest](checkpoints.json) and [external assets](external-assets.json).
- [Environment records](environments.json) and [analysis dependency pins](analysis-requirements.txt).
- [Run catalog](run-catalog.json), [outputs](outputs.json), and
  [historical source archives](historical-sources.json).
- [Repository snapshot identity](repository-state.json),
  [verification results](verification.json), and [sealed inventory](package-index.json).

The archive includes retained metadata under `outputs/`; a fresh Git checkout
alone would omit those inputs and much of the completed work. Current code and
historical executed code are identified separately. Model weights, raw robot
recordings, demonstration videos and locked full-10K/test panels are not bundled.

The existing results are at [Steps 2–3](../week12-results-v1/README.md).
The [status ledger](../../WEEK12_STATUS.md) records Steps 1–4 complete and
Steps 5–6 unstarted. Prior sealed status snapshots remain historical records.
