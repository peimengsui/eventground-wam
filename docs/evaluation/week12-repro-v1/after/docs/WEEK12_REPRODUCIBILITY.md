# Week 12 reproducibility handoff

**November 1, 2026: Step 4 is complete.** The
[reproduction package](evaluation/week12-repro-v1/README.md) captures the current
source/config/test files, the inputs needed to rebuild the consolidated analysis,
historical execution references, dependency versions, and checkpoint/output
manifests. The [scope](WEEK12_SCOPE.md) and previous evidence packages are unchanged.

Start with the [reproduction guide](evaluation/week12-repro-v1/GUIDE.md).
The [portable bundle](evaluation/week12-repro-v1/analysis-reproduction.tar.gz)
contains the local inputs needed for a CPU analysis rebuild, including retained
metadata that normally lives in ignored `outputs/` directories. It also includes
the current implementation and tests, the exact Week 9 runner archive, and
historical source manifests. A Git checkout of the recorded HEAD alone is insufficient:
the completed work includes uncommitted and untracked files captured by hash.

The bundle was extracted into a new directory and used to rerun all four
reporting scripts. The rebuild passed all 2,210 numerical checks and reproduced
the three numerical/provenance JSON outputs, five CSV tables and five chart PNGs
exactly. See [verification](evaluation/week12-repro-v1/verification.json).

The package distinguishes three operations:

| Operation | Available now | Boundary |
| --- | --- | --- |
| Rebuild tables and figures | Portable CPU inputs, scripts, pinned analysis dependencies and an equality check | No models, simulator or network access required after dependencies are installed |
| Audit original recordings/checkpoints | Exact paths, hashes, prior audit receipts and original source identities | Large payloads remain outside the bundle; no new remote audit was performed |
| Rerun training or rollouts | Historical entry points, configs, source identities and runtime requirements | Requires a separate scoped decision and new output identities; not executed by this package |

The [checkpoint manifest](evaluation/week12-repro-v1/checkpoints.json),
[external assets](evaluation/week12-repro-v1/external-assets.json),
[environment records](evaluation/week12-repro-v1/environments.json),
[run catalog](evaluation/week12-repro-v1/run-catalog.json) and
[output index](evaluation/week12-repro-v1/outputs.json) supply the report's
reproducibility references. Historical source snapshots take precedence over
current working-tree code when reconstructing an old execution.

This completes packaging, not an independent replication of GPU training or
robot behavior. Checkpoints still depend on the pinned backbone, processor,
normalizers and benchmark assets. Renderer differences and original scientific
qualifications remain. Full-10K/test stay locked, and the adapter screen remains
closed at 27/81. [Step 5, the technical report](WEEK12_STATUS.md), is next.
