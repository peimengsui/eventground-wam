# Reproduction guide

## Rebuild the reported analysis

The default workflow uses retained local metadata and images. It requires no GPU,
checkpoint, simulator or remote connection. Use Python 3.11 with the versions in
[analysis-requirements.txt](analysis-requirements.txt). These pins describe the
working analysis runtime, including its required transitive dependencies; they
are separate from the Python 3.10 experiment runtimes. No project installation
or `uv sync` is needed for these standalone reporting scripts.

From this repository root, verify and extract into a new directory:

```bash
python3 scripts/verify_week12_reproducibility.py \
  --package docs/evaluation/week12-repro-v1 \
  --extract-to /tmp/eventground-wam-step4-reproduction
```

The destination must not exist. Verification checks the archive SHA-256, every
member's path/size/hash, and the final package inventory. Extraction accepts only
regular files with safe relative paths; it refuses existing destinations. For a
standalone copy of this package, run its identical `verify.py` with `--package .`.

Then, from the extracted root:

```bash
cd /tmp/eventground-wam-step4-reproduction
python3.11 scripts/reproduce_week12_analysis.py --output /tmp/eventground-wam-step4-results
```

The second destination must also be new. The wrapper verifies all 120 original
analysis inputs, runs the numerical builder, figure renderer, prose/table writer
and artifact verifier, and compares thirteen deterministic outputs with the
sealed reference. Success produces `reproduction-check.json` with 2,210 numerical
checks and exact hashes for three JSON files, five CSV tables and five chart PNGs.
The regenerated `gallery.html`, `RESULTS.md`, `CAPTIONS.md` and `QUALITATIVE.md`
are the review entry points. PDF creation metadata and SVG element IDs can differ;
PNG equality is tied to the recorded rendering dependencies and fonts.

On a machine without these dependencies, create a separate Python 3.11 environment
and install the supplied pins before running the wrapper:

```bash
python3.11 -m venv /tmp/eventground-wam-analysis-env
/tmp/eventground-wam-analysis-env/bin/python -m pip install \
  -r docs/evaluation/week12-repro-v1/analysis-requirements.txt
```

The pins and rebuild were checked in the available runtime. A clean dependency
installation and cross-platform pixel equivalence were not tested. This package
does not contain a wheelhouse or assert an offline installation of Python packages.
Use the environment's Python in place of `python3.11` after installation.

To check only the existing sealed results, use:

```bash
python3.11 scripts/verify_week12_results.py --output docs/evaluation/week12-results-v1
```

These commands never invoke historical launch scripts. The artifact verifier
checks local files and prior metadata; it does not reopen remote raw recordings.

## Locate the implementation and exact historical source

[bundle-manifest.json](bundle-manifest.json) identifies every captured file.
The archive contains `src/`, `scripts/`, `tests/`, configs, environment locks,
allowed evaluation panels, top-level design documents, model/normalizer metadata,
the sealed Steps 1–3 packages, and consumed analysis evidence. Unneeded historical
evidence directories are referenced rather than copied in full; this is not a
mirror of every repository document or every old experiment output.

[repository-state.json](repository-state.json) records HEAD and the dirty-tree
status. HEAD alone does not identify the completed implementation. The captured
file hashes identify this reporting snapshot; original source-manifest digests
identify historical executions. Do not replace old source archives with current
code merely because filenames match.

[historical-sources.json](historical-sources.json) binds the executed Week 9
runner, Week 11 formal runner and Week 11 closeout analyzer to their source
digests and archive hashes. All three match their original package receipts.
The Week 9 archive is bundled unchanged. The two Week 11 archives remain at
their recorded repository paths; their source manifests and archive hashes are
bundled. Those original archives contain full-10K panel metadata, so they are
referenced rather than copied into this analysis bundle. Week 10 and training source/manifests are indexed
through their original packages and endpoint admissions in the run catalog.
The current source tree provides implementation access, not a claim that every
file was identical in every historical run.

## Experiment environments and inputs

[environments.json](environments.json) preserves the root and four specialized
`uv` projects, lock-file hashes, full Week 9/11 package inventories, CUDA/cuDNN,
GPU/driver identity, environment paths and numerical flags. Training and robot
evaluation used Python 3.10.21, PyTorch 2.13.0, Transformers 5.17.0 and MuJoCo
2.3.7 in the B0 runtime. The auxiliary reference-policy environment has its own
Transformers fork; it is not the B0 runtime.

The original setup commands, for a separately authorized reconstruction, are:

```bash
uv sync --project environments/b0 --locked
uv sync --project environments/demonstrations --locked
uv sync --project environments/libero --locked
# Only for the separate public reference-policy comparison:
uv sync --project environments/reference-policy --locked
```

These locks are retained inputs, not proof that every source checkout or hardware
backend yields the same rollout. Use the recorded runtime from the relevant
cohort. In particular, Week 9 recorded `deterministic_algorithms=false`, whereas
Week 11 recorded `true`. Both retained their full numerical flag sets. The shell
runners and systemd definitions are preserved as historical execution references;
do not start completed services as part of analysis reproduction.

[external-assets.json](external-assets.json) names the immutable Qwen revision,
benchmark revision and asset checksum, demonstration revision/source-manifest
hash, trajectory-v2 split, future-pair/order manifests and train-only action/state
normalizers. Large assets are referenced by identity and expected location.
Test split metadata describes the frozen partition, but no test payload or
full-10K panel is copied or evaluated in this workflow.

[checkpoints.json](checkpoints.json) lists B0/M1–M4, predictor exports, shared
pixel/JEPA modules, all five revised-task event endpoints (including the
provisional event predictor), and retained exploratory robot endpoints.
Hashes are inherited from admissions, epoch reports and closeout
receipts. Step 4 verifies those local receipts, not current remote checkpoint
bytes. The prior Week 11 closure freshly hashed its nine retained files; that
dated observation is preserved. Policy exports need the pinned backbone,
processor, normalizers and runtime; they are not self-contained model bundles.

## Historical commands and experiment lineage

[run-catalog.json](run-catalog.json) maps each stage to its entry point, config,
protocol, source/endpoint records and output evidence. The following are argument
templates describing the original execution surfaces. Variables denote the
matching frozen paths from that stage's package, not arbitrary replacements.
They are not executed by Step 4.

| Stage | Historical command form | Required lineage |
| --- | --- | --- |
| B0 fixed epoch 2 | `train_b0_fresh.py --phase train --until-epoch 2 --work-root <new-root>` | Fresh all-L1 trajectory-v2 recipe; accepted recovery pilot; pinned Qwen and normalizers |
| Pixel / JEPA CPT | `train_pixel_cpt.py` / `train_jepa_cpt.py --phase train --order <order> --pairs <pairs> --output <new-root> --pilot-report <report>` | Shared one-pass order, 221,633 updates; original predictor gates retained |
| M1 auxiliary | `train_auxiliary_policy.py --recipe <M1-recipe> --output <new-root>` | Accepted production source and two-epoch action/future exposure |
| M2 auxiliary | `train_auxiliary_lambda.py --profile <profile> --output <new-root>` | Lambda qualification, M2 recipe and budget amendment |
| M3 coupled | `train_coupled_policy.py --arm M3 --profile <profile> --qualification <qualification> --output <new-root>` | Coupled source, executable detached proposals and fixed exposure |
| M4 coupled | `train_coupled_lambda.py --profile <profile> --qualification <qualification> --output <new-root>` | Lambda source/qualification and M4 budget amendment |
| Week 9 evaluation | `evaluate_week9.py run --root <cohort> --arm <arm> --panel <perturbed-or-clean>` | Existing immutable cohort, all endpoint admissions and runtime lock |
| Revised event training | `train_week10_future_onset.py --base <frozen-base> --stage <stage>` | Protocol, source index, prompt manifests and schedule; closed stages stay closed |
| Preservation comparison | `train_week10_preservation.py --base <frozen-base> --stage <stage>` | Matched SFT parent, selected preservation weight and schedule |
| Week 11 formal | `run_week11_formal.py --output <frozen-output>` | Frozen three-arm contract, mapping, admitted endpoints and qualified source |
| Matched continuation | `run_week11_multihorizon_training.py --root <root> --arm <arm> --phase train` | Two matched 2,000-update endpoints and separate fixed screen |
| Frozen-H8 adapters | `run_week11_motion_training.py --bundle <bundle> --root <root> --arm <arm> --phase train` | Matched motion/control arms; screen closed at latency stop |

Host/base/output checks are intentionally strict in the original scripts. A new
experiment requires a new scoped contract and output identities; editing a path
or disabling a guard is not faithful reproduction. Follow the prior source
package's setup and environment before using an entry point. Do not treat these
templates as a one-command training pipeline.

For a raw-recording audit, the original Week 9 `analyze_week9.py --root <cohort>`
and Week 11 `analyze_week11_formal.py` require their executed source archives,
the full raw inventories, timing sidecars and resource-ledger handling. The
Week 11 closeout method is retained in the source snapshot as
`docs/WEEK11_CLOSEOUT_METHOD.md`. That operation is separate from the portable
metadata rebuild tested here.

## Output coverage and scientific limits

[outputs.json](outputs.json) links main results and exploratory follow-ups to
local package/index hashes and remote raw locations. The formal raw inventory
lists 10,048 files; the prior closure checked their presence and sizes. Step 4
does not claim to have freshly rehashed those raw bytes. Week 9 episode reports
retain individual artifact paths and receipts. Event prediction JSONL files and
selected images are bundled; raw observation recordings remain external.

The known prior prose/hash discrepancy in the adapter training report is carried
forward from the Week 11 evidence ledger. It is not silently repaired or hidden
by regenerating an old package index. The source archive includes its current
receipt/reference, and the original scientific decision stays unchanged.

Formal protocols remain subject to one training seed, reused panels/development
cases, renderer/input differences, historical source-alias exposure, JEPA's failed
gate and research-use exception, and unresolved independent/on-policy event QA.
Metadata equality is not an independent replication of training or robot behavior.
Full-10K/test, new seeds and new runs remain outside the frozen reporting scope.
