# M1 evaluation workflow

Prepared October 5, 2026; **completed October 6**. See the
[results and package](WEEK7_M1_RESULTS.md) and [completion record](evaluation/week7-m1-final-v1/completion.json).
The actual successful v2 launch used a separate qualified GPU-guard wrapper
and a recorded B0-compatible environment drop-in. Two startup failures consumed
no episodes. The original v1 preparation and commands below are retained as
historical evidence, not instructions to rerun this completed experiment.

The frozen
[contract](../configs/evaluation/week7_m1_dev_v1.json) accepts only M1's completed
epoch-2 endpoint: 117,132 updates, 3,412,044 action presentations and 16,259,770
future pairs. The M3/M4 [design freeze](WEEK8_COUPLED_DESIGN.md) predates all new
rollout outcomes. Epoch 1 is not a substitute endpoint or a selection candidate.

## Prepared workflow

The separate `evaluate_m1_dev.py` runner strictly links the production manifest,
epoch report, completion record and both exported file hashes. Its loader accepts
the M1 action-only tensor envelope and restores the original OFT architecture,
normalizers and exact trainable parameter names. The auxiliary predictor is
preserved as evidence but is not loaded for inference. Existing accepted training
and evaluator source files are unchanged.

The package reuses all 215 files from M1's accepted production source and adds
the wrapper, contract, analysis helpers, tests and inactive service template.
Both the live training source and packaged evaluation source are hash-checked.
The recorder's Git revision denotes the base commit; the additional full source
digest identifies the executed code, including changes outside that commit.

Preflight checks the paired dev-210 manifest, protocol, action/state normalizers,
exact B0 package versions, LIBERO revision, and availability of all 210 baseline
episode files. It reports `waiting_for_epoch_2` until both the export and final
completion record exist. It does not allocate a model on the GPU or run a rollout.
Launch additionally requires the training owner lock and an idle GPU; competing
training cannot be displaced. Evaluation has a 48-active-hour cap, 100-GiB output
cap and 50-GiB free-space reserve. The service has a 49-hour backstop.

## After the fixed training endpoint

On Feifei, from the isolated evaluation package directory:

```bash
export PYTHONPATH=/home/peimengsui/code/eventground-wam-m1-evaluation-v1/src
export PYTHONDONTWRITEBYTECODE=1
/home/peimengsui/code/eventground-wam-b0-oft-v1/environments/b0/.venv/bin/python scripts/evaluate_m1_dev.py preflight --output-root /home/peimengsui/data/eventground-wam/libero-plus/evaluation/week7-m1-dev-v1
```

Proceed only with `endpoint_verified_launch_pending`, after checking the finished
training service and GPU. Install and start the supplied service manually:

```bash
cp configs/systemd/eventground-m1-evaluation-v1.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start eventground-m1-evaluation-v1.service
```

The preparation step does not install, enable or start this service. The run
records immutable per-task receipts, permitting resume while preserving scored
policy failures and exhausted infrastructure retries. No extra attempt is
granted merely because the service restarts. Inherited evaluation allows one
infrastructure retry and retains partial attempt artifacts. Policy exceptions,
invalid actions and timeouts remain failures under the frozen protocol.

When all 210 tasks finish, run analysis in the same environment:

```bash
/home/peimengsui/code/eventground-wam-b0-oft-v1/environments/b0/.venv/bin/python scripts/evaluate_m1_dev.py analyze --output-root /home/peimengsui/data/eventground-wam/libero-plus/evaluation/week7-m1-dev-v1
```

Analysis validates recorded outcomes/provenance and hashes every canonical
settling frame and first-query camera/state input, including simulator state and
pre-query executed actions, against the frozen B0 episodes. All 210 rows remain
in the comparison. Any mismatch makes the entire contrast descriptive; missing
or corrupt artifacts also leave the artifact audit incomplete. Do not substitute
another B0 report silently. If baseline/runtime compatibility prevents reuse,
record the issue and plan a same-checkpoint B0 reevaluation before drawing paired
conclusions. Week 4's historical rendering decision remains unchanged.

The analysis emits counts/rates, Wilson 95% intervals, paired wins/losses,
10,000 task-bootstrap draws with PCG64 seed 7 and linear 2.5/97.5 percentiles,
suite/category/difficulty breakdowns including Long, and query-latency summaries.
Identical task-index draws are used when future arms are compared. These are
single-seed development summaries; they do not establish universal superiority.

## Packaging and completion after analysis

1. Preserve the endpoint checkpoint and adapted predictor separately, their SHA256
   hashes, production completion/exposure chains, full training logs, and recipes.
2. Retain all rollout attempts, immutable receipts, source/runtime/panel identities,
   paired-input audit and summary. Copy a checksum-verified backup before reusing
   storage. Preserve artifact errors and prior failed qualifications.
3. Add training wall time, peak memory, parameter counts, inference latency,
   inherited CPT cost and all pilot/tuning exposure to the results narrative.
   Report positive, null or negative results under the same rules.
4. Review representative failures/limitations and seal a final artifact inventory.
   The analysis file deliberately does not claim this narrative/backup review is
   complete. Trajectory-v2 endpoint validation, if run, is diagnostic only and
   cannot select or retune checkpoints. Test and larger panels stay locked.

M1 can be packaged independently while M2 continues on Lambda. Week 7 as a whole
remains open until both arms have completed their required evaluation and evidence.
M3/M4 implementation and train-only hardware qualification can proceed against
their sealed design; full M3 training on Feifei follows release of the GPU after
M1 evaluation. An A100 for M4 should be provisioned when its qualification package
is ready, rather than left waiting for code.

## Preparation evidence

See [package and verification](evaluation/week7-m1-evaluation-v1/) and
[M3/M4 freeze verification](evaluation/week8-design-freeze-v1/verification.json).
The original preparation marked GPU inference, simulator rollout, endpoint
reload and packaging as pending. Those steps are now complete in the
[October 6 closeout](WEEK7_M1_RESULTS.md), with 207/210 exact canonical prefixes
and the entire comparison retained as descriptive under the frozen rule.
