# Week 4 paired clean comparison execution

The immutable acceptance rule is `week4-clean-acceptance-v1`, digest
`0e3404fbe44d16ad5d6742948d8aeaa8a388cfd04ac6527d21454f1dd0e82f03`.
The panel contains 40 canonical clean tasks × 50 distinct initial states: 2,000
fresh episodes for each policy. All B0 episodes precede all reference episodes.
This is an evaluation-matched external reference, with different training,
architecture, native preprocessing and normalization retained.

## Implementation

`evaluate_clean_comparison.py` uses the existing `run_episode` control loop
unchanged. The new sealed protocol adds trial identity and selects each frozen
initial state and seed. Preflight checks the original acceptance freeze, all
2,000 state hashes, task inventory, runtime locks and checkpoint bytes. Each
policy runs in its existing locked virtual environment; no environment sync or
model download occurs.

A durable attempt journal is written before environment construction. Each
completed HDF5 artifact and receipt is verified on resume. Crashes after HDF5
publication recover the outcome without rerunning inference. Interruptions
before or during an episode consume an attempt; only one infrastructure retry
is allowed, with exactly the same task, state and seed. Partial artifacts and
attempt journals remain on disk. Completed policy failures are never retried.
Exhausted infrastructure failures remain zero outcomes and block acceptance.

The journal captures raw inputs before the first policy query, including when
that query fails. The reference checks those inputs against B0; final analysis
also compares the full recorded settling observations and simulator states.
All receipts and HDF5 files are revalidated before the frozen NumPy 1.26.4
bootstrap analysis can produce an acceptance decision.

## Host execution

Run `scripts/run_clean_comparison.sh OUTPUT_ROOT` in a persistent user systemd
service on Feifei. The workflow handles B0, reference and final analysis in
sequence and resumes after a service restart. Use `flock` on `OUTPUT_ROOT/job.lock`
to prevent concurrent launches; set `Restart=on-failure`, `RestartSec=60`,
`RestartPreventExitStatus=78`, `KillMode=control-group`, and enable the service
under `default.target`. User lingering must remain enabled so logout does not
stop execution. Exit 78 indicates an integrity issue needing investigation.

Before formal execution, use a separate output root with `--pilot-pairs 2`.
These engineering episodes and their restart evidence are excluded from formal
analysis. Pilot mode cannot produce a scientific acceptance decision.

Outputs include per-policy `identity.json`, `progress.json`, sealed episode
receipts, attempt journals, HDF5 recordings and `evaluation-report.json`. The
root `comparison-report.json` contains paired validity checks, task outcomes,
contingency counts, policy summaries and the final interval/decision.

Budget roughly 200 GB for recordings and evidence, based on the completed
clean-40 runs; actual size varies with episode length. The comparison writes
no new model checkpoints. Completion of this run alone does not close Week 4:
`dev-210`, the seven-category failure analysis and final evidence freeze remain.

## Launch evidence — 2026-09-18

Source revision: `7e40768d74fa37f856292d7a1b91e5fd1403122f`.
All 248 local tests and Ruff passed. The [pilot report](evaluation/week4-clean-comparison-pilot-v1.json)
verifies two paired initial states, exact raw-input equality, and one intentional
B0 infrastructure interruption followed by successful recovery. Both policies
completed the engineering pilot; its service exited successfully.

The formal service `eventground-week4-clean-comparison.service` started on
Feifei at 08:04:26 PDT. The [launch audit](evaluation/week4-clean-comparison-launch-v1.json)
was captured from a fresh SSH connection at 08:05:32 PDT: 11/2,000 B0 episodes
complete, zero formal infrastructure retries, and the service enabled/running
with user lingering. Early counts are operational checks and do not determine
the acceptance result.

Formal output: `/home/peimengsui/data/eventground-wam/comparisons/week4-clean-comparison-v1`.
Pilot output: `/home/peimengsui/data/eventground-wam/comparisons/week4-clean-comparison-pilot-v1`.
Both remain on Feifei; compact sealed launch/pilot evidence is copied here.

Rough planning allowance from the previous clean-40 runs: 7–12 hours for this
4,000-episode comparison, with artifact verification adding time. The subsequent
B0-only `dev-210` evaluation is roughly 30–60 minutes of GPU time, or 1–3 hours
including artifact checks, representative previews and an initial failure
summary. These are estimates, not measured durations for the full panels.

## Transition recovery — September 18, evening PDT

B0 completed all 2,000 episodes (1,365 successes, 635 timeouts). A generated
MuJoCo log made the checkout appear dirty and blocked reference startup; five
automatic service retries exhausted without starting reference episodes.
The [recovery audit](evaluation/week4-clean-comparison-recovery-v1.json) records
the original diagnostic bytes/hash, exact affected episodes and service state.
The log is now stored under the run's `diagnostics/` directory, with its expected
checkout path a symlink excluded by an exact `/MUJOCO_LOG.TXT` host Git rule.
No tracked code or frozen scientific input changed. Service recovery revalidated
the completed B0 outputs and continued into the reference phase without reruns.
Two simulator warnings occurred during episodes already recorded as failures;
they remain explicit items for the final validity/failure review.
