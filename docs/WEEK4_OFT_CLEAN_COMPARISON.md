# New B0 clean LIBERO comparison

The selected OFT-style Qwen3.5-2B checkpoint (update 72,000) runs on the original
40 clean LIBERO tasks × 50 initial states. Compare its 2,000 new outcomes with
the complete verified OpenVLA and previous B0 runs. The checkpoint was selected
by full validation before these new rollouts.

## Frozen scope

The sealed `configs/evaluation/week4_oft_clean_reuse_v1.json` addendum changes
the B0 checkpoint and authorizes reuse of the two completed 2,000-episode policy
runs. Its parent panel describes the original fresh 4,000-episode comparison;
the addendum's explicit reuse override applies to this new comparison.
The original rule, protocol, panel, results and source identities remain intact.
Simulation, task/state/seed order, native policy preprocessing, outcome handling,
five-percentage-point margin, and 20,000-replicate paired task-cluster bootstrap
are inherited unchanged. The episode protocol ID remains the original protocol;
new run identities bind the new comparison addendum and checkpoint.

OpenVLA's fixed baseline is 1,935/2,000 (96.75%); previous B0 is 1,365/2,000
(68.25%). Acceptance uses new B0 minus OpenVLA and the original paired confidence
interval. New B0 minus old B0 has a descriptive paired interval, without another
acceptance gate. All failures remain included. This is a previously inspected
development panel, not an unseen holdout or a training-matched reproduction.

## Evidence and recovery

Before loading the policy, verify the original sealed final report, both policy
identities, all 4,000 receipt digests, and SHA-256 bytes of all saved recordings
and attempt evidence. These hashes bind exactly the HDF5 contents structurally
validated by the original final audit; no old identities or recordings are
rewritten or copied. Every new first policy input must match cached OpenVLA;
the completed settling camera/proprioception/simulator hashes must also match
both original policies. Any mismatch stops execution and withholds acceptance.

The shared durable attempt journals and atomic episode receipts provide resume.
Completed episodes, including policy failures, are reused after verification;
an interrupted episode has at most one recorded retry with identical task,
state and seed. Exhausted infrastructure errors stop acceptance. Final analysis
fully validates every new HDF5 file and writes both paired comparisons, per-task
scores, termination counts and provenance to `comparison-report.json`.

Run `scripts/run_oft_clean_reuse.sh OUTPUT_ROOT` under a persistent, enabled user
systemd service, with user lingering, `flock`, `Restart=on-failure`,
`RestartPreventExitStatus=78`, `RestartSec=60`, and `KillMode=control-group`.
MuJoCo diagnostics are written in the output directory, preserving clean source.
A separate `--pilot-pairs 2` run checks integration before formal execution.
Pilot outcomes never enter the formal comparison. This launch covers clean
LIBERO; the new checkpoint's `dev-210` evaluation remains separate.
