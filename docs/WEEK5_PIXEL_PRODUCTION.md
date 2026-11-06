# Week 5 Step 5: fresh production training

**Status: complete and verified.** The persistent Feifei service ran from September
25, 2026, at **20:24:33 PDT** to September 26 at **10:54:13 PDT**, with zero restarts
and exit code 0. It completed all **221,633 updates and 8,221,365 pairs** in
**14.49 active hours**. [Step 6 held-out validation](WEEK5_PIXEL_VALIDATION.md)
passed all frozen gates and independent checks; `C_pixel-v1` is accepted.

The [final verification](evaluation/week5-pixel-cpt-final-v1/final-checkpoint-verification.json)
checked every recovery-state component, optimizer/scheduler counters, RNG states,
and exported weights. An independent replay of the common order verified all
8,221,365 pair identities exactly once and reproduced the complete sample chain.
All 11,435 train episodes were used. Frozen base weights remained unchanged,
and training read zero validation/test payloads. The
[Step 5 completion ledger](evaluation/week5-step5-completion-v1.json) preserves
the preceding launch documents and pins the final evidence.

The [launch ledger](evaluation/week5-step5-launch-v1.json) pins the
[service definition](../configs/systemd/eventground-pixel-cpt-v1.service),
preflight, initialization and checkpoint receipts. The source is an independent,
read-only copy of the exact 56-file Step 4 archive. Code, recipe, common order,
runtime and dependency-lock checks match the passed pilot. No training code or
numerical setting changed for the launch.

## Frozen run

| Setting | Production value |
| --- | --- |
| Run / service | `week5-pixel-cpt-v1` / `eventground-pixel-cpt-v1.service` |
| Host | Feifei, one NVIDIA RTX 5090 |
| Exposure | One full pass; 8,221,365 train pairs, both cameras |
| Fixed endpoint | 221,633 optimizer updates |
| Trainable model | Fresh 4,631,427-parameter pixel predictor; base Qwen frozen |
| Objective | L1 + 0.1 × (1 − SSIM) |
| Recipe | [Frozen Step 4 recipe](WEEK5_PIXEL_TRAINING.md) |
| Duration estimate | 22.90 hours from the pilot, including a 1.5× margin |
| Runtime limit | 72 cumulative active hours across process restarts |
| Endpoint artifact | `C_pixel-candidate.pt`, subsequently accepted byte-for-byte as `C_pixel-v1.pt` after Step 6 |

There is no validation-based early stopping or checkpoint search. The duration
is a pilot estimate; final validation requires additional time. Production
starts from the pinned base model and deterministic fresh predictor seed, with
no B0-trained, Step 3 or pilot weights.

## Verified launch evidence

The [preflight](evaluation/week5-pixel-cpt-launch-v1/preflight.json) passed exact
source/runtime/recipe/order comparisons, full order-array hashes, empty-output
and disk-reserve checks. `systemd-analyze --user verify` accepted the service;
its only warning concerned an unrelated installed `spice-vdagent.service`.

The [fresh-start verification](evaluation/week5-pixel-cpt-launch-v1/fresh-start-verification.json)
checked the step-zero checkpoint checksum and contents: zero updates and pairs,
empty optimizer, matching fresh predictor and pinned-base digests, zero trainable
base parameters, and a production identity distinct from the pilot.

The [first periodic checkpoint verification](evaluation/week5-pixel-cpt-launch-v1/first-periodic-checkpoint-verification.json)
passed at **20:28:05 PDT**. Checkpoint 500 contains **18,277 acknowledged pairs**,
matching the common order. Its SHA-256 and size match the sealed commit; all
model/optimizer tensors are finite, optimizer counters and scheduler are at 500,
Python/NumPy/CPU/CUDA RNG states are present, and the sample chain matches the
logged update. Predictor weights have changed from initialization. This is a
read-only checkpoint inspection; production was not interrupted for a recovery
test. The actual kill/restart equality test remains the passed Step 4 pilot.

At that check, the service was enabled and running with zero restarts. Full-device
GPU usage was 6,409 MiB of 32,607 MiB; process peak RSS was about 5.17 GiB.
These are launch observations, not full-run peak measurements or evidence of
held-out predictor quality.

## Supervision and recovery

The user service is enabled with lingering active, so training survives SSH
disconnects and the service starts again after a host reboot. It saves full
recovery state every 500 updates or 15 minutes and retains three committed
generations. Unexpected process failures can restart after 60 seconds, bounded
to three starts per hour. Ordinary Python failures (exit 1) and the active-budget
stop (exit 75) require review instead of automatic retries.

The service has a six-CPU quota, 24 GiB memory limit and 73-hour process backstop;
the trainer's stricter 72-hour cumulative budget persists across restarts.
SIGTERM requests a checkpoint at the next complete optimizer boundary. A
completed `report.json` prevents the enabled service from rerunning a finished
endpoint. Integrity changes are rejected by the runner's recovery identity.

```text
Production source:
/home/peimengsui/code/eventground-wam-pixel-cpt-v1

Run outputs, status.json, service.log, events.jsonl and recovery/:
/home/peimengsui/data/eventground-wam/libero-plus/training/week5-pixel-cpt-v1

Launch evidence and matching passed pilot report:
/home/peimengsui/data/eventground-wam/analysis/week5-pixel-cpt-production-v1
```

Read current progress without changing the run:

```bash
ssh feifei 'systemctl --user status eventground-pixel-cpt-v1.service --no-pager'
ssh feifei 'cat /home/peimengsui/data/eventground-wam/libero-plus/training/week5-pixel-cpt-v1/status.json'
```

The final export SHA-256 is
`2ddd8bcb825f6a2905c4036a0d0e4b60e973f8f0fbe32066989eaabd4f076eaf`.
Its predictor tensor digest is
`2540b4bc6dd43101181a9dccc8dc36a5f006eb6febe576ca84277925451ff2ea`;
the export exactly matches the final recovery weights. Step 6 evaluated this
fixed candidate against the unchanged held-out reconstruction gates and reviewed
all 480 fixed examples. The [accepted copy](evaluation/week5-pixel-validation-v1/acceptance.json)
is `/home/peimengsui/data/eventground-wam/libero-plus/checkpoints/week5/C_pixel-v1.pt`
on Feifei, with the same SHA-256. No weights changed during validation or freezing.
