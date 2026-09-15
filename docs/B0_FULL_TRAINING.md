# Full B0 training and recovery

## Declared five-epoch recipe

[`week4_b0_5epoch_v1.json`](../configs/training/week4_b0_5epoch_v1.json) declares
five complete epochs over all 1,715,954 frozen training chunks: **8,579,770
sample exposures**. Every episode's final short batch is included. Batch size is
16, with two CPU input workers and two prefetched batches per worker. Each worker
caches at most one decoded episode, capped at 4,096 source frames. No full-dataset
image or processor cache is materialized.

- Pinned Qwen3.5-2B initialization; fresh rank-16 LoRA, projectors, and decoder.
  Training RNG seed 17, projector initialization seed 7, decoder seed 11.
- Frozen Week 2 assignments and verified train-only action normalization.
- The existing deterministic episode shuffle, then PCG64 anchor shuffle keyed
  by seed, epoch, and episode. Each epoch's complete batch-order digest is sealed
  before optimization. Worker scheduling does not change batch order.
- AdamW: projector/decoder LR `3e-4`, LoRA LR `1e-4`, betas `(0.9, 0.95)`,
  epsilon `1e-8`, weight decay `0.01`, gradient clipping `1.0`.
- Linear warmup over 2% of all five epochs' optimizer updates, followed by
  cosine decay to 10% of each group's peak LR. No gradient accumulation.
- Continuous-action L1 plus unit-weight gripper BCE; existing Qwen BF16 and
  projector/action-head precision behavior; unchanged camera and action contracts.
- Evaluate **every validation chunk** after each epoch. Select the epoch with
  the lowest sample-weighted validation hybrid objective; ties retain the earlier
  epoch. Test data and rollout scores do not participate in checkpoint selection.
- Preserve all five epoch policy exports. Reload the selected one and repeat
  full validation before publishing `B0.ckpt` and the sealed training report.

The full-training recipe is intentionally a new experiment, separate from the
Week 3 memorization schedules. The five-day launch guard requires a matching
real-data pilot, including exact disk reload and next-update equivalence. Its
projection charges all five training and validation passes at observed training
throughput, with a further 20% margin. This guard checks feasibility; it does not
guarantee wall time or policy quality.

## Checkpoints and interruption behavior

The training process saves every **five minutes or 2,000 updates**, whichever
comes first, at epoch boundaries, and after a requested termination reaches a
completed update. A recovery checkpoint includes:

- every allowlisted trainable model tensor;
- complete AdamW moments and scheduler state;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- epoch, exact next batch, optimizer update and sample counts;
- the rolling consumed-sample digest, epoch loss sums, validation history, and
  selected-checkpoint identity;
- a binding to the source commit, recipe, pinned model, runtime lock, data,
  normalizer, and all epoch-order digests.

Writes use a temporary file, file `fsync`, atomic rename, and directory `fsync`.
The tensor file is committed before its digest-sealed JSON record. The last
three committed recovery generations are retained. An incomplete write is
ignored; a damaged newest checkpoint falls back to the newest intact generation
and records that recovery. If no intact generation exists, or the run identity
has changed, training fails explicitly instead of silently starting over.

Prefetched batches are disposable. Resumption starts at the saved next batch and
decodes only the needed episode, without replaying earlier episodes. Abrupt
termination may lose work since the last committed checkpoint; that work is
recomputed. Validation interrupted mid-pass restarts that validation pass.

One advisory file lock prevents two processes from writing the same run.
`events.jsonl` is an append-only attempt/progress log; checkpoint state is the
authority for completed updates. Replayed updates can appear twice in the log
after an abrupt interruption, distinguished by process ID and timestamp.

## Host-managed execution

Run from a dedicated clean source checkout on `feifei`, using the existing locked
B0 virtual environment. `scripts/manage_b0_job.py` installs a **persistent user
systemd service**, enables it at user-manager startup, and requires user
lingering. The host owns the training process; closing SSH has no effect.

The service restarts failures after 60 seconds, with a limit of four starts per
hour. Recipe/integrity failures (exit 78) are not retried. A clean completion
does not restart. `systemctl --user stop` intentionally stops the job. Lingering
and the enabled unit allow recovery after a host reboot when the home directory,
GPU driver, and local data are available. No machine-wide service or reference
runtime is modified.

Start the pilot through the same launcher, supplying real host paths:

```bash
python scripts/manage_b0_job.py start --name eventground-b0-week4-pilot \
  --python ~/code/eventground-wam/environments/b0/.venv/bin/python \
  --recipe configs/training/week4_b0_5epoch_v1.json \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --normalization ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --output-root ~/data/eventground-wam/libero-plus/derived/b0/week4-b0-pilot-v1 \
  --pilot-steps 2000
```

Production uses the same arguments with a fresh output root and service name,
replacing `--pilot-steps` with `--pilot-report <pilot-root>/pilot-report.json`.
The production process starts from fresh initialization; it never continues the
pilot weights. A restart of an existing production output automatically resumes
its latest intact checkpoint.

```bash
systemctl --user status eventground-b0-week4 --no-pager
journalctl --user -u eventground-b0-week4 -n 30 --no-pager
systemctl --user stop eventground-b0-week4
systemctl --user start eventground-b0-week4
```

If repeated failures exhaust the rate limit, inspect the logs and repair the
underlying problem, then use `systemctl --user reset-failed eventground-b0-week4`
followed by `start`. Preserve the source checkout and recipe for the duration of
the run; changed code belongs in a new versioned run.

## Evaluation handoff

The selected checkpoint bundle contains the full-training manifest, sealed
completion report, normalizer identity, and `B0.ckpt`. The Week 3 cohort review
is not part of this new bundle's acceptance contract. Complete Standard LIBERO
clean and `dev-210` rollouts remain required after training. A matched public
clean comparator has not yet been established; training completion alone does
not close the Week 4 scientific gate.
