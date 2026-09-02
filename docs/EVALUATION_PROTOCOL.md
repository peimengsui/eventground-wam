# Locked evaluation protocol

The shared LIBERO-Plus evaluation protocol is
[`configs/evaluation/libero_plus_v1.json`](../configs/evaluation/libero_plus_v1.json).
It is a machine-readable, validated scientific artifact. Every evaluation result
must record both `protocol_id` and its canonical SHA-256 digest.

## Locked choices

| Decision | Version 1 setting | Rationale |
|---|---|---|
| Benchmark | LIBERO-Plus at `4976dc30028e805ff8094b55501d532c48fec182` | Matches the installed, smoke-tested checkout. |
| Task inventory | 10,030 tasks across spatial, object, goal, and LIBERO-10 | Matches the pinned classification file. |
| Trials | One initial state and rollout per task | Required by the LIBERO-Plus evaluation instructions. |
| Cameras | Agent and wrist RGB | Required by the public OpenVLA-OFT+ reference policy and future shared models. |
| Capture | 256×256 RGB, rotated 180° into canonical orientation | Matches the upstream evaluator before model-specific resizing. |
| Proprioception | 8D end-effector position, axis-angle orientation, and gripper state | Matches the public reference policy. |
| Control | 20 Hz OSC pose control | Pinned LIBERO-Plus simulator default. |
| Action | 7D actions, chunks of 8, replanned every 8 steps | Matches the public OpenVLA-OFT+ LIBERO profile. |
| Stabilization | Ten recorded no-op steps outside the policy budget | Matches the upstream evaluation behavior. |
| Timeouts | 220/280/300/520 policy steps by suite | Matches the published OpenVLA-OFT LIBERO evaluator. |
| Failure accounting | Timeouts, invalid actions, and policy exceptions are failures | Prevents selective dropping of hard episodes. |
| Primary metric | Micro success rate over tasks with Wilson 95% interval | Every LIBERO-Plus task receives equal weight. |

The raw simulator images remain 256×256. Each model profile may apply its own
deterministic resize or crop, which must be recorded, but changing model-native
preprocessing does not change the shared environment capture.

## Task identity and missing metadata

The classification JSON uses one-based `id` values while the runtime benchmark
API uses zero-based task indices. Results and manifests must store both fields.

The pinned classification contains 121 Light Conditions tasks with no difficulty
label. Version 1 preserves these as `unlabeled`; it never imputes a difficulty.
They are excluded from the strictly 7×5 `sanity-35` and `dev-210` grids, included
proportionally in `stratified-1K`, and included in `full-10K`.

The split hierarchy is nested:

```text
sanity-35 ⊂ dev-210 ⊂ stratified-1K ⊂ full-10K
```

Split manifests are generated separately so they can contain the benchmark's
full task names and classification metadata. Stable SHA-256 ranking and seed 7
make sampling independent of Python or NumPy random-number-generator versions.

## Seeding and reproducibility

The environment seed is 0, matching the public evaluator. Each episode receives
an order-independent seed derived from the protocol ID, suite, zero-based task
index, trial index, and base seed 7. This makes resumed and sharded evaluations
equivalent to an uninterrupted run.

The protocol fixes inputs and evaluator behavior; it does not claim bitwise
identity across GPU architectures or dependency versions. Each run must record
the software revisions, checkpoint revision, and hardware alongside its result.

## Recording and retry rules

The source-of-record episode artifact is HDF5 and includes every 20 Hz simulator
step, including stabilization. MP4 output is a preview only. Required episode and
step fields are enumerated in the protocol JSON and will be enforced by the
episode recorder.

Policy errors, invalid actions, and timeouts are failures. A clearly identified
infrastructure failure may be retried once; the original failure and retry remain
in the report. A repeated infrastructure failure counts as a failed episode.

## Reference checkpoint

The Week 1 sanity check is bound to:

- Checkpoint: `Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`
- Checkpoint revision: `a85655ec941bae6644c9fbdf62db02b9726d7cf5`
- Evaluator: `moojink/openvla-oft`
- Evaluator revision: `e4287e94541f459edc4feabc4e181f537cd569a8`

These pins make later changes to either upstream repository unable to silently
alter the sanity-check protocol.

## Validation and change control

Validate the protocol from the repository root:

```bash
uv run python scripts/validate_eval_protocol.py
```

Version 1 is immutable. A scientific change to task selection, observations,
control, failure accounting, or metrics requires a new protocol ID and a new
configuration file. Corrections that alter results must not overwrite v1.
