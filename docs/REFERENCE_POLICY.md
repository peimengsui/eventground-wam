# OpenVLA-OFT reference policy

The Week 1 reference policy is the public OpenVLA-OFT+ mix-SFT checkpoint for
LIBERO-Plus. Model inference and simulation run on `feifei`; the local checkout
only holds code and lockfiles.

## Pinned inputs

- evaluator: `https://github.com/moojink/openvla-oft.git`
- evaluator revision: `e4287e94541f459edc4feabc4e181f537cd569a8`
- checkpoint: `Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`
- checkpoint revision: `a85655ec941bae6644c9fbdf62db02b9726d7cf5`
- custom Transformers revision:
  `bc339d9ad707454c0c115970db43c260067c61ab` (version 4.40.1)
- dlimp revision: `040105d256bd28866cc6620621a3d5f7b6b91b46`

The checkpoint snapshot contains 25 files and 15,939,183,790 bytes (14.845
GiB). `check_reference_policy.py` pins and validates the exact size and SHA-256
of all 25 files, including the Hugging Face remote-code Python files. The
EventGround provenance marker is a sibling of the snapshot, so the snapshot is
never modified.

The four BF16 safetensor shards collectively contain 15.08 GB and already form
a complete fine-tuned model. This is also the checkpoint's documented direct
`AutoModelForVision2Seq.from_pretrained` interface. The separately published
LoRA adapter is retained and hashed as provenance, but is not applied a second
time. The action head and proprioception projector are loaded from their
separate, pinned state dictionaries.

## Runtime compatibility

The upstream evaluator reports results with Python 3.10, PyTorch 2.2.0, and its
custom Transformers 4.40.1. PyTorch 2.2.0 predates the RTX 5090's `sm_120`
architecture and cannot execute CUDA kernels on `feifei`. The locked runtime
therefore keeps Python 3.10 and the exact evaluator/Transformers logic, while
using PyTorch 2.13.0 + CUDA 13.0 and torchvision 0.28.0. This hardware-required
deviation is reported alongside results. Flash Attention is omitted because
the upstream project only requires it for training and its inference call does
not enable it.

The reference environment is isolated from the smaller LIBERO-only environment
under `environments/reference-policy`. NumPy 1.26.4 satisfies TensorFlow 2.15's
lower bound; the LIBERO-only environment remains on its historical NumPy pin.

## Setup and verification on `feifei`

From `~/code/eventground-wam`:

```bash
./scripts/setup_reference_policy.sh
```

The setup is resumable. It checks out the exact evaluator revision, synchronizes
the locked `uv` environment, downloads only the exact checkpoint revision, hashes
the entire snapshot, validates CUDA, and exercises the evaluator imports. The
runner refuses to add the project, evaluator, or LIBERO-Plus checkout to
`PYTHONPATH` unless all three source trees are at their expected revisions and
have no tracked or non-ignored untracked changes.

Re-run the offline integrity and runtime check without downloading:

```bash
./scripts/run_reference_policy.sh python scripts/check_reference_policy.py \
  --full-hash --require-cuda
```

## Manifest-driven evaluation

Inspect the first locked task without loading the model:

```bash
./scripts/run_reference_policy.sh python scripts/evaluate_reference_policy.py \
  --manifest evaluation/splits/libero-plus-v1/sanity-35.json \
  --task-limit 1 --dry-run
```

Load the full model and both continuous-control heads, without starting MuJoCo:

```bash
./scripts/run_reference_policy.sh python scripts/evaluate_reference_policy.py \
  --task-limit 1 --load-only
```

Run one real manifest task:

```bash
./scripts/run_reference_policy.sh python scripts/evaluate_reference_policy.py \
  --manifest evaluation/splits/libero-plus-v1/sanity-35.json \
  --task-limit 1 --resume
```

Run the complete Week 1 gate:

```bash
./scripts/run_reference_policy.sh python scripts/evaluate_reference_policy.py \
  --manifest evaluation/splits/libero-plus-v1/sanity-35.json
```

The adapter follows the official input path: both locked 256x256 images are
rotated 180 degrees, then the upstream evaluator performs its JPEG/Lanczos
resize to 224x224 and 90% center crop. It predicts 8x7 action chunks, applies
the official gripper normalization and sign inversion, measures every model
query, and executes all eight actions open-loop. Every settling and policy
transition is written through the unified HDF5 recorder.

LIBERO-Plus motion-blur tasks use the same ImageMagick primitive, severity
table, and seeded random angle as upstream, but observations are transferred
through raw RGB pixels. This avoids an undeclared dependency on host-specific
ImageMagick PNG delegates and keeps sensor-noise evaluation portable.

Evaluation resumes completed base or retry artifacts after validating their
checkpoint and project-revision provenance. A post-commit run will therefore
refuse an episode recorded by a different source revision rather than silently
mixing results. Use `--no-resume` to require fresh output.
Infrastructure failures are retried once into an explicit `-retry-1` artifact;
policy exceptions and invalid actions are recorded as task failures. The
incrementally written report contains micro success, Wilson confidence
interval, required latency/timeout/exception/retry metrics, and all four locked
category/suite/difficulty breakdowns. It also records the complete manifest
task count, selected offset and count, and whether the selection covers the
whole manifest, so a one-task smoke run cannot be mistaken for the 35-task
Week 1 gate.
Partial selections use qualified names such as
`summary-offset-0000-count-0001.json`; only a full-manifest run writes
`summary.json`. Every report embeds Python/library/CUDA/device versions and the
SHA-256 of the reference-policy `uv.lock`.

Outputs default to:

```text
~/data/eventground-wam/reference-policy/evaluations/sanity-35/
```

To expose that directory locally using the same working SSHFS pattern as the
LIBERO output mount, mount the parent data directory or create a second mount:

```bash
mkdir -p ~/feifei-reference-policy
rclone nfsmount \
  :sftp:/home/peimengsui/data/eventground-wam/reference-policy \
  ~/feifei-reference-policy \
  --sftp-ssh "ssh -o ServerAliveInterval=20 feifei" \
  --read-only --dir-cache-time 5s --daemon
```
