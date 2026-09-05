#!/usr/bin/env python3
"""Download and verify the pinned OpenVLA-OFT LIBERO-Plus reference policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

MODEL_ID = "Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata"
MODEL_REVISION = "a85655ec941bae6644c9fbdf62db02b9726d7cf5"
EVALUATOR_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
TRANSFORMERS_REVISION = "bc339d9ad707454c0c115970db43c260067c61ab"
DLIMP_REVISION = "040105d256bd28866cc6620621a3d5f7b6b91b46"
MODEL_WEIGHT_BYTES = 15_082_474_368
SNAPSHOT_BYTES = 15_939_183_790

# Hugging Face LFS object IDs are the SHA-256 digests of the downloaded bytes.
EXPECTED_LFS_FILES: dict[str, tuple[int, str]] = {
    "action_head--150000_checkpoint.pt": (
        302_242_674,
        "adc70a2da8cadd90b5f5cdec1771058130e62e48ede4c8cd351a8c35ee740330",
    ),
    "lora_adapter/adapter_model.safetensors": (
        484_458_600,
        "dbd9440767b9b70994f9f0a6bd67dfc250f4ec37649ec2e10cdf34f46e1c9244",
    ),
    "model-00001-of-00004.safetensors": (
        4_925_122_448,
        "45f48b9bc6d7535f5498e826b2b0d9da4aae03853bd4681a10decb480de55fd3",
    ),
    "model-00002-of-00004.safetensors": (
        4_947_392_496,
        "47500ca78f427bfb675879d93bea6f514480f460740610458addef1921c4522b",
    ),
    "model-00003-of-00004.safetensors": (
        4_947_417_456,
        "17cc127c4ed727708a6138103392b79290d45bb8c9edac949674625b0d2d0bf5",
    ),
    "model-00004-of-00004.safetensors": (
        262_668_432,
        "a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60",
    ),
    "proprio_projector--150000_checkpoint.pt": (
        67_275_256,
        "491e30d2aaa1d9ff33521a93a511b1828f8deb33515dd2b43b470ade74569221",
    ),
    "tokenizer.model": (
        499_723,
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    ),
}
EXPECTED_SMALL_FILES: dict[str, tuple[int, str]] = {
    ".gitattributes": (
        401,
        "8eb2c58ae70925819b314305dd24765a7306e3de2692b886e4ddcf9d95e6e54b",
    ),
    "README.md": (
        5_920,
        "fa08a4f50b3fa0066fc40f86bd167880919686b34258b82fc301273f1c815072",
    ),
    "added_tokens.json": (
        21,
        "ab43123267b190cb7990bc9f2ae4ad32dcb7ed029fb0fccb1b4e062c6f54a2a1",
    ),
    "config.json": (
        73_547,
        "7c17f7d9eb1aab2743abb3aee0e901bf38a308e9837abf0a30faecd046a6dc68",
    ),
    "configuration_prismatic.py": (
        5_865,
        "68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68",
    ),
    "dataset_statistics.json": (
        11_705,
        "fc739fa84887047a65226f928117ddf411eec7afce3341d20990c19d9ede4e7d",
    ),
    "generation_config.json": (
        136,
        "67fbe17b12e8beb5be98eb911483971cc78589e51a37ac0cce9cd499d06a8bf1",
    ),
    "lora_adapter/README.md": (
        5_136,
        "daf9c481ae53315db570a75d1beb858e7fba07dc7c6a7ecfb070e549ea67b4c6",
    ),
    "lora_adapter/adapter_config.json": (
        1_086,
        "7c3915807f98415ef9e31a006439a09857a0236cf09655181003b52a5f662c70",
    ),
    "model.safetensors.index.json": (
        94_764,
        "ca8b53fed8133ee2afcd2fc483de8febf7f5bb0f6bcb09f91189772e59e8f659",
    ),
    "modeling_prismatic.py": (
        48_906,
        "b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6",
    ),
    "preprocessor_config.json": (
        1_650,
        "bd024783fb785cbdfe35b710b621c5e8678d4c157a9ba625675e67d40d1757da",
    ),
    "processing_prismatic.py": (
        12_718,
        "06563586bbc73d36501a14a231a3174904ede830d33174d0c625ece8e931ef1e",
    ),
    "processor_config.json": (
        130,
        "167e026d8121f6cb0bf5c18310b97ba8139152f970822bb3824213bd37dc7a76",
    ),
    "special_tokens_map.json": (
        552,
        "cb90ee5cf5793aa444039af9795b00db59d3fef8942e7eca9cf1990bab370d61",
    ),
    "tokenizer.json": (
        1_842_976,
        "8f5e2869e1807b8bb3c7717a294539c37e9de5d728ad60e31ef57e83cc5ea527",
    ),
    "tokenizer_config.json": (
        1_192,
        "f5f1d3ed015ebb71cf11686ff00bc8e0c25957bbf3952f95312cecc9b168fbea",
    ),
}


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def marker_path(checkpoint_dir: Path) -> Path:
    """Keep EventGround provenance beside, never inside, the immutable snapshot."""

    return checkpoint_dir.with_name(checkpoint_dir.name + ".eventground.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=expanded_path,
        default=expanded_path(
            "~/data/eventground-wam/reference-policy/checkpoints/"
            "openvla-7b-oft-libero-plus-a85655ec"
        ),
    )
    parser.add_argument(
        "--evaluator-root",
        type=expanded_path,
        default=expanded_path("~/code/vendor/openvla-oft"),
    )
    parser.add_argument(
        "--libero-root",
        type=expanded_path,
        default=expanded_path("~/code/vendor/LIBERO-plus"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Resume/download the exact Hugging Face snapshot before validation.",
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="SHA-256 all 15.8 GB of LFS payloads, not just sizes and metadata.",
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Skip heavyweight evaluator import checks.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require a working CUDA device and execute a small kernel.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
    ).strip()


def verify_evaluator(evaluator_root: Path) -> dict[str, Any]:
    if not (evaluator_root / ".git").is_dir():
        raise RuntimeError(f"Evaluator checkout is missing: {evaluator_root}")
    revision = git_output(evaluator_root, "rev-parse", "HEAD")
    if revision != EVALUATOR_REVISION:
        raise RuntimeError(
            f"Evaluator revision mismatch: expected {EVALUATOR_REVISION}, found {revision}"
        )
    status = git_output(
        evaluator_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(f"Evaluator checkout is not clean:\n{status}")
    return {"path": str(evaluator_root), "revision": revision, "clean": True}


def verify_libero(libero_root: Path) -> dict[str, Any]:
    if not (libero_root / ".git").is_dir():
        raise RuntimeError(f"LIBERO-Plus checkout is missing: {libero_root}")
    revision = git_output(libero_root, "rev-parse", "HEAD")
    if revision != LIBERO_REVISION:
        raise RuntimeError(
            f"LIBERO-Plus revision mismatch: expected {LIBERO_REVISION}, found {revision}"
        )
    status = git_output(
        libero_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(f"LIBERO-Plus checkout is not clean:\n{status}")
    return {"path": str(libero_root), "revision": revision, "clean": True}


def download_checkpoint(checkpoint_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    available_bytes = shutil.disk_usage(checkpoint_dir.parent).free
    minimum_free = SNAPSHOT_BYTES + 5 * 1024**3
    if available_bytes < minimum_free:
        raise RuntimeError(
            f"Need at least {minimum_free} free bytes before download; found {available_bytes}"
        )
    downloaded = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )
    ).resolve()
    if downloaded != checkpoint_dir:
        raise RuntimeError(f"Snapshot downloaded to unexpected path: {downloaded}")


def verify_checkpoint(checkpoint_dir: Path, *, full_hash: bool) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Checkpoint directory is missing: {checkpoint_dir}")

    checked_hashes: dict[str, str] = {}
    for relative_path, (expected_size, expected_sha256) in EXPECTED_LFS_FILES.items():
        path = checkpoint_dir / relative_path
        if not path.is_file():
            raise RuntimeError(f"Checkpoint file is missing: {relative_path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Checkpoint size mismatch for {relative_path}: "
                f"expected {expected_size}, found {actual_size}"
            )
        if full_hash:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Checkpoint SHA-256 mismatch for {relative_path}: {actual_sha256}"
                )
            checked_hashes[relative_path] = actual_sha256

    for relative_path, (expected_size, expected_sha256) in EXPECTED_SMALL_FILES.items():
        path = checkpoint_dir / relative_path
        if not path.is_file():
            raise RuntimeError(f"Checkpoint metadata file is missing: {relative_path}")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Checkpoint metadata size changed: {relative_path}")
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Checkpoint metadata SHA-256 changed: {relative_path}")

    expected_paths = set(EXPECTED_LFS_FILES) | set(EXPECTED_SMALL_FILES)
    actual_paths = {
        str(path.relative_to(checkpoint_dir))
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint_dir).parts
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "Checkpoint snapshot file set changed: "
            f"missing={sorted(expected_paths - actual_paths)!r}, "
            f"extra={sorted(actual_paths - expected_paths)!r}"
        )

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    expected_auto_map = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }
    if config.get("model_type") != "openvla" or config.get("auto_map") != expected_auto_map:
        raise RuntimeError("Checkpoint config is not the expected OpenVLA remote-code profile")

    statistics = json.loads(
        (checkpoint_dir / "dataset_statistics.json").read_text(encoding="utf-8")
    )
    expected_suites = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    if set(statistics) != expected_suites:
        raise RuntimeError("Checkpoint action statistics do not cover the four locked suites")

    index = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    if index.get("metadata", {}).get("total_size") != MODEL_WEIGHT_BYTES:
        raise RuntimeError("Checkpoint weight index total_size changed")

    provenance_path = marker_path(checkpoint_dir)
    marker = None
    if provenance_path.is_file():
        marker = json.loads(provenance_path.read_text(encoding="utf-8"))
        if marker.get("model_id") != MODEL_ID or marker.get("revision") != MODEL_REVISION:
            raise RuntimeError("Checkpoint provenance marker does not match the locked model")

    return {
        "path": str(checkpoint_dir),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "lfs_file_count": len(EXPECTED_LFS_FILES),
        "snapshot_file_count": len(expected_paths),
        "full_hash_verified": full_hash,
        "verified_sha256": checked_hashes,
        "marker_present": marker is not None,
    }


def write_marker(checkpoint_dir: Path, *, full_hash: bool) -> None:
    marker = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "lfs_sha256_verified": full_hash,
    }
    marker_path(checkpoint_dir).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_runtime(*, require_cuda: bool) -> dict[str, Any]:
    import numpy
    import tensorflow
    import torch
    import transformers

    if transformers.__version__ != "4.40.1":
        raise RuntimeError(f"Expected custom Transformers 4.40.1, found {transformers.__version__}")
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    device = None
    capability = None
    if cuda_available:
        device = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
        kernel_result = (torch.ones(4, device="cuda") * 2).tolist()
        if kernel_result != [2.0, 2.0, 2.0, 2.0]:
            raise RuntimeError("CUDA smoke kernel returned unexpected values")

    # These imports exercise the evaluator's inference path without loading 15 GB of weights.
    import prismatic  # noqa: F401
    from experiments.robot.openvla_utils import get_vla_action  # noqa: F401
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tensorflow": tensorflow.__version__,
        "numpy": numpy.__version__,
        "cuda_available": cuda_available,
        "cuda_device": device,
        "cuda_capability": capability,
    }


def main() -> None:
    args = parse_args()
    evaluator = verify_evaluator(args.evaluator_root)
    libero = verify_libero(args.libero_root)
    if args.download:
        download_checkpoint(args.checkpoint_dir)
    checkpoint = verify_checkpoint(args.checkpoint_dir, full_hash=args.full_hash)
    if args.download:
        write_marker(args.checkpoint_dir, full_hash=args.full_hash)
        checkpoint["marker_present"] = True
    runtime = None if args.skip_imports else verify_runtime(require_cuda=args.require_cuda)
    print(
        json.dumps(
            {
                "evaluator": evaluator,
                "libero": libero,
                "checkpoint": checkpoint,
                "runtime": runtime,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
