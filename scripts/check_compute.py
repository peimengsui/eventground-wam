"""Validate the Python and PyTorch compute environment."""

from __future__ import annotations

import argparse
import platform

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail when PyTorch cannot use a CUDA device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        if args.require_cuda:
            raise SystemExit("CUDA is required but unavailable")
        return

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"device: {properties.name}")
    print(f"compute capability: {properties.major}.{properties.minor}")
    print(f"device memory: {properties.total_memory / 2**30:.1f} GiB")

    matrix = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
    result = matrix @ matrix
    torch.cuda.synchronize(device)
    if not torch.isfinite(result).all().item():
        raise SystemExit("CUDA BF16 smoke test produced non-finite values")
    print("BF16 matrix multiplication: passed")


if __name__ == "__main__":
    main()
