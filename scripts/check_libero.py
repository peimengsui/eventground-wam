"""Create, reset, render, and step one LIBERO-Plus environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=expanded_path,
        default=expanded_path("~/data/eventground-wam/libero-plus/smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    import cv2
    import numpy as np
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_class = benchmark.get_benchmark_dict().get(args.suite.lower())
    if benchmark_class is None:
        available = ", ".join(sorted(benchmark.get_benchmark_dict()))
        raise SystemExit(f"Unknown suite {args.suite!r}; available suites: {available}")

    suite = benchmark_class()
    task = suite.get_task(args.task_id)
    bddl_path = suite.get_task_bddl_file_path(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)

    environment = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    try:
        environment.reset()
        observation = environment.set_init_state(init_states[0])
        for _ in range(args.steps):
            observation, _, _, _ = environment.step(np.zeros(7, dtype=np.float32))
    finally:
        environment.close()

    image = observation["agentview_image"]
    if image.shape != (128, 128, 3):
        raise SystemExit(f"Unexpected agent-view image shape: {image.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"{args.suite}-task-{args.task_id}.png"
    report_path = args.output_dir / f"{args.suite}-task-{args.task_id}.json"
    if not cv2.imwrite(str(image_path), image[::-1, :, ::-1]):
        raise SystemExit(f"Failed to write smoke-test image: {image_path}")

    report = {
        "bddl_file": bddl_path,
        "cuda_available": torch.cuda.is_available(),
        "image_shape": list(image.shape),
        "language": task.language,
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "steps": args.steps,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"render: {image_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
