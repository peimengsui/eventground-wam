#!/usr/bin/env python3
"""Profile process-level GPU and host memory during a recorded B0 rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from eventground_wam.b0_rollout_validation import validate_b0_rollout_bundle
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import sha256_file
from eventground_wam.evaluation_protocol import (
    B0_TRAINING_ROLLOUT_PROTOCOL_PATH,
    load_evaluation_protocol,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


def _expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-output-dir", type=_expanded_path, required=True)
    parser.add_argument("--rollout-output-dir", type=_expanded_path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def parse_nvidia_csv(output: str, field_count: int) -> list[list[str]]:
    """Parse no-header, no-unit nvidia-smi CSV output."""

    rows: list[list[str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != field_count:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(fields)
    return rows


def _run_nvidia_smi(query: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout


def _process_group_id(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = stat[closing_parenthesis + 2 :].split()
    try:
        return int(fields[2])
    except (IndexError, ValueError):
        return None


def _process_group_rss_bytes(process_group: int) -> int:
    total_kib = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or _process_group_id(int(entry.name)) != process_group:
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total_kib += int(line.split()[1])
                break
    return total_kib * 1024


def _sample_runtime(process_group: int, elapsed_seconds: float) -> dict[str, object]:
    gpu_rows = parse_nvidia_csv(
        _run_nvidia_smi("gpu=index,name,memory.total,memory.used"),
        4,
    )
    app_rows = parse_nvidia_csv(
        _run_nvidia_smi("compute-apps=pid,process_name,used_gpu_memory"),
        3,
    )
    target_apps = [
        {
            "pid": int(pid),
            "process_name": process_name,
            "used_gpu_memory_mib": int(used_memory),
        }
        for pid, process_name, used_memory in app_rows
        if _process_group_id(int(pid)) == process_group
    ]
    return {
        "elapsed_seconds": elapsed_seconds,
        "process_group_rss_bytes": _process_group_rss_bytes(process_group),
        "target_gpu_memory_mib": sum(
            int(app["used_gpu_memory_mib"]) for app in target_apps
        ),
        "target_gpu_processes": target_apps,
        "gpus": [
            {
                "index": int(index),
                "name": name,
                "total_memory_mib": int(total_memory),
                "used_memory_mib": int(used_memory),
            }
            for index, name, total_memory, used_memory in gpu_rows
        ],
    }


def _write_sealed_json(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
        encoding="utf-8",
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_interval_seconds <= 0:
        raise ValueError("--sample-interval-seconds must be positive")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("a rollout command is required after --")
    if args.profile_output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite profile output {args.profile_output_dir}"
        )
    if args.rollout_output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite rollout output {args.rollout_output_dir}"
        )
    args.profile_output_dir.mkdir(parents=True)
    stdout_path = args.profile_output_dir / "rollout.stdout.log"
    stderr_path = args.profile_output_dir / "rollout.stderr.log"
    samples: list[dict[str, object]] = []
    sampling_errors: list[str] = []
    started_wall = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
        )
        process_group = process.pid
        try:
            while process.poll() is None:
                sample_started = time.perf_counter()
                try:
                    samples.append(
                        _sample_runtime(process_group, sample_started - started)
                    )
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    message = f"{type(error).__name__}: {error}"
                    if message not in sampling_errors:
                        sampling_errors.append(message)
                remaining = args.sample_interval_seconds - (
                    time.perf_counter() - sample_started
                )
                if remaining > 0:
                    time.sleep(remaining)
        except BaseException:
            os.killpg(process_group, signal.SIGTERM)
            process.wait(timeout=10)
            raise
        return_code = process.wait()
    duration_seconds = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            f"rollout command exited {return_code}; inspect {stderr_path}"
        )

    artifacts = sorted(args.rollout_output_dir.glob("*.hdf5"))
    if len(artifacts) != 1:
        raise RuntimeError(
            f"expected one rollout artifact, found {len(artifacts)}"
        )
    rollout_report_path = args.rollout_output_dir / "training-rollout-report.json"
    protocol = load_evaluation_protocol(B0_TRAINING_ROLLOUT_PROTOCOL_PATH)
    validation = validate_b0_rollout_bundle(
        artifacts[0],
        rollout_report_path,
        protocol,
    )
    rollout_report = json.loads(rollout_report_path.read_text(encoding="utf-8"))
    target_samples = [
        sample for sample in samples if int(sample["target_gpu_memory_mib"]) > 0
    ]
    if not target_samples:
        raise RuntimeError("nvidia-smi did not attribute GPU memory to the rollout process")
    gpu_zero_samples = [
        gpu
        for sample in samples
        for gpu in sample["gpus"]  # type: ignore[union-attr]
        if gpu["index"] == 0
    ]
    if not gpu_zero_samples:
        raise RuntimeError("GPU 0 telemetry is missing")
    peak_target_mib = max(int(sample["target_gpu_memory_mib"]) for sample in samples)
    total_gpu_mib = int(gpu_zero_samples[0]["total_memory_mib"])
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_concurrent_runtime_profile",
            "schema_version": 1,
            "status": "passed",
            "started_at_utc": started_wall,
            "duration_seconds": duration_seconds,
            "command": command,
            "sample_interval_seconds": args.sample_interval_seconds,
            "concurrency_claim": (
                "The same process group held the B0 CUDA model while producing a "
                "successful, recorder-validated LIBERO simulator episode."
            ),
            "telemetry": {
                "sample_count": len(samples),
                "target_gpu_sample_count": len(target_samples),
                "sampling_errors": sampling_errors,
                "gpu": {
                    "index": 0,
                    "name": gpu_zero_samples[0]["name"],
                    "total_memory_bytes": total_gpu_mib * MIB,
                    "peak_target_process_memory_bytes": peak_target_mib * MIB,
                    "capacity_minus_peak_target_process_bytes": (
                        total_gpu_mib - peak_target_mib
                    )
                    * MIB,
                    "peak_global_used_memory_bytes": max(
                        int(gpu["used_memory_mib"]) for gpu in gpu_zero_samples
                    )
                    * MIB,
                    "minimum_global_free_memory_bytes": (
                        total_gpu_mib
                        - max(int(gpu["used_memory_mib"]) for gpu in gpu_zero_samples)
                    )
                    * MIB,
                },
                "peak_process_group_rss_bytes": max(
                    int(sample["process_group_rss_bytes"]) for sample in samples
                ),
                "torch_cuda": {
                    "peak_allocated_bytes": rollout_report["runtime"][
                        "peak_allocated_cuda_bytes"
                    ],
                    "peak_reserved_bytes": rollout_report["runtime"][
                        "peak_reserved_cuda_bytes"
                    ],
                },
                "policy_latency_ms": rollout_report["latency_ms"],
            },
            "rollout_validation": validation,
            "logs": {
                "stdout": str(stdout_path),
                "stdout_sha256": sha256_file(stdout_path),
                "stderr": str(stderr_path),
                "stderr_sha256": sha256_file(stderr_path),
            },
            "samples": samples,
        }
    )
    _write_sealed_json(args.profile_output_dir / "concurrent-runtime-report.json", report)
    return report


def main() -> int:
    report = _run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
