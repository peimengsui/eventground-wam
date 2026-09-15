#!/usr/bin/env python3
"""Install/control a persistent user-systemd B0 job; SSH never owns its lifetime."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def unit_quote(value: str) -> str:
    if any(char in value for char in "\n\r\x00"):
        raise ValueError("systemd arguments cannot contain newline or NUL")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def render_unit(args: argparse.Namespace) -> str:
    command = [str(args.python), "-u", str(PROJECT_ROOT / "scripts/train_b0.py")]
    for name in ("recipe", "repository_root", "frozen_root", "normalization", "cache_dir",
                 "output_root"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.pilot_steps:
        command.extend(["--pilot-steps", str(args.pilot_steps)])
    elif args.pilot_report:
        command.extend(["--pilot-report", str(args.pilot_report)])
    else:
        raise ValueError("production requires a matching pilot report")
    environment = {
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'scripts'}",
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    lines = [
        "[Unit]", "Description=EventGround B0 five-epoch training",
        "StartLimitIntervalSec=3600", "StartLimitBurst=4", "", "[Service]",
        "Type=exec", "WorkingDirectory=" + str(PROJECT_ROOT).replace("%", "%%"),
        "ExecStart=" + " ".join(unit_quote(item) for item in command),
        *["Environment=" + unit_quote(f"{key}={value}") for key, value in environment.items()],
        "Restart=on-failure", "RestartSec=60", "RestartPreventExitStatus=78",
        "TimeoutStopSec=300", "KillMode=mixed", "Nice=5", "UMask=0077",
        "StandardOutput=journal",
        "StandardError=inherit", "", "[Install]", "WantedBy=default.target", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "start", "status", "stop"))
    parser.add_argument("--name", default="eventground-b0-week4")
    parser.add_argument("--python", type=Path,
                        default=PROJECT_ROOT / "environments/b0/.venv/bin/python")
    for name in ("recipe", "repository-root", "frozen-root", "normalization", "cache-dir",
                 "output-root", "pilot-report"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--replace-inactive", action="store_true")
    args = parser.parse_args()
    if not args.name.startswith("eventground-b0-") or any(
        not (c.isalnum() or c == "-") for c in args.name
    ):
        parser.error("name must be an eventground-b0- name containing letters/digits/hyphens")
    unit_name = args.name + ".service"
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # Resolving the interpreter symlink would bypass its virtual environment.
            absolute = value.expanduser().absolute() if key == "python" else value.expanduser().resolve()
            setattr(args, key, absolute)
    if args.action in {"status", "stop"}:
        subprocess.run(["systemctl", "--user", args.action, unit_name, "--no-pager"], check=True)
        return
    for field in ("recipe", "repository_root", "frozen_root", "normalization", "cache_dir",
                  "output_root"):
        if getattr(args, field) is None:
            parser.error(f"--{field.replace('_', '-')} is required")
    unit = render_unit(args)
    if args.action == "render":
        print(unit, end="")
        return
    if platform.system() != "Linux":
        raise RuntimeError("install this service on the Linux GPU host")
    if not args.python.is_file():
        raise RuntimeError(f"locked B0 interpreter is missing: {args.python}")
    subprocess.run(["systemctl", "--user", "show-environment"], check=True, capture_output=True)
    user = str(os.getuid())
    linger = subprocess.check_output(
        ["loginctl", "show-user", user, "--property=Linger", "--value"], text=True,
    ).strip()
    if linger != "yes":
        subprocess.run(["loginctl", "enable-linger", user], check=True)
        linger = subprocess.check_output(
            ["loginctl", "show-user", user, "--property=Linger", "--value"], text=True,
        ).strip()
        if linger != "yes":
            raise RuntimeError("user lingering must be enabled for logout/reboot survival")
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    unit_path = directory / unit_name
    if unit_path.exists() and unit_path.read_text() != unit:
        state = subprocess.check_output(
            ["systemctl", "--user", "show", unit_name, "--property=ActiveState", "--value"],
            text=True,
        ).strip()
        if not args.replace_inactive or state not in {"inactive", "failed"}:
            raise RuntimeError("existing unit differs; explicitly replace an inactive unit")
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = unit_path.with_suffix(".partial")
    temporary.write_text(unit)
    temporary.replace(unit_path)
    subprocess.run(["systemd-analyze", "--user", "verify", str(unit_path)], check=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", unit_name], check=True)
    subprocess.run(["systemctl", "--user", "show", unit_name,
                    "--property=ActiveState,SubState,MainPID,Restart,Linger"], check=True)
    print(f"Host-owned service: {unit_name}; linger={linger}")
    print(f"Logs: journalctl --user -u {unit_name}; progress: {args.output_root / 'events.jsonl'}")


if __name__ == "__main__":
    main()
