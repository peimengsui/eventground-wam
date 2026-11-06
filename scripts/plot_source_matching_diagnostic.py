"""Plot saved, independently verified source-pilot gripper discrepancies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--numeric-root", type=Path)
    args = parser.parse_args()
    numeric_root = args.numeric_root or args.base / "run"
    result = json.loads((args.base / "recorded-state-diagnostic/results.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharey=True)
    for axis, row in zip(axes, result["results"], strict=True):
        case = f"candidate-{row['rank']:02d}"
        states = np.load(
            args.base / "recorded-state-diagnostic" / case / "recorded-states.npy",
            allow_pickle=False,
        )
        with np.load(numeric_root / case / "target-numeric.npz", allow_pickle=False) as target:
            gripper = target["states"][:, 6:8].astype(np.float64)
        error = np.abs(gripper - states[:, row["gripper_state_columns"]]).max(axis=1) * 1000
        axis.plot(np.arange(len(error)), error, color="#b34836", linewidth=1.5)
        axis.axhline(0.01, color="#344955", linestyle="--", linewidth=1)
        axis.set_title(
            f"Candidate {row['rank']} · {row['demo_id']} · maximum error {error.max():.2f} mm",
            loc="left",
            fontsize=11,
        )
        axis.set_ylabel("Finger qpos error (mm)")
        axis.set_xlabel("Recorded frame (20 Hz)")
        axis.set_xlim(0, len(error) - 1)
        axis.set_ylim(-0.1, 5.25)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Matching action sequences do not establish matching physical states",
        fontsize=14,
        x=0.08,
        ha="left",
    )
    fig.text(
        0.08,
        0.01,
        "Maximum absolute error across the two finger joints; dashed line: frozen 0.01 mm state tolerance.\nFirst three queue-ranked action matches. Recorded states only; no replay or physical labels admitted.",
        fontsize=9,
        color="#44515c",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(args.base / "recorded-gripper-discrepancy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
