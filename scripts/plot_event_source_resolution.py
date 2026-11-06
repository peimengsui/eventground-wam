"""Plot the six preselected diagnostic pairs; no scene-admission claim."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
args = parser.parse_args()
run = args.base / "instrumentation-recovery/run"
fig, axes = plt.subplots(3, 4, figsize=(10, 8.2))
for row, frame in enumerate((0, 44, 87)):
    for col, (camera, source) in enumerate(
        (("front", "target"), ("front", "replayed"), ("wrist", "target"), ("wrist", "replayed"))
    ):
        suffix = "-raw" if source == "replayed" else ""
        with Image.open(run / f"{source}-{camera}-{frame:03d}{suffix}.png") as im:
            pixels = np.asarray(im)
        if source == "replayed":
            pixels = pixels[::-1, ::-1]
        axes[row, col].imshow(pixels)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(f"{camera.title()} · {source}", fontsize=11)
        if col == 0:
            axes[row, col].set_ylabel(f"Frame {frame}", fontsize=11)
fig.suptitle("Episode 7856: diagnostic camera pairs", fontsize=16)
fig.text(
    0.5,
    0.018,
    "Replayed images use the predeclared 180° orientation. Target front is blurred.\n"
    "Sparse visual agreement does not establish historical scene identity or contact/grasp labels.",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.065, 1, 0.96), pad=0.6)
fig.savefig(args.base / "camera-comparison.png", dpi=150)
plt.close(fig)
