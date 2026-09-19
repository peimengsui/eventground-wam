#!/usr/bin/env python3
"""Fit OFT proprioception quantiles from the existing training split only."""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow.parquet as pq

from eventground_wam.b0_data import load_frozen_b0_split_index
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = load_frozen_b0_split_index(args.frozen_root)
    train = {item.episode_index for item in index.assignments if item.split == "train"}
    rows = []
    for episode in DemonstrationReader(args.repository_root).iter_episodes():
        if episode.episode_index in train:
            column = pq.read_table(episode.parquet_path, columns=["observation.state"])
            array = column.column(0).combine_chunks().values.to_numpy().reshape(-1, 8)
            if array.shape[0] != episode.length or not np.isfinite(array).all():
                raise ValueError("invalid state rows in training episode")
            rows.append(array)
    values = np.concatenate(rows)
    if len(rows) != 11521 or len(values) != 1796601:
        raise ValueError("training state coverage differs from the frozen dataset")
    artifact = seal_contract({
        "artifact_type": "eventground_wam.b0_oft_state_statistics",
        "split": "train", "episode_count": len(rows), "frame_count": len(values),
        "source": index.source_identity,
        "normalization": "per-dimension-q01-q99-affine-clipped-to-[-1,1]-constant-zero",
        "lower_q01": np.quantile(values, .01, axis=0).tolist(),
        "upper_q99": np.quantile(values, .99, axis=0).tolist(),
    })
    atomic_json(args.output, artifact)
    print(f"Saved train-only state normalization: {args.output}")


if __name__ == "__main__":
    main()
