#!/usr/bin/env python3
"""Fit v2 train-only normalizers and verify source bytes before fresh training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eventground_wam.artifact_provenance import project_state
from eventground_wam.b0_data import fit_b0_action_statistics, load_trajectory_v2_split_index
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=home / 'data/eventground-wam/libero-plus')
    parser.add_argument('--evidence', type=Path, default=home / 'data/eventground-wam/analysis/week4-trajectory-split-v2/result')
    args = parser.parse_args()
    output = args.work_root / 'prepared'
    if output.exists():
        raise ValueError('refusing to overwrite prepared statistics')
    revision, clean = project_state(ROOT)
    if not clean:
        raise ValueError('preparation requires committed source')
    split_root = ROOT / 'configs/datasets/libero-plus-trajectory-v2'
    manifest = read_sealed(split_root / 'manifest.json')
    index = load_trajectory_v2_split_index(args.data_root / 'derived/frozen/week2-dataset-v1', split_root)
    reader = DemonstrationReader(args.data_root / 'datasets/libero-plus-lerobot/snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/repo')
    fp_path = args.evidence / 'fingerprints.jsonl'
    if sha256_file(fp_path) != manifest['outputs']['fingerprints.jsonl']['sha256']:
        raise ValueError('source fingerprint evidence changed')
    fingerprints = {r['episode_index']: r for r in map(json.loads, fp_path.read_text().splitlines())}
    assignments = {a.episode_index: a for a in index.assignments}
    states = []
    counts = {'train': 0, 'validation': 0}
    for episode in reader.iter_episodes():
        split = assignments[episode.episode_index].split
        if split == 'test':
            continue
        if sha256_file(episode.parquet_path) != fingerprints[episode.episode_index]['parquet_sha256']:
            raise ValueError(f'source bytes changed: {episode.episode_index}')
        counts[split] += 1
        if split == 'train':
            array = pq.read_table(episode.parquet_path, columns=['observation.state']).column(0).combine_chunks().values.to_numpy().reshape(-1, 8)
            if len(array) != episode.length or not np.isfinite(array).all():
                raise ValueError('invalid training state')
            states.append(array)
    values = np.concatenate(states)
    if counts != {'train': 11435, 'validation': 1528} or len(values) != 1786067:
        raise ValueError('v2 coverage mismatch')
    state = seal_contract({
        'artifact_type': 'eventground_wam.b0_oft_state_statistics', 'split': 'train',
        'source': index.source_identity, 'episode_count': len(states), 'frame_count': len(values),
        'normalization': 'per-dimension-q01-q99-affine-clipped-to-[-1,1]-constant-zero',
        'lower_q01': np.quantile(values, .01, axis=0).tolist(),
        'upper_q99': np.quantile(values, .99, axis=0).tolist(),
    })
    action = fit_b0_action_statistics(reader, index, project_git_revision=revision, project_tree_clean=True)
    if action['split']['valid_action_chunk_count'] != 1706022:
        raise ValueError('wrong action-statistics partition')
    atomic_json(output / 'action-statistics.json', action)
    atomic_json(output / 'state-statistics.json', state)
    receipt = seal_contract({
        'artifact_type': 'eventground_wam.fresh_v2_preparation', 'status': 'verified',
        'project_git_revision': revision, 'source': index.source_identity,
        'source_parquet_hashes_verified': counts, 'test_payloads_read': 0,
        'action_statistics_sha256': sha256_file(output / 'action-statistics.json'),
        'state_statistics_sha256': sha256_file(output / 'state-statistics.json'),
        'counts': manifest['counts'],
    })
    atomic_json(output / 'preparation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
