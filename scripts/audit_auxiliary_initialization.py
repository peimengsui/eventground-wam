#!/usr/bin/env python3
"""Compare a fresh shared policy to the completed B0 run's initial tensor digest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_trajectory_v2_split_index,
)
from eventground_wam.b0_diagnostics import tensor_tree_digest
from eventground_wam.b0_fresh import fresh_plan, load_fresh_recipe
from eventground_wam.b0_model import b0_trainable_state_dict, load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder, build_oft_policy
from eventground_wam.b0_training import read_sealed, seed_training, sha256_file
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.pixel_predictor import tensor_state_digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    recipe = load_fresh_recipe(root/'configs/training/week4_b0_fresh_all_l1_v2.json')
    controls = read_sealed(root/'configs/training/phase2_two_epoch_v1.json')
    seed_training(17)
    torch.set_num_threads(1)
    policy, names = build_oft_policy(recipe, str(args.cache), torch.device('cpu'))
    state = b0_trainable_state_dict(policy, allowed_exact_names=names)
    original = tensor_tree_digest(state)
    new = tensor_state_digest(state)
    assert original == controls['initialization']['b0_initial_task_state_digest']
    for method in ('pixel', 'jepa'):
        assert read_sealed(args.pilot/f'{method}-calibration.json')['shared_initialization'] == new
    manifest = read_sealed(args.pairs/'manifest.json')
    assert sha256_file(args.pairs/'manifest.json') == read_sealed(
        root/'configs/training/week6_jepa_cpt_v1.json')['pair_manifest_sha256']
    metadata_hash = sha256_file(args.pairs/'sources.jsonl')
    assert metadata_hash == manifest['outputs']['sources.jsonl']['sha256']
    rows = [json.loads(line) for line in (args.pairs/'sources.jsonl').read_text().splitlines()]
    instructions = sorted({r['instruction'] for r in rows if r['split'] == 'train'})
    builder = OFTInputBuilder(load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True))
    image = np.zeros((3, 256, 256), dtype=np.uint8)
    lengths = []
    for start in range(0, len(instructions), 32):
        selected = instructions[start:start+32]
        encoded = builder(selected, [image]*len(selected), [image]*len(selected))
        lengths.extend(encoded['attention_mask'].sum(1).tolist())
    assert max(lengths) <= 512
    index = load_trajectory_v2_split_index(args.data/'derived/frozen/week2-dataset-v1',
                                         root/'configs/datasets/libero-plus-trajectory-v2')
    normalizer = load_b0_action_normalizer(root/'docs/model/week4-fresh-v2/action-statistics.json')
    dataset = B0StreamingDataset(DemonstrationReader(args.data/
        f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo'),
        index, normalizer, config=B0DataConfig(seed=17))
    episode_lengths = {e.episode_index: e.length for e in dataset.episodes}
    window_bounds = []
    for epoch in range(2):
        plan = fresh_plan(dataset, recipe, epoch)
        assert batches_digest(plan) == controls['batching_and_sampling']['mixed_plan_digests'][epoch]
        worst = {'frames': 0}
        for stage in range(0, len(plan), 15000):
            end = min(stage+15000, len(plan))
            for start in range(stage, end, 128):
                episodes = {t[1] for batch in plan[start:min(start+128, end)] for t in batch}
                frames = sum(episode_lengths[e] for e in episodes)
                if frames > worst['frames']:
                    worst = {'frames': frames, 'epoch': epoch, 'first_batch': start,
                             'episodes': len(episodes)}
        window_bounds.append(worst)
    assert max(w['frames'] for w in window_bounds) <= 8192
    print(json.dumps(seal_contract({'artifact_type': 'eventground_wam.week7_b0_initialization_audit',
        'b0_task_state_digest': original, 'shared_tensor_digest': new,
        'both_arms_equal_original_b0_initialization': True, 'device': 'cpu',
        'training_instruction_count': len(instructions), 'maximum_prompt_tokens': max(lengths),
        'all_training_prompts_within_interface_bound': True,
        'source_metadata_sha256': metadata_hash, 'real_image_payload_reads': 0,
        'maximum_distinct_episode_frames_per_mixing_window': window_bounds,
        'every_mixing_window_fits_decoded_and_teacher_cache_bounds': True,
        'shared_trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
        'source_sha256': {str(p.relative_to(root)): sha256_file(p) for p in [
            root/'src/eventground_wam/b0_oft.py', root/'src/eventground_wam/b0_model.py',
            root/'src/eventground_wam/b0_training.py']},
        'audit_script_sha256': sha256_file(Path(__file__)),
        'controls_digest': controls['digest'], 'recipe_digest': recipe['digest']}), indent=2))


if __name__ == '__main__':
    main()
