"""Reuse pinned training inputs without constructing demonstrated future targets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from eventground_wam.auxiliary_data import AuxiliaryBatches, AuxiliaryPairDataset
from eventground_wam.b0_data import (
    B0DataConfig,
    B0Sample,
    B0SampleIdentity,
    B0StreamingDataset,
    collate_b0_samples,
    load_trajectory_v2_split_index,
)
from eventground_wam.b0_model import load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, canonical_json_bytes
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.pixel_baselines import box_rgb


class MotionBatches(AuxiliaryBatches):
    def __getitem__(self, index):
        tokens = self.plan[index]
        if any(epoch != 0 for epoch, _, _ in tokens) or self.pairs.split != 'train':
            raise ValueError('motion training must use the frozen train-only schedule')
        samples = {}
        for ep in dict.fromkeys(t[1] for t in tokens):
            source = self.pairs.sources[ep]
            if source['split'] != 'train':
                raise ValueError('non-training source')
            actions, states, videos = self.pairs._episode(source)
            assignment = self.assignments[ep]
            for slot, (_, episode, anchor) in enumerate(tokens):
                if episode != ep:
                    continue
                if not 0 <= anchor <= len(actions)-8:
                    raise ValueError('invalid eight-action anchor')
                samples[slot] = B0Sample(B0SampleIdentity('train', ep, source['task_index'],
                    anchor, anchor+8, assignment.digest), source['instruction'],
                    videos[anchor, 0].transpose(2, 0, 1).copy(),
                    videos[anchor, 1].transpose(2, 0, 1).copy(), states[anchor].copy(),
                    self.pairs.normalizer.normalize(actions[anchor:anchor+8]).astype(np.float32))
        batch = collate_b0_samples([samples[i] for i in range(len(tokens))])
        if self.builder is None:
            self.builder = OFTInputBuilder(load_pinned_qwen_processor(
                cache_dir=self.cache_dir, local_files_only=True))
        encoding = self.builder(batch.instructions, list(batch.front_rgb), list(batch.wrist_rgb))
        images = torch.from_numpy(np.stack([np.stack([
            box_rgb(image.transpose(1, 2, 0)) for image in (front, wrist)])
            for front, wrist in zip(batch.front_rgb, batch.wrist_rgb, strict=True)]))
        return {'index': index, 'tokens': tokens, 'sample_count': len(tokens),
            'inputs': {k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in encoding.items()},
            'proprioception': torch.from_numpy(batch.proprioception),
            'targets': torch.from_numpy(batch.actions), 'current_images': images,
            'token_digest': batches_digest([tokens]),
            'identity_digest': hashlib.sha256(canonical_json_bytes(
                [tokens, [i.digest for i in batch.identities]])).hexdigest()}


def load_batches(normalizer, schedule, audit, *, root, data, cache):
    from week11_replay_common import read
    base = read(root/'configs/training/week4_b0_fresh_all_l1_v2.json')
    index = load_trajectory_v2_split_index(data/'derived/frozen/week2-dataset-v1',
        root/'configs/datasets/libero-plus-trajectory-v2')
    repo = data/f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo'
    dataset = B0StreamingDataset(DemonstrationReader(repo), index, normalizer, config=B0DataConfig(seed=17))
    pairs = AuxiliaryPairDataset(data/'derived/future-pairs/trajectory-v2-future-pairs-v1/primary', repo,
        split='train', expected_manifest_sha256=audit['input_files']['future_pair_manifest']['sha256'],
        normalizer=normalizer, state_bounds=base['state_normalization'], maximum_cache_frames=8192)
    plan = [[tuple(t) for t in row['tokens']] for row in schedule['batches']]
    return MotionBatches(dataset, pairs, plan, cache)


def verify_training_payloads(audit):
    from week11_replay_common import sha
    checked = {}
    for item in audit['input_files'].values():
        assert sha(item['path']) == item['sha256'], item['path']
        checked[item['path']] = item['sha256']
    for row in audit['selected_episode_metadata']:
        for item in row['payloads']:
            assert sha(Path(item['path'])) == item['expected_sha256'], item['path']
            checked[item['path']] = item['expected_sha256']
    return checked
