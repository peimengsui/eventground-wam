#!/usr/bin/env python3
"""Read-only continuation planning audit; no GPU, media decode or model forward."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--verify-media-hashes', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root / 'src'))

    import torch

    from eventground_wam.b0_data import (
        B0DataConfig,
        B0StreamingDataset,
        load_b0_action_normalizer,
        load_trajectory_v2_split_index,
    )
    from eventground_wam.b0_fresh import fresh_plan, load_fresh_recipe
    from eventground_wam.b0_training import read_sealed, sha256_file
    from eventground_wam.continuation_freeze import batches_digest
    from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
    from eventground_wam.demonstration_reader import DemonstrationReader

    root, data = args.source_root, args.data
    integration = read_sealed(root / 'configs/evaluation/week11_integration_v1.json')
    pins = integration['endpoints']
    epoch = Path(pins['m4']['checkpoint_path']).parent
    files = {
        'action_checkpoint': (epoch / 'M4.ckpt', pins['m4']['policy_sha256']),
        'full_predictor': (epoch / 'full-training-predictor.pt', pins['m4']['predictor_sha256']),
        'event_checkpoint': (Path(pins['event']['remote_endpoint']), pins['event']['endpoint_sha256']),
    }
    pair_root = data / 'derived/future-pairs/trajectory-v2-future-pairs-v1/primary'
    cpt = read_sealed(root / 'configs/training/week6_jepa_cpt_v1.json')
    files['future_pair_manifest'] = (pair_root / 'manifest.json', cpt['pair_manifest_sha256'])
    manifest = read_sealed(pair_root / 'manifest.json')
    # No validation payload is opened. The source catalog includes split metadata.
    for name in ('sources.jsonl', 'train.bin'):
        files[name] = (pair_root / name, manifest['outputs'][name]['sha256'])
    inventory = {}
    for name, (path, expected) in files.items():
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f'pinned input changed: {name}')
        inventory[name] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}

    # The inference export omits the trained JEPA projection. Both exports are
    # required; replacing that projection with its original CPT state is invalid.
    action = torch.load(epoch / 'M4.ckpt', map_location='cpu', weights_only=True)
    predictor = torch.load(epoch / 'full-training-predictor.pt', map_location='cpu', weights_only=True)
    if predictor['global_step'] != 117132 or predictor['identity_digest'] != pins['m4']['identity_digest']:
        raise ValueError('full predictor belongs to a different endpoint')
    core = action['states']['future_core']
    if not all(k in predictor['state_dict'] and torch.equal(v, predictor['state_dict'][k])
               for k, v in core.items()):
        raise ValueError('action export and full predictor do not share the same trained core')
    tensor_check = {'shared_core_tensors': len(core), 'bitwise_equal': True,
                    'training_only_tensor_names': sorted(set(predictor['state_dict']) - set(core))}
    del action, predictor, core

    recipe = load_fresh_recipe(root / 'configs/training/week4_b0_fresh_all_l1_v2.json')
    index = load_trajectory_v2_split_index(data / 'derived/frozen/week2-dataset-v1',
                                         root / 'configs/datasets/libero-plus-trajectory-v2')
    repository = data / f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo'
    dataset = B0StreamingDataset(
        DemonstrationReader(repository), index,
        load_b0_action_normalizer(root / 'docs/model/week4-fresh-v2/action-statistics.json'),
        config=B0DataConfig(seed=17))
    plan = fresh_plan(dataset, recipe, 0)
    expected = read_sealed(root / 'configs/training/week7_m2_auxiliary_v1.json')['mixed_plan_digests'][0]
    if batches_digest(plan) != expected or len(plan) != 58566 or dataset.sample_count != 1706022:
        raise ValueError('original training plan does not reproduce')
    selected = plan[:2000]
    sources = {r['episode_index']: r for r in map(json.loads, (pair_root / 'sources.jsonl').read_text().splitlines())}
    selected_episodes = sorted({t[1] for b in selected for t in b})
    futures = Counter()
    task_presentations = Counter()
    episode_rows = []
    for episode in selected_episodes:
        source = sources[episode]
        if source['split'] != 'train':
            raise ValueError('non-training episode in continuation plan')
        payloads = []
        for entry in source['files'].values():
            path = repository / entry['path']
            if not path.is_file() or path.stat().st_size != entry['bytes']:
                raise ValueError(f'missing or truncated training media: {path}')
            if args.verify_media_hashes and sha256_file(path) != entry['sha256']:
                raise ValueError(f'training media checksum differs: {path}')
            payloads.append({'path': str(path), 'bytes': path.stat().st_size,
                             'expected_sha256': entry['sha256'],
                             'content_hash_checked': args.verify_media_hashes})
        episode_rows.append({'episode_index': episode, 'episode_length': source['episode_length'],
                             'task_index': source['task_index'], 'split': source['split'],
                             'source_digest': source['digest'], 'payloads': payloads})
    for batch in selected:
        for _, episode, anchor in batch:
            source = sources[episode]
            task_presentations[source['task_index']] += 1
            futures.update(h for h in (2, 4, 8, 16, 32) if anchor + h < source['episode_length'])
    source_hashes = {str(p.relative_to(root)): sha256_file(p) for p in (
        root / 'src/eventground_wam/b0_fresh.py', root / 'src/eventground_wam/b0_training.py',
        root / 'src/eventground_wam/b0_data.py', root / 'src/eventground_wam/continuation_freeze.py')}
    report = {
        'artifact_type': 'eventground_wam.week11_multihorizon_input_audit',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'host': subprocess.check_output(['hostname'], text=True).strip(),
        'data_free_bytes': shutil.disk_usage(data).free,
        'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                                                 '--format=csv,noheader'], text=True).strip(),
        'input_files': inventory, 'predictor_core_consistency': tensor_check,
        'source_root': str(root), 'source_sha256': source_hashes,
        'full_epoch_plan_digest': expected, 'selected_plan_digest': batches_digest(selected),
        'selection': 'First 2000 unchanged logical batches from original epoch-0 mixed plan',
        'selected_batches': selected, 'selected_episode_metadata': episode_rows,
        'updates_per_arm': len(selected), 'action_presentations_per_arm': sum(map(len, selected)),
        'future_pairs_per_arm_by_horizon': {str(k): v for k, v in sorted(futures.items())},
        'task_presentations': {str(k): v for k, v in sorted(task_presentations.items())},
        'short_batch_size_counts': {str(k): v for k, v in sorted(Counter(map(len, selected)).items())},
        'selected_training_episodes': len(selected_episodes),
        'selected_training_tasks': len(task_presentations),
        'payload_check_scope': ('All selected training payload byte sizes and SHA256'
                                if args.verify_media_hashes else
                                'Existence and byte sizes only; hashes required during execution qualification'),
        'gpu_model_loads': 0, 'model_forwards': 0, 'optimizer_updates': 0,
        'media_decodes': 0, 'validation_or_test_payload_reads': 0,
    }
    print(json.dumps(seal_contract(report), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
