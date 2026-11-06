#!/usr/bin/env python3
"""Read-only train-v2 exposure audit and deterministic demonstration contact sheets.

Stage proxies describe recorded commands, not simulator success or object predicates.
No model, optimizer, validation/test payload, or simulator rollout is used.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_trajectory_v2_split_index,
)
from eventground_wam.b0_fresh import fresh_plan, load_fresh_recipe
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader

TARGET_TASKS = ('libero_10/task-01', 'libero_10/task-02', 'libero_10/task-07')


def command_events(gripper, minimum_run=4):
    """Count adjacent stable +/-1 runs; no interpolation across short command runs."""
    cuts = np.r_[0, np.flatnonzero(np.diff(gripper) != 0) + 1, len(gripper)]
    events = []
    for j in range(1, len(cuts) - 1):
        before, at, after = map(int, cuts[j - 1:j + 2])
        if at - before >= minimum_run and after - at >= minimum_run:
            events.append({'frame': at, 'kind': 'release' if gripper[at] == -1 else 'close'})
    return events


def anchor_statistics(action):
    n = len(action) - 7
    events = command_events(action[:, 6])
    releases = [x['frame'] for x in events if x['kind'] == 'release']
    first_release = releases[0] if releases else None
    # Every frame is a target, but terminal observations lack a complete future chunk.
    target_multiplicity = np.convolve(np.ones(n, dtype=int), np.ones(8, dtype=int))
    assert len(target_multiplicity) == len(action) and target_multiplicity.sum() == n * 8
    post_release = max(0, n - first_release) if first_release is not None else 0
    return {
        'stable_command_events': events, 'first_stable_release_frame': first_release,
        'valid_anchors': n, 'last_40pct_anchors': n - math.ceil(.6 * n),
        'post_first_stable_release_anchors': post_release,
        'last_20_frames_target_terms': int(target_multiplicity[-20:].sum()),
        'last_20_frames_anchor_count': max(0, n - max(0, len(action) - 20)),
        'last_7_frames_used_as_observation_anchors': 0,
        'last_7_frames_target_terms': int(target_multiplicity[-7:].sum()),
        'raw_gripper_transitions': int(np.count_nonzero(np.diff(action[:, 6]))),
        'motion_norm_median': float(np.median(np.linalg.norm(action[:, :6], axis=1))),
    }


def quantiles(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1])), strict=True))


def write_report(root, name, value):
    result = seal_contract(value)
    atomic_json(root / name, result)
    return result


def make_previews(output, selected, episodes, numeric):
    import av
    from PIL import Image, ImageDraw

    root = output / 'previews'
    root.mkdir(exist_ok=True)
    previews = []
    for item in selected:
        episode = episodes[item['episode_index']]
        row = numeric[episode.episode_index]
        points = list(map(int, np.linspace(0, episode.length - 1, 12)))
        canvas = Image.new('RGB', (6 * 256, 4 * 280 + 50), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f"{item['task_key']} | train episode {episode.episode_index} | "
                  f"{episode.length} frames | group {item['group_digest'][:12]}", fill='black')
        draw.text((10, 25), 'Source frames; each time has front then wrist below. '
                  'These are demonstration images, not policy rollouts.', fill='black')
        files = {}
        for camera_index, (camera, path) in enumerate(episode.video_paths):
            files[camera] = {'path': str(path), 'sha256': sha256_file(path)}
            frames = {}
            with av.open(str(path)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i in points:
                        frames[i] = frame.to_image()
            if len(frames) != len(set(points)):
                raise ValueError('missing selected video frame')
            for i, frame in enumerate(points):
                x, y = (i % 6) * 256, 50 + (2 * (i // 6) + camera_index) * 280
                canvas.paste(frames[frame], (x, y + 20))
                draw.text((x + 4, y + 2), f'{camera.rsplit(".", 1)[-1]} f={frame} '
                          f't={frame / 20:.2f}s', fill='black')
        name = f"{item['task_key'].replace('/', '-')}-episode-{episode.episode_index:06d}.jpg"
        canvas.save(root / name, quality=92)
        previews.append({**item, 'frames': points, 'source_videos': files,
                         'sheet': f'previews/{name}', 'sheet_sha256': sha256_file(root / name),
                         'stable_command_events': row['stable_command_events']})
    return previews


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    split_root = args.source / 'configs/datasets/libero-plus-trajectory-v2'
    split = read_sealed(split_root / 'manifest.json')
    fp_path = args.evidence / 'fingerprints.jsonl'
    if sha256_file(fp_path) != split['outputs']['fingerprints.jsonl']['sha256']:
        raise ValueError('fingerprint evidence changed')
    fingerprints = {r['episode_index']: r for r in
                    map(json.loads, fp_path.read_text().splitlines())}
    index = load_trajectory_v2_split_index(args.data / 'derived/frozen/week2-dataset-v1', split_root)
    reader = DemonstrationReader(args.data / 'datasets/libero-plus-lerobot/snapshots/'
                                 '22c57433fef692b5b9ecc0795344daac7fa867a5/repo')
    episodes = {e.episode_index: e for e in reader.iter_episodes()}
    groups = [g for g in map(json.loads, (split_root / 'groups.jsonl').read_text().splitlines())
              if g['split'] == 'train']
    memberships = {i: g for g in groups for i in g['episode_indices']}
    train = [e for i, e in episodes.items() if index.assignments[i].split == 'train']
    assert set(memberships) == {e.episode_index for e in train}
    assert len(train) == 11435 and len(groups) == 1341

    def inspect(episode):
        i = episode.episode_index
        if sha256_file(episode.parquet_path) != fingerprints[i]['parquet_sha256']:
            raise ValueError(f'source parquet changed: {i}')
        table = pq.read_table(episode.parquet_path, columns=['action'], use_threads=False)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        if (action.shape != (episode.length, 7) or not np.isfinite(action).all()
                or not set(np.unique(action[:, 6])) <= {-1., 1.}):
            raise ValueError(f'invalid source actions: {i}')
        g = memberships[i]
        return {'episode_index': i, 'task_key': g['task_key'], 'group_digest': g['group_digest'],
                'length': episode.length, 'variants_in_group': len(g['episode_indices']),
                'parquet_sha256': fingerprints[i]['parquet_sha256'], **anchor_statistics(action)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(inspect, train))
    numeric = {r['episode_index']: r for r in rows}
    assert sum(r['valid_anchors'] for r in rows) == 1706022
    # Numeric trajectories in each frozen group must have identical proxy boundaries.
    for group in groups:
        assert len({json.dumps(numeric[i]['stable_command_events'])
                    for i in group['episode_indices']}) == 1
    (args.output / 'train-episodes.jsonl').write_text(''.join(
        json.dumps(r, sort_keys=True) + '\n' for r in rows))
    print(json.dumps({'verified_train_episodes': len(rows), 'numeric_groups': len(groups)}), flush=True)

    by_task = {}
    for task in sorted({g['task_key'] for g in groups}):
        task_rows = [r for r in rows if r['task_key'] == task]
        task_groups = [g for g in groups if g['task_key'] == task]
        chunks = sum(r['valid_anchors'] for r in task_rows)
        by_task[task] = {
            'episodes': len(task_rows), 'numeric_groups': len(task_groups), 'chunks_per_epoch': chunks,
            'sample_share_pct': chunks / 1706022 * 100,
            'frames_per_group': quantiles([g['episode_length'] for g in task_groups]),
            'variants_per_group': quantiles([len(g['episode_indices']) for g in task_groups]),
            'group_exposure_chunks': quantiles([(g['episode_length'] - 7) * len(g['episode_indices'])
                                               for g in task_groups]),
            'groups_with_stable_release': sum(numeric[g['episode_indices'][0]]['first_stable_release_frame']
                                              is not None for g in task_groups),
            'post_release_chunks': sum(r['post_first_stable_release_anchors'] for r in task_rows),
            'last_40pct_chunks': sum(r['last_40pct_anchors'] for r in task_rows),
        }
    # Selection is based only on training group length, never policy losses/outcomes.
    selected = []
    for task in TARGET_TASKS:
        ordered = sorted([g for g in groups if g['task_key'] == task],
                         key=lambda g: (g['episode_length'], g['group_digest']))
        for q in (.25, .5, .75):
            g = ordered[round((len(ordered) - 1) * q)]
            selected.append({'task_key': task, 'group_digest': g['group_digest'],
                             'episode_index': min(g['episode_indices']), 'length_quantile': q})
    previews = make_previews(args.output, selected, episodes, numeric)
    write_report(args.output, 'preview-manifest.json', {'selection': 'Training-only length '
                 'quartiles per target task; group-digest tie break; smallest episode index per group',
                 'previews': previews, 'not_population_success_estimate': True})

    recipe = load_fresh_recipe(args.source / 'configs/training/week4_b0_fresh_all_l1_v2.json')
    normalizer = load_b0_action_normalizer(args.work / 'prepared/action-statistics.json')
    dataset = B0StreamingDataset(reader, index, normalizer, config=B0DataConfig(seed=17, split='train'))
    manifest = read_sealed(args.work / 'training/training-manifest.json')
    plan_reports = []
    for epoch in range(2):
        plan = fresh_plan(dataset, recipe, epoch)
        digest = batches_digest(plan)
        assert digest == manifest['mixed_plan_digests'][epoch]
        assert len(plan) == 58566 and sum(map(len, plan)) == 1706022
        task_mass, phase_mass = defaultdict(float), defaultdict(float)
        sizes, task_counts, group_counts, exposures = Counter(), Counter(), Counter(), Counter()
        for batch in plan:
            sizes[len(batch)] += 1
            task_counts[len({numeric[t[1]]['task_key'] for t in batch})] += 1
            group_counts[len({numeric[t[1]]['group_digest'] for t in batch})] += 1
            for _, episode, anchor in batch:
                row = numeric[episode]
                exposures[episode, anchor] += 1
                task_mass[row['task_key']] += 1 / len(batch)
                if anchor >= math.ceil(.6 * row['valid_anchors']):
                    phase_mass[row['task_key']] += 1 / len(batch)
        assert len(exposures) == 1706022 and set(exposures.values()) == {1}
        report = {'epoch': epoch + 1, 'plan_digest': digest, 'matches_training_manifest': True,
                  'every_valid_train_anchor_exactly_once': True,
                  'batch_sizes': {str(k): v for k, v in sizes.items()},
                  'tasks_per_batch': {str(k): v for k, v in task_counts.items()},
                  'numeric_groups_per_batch': {str(k): v for k, v in group_counts.items()},
                  'nominal_mean_loss_task_share_pct': {k: v / len(plan) * 100 for k, v in task_mass.items()},
                  'nominal_last40_task_share_pct': {k: v / len(plan) * 100 for k, v in phase_mass.items()},
                  'weighting_limit': 'Sum of 1/batch_size coefficients, not measured gradients or Adam influence'}
        plan_reports.append(report)
        print(json.dumps({'epoch_plan_verified': epoch + 1, 'digest': digest}), flush=True)
        del plan, exposures
    result = write_report(args.output, 'sampling-audit.json', {
        'artifact_type': 'eventground_wam.fresh_v2_stage_sampling_audit', 'status': 'complete',
        'script_sha256': sha256_file(Path(__file__)), 'split_identity': index.source_identity,
        'train_episodes': len(rows), 'train_groups': len(groups), 'train_chunks_per_epoch': 1706022,
        'training_manifest_digest': manifest['digest'], 'plan_reports': plan_reports,
        'per_task': by_task, 'target_tasks': list(TARGET_TASKS),
        'numeric_rows_sha256': sha256_file(args.output / 'train-episodes.jsonl'),
        'payload_scope': 'train only; no validation/test numeric or image payloads',
        'no_optimizer_updates': True, 'no_policy_rollouts': True,
        'stage_proxy_definitions': {
            'post_release': 'After first +1 to -1 command transition with >=4 frames in both adjacent runs; '
                            'does not establish an object was released or placed',
            'last40': 'Anchor index >= ceil(0.6 * number_of_valid_anchors); temporal proxy only',
            'terminal': 'Final 7 observations excluded as anchors because targets require 8 actions; '
                        'their actions still appear in earlier targets, with decreasing multiplicity',
        },
    })
    print(json.dumps({'complete': True, 'digest': result['digest']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'data', 'work', 'evidence', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    main(parser.parse_args())
