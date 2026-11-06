#!/usr/bin/env python3
"""Freeze two matched continuation sample plans without loading a model or training."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_trajectory_v2_split_index,
)
from eventground_wam.b0_fresh import fresh_plan, load_fresh_recipe
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader

EXPERIMENT = 'week4-late-stage-screen-v1'
TARGETS = ('libero_10/task-01', 'libero_10/task-02', 'libero_10/task-07')
PARENT_SHA = '7a385fea68fe5d14f642b31e18f8e2e166ccd984d9b4f1ff5b68484191a9b720'


def weighted_anchor(n, update, slot, episode):
    """Weight each final-40% anchor twice; other valid anchors once."""
    cut = math.ceil(.6 * n)
    mass = cut + 2 * (n - cut)
    seed = f'{EXPERIMENT}|17|{update}|{slot}|{episode}'.encode()
    draw = int.from_bytes(hashlib.sha256(seed).digest(), 'big') % mass
    return draw if draw < cut else cut + (draw - cut) // 2


def summarize(plan, rows):
    tasks, episodes, groups, phase, target_total = (Counter() for _ in range(5))
    unique = set()
    for batch in plan:
        for _, episode, anchor in batch:
            row = rows[episode]
            if not 0 <= anchor < row['valid_anchors']:
                raise ValueError('invalid target boundary')
            tasks[row['task_key']] += 1
            episodes[episode] += 1
            groups[row['group_digest']] += 1
            unique.add((episode, anchor))
            if row['task_key'] in TARGETS:
                target_total[row['task_key']] += 1
                phase[row['task_key']] += anchor >= math.ceil(.6 * row['valid_anchors'])
    return {'updates': len(plan), 'samples': sum(map(len, plan)), 'unique_episode_anchors': len(unique),
            'episodes': len(episodes), 'numeric_groups': len(groups), 'task_exposures': dict(tasks),
            'target_late_exposures': dict(phase), 'target_exposures': dict(target_total),
            'episode_exposure_digest': hashlib.sha256(canonical_json_bytes(
                sorted(episodes.items()))).hexdigest(),
            'group_exposure_digest': hashlib.sha256(canonical_json_bytes(
                sorted(groups.items()))).hexdigest(), 'plan_digest': batches_digest(plan)}


def main(args):
    audit = read_sealed(args.audit / 'sampling-audit.json')
    if sha256_file(args.audit / 'train-episodes.jsonl') != audit['numeric_rows_sha256']:
        raise ValueError('audited numeric rows changed')
    rows = {r['episode_index']: r for r in map(json.loads,
            (args.audit / 'train-episodes.jsonl').read_text().splitlines())}
    index = load_trajectory_v2_split_index(args.data / 'derived/frozen/week2-dataset-v1',
                                          args.source / 'configs/datasets/libero-plus-trajectory-v2')
    reader = DemonstrationReader(args.data / 'datasets/libero-plus-lerobot/snapshots/'
                                 '22c57433fef692b5b9ecc0795344daac7fa867a5/repo')
    normalizer = load_b0_action_normalizer(args.work / 'prepared/action-statistics.json')
    dataset = B0StreamingDataset(reader, index, normalizer, config=B0DataConfig(seed=17, split='train'))
    assert {e.episode_index for e in dataset.episodes} == set(rows)
    recipe = load_fresh_recipe(args.source / 'configs/training/week4_b0_fresh_all_l1_v2.json')
    parent = args.work / 'epochs/epoch-02/B0.ckpt'
    if sha256_file(parent) != PARENT_SHA:
        raise ValueError('parent checkpoint changed')
    control = fresh_plan(dataset, recipe, 2)[:15000]
    treatment = []
    for update, batch in enumerate(control):
        new = []
        for slot, token in enumerate(batch):
            epoch, episode, anchor = token
            if rows[episode]['task_key'] in TARGETS:
                anchor = weighted_anchor(rows[episode]['valid_anchors'], update, slot, episode)
            new.append((epoch, episode, anchor))
        assert [t[1] for t in new] == [t[1] for t in batch]
        treatment.append(new)
    arms = {'C': summarize(control, rows), 'S': summarize(treatment, rows)}
    for key in ('updates', 'samples', 'episodes', 'numeric_groups', 'task_exposures',
                'episode_exposure_digest', 'group_exposure_digest'):
        assert arms['C'][key] == arms['S'][key], key
    assert all(c == s for cb, sb in zip(control, treatment, strict=True)
               for c, s in zip(cb, sb, strict=True) if rows[c[1]]['task_key'] not in TARGETS)
    args.output.mkdir(parents=True, exist_ok=False)
    for arm, plan in (('C', control), ('S', treatment)):
        path = args.output / f'arm-{arm}-batches.jsonl.gz'
        with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as out:
            for batch in plan:
                out.write(canonical_json_bytes(batch) + b'\n')
        arms[arm]['file_sha256'] = sha256_file(path)
    parent_panel = read_sealed(args.source / 'evaluation/splits/week4-clean-comparison-v1/paired-2000.json')
    development = read_sealed(args.source / 'evaluation/splits/week4-oft-continuation-v1/development-200.json')
    ids = {p['pair_id'] for p in development['pairs']}
    panel = [p for p in parent_panel['pairs'] if p['pair_id'] in ids or
             f"{p['suite']}/task-{p['runtime_task_index']:02d}" in TARGETS]
    assert len(panel) == 335 and len({p['pair_id'] for p in panel}) == 335
    task_counts = Counter(f"{p['suite']}/task-{p['runtime_task_index']:02d}" for p in panel)
    assert all(n == (50 if task in TARGETS else 5) for task, n in task_counts.items())
    sealed_panel = seal_contract({'artifact_type': 'eventground_wam.stage_screen_panel',
            'experiment_id': EXPERIMENT, 'pair_count': 335, 'pairs': panel,
            'parent_panel_digest': parent_panel['digest'], 'development_panel_digest': development['digest'],
            'selection': 'All 50 existing states for each of three target tasks; existing five '
                         'development states for each of the other 37 tasks',
            'interpretation': 'Previously observed development states; not independent held-out confirmation'})
    atomic_json(args.output / 'development-335.json', sealed_panel)
    manifest = seal_contract({
        'artifact_type': 'eventground_wam.stage_screen_plan', 'experiment_id': EXPERIMENT,
        'status': 'design_and_sample_plans_frozen_not_launched', 'script_sha256': sha256_file(Path(__file__)),
        'audit_digest': audit['digest'], 'parent_checkpoint_sha256': PARENT_SHA,
        'parent_recipe_digest': recipe['digest'], 'dataset_version': index.dataset_version,
        'seed': 17, 'target_tasks': list(TARGETS), 'arms': arms,
        'C': 'Unchanged fresh_plan epoch index 2, first 15000 updates',
        'S': 'Identical batch sizes and per-slot episode IDs. Target tasks only: redraw valid '
             'anchors with replacement, weight 2 if anchor>=ceil(0.6*N), otherwise 1. '
             'Hash-to-integer mapping defined by weighted_anchor. All other tokens identical.',
        'temporal_proxy_not_semantic_stage_labels': True,
        'optimizer': {'initialization': 'fresh AdamW for both arms from identical parent weights',
                      'learning_rate': 1e-5, 'scheduler': 'constant', 'warmup_updates': 0,
                      'betas': [.9, .95], 'epsilon': 1e-8, 'weight_decay': .01,
                      'maximum_gradient_norm': 1.0},
        'fixed': {'backbone': 'Qwen3.5-2B', 'objective': 'all-action-l1-binary-gripper-v2',
                  'architecture': 'parent unchanged', 'normalizers': 'parent train-v2 unchanged',
                  'augmentation': 'none', 'predict_actions': 8, 'execute_actions': 8,
                  'control_hz': 20, 'simulator_limits': 'frozen parent protocol'},
        'measurement': {'checkpoint_selection': 'fixed update 15000 in each arm',
                        'validation': 'full v2 validation once per endpoint, micro and hierarchical; '
                                      'also early/late temporal bins, not for selecting checkpoints',
                        'rollout_panel_digest': sealed_panel['digest'], 'rollouts_per_arm': 335,
                        'additional_formal_eval': 'Not launched by this plan; eligible candidate '
                                                 'requires clean-2000 and dev-210 with original acceptance rules'},
        'engineering_screen': {
            'interpretation': 'Prospective resource-allocation screen, not a new Week 4 acceptance rule',
            'target_gain_vs_both_C_and_parent_min': 8,
            'target_tasks_improving_vs_C_min': 2, 'max_loss_on_any_target_task_vs_C_or_parent': 2,
            'non_target_185_max_loss_vs_C_or_parent': 3,
            'each_other_suite_max_loss_vs_C_or_parent': 2,
            'otherwise': 'Keep parent; no automatic longer continuation or weight search',
        },
        'reliability': {'host_owned_systemd': True, 'recovery_seconds': 300,
                        'recovery_updates': 2000, 'retained_complete_generations': 3,
                        'required_before_launch': 'Separate child identity/source freeze; correct parent reload; '
                        'train-only plan and exact resume verification; bounded smoke/pilot; never overwrite parent',
                        'training_active_hours_per_arm_cap': 10,
                        'validation_and_screen_active_hours_total_cap': 12,
                        'output_bytes_cap': 107374182400, 'free_bytes_reserve': 53687091200},
        'model_loaded': False, 'optimizer_updates': 0, 'policy_rollouts': 0,
    })
    atomic_json(args.output / 'plan-manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'digest': manifest['digest'], 'arms': arms}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'data', 'work', 'audit', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
