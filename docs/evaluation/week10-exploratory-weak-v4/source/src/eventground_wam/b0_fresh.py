"""Fresh all-L1 v2 plans, grouped validation and strict checkpoint verification."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eventground_wam.b0_continuation import OBJECTIVE, action_metrics
from eventground_wam.b0_data import TRAJECTORY_V2_VERSION, load_b0_action_normalizer
from eventground_wam.b0_model import QWEN_REVISION
from eventground_wam.b0_training import (
    TrainingContractError,
    build_batch_plan,
    capture_rng,
    move_batch,
    read_sealed,
    restore_rng,
    sha256_file,
)
from eventground_wam.continuation_freeze import mix_batches

RECIPE_ID = 'week4-fresh-all-l1-v2'
MANIFEST_TYPE = 'eventground_wam.b0_fresh_v2_training_manifest'


def load_fresh_recipe(path: Path) -> dict[str, Any]:
    recipe = read_sealed(path)
    expected = {'recipe_id': RECIPE_ID, 'epochs': 2, 'training_sample_count': 1706022,
                    'steps_per_epoch': 58566, 'maximum_updates': 117132, 'dataset_version': TRAJECTORY_V2_VERSION,
                    'objective': OBJECTIVE, 'qwen_revision': QWEN_REVISION, 'batch_size': 32, 'lora_rank': 32,
                    'lora_alpha': 16, 'policy_family': 'qwen35-oft-b0-v1', 'mix_window_updates': 128,
                    'checkpoint_selection': 'fixed-epoch-2-endpoint', 'augmentation': 'none',
                    'deterministic_algorithms': True, 'drop_last': False}
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise TrainingContractError('fresh recipe differs from the declared experiment')
    stats = recipe['state_normalization']
    from eventground_wam.demonstration_contracts import contract_digest
    if (stats['digest'] != contract_digest(stats) or stats['split'] != 'train'
            or stats['frame_count'] != 1786067 or stats['episode_count'] != 11435
            or stats['source']['dataset_version'] != TRAJECTORY_V2_VERSION):
        raise TrainingContractError('fresh state statistics are not v2 train-only')
    return recipe


def fresh_plan(dataset: Any, recipe: Any, epoch: int):
    blocked = build_batch_plan(dataset.ordered_episodes(epoch), batch_size=recipe['batch_size'],
                               seed=recipe['seed'], epoch=epoch, shuffle=True)
    tokens = [[(epoch, episode, anchor) for anchor in anchors] for episode, anchors in blocked]
    # Same 128-update windows and 15k stage fences as Run B, restarted per epoch.
    return mix_batches(tokens)


def aggregate_validation(episode_metrics, memberships):
    """Equal chunks -> variants -> numeric groups -> tasks; retain micro totals."""
    groups = defaultdict(list)
    totals = defaultdict(float)
    count = 0
    for episode, entry in episode_metrics.items():
        n = entry['sample_count']
        task, group = memberships[episode]
        mean = {k: v / n for k, v in entry['sums'].items()}
        groups[(task, group)].append(mean)
        for key, value in entry['sums'].items():
            totals[key] += value
        count += n
    tasks = defaultdict(list)
    for (task, _), variants in groups.items():
        tasks[task].append({k: float(np.mean([v[k] for v in variants])) for k in totals})
    task_means = {task: {k: float(np.mean([v[k] for v in values])) for k in totals}
                  for task, values in tasks.items()}
    return {'sample_count': count, 'episode_count': len(episode_metrics), 'group_count': len(groups),
            'task_count': len(tasks), 'micro': {k: v / count for k, v in totals.items()},
            'task_macro': {k: float(np.mean([v[k] for v in task_means.values()])) for k in totals},
            'per_task': task_means, 'episodes_per_group': {f'{t}|{g}': len(v) for (t, g), v in groups.items()},
            'weighting': 'equal-task/group/variant/chunk-hierarchy', 'test_partition_used': False}


def grouped_validation(policy, loader, plan, memberships, device):
    rng = capture_rng()
    modes = {name: module.training for name, module in policy.named_modules()}
    episodes = {}
    try:
        policy.eval()
        with torch.no_grad():
            for batch in loader:
                episode, anchors = plan[batch['index']]
                inputs, state, targets = move_batch(batch, device)
                result = policy(inputs, state, normalized_action_targets=targets)
                if not torch.isfinite(result.loss):
                    raise TrainingContractError('nonfinite fresh validation')
                metrics = action_metrics(result, targets)
                entry = episodes.setdefault(episode, {'sample_count': 0, 'sums': defaultdict(float)})
                n = len(anchors)
                if n != batch['sample_count']:
                    raise TrainingContractError('validation batch coverage mismatch')
                entry['sample_count'] += n
                for key, value in metrics.items():
                    if key not in {'raw_gripper_min', 'raw_gripper_max'}:
                        entry['sums'][key] += n * value
    finally:
        restore_rng(rng)
        for name, module in policy.named_modules():
            module.training = modes[name]
    return aggregate_validation(episodes, memberships)


def verify_fresh_bundle(root: Path, normalization: Path, *, allow_pilot=False):
    manifest = read_sealed(root / 'training-manifest.json')
    report = read_sealed(root / 'training-report.json')
    recipe = manifest['recipe']
    epoch = report['selected_epoch']
    pilot = manifest['mode'] == 'pilot'
    if (manifest['artifact_type'] != MANIFEST_TYPE or recipe['recipe_id'] != RECIPE_ID
            or recipe['objective'] != OBJECTIVE or recipe['qwen_revision'] != QWEN_REVISION
            or report['identity_digest'] != manifest['digest'] or report['status'] != 'complete'
            or (pilot and not allow_pilot) or epoch not in ({0} if pilot else {1, 2})
            or report['global_step'] != (128 if pilot else epoch * 58566)
            or (not pilot and report['samples_seen'] != epoch * 1706022)
            or report['reload_validation_passed'] is not True):
        raise TrainingContractError('invalid fresh checkpoint bundle')
    if sha256_file(root / 'B0.ckpt') != report['checkpoint_sha256']:
        raise TrainingContractError('fresh checkpoint bytes changed')
    normalizer = load_b0_action_normalizer(normalization)
    if (sha256_file(normalization) != manifest['source']['normalization_file_sha256']
            or normalizer.source_identity['dataset_version'] != TRAJECTORY_V2_VERSION):
        raise TrainingContractError('fresh normalizer identity changed')
    return manifest, report
