"""Descriptive rollout diagnostics; motion flags do not establish failure causes."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import h5py
import numpy as np


def success_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {'count': 0, 'successes': 0, 'success_rate': None, 'wilson_95': None}
    k = sum(row['success'] for row in rows)
    p, z = k / n, 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {'count': n, 'successes': k, 'failures': n - k, 'success_rate': p,
            'wilson_95': [max(0.0, center - half), min(1.0, center + half)],
            'terminations': dict(Counter(row['termination_reason'] for row in rows))}


def grouped_success(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[' | '.join(str(row[field]) for field in fields)].append(row)
    return {key: success_summary(value) for key, value in sorted(groups.items())}


def motion_metrics(actions: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    """Actions in simulator units and recorded pre-action proprioception only."""
    actions = np.asarray(actions, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or state.shape != (len(actions), 8):
        raise ValueError('expected aligned [T,7] actions and [T,8] proprioception')
    if not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError('non-finite rollout diagnostics')
    if not len(actions):
        return {'policy_steps': 0, 'late_low_motion': None, 'late_low_motion_with_command': None}
    tail = state[-min(40, len(state)):]
    tail_actions = actions[-len(tail):]
    tail_span = float(np.linalg.norm(np.ptp(tail[:, :3], axis=0)))
    tail_path = float(np.linalg.norm(np.diff(tail[:, :3], axis=0), axis=1).sum())
    tail_command = float(np.linalg.norm(tail_actions[:, :3], axis=1).mean())
    close = actions[:, 6] > 0
    closed = np.flatnonzero(close)
    continuous = actions[:, :6]
    return {
        'policy_steps': len(actions), 'gripper_close_command_fraction': float(close.mean()),
        'gripper_command_switches': int(np.count_nonzero(np.diff(close.astype(int)))),
        'first_close_command_policy_step': int(closed[0]) if len(closed) else None,
        'continuous_saturation_fraction': float((np.abs(continuous) >= 0.99999).mean()),
        'translation_command_norm_mean': float(np.linalg.norm(actions[:, :3], axis=1).mean()),
        'rotation_command_norm_mean': float(np.linalg.norm(actions[:, 3:6], axis=1).mean()),
        'eef_path_length_m': float(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).sum()),
        'eef_net_displacement_m': float(np.linalg.norm(state[-1, :3] - state[0, :3])),
        'gripper_width_min_m': float(np.abs(state[:, 6:8]).sum(axis=1).min()),
        'gripper_width_max_m': float(np.abs(state[:, 6:8]).sum(axis=1).max()),
        'tail_window_steps': len(tail), 'tail_eef_span_m': tail_span,
        'tail_eef_path_length_m': tail_path,
        'tail_eef_net_displacement_m': float(np.linalg.norm(tail[-1, :3] - tail[0, :3])),
        'tail_translation_command_norm_mean': tail_command,
        'late_low_motion': bool(len(tail) == 40 and tail_span < 0.01),
        'late_low_motion_with_command': bool(len(tail) == 40 and tail_span < 0.01
                                             and tail_command >= 0.1),
    }


def inspect_trajectory(path: Any) -> dict[str, Any]:
    with h5py.File(path, 'r') as f:
        policy = f['steps/phase'][:] == 1
        actions = f['steps/executed_action'][:][policy]
        state = f['steps/proprioception'][:][policy]
        latency = f['steps/policy_latency_ms'][:]
        latency = latency[np.isfinite(latency)]
    metrics = motion_metrics(actions, state)
    metrics['policy_latency_median_ms'] = float(np.median(latency)) if len(latency) else None
    metrics['policy_latency_p95_ms'] = float(np.quantile(latency, 0.95)) if len(latency) else None
    metrics['policy_query_count'] = len(latency)
    return metrics


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ('continuous_saturation_fraction', 'gripper_close_command_fraction',
            'gripper_command_switches', 'translation_command_norm_mean',
            'tail_eef_span_m', 'tail_eef_path_length_m', 'policy_latency_median_ms')
    result = {'episode_count': len(rows)}
    for key in keys:
        values = [row['motion'][key] for row in rows if row['motion'].get(key) is not None]
        result[key] = {'median': float(np.median(values)), 'mean': float(np.mean(values))} if values else None
    for key in ('late_low_motion', 'late_low_motion_with_command'):
        result[key] = sum(row['motion'].get(key) is True for row in rows)
    return result
