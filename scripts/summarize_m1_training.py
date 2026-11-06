#!/usr/bin/env python3
"""Summarize the completed M1 training ledger and plot its observed training losses."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_auxiliary_losses import read_progress, trailing_mean

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--host-accounting', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe = read_sealed(ROOT / 'configs/training/week7_m1_auxiliary_v1.json')
    training = read_sealed(args.training_root / 'training-manifest.json')
    completion = read_sealed(args.training_root / 'training-report.json')
    budget = read_sealed(args.training_root / 'active-budget.json')
    assert completion['identity_digest'] == training['digest'] and completion['recipe_digest'] == recipe['digest']
    assert completion['global_step'] == 117132 and not budget['running']
    rows, duplicates = read_progress(args.training_root / 'service.log', recipe['loss']['outer_future_weight'])
    assert rows[-1]['identity_digest'] == training['digest']
    events = []
    for line in (args.training_root / 'service.log').read_text().splitlines():
        if line.startswith('{'):
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    accounting = json.loads(args.host_accounting.read_text())
    service = dict(line.split('=', 1) for line in
                   accounting['services']['eventground-m1-auxiliary-v2.service'].splitlines())
    cpt_path = ROOT / 'docs/evaluation/week5-pixel-cpt-final-v1/training-report.json'
    cpt = read_sealed(cpt_path)
    times = np.array([row['global_step'] for row in rows])
    fields = ['action_loss', 'future_loss', 'joint_loss']
    values = {field: np.array([row[field] for row in rows]) for field in fields}
    trends = {field: trailing_mean(times, values[field], 500) for field in fields}
    start = datetime.fromtimestamp(events[0]['time_unix'], ZoneInfo('America/Los_Angeles'))
    end = datetime.fromtimestamp(events[-1]['time_unix'], ZoneInfo('America/Los_Angeles'))
    summary = seal_contract({
        'artifact_type': 'eventground_wam.week7_m1_final_training_summary',
        'training_identity_digest': training['digest'], 'completion_digest': completion['digest'],
        'recipe_digest': recipe['digest'], 'checkpoint_sha256': completion['checkpoint_sha256'],
        'epochs': 2, 'updates': completion['global_step'], 'action_presentations': completion['samples_seen'],
        'future_pairs': completion['future_pairs_seen'], 'first_model_loading_event_local': start.isoformat(),
        'completion_local': end.isoformat(), 'active_training_hours': budget['active_seconds'] / 3600,
        'training_service_result': service['Result'], 'training_service_restarts': int(service['NRestarts']),
        'production_host_cgroup_peak_bytes': int(service['MemoryPeak']),
        'production_gpu_continuous_peak': None,
        'memory_qualification': 'Host peak is service cgroup accounting. Continuous production GPU peak was not recorded; GPU figures below are qualification-pilot measurements.',
        'pilot_gpu_peak_allocated_bytes': recipe['resources']['gpu_peak_allocated_bytes'],
        'pilot_gpu_peak_reserved_bytes': recipe['resources']['gpu_peak_reserved_bytes'],
        'pilot_device_used_mib': recipe['resources']['four_update_peak_device_used_mib'],
        'shared_trainable_parameters': recipe['shared_trainable_parameters'],
        'training_only_predictor_parameters': recipe['future_trainable_parameters'],
        'total_trainable_parameters': recipe['shared_trainable_parameters'] + recipe['future_trainable_parameters'],
        'inference_predictor_parameters': 0,
        'cpt': {'module': 'C_pixel-v1', 'pairs': cpt['pairs_seen'], 'updates': cpt['global_step'],
                'active_hours': cpt['active_seconds_all_processes'] / 3600, 'report_digest': cpt['digest']},
        'production_plus_cpt_hours_excluding_pilots': (budget['active_seconds'] + cpt['active_seconds_all_processes']) / 3600,
        'pilot_and_tuning_exposure': {'accepted_attempt': 4, 'accepted_overfit_updates': 64,
            'accepted_timing_updates': 128, 'edge_batches': 3, 'calibration_optimizer_updates': 0,
            'failed_attempt_1_updates': 0, 'failed_attempt_2_updates': 0,
            'failed_attempt_3_overfit_updates': 64, 'failed_attempt_3_timing_updates': 128,
            'additional_recovery_precision_and_CPT_development_work': 'Preserved in supporting-evidence; excluded from the production+CPT hour sum. Full historical tuning wall time is not consolidated.',
            'pilot_weights_loaded_into_production': False},
        'losses': {'logged_points': len(rows), 'duplicate_steps': duplicates, 'last_logged_step': int(times[-1]),
            'last_logged_batch': {field: float(values[field][-1]) for field in fields},
            'last_500_update_logged_mean': {field: float(trends[field][-1]) for field in fields},
            'first_500_update_logged_mean': {field: float(values[field][times <= 500].mean()) for field in fields},
            'scope': 'Sparse training-batch samples; no interpolation of final unlogged updates and no held-out validation claim.'},
        'source_sha256': {'training_log': sha256_file(args.training_root / 'service.log'),
                          'host_accounting': sha256_file(args.host_accounting), 'cpt_report': sha256_file(cpt_path),
                          'summary_script': sha256_file(Path(__file__))}})
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'training-summary.json', summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor='white')
    for ax, field, title, color in zip(axes, fields, ['Action L1', 'Pixel L1 + 0.1 SSIM', 'Combined objective'],
                                      ['#2463B3', '#087F78', '#9253A1'], strict=True):
        ax.plot(times, values[field], color=color, alpha=.2, linewidth=.5)
        ax.plot(times, trends[field], color=color, linewidth=1.8)
        ax.axvline(58566, color='#777777', linestyle='--', linewidth=.7)
        ax.set(title=title, xlabel='Optimizer update', ylabel='Training loss', xlim=(0, 117132), ylim=(0, None))
        ax.grid(axis='y', alpha=.2)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
    fig.suptitle('M1 completed two-epoch training · 117,132 updates', fontweight='bold')
    fig.text(.06, .02, f'{len(rows):,} sparse logged batches; solid line: mean of logged points in trailing 500 updates. '
             f'Last logged update: {times[-1]:,}. Dashed line: epoch boundary.', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .92))
    for suffix in ('png', 'svg'):
        fig.savefig(args.output / f'training-losses.{suffix}', dpi=170)
    plt.close(fig)
    print(json.dumps({'active_training_hours': summary['active_training_hours'],
                      'completion_local': summary['completion_local'], 'losses': summary['losses']}))


if __name__ == '__main__':
    main()
