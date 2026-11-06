#!/usr/bin/env python3
"""Plot an immutable read-only Lambda M2 snapshot using the frozen loss parser."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_auxiliary_losses import read_progress, trailing_mean

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    output = args.snapshot.parent
    snapshot = json.loads(args.snapshot.read_text())
    recipe = read_sealed(Path(__file__).resolve().parents[1] / 'configs/training/week7_m2_auxiliary_v1.json')
    log = output / 'service.log'
    log.write_text(snapshot['log_text'])
    rows, duplicates = read_progress(log, recipe['loss']['outer_future_weight'])
    last, live = rows[-1], snapshot['status']
    assert last['identity_digest'] == live['identity_digest'] == snapshot['training_manifest']['digest']
    assert snapshot['training_manifest']['recipe_digest'] == recipe['digest']
    assert snapshot['checkpoint_checksum_valid']
    steps = np.array([r['global_step'] for r in rows])
    zone = ZoneInfo('America/Los_Angeles')
    as_of = datetime.fromtimestamp(last['time_unix'], zone)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    trends = {}
    for ax, field, label, color in zip(axes,
            ('action_loss', 'future_loss', 'joint_loss'),
            ('Action L1', 'JEPA auxiliary: cosine + 10 × variance penalty',
             f"Joint: action + {recipe['loss']['outer_future_weight']:.4f} × JEPA auxiliary"),
            ('#2463b3', '#087f78', '#9253a1'), strict=True):
        values = np.array([r[field] for r in rows])
        smooth = trailing_mean(steps, values, 500)
        trends[field] = float(smooth[-1])
        ax.plot(steps, values, color=color, alpha=.2, linewidth=.8)
        ax.plot(steps, smooth, color=color, linewidth=2)
        ax.set_title(f'{label}  |  latest trend {smooth[-1]:.4f}', loc='left', fontsize=11)
        ax.set(ylabel='Loss', ylim=(0, max(values)*1.08), xlim=(0, steps[-1]*1.01))
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Optimizer update')
    fig.suptitle(f"M2 · Lambda A10 · {steps[-1]:,}/{recipe['total_updates']:,} updates "
                 f"({100*steps[-1]/recipe['total_updates']:.1f}%)\n{as_of:%b %d, %Y · %H:%M %Z}", fontsize=15)
    fig.text(.5, .015, 'Faint: logged batch. Solid: trailing 500-update mean of sparse logged points.\n'
             'Training losses only; rollout quality and representation quality remain unverified.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(output / f'loss-curves.{ext}', dpi=180)
    plt.close(fig)
    remaining = (recipe['total_updates']-live['global_step'])*live['active_seconds']/live['global_step']/3600
    rolling = [r for r in rows if r['global_step'] >= steps[-1]-500]
    seconds_per_update = (rolling[-1]['time_unix']-rolling[0]['time_unix'])/(rolling[-1]['global_step']-rolling[0]['global_step'])
    summary = seal_contract({'artifact_type':'eventground_wam.m2_progress_check',
        'as_of_local':datetime.fromtimestamp(snapshot['captured_at_unix'],zone).isoformat(),
        'live_step':live['global_step'], 'plot_last_step':int(steps[-1]),
        'total_updates':recipe['total_updates'], 'percent_complete':100*live['global_step']/recipe['total_updates'],
        'latest_trailing_500_update_means':trends,'finite_logged_points':len(rows),
        'duplicate_logged_updates':duplicates,'checkpoint_checksum_valid':True,
        'checkpoint_step':snapshot['latest_checkpoint']['global_step'],
        'checkpoint_age_seconds':snapshot['captured_at_unix']-snapshot['checkpoint_modified_unix'],
        'active_hours':live['active_seconds']/3600,'remaining_hours_cumulative':remaining,
        'estimated_finish_local':(datetime.fromtimestamp(live['time_unix'],zone)+timedelta(hours=remaining)).isoformat(),
        'recent_seconds_per_update':seconds_per_update,
        'remaining_hours_recent':(recipe['total_updates']-live['global_step'])*seconds_per_update/3600,
        'service':snapshot['service'],'gpu':snapshot['gpu'],
        'limitations':'Throughput extrapolation, not a deadline guarantee. Original JEPA quality gate remains FAIL under research-use exception; finite losses do not establish latent quality or downstream success.',
        'source_sha256':{'snapshot':sha256_file(args.snapshot),'plot_script':sha256_file(Path(__file__)),
                         'loss_parser':sha256_file(Path(__file__).with_name('plot_auxiliary_losses.py'))},
        'outputs':{f'loss-curves.{ext}':sha256_file(output/f'loss-curves.{ext}') for ext in ('png','svg')}})
    atomic_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
