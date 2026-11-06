#!/usr/bin/env python3
"""Plot read-only M1/M2 log snapshots on separate optimizer-update axes."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from plot_auxiliary_losses import read_progress, trailing_mean

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo('America/Los_Angeles')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--window', type=int, default=500)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError('positive smoothing window required')
    output = args.snapshots/'charts'
    output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#D8DFE7',
        'text.color': '#182333', 'axes.labelcolor': '#364152', 'xtick.color': '#526174',
        'ytick.color': '#526174', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.4), facecolor='white')
    fig.subplots_adjust(left=.075, right=.97, bottom=.16, top=.80, hspace=.65, wspace=.25)
    summaries = {}
    for col, arm in enumerate(('M1', 'M2')):
        snapshot_path = args.snapshots/arm/'snapshot.json'
        snapshot = json.loads(snapshot_path.read_text())
        recipe = json.loads((ROOT/f'configs/training/week7_{arm.lower()}_auxiliary_v1.json').read_text())
        log = args.snapshots/arm/'service.log'
        log.write_text(snapshot['log_text'])
        weight = recipe['loss']['outer_future_weight']
        rows, duplicates = read_progress(log, weight)
        latest = rows[-1]
        assert latest['identity_digest'] == snapshot['training_manifest']['digest']
        assert snapshot['training_manifest']['recipe_digest'] == recipe['digest']
        assert snapshot['latest_checkpoint']['identity_digest'] == latest['identity_digest']
        assert latest['global_step'] <= snapshot['status']['global_step']
        steps = np.array([r['global_step'] for r in rows])
        fields = ('action_loss', 'future_loss', 'joint_loss')
        values = {key: np.array([r[key] for r in rows]) for key in fields}
        trends = {key: trailing_mean(steps, value, args.window) for key, value in values.items()}
        warmup = int(recipe['total_updates']*recipe['optimizer']['warmup_fraction'])
        method, host = ('Pixel auxiliary', 'Feifei RTX 5090') if arm == 'M1' else ('JEPA auxiliary', 'Lambda A10')
        auxiliary = 'Pixel: L1 + 0.1 × SSIM penalty' if arm == 'M1' else 'JEPA: cosine + 10 × variance penalty'
        colors = ('#2463B3', '#087F78', '#9253A1')
        titles = ('Action prediction', 'Auxiliary prediction', 'Combined training objective')
        labels = ('Action L1', auxiliary, f'Action + {weight:.4f} × auxiliary')
        for row, (key, color, title, label) in enumerate(zip(fields, colors, titles, labels, strict=True)):
            ax = axes[row, col]
            ax.plot(steps, values[key], color=color, alpha=.24, linewidth=.9)
            ax.plot(steps, trends[key], color=color, linewidth=2.1)
            ax.scatter([steps[-1]], [trends[key][-1]], color=color, s=18, zorder=4)
            ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=20)
            ax.text(0, 1.025, label, transform=ax.transAxes, fontsize=9, color='#647184')
            ax.text(.985, .93, f'Trend {trends[key][-1]:.4f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, color=color,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .87, 'pad': 2})
            if steps[0] <= warmup <= steps[-1]:
                ax.axvline(warmup, color='#89929D', linewidth=.85, linestyle=(0, (4, 4)))
            ax.set(xlim=(0, steps[-1]*1.025), ylim=(0, max(values[key])*1.08),
                   xlabel='Optimizer update', ylabel='Loss')
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis='y', color='#E7EBF0', linewidth=.75)
            ax.set_axisbelow(True)
        x = .075 if col == 0 else .573
        fig.text(x, .915, f'{arm} · {method}', fontsize=17, fontweight='bold')
        progress = 100*steps[-1]/recipe['total_updates']
        fig.text(x, .883, f'{host}   |   {steps[-1]:,} / {recipe["total_updates"]:,} updates ({progress:.2f}%)',
                 fontsize=10.5, color='#526174')
        phase = f'Warmup: {steps[-1]:,} / {warmup:,}' if steps[-1] < warmup else 'Warmup complete'
        observed = datetime.fromtimestamp(latest['time_unix'], ZONE)
        fig.text(x, .856, f'Epoch {latest["epoch"]} of 2   |   {phase}   |   {observed:%H:%M:%S %Z}',
                 fontsize=10, color='#647184')
        remaining = recipe['total_updates']-int(steps[-1])
        wall_hours = remaining*latest['active_seconds']/int(steps[-1])/3600
        capture = snapshot['captured_at_unix']
        summaries[arm] = {
            'host': host, 'recipe_digest': recipe['digest'], 'identity_digest': latest['identity_digest'],
            'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            'service_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
            'captured_at_utc': snapshot['captured_at_utc'], 'as_of_local': observed.isoformat(),
            'step': int(steps[-1]), 'total_updates': recipe['total_updates'], 'percent_complete': progress,
            'epoch': latest['epoch'], 'logged_points': len(rows), 'duplicates_kept_latest': duplicates,
            'warmup_updates': warmup, 'service': snapshot['service'], 'gpu': snapshot['gpu'],
            'last_log_age_seconds': capture-latest['time_unix'],
            'latest_checkpoint_step': snapshot['latest_checkpoint']['global_step'],
            'checkpoint_age_seconds': capture-snapshot['checkpoint_modified_unix'],
            'checkpoint_exists': snapshot['checkpoint_exists'],
            'latest_batch_losses': {key: float(values[key][-1]) for key in fields},
            'latest_trailing_mean': {key: float(trends[key][-1]) for key in fields},
            'first_500_update_mean': {key: float(values[key][steps <= 500].mean()) for key in fields},
            'last_window_logged_points': int(sum(steps > steps[-1]-args.window)),
            'active_hours': latest['active_seconds']/3600,
            'remaining_step_time_hours': latest['remaining_mean_training_hours'],
            'remaining_active_time_hours': wall_hours,
            'estimated_finish_local': (observed+timedelta(hours=wall_hours)).isoformat(),
            'eta_method': 'remaining updates × cumulative active seconds / completed updates; includes observed startup/checkpoint overhead; assumes same future throughput',
            'completion': snapshot['completion'],
        }
        for i, item in enumerate(rows):
            item['smoothed_losses'] = {key: float(trends[key][i]) for key in fields}
        (output/f'{arm}-plotted-points.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    captured = datetime.fromisoformat(summaries['M1']['captured_at_utc']).astimezone(ZONE)
    fig.text(.075, .973, 'M1 & M2 · current training loss curves', fontsize=22, fontweight='bold')
    fig.text(.97, .975, f'{captured:%b %d, %Y · %H:%M %Z}', ha='right', fontsize=10, color='#647184')
    handles = [Line2D([0], [0], color='#526174', alpha=.3, linewidth=1, label='Logged batch loss'),
        Line2D([0], [0], color='#526174', linewidth=2.1, label=f'Trailing {args.window:,}-update mean of logged points'),
        Line2D([0], [0], color='#89929D', linestyle=(0, (4, 4)), label='Warmup end, when visible')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.07, .086), ncol=3,
               frameon=False, fontsize=10, columnspacing=2)
    fig.text(.075, .065, 'Each column uses its own update range. Logging is sparse and uneven; smoothing averages the available logged batches.',
             fontsize=9.5, color='#647184')
    fig.text(.075, .043, 'Pixel and JEPA auxiliary losses use different objectives and scales. Training curves do not measure rollout success.',
             fontsize=9.5, color='#647184')
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'loss-curves.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    summary = {'artifact_type': 'eventground_wam.auxiliary_pair_progress', 'arms': summaries,
        'smoothing_window_updates': args.window,
        'plotting_environment': {'python': platform.python_version(),
                                 'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parser_script_sha256': hashlib.sha256((ROOT/'scripts/plot_auxiliary_losses.py').read_bytes()).hexdigest(),
        'output_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(output.iterdir())}}
    (output/'progress-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
