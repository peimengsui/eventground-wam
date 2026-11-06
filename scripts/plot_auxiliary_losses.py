#!/usr/bin/env python3
"""Plot logged Week 7 training losses without modifying a running experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


def read_progress(path, weight):
    by_step = {}
    identities = set()
    duplicates = 0
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A copied live log can end with an incomplete final line.
        if row.get('status') != 'running':
            continue
        if any(not math.isfinite(row[field]) for field in
               ('action_loss', 'future_loss', 'joint_loss')):
            raise ValueError('nonfinite logged loss')
        if not math.isclose(row['joint_loss'], row['action_loss']+weight*row['future_loss'],
                            rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError('logged objective disagrees with the frozen loss weight')
        identities.add(row['identity_digest'])
        step = row['global_step']
        duplicates += step in by_step
        by_step[step] = row
    if len(identities) != 1 or not by_step:
        raise ValueError('expected one nonempty training identity')
    return [by_step[step] for step in sorted(by_step)], duplicates


def trailing_mean(steps, values, window):
    start = np.searchsorted(steps, steps-window+1, side='left')
    sums = np.r_[0., np.cumsum(values)]
    end = np.arange(1, len(values)+1)
    return (sums[end]-sums[start])/(end-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=500)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError('smoothing window must be positive')
    recipe = json.loads(args.recipe.read_text())
    weight = recipe['loss']['outer_future_weight']
    rows, duplicates = read_progress(args.log, weight)
    steps = np.array([row['global_step'] for row in rows])
    values = {key: np.array([row[key] for row in rows]) for key in
              ('action_loss', 'future_loss', 'joint_loss')}
    smoothed = {key: trailing_mean(steps, value, args.window) for key, value in values.items()}
    latest = rows[-1]
    timestamp = datetime.fromtimestamp(latest['time_unix'], ZoneInfo('America/Los_Angeles'))
    warmup = int(recipe['total_updates']*recipe['optimizer']['warmup_fraction'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.labelcolor': '#364152', 'text.color': '#182333', 'xtick.color': '#526174',
        'ytick.color': '#526174', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#D8DFE7', 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(12.5, 8.2), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.08, right=.97, bottom=.20, top=.82,
                            hspace=.63, wspace=.24)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    panels = [
        ('action_loss', 'Action prediction', 'Action L1', '#2463B3'),
        ('future_loss', 'Pixel auxiliary prediction', 'L1 + 0.1 × SSIM penalty', '#087F78'),
        ('joint_loss', 'Combined training objective',
         f'Action + {weight:.4f} × pixel auxiliary', '#9253A1'),
    ]
    for ax, (key, title, subtitle, color) in zip(axes, panels, strict=True):
        ax.plot(steps, values[key], color=color, alpha=.23, linewidth=.95)
        ax.plot(steps, smoothed[key], color=color, linewidth=2.3)
        ax.scatter([steps[-1]], [smoothed[key][-1]], color=color, s=25, zorder=4)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=13, pad=22)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=9.5, color='#647184')
        ax.text(.985, .94, f'Latest trend: {smoothed[key][-1]:.4f}', transform=ax.transAxes,
                ha='right', va='top', color=color, fontsize=10,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 3})
        if steps[0] <= warmup <= steps[-1]:
            ax.axvline(warmup, color='#89929D', linewidth=.85, linestyle=(0, (4, 4)))
        ax.set(xlim=(0, steps[-1]*1.025), ylim=(0, max(values[key])*1.08),
               xlabel='Optimizer update', ylabel='Loss')
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y', color='#E7EBF0', linewidth=.75)
        ax.set_axisbelow(True)
    fig.text(.08, .945, f"{recipe['arm']} · training loss curves", fontsize=22, fontweight='bold')
    fig.text(.08, .905,
        f"Through update {steps[-1]:,} / {recipe['total_updates']:,} "
        f"({100*steps[-1]/recipe['total_updates']:.1f}%)  ·  Epoch {latest['epoch']} of {recipe['epochs']}",
        fontsize=12, color='#526174')
    fig.text(.97, .95, timestamp.strftime('%b %d, %Y · %H:%M %Z'), ha='right', fontsize=10,
             color='#647184')
    handles = [Line2D([0], [0], color='#526174', alpha=.3, linewidth=1, label='Logged batch'),
               Line2D([0], [0], color='#526174', linewidth=2.3,
                      label=f'Trailing {args.window:,}-update mean of logged points'),
               Line2D([0], [0], color='#89929D', linestyle=(0, (4, 4)),
                      label=f'Warmup ends ({warmup:,})')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.073, .075),
               frameon=False, ncol=3, fontsize=9.5, handlelength=2.7, columnspacing=2.4)
    fig.text(.08, .047,
        f'{len(rows):,} logged batches; logging is sparse and uneven. Curves show training losses; '
        'no validation or rollout results yet.', fontsize=9, color='#647184')
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'svg'):
        fig.savefig(args.output/f'loss-curves.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    for i, row in enumerate(rows):
        row['smoothed_losses'] = {key: float(smoothed[key][i]) for key in values}
    trace = args.output/'plotted-points.json'
    trace.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    summary = {'artifact_type': 'eventground_wam.auxiliary_loss_plot',
        'recipe_digest': recipe['digest'], 'training_identity_digest': latest['identity_digest'],
        'source_log': str(args.log), 'source_log_sha256': hashlib.sha256(args.log.read_bytes()).hexdigest(),
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'logged_points': len(rows), 'first_step': int(steps[0]), 'last_step': int(steps[-1]),
        'as_of_local': timestamp.isoformat(), 'warmup_updates': warmup,
        'smoothing': f'unweighted arithmetic mean of logged points within trailing {args.window} updates',
        'duplicate_updates_kept_latest': duplicates,
        'loss_weight': weight, 'latest_batch_losses': {key: float(values[key][-1]) for key in values},
        'latest_smoothed_losses': {key: float(smoothed[key][-1]) for key in values},
        'first_500_update_logged_mean': {key: float(values[key][steps <= 500].mean()) for key in values},
        'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}}
    (args.output/'plot-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
