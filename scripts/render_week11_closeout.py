"""Render the sealed paired effects and deterministically selected recorded cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json, utc_now
from eventground_wam.demonstration_contracts import seal_contract


def render(root, output=None):
    started = time.monotonic()
    analysis = read_sealed(root/'analysis.json')
    output = output or root
    output.mkdir(parents=True, exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    contrasts = ['M4-EG-adaptive minus M4-fixed', 'M4-EG-adaptive minus M4-EG-fixed8', 'M4-EG-fixed8 minus M4-fixed']
    labels = ['Adaptive − fixed', 'Adaptive − event/fixed8', 'Event/fixed8 − fixed']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, panel, title in zip(axes, ['targeted-467', 'clean-200'], ['Targeted · 467 pairs', 'Clean · 40 tasks × 5 states'], strict=True):
        for i, contrast in enumerate(contrasts):
            row = analysis['statistics'][panel][contrast]
            point, (lo, hi) = row['difference_pp'], row['paired_95_ci_pp']
            color = '#b84040' if point < 0 else '#245c80'
            ax.errorbar(point, 2-i, xerr=[[point-lo], [hi-point]], fmt='o', capsize=5, color=color, lw=2, markersize=7)
            ax.annotate(f'{point:+.2f} [{lo:+.2f}, {hi:+.2f}]', (point, 2-i),
                        xytext=(0, 14), textcoords='offset points', ha='center', fontsize=9, color=color)
        ax.axvline(0, color='#6b7280', lw=1)
        ax.set_yticks([2, 1, 0], labels)
        ax.set_ylim(-.5, 2.6)
        ax.set_xlim(-15, 5)
        ax.set_xlabel('Success difference (percentage points)')
        ax.set_title(title, fontsize=12, pad=14)
        ax.grid(axis='x', color='#e5e7eb')
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('Week 11: adaptive horizon regresses on both panels', fontsize=14, y=.98)
    fig.text(.5, .015, 'Frozen 10,000-draw paired bootstrap · clean resamples whole tasks · conditional panel uncertainty',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .06, 1, .92])
    fig.savefig(figures/'paired-effects.png', dpi=180)
    fig.savefig(figures/'paired-effects.svg')
    plt.close(fig)
    cases = read_sealed(root/'selected-case-index.json')
    snapshots = []
    for panel, entries in cases['panels'].items():
        for category, case in entries.items():
            fig, axes = plt.subplots(3, 3, figsize=(12, 8.6))
            records = []
            for row, (arm, path) in enumerate(case['artifacts'].items()):
                with h5py.File(path) as h:
                    n = len(h['steps/step_index'])
                    indices = [10, max(10, n//2), n-1]
                    for col, index in enumerate(indices):
                        frame = np.concatenate([h['steps/agent_rgb'][index], h['steps/wrist_rgb'][index]], axis=1)
                        axes[row, col].imshow(frame)
                        axes[row, col].axis('off')
                        axes[row, col].set_title(f'{arm} · recorded step {index}', fontsize=9)
                records.append({'arm': arm, 'artifact': path, 'recorded_steps': indices})
            fig.suptitle(f'{panel} · pair {case["ordinal"]} · {category.replace("_", " ")}', fontsize=13)
            fig.text(.5, .018, 'First query / midpoint / last recorded pre-action observation · agent camera | wrist camera',
                     ha='center', fontsize=9)
            fig.subplots_adjust(left=.015, right=.985, bottom=.07, top=.90, hspace=.32, wspace=.05)
            name = f'{panel}-{category}.png'
            fig.savefig(figures/name, dpi=140)
            plt.close(fig)
            snapshots.append({'panel': panel, 'outcome_class': category, 'ordinal': case['ordinal'],
                              'file': name, 'records': records})
    receipt = seal_contract({'analysis_digest': analysis['digest'], 'created_at': utc_now(),
        'script_sha256': sha256_file(Path(__file__)), 'render_seconds': time.monotonic()-started,
        'selected_case_index_digest': cases['digest'], 'snapshots': snapshots,
        'files_sha256': {p.name: sha256_file(p) for p in sorted(figures.iterdir()) if p.is_file()},
        'causal_interpretation': 'recorded examples only; no new event labels or failure-cause claims'})
    atomic_json(output/'presentation.json', receipt)
    print(json.dumps({'files': len(receipt['files_sha256']), 'render_seconds': receipt['render_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    render(args.root, args.output)
