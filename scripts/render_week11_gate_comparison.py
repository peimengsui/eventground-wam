"""Static scientific comparison figure from audited diagnostic outcomes."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.week11_closing_gate import ADAPTIVE, FIXED, GATED

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def run(base, output):
    summary = read_sealed(base/'summary.json')
    output.mkdir(parents=True, exist_ok=False)
    rows = summary['rows']
    arms, labels = [FIXED, ADAPTIVE, GATED], ['Fixed8', 'Original adaptive', 'Adaptive after closing']
    colors = ['#455b68', '#c2764b', '#39847a']
    figure = plt.figure(figsize=(11.8, 11.5), layout='constrained')
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 3.8], width_ratios=[1, 1])
    ax = figure.add_subplot(grid[0, :])
    counts = [summary['groups']['all']['successes'][a] for a in arms]
    ax.barh(np.arange(3), counts, color=colors, height=.62)
    ax.set_yticks(np.arange(3), labels)
    ax.invert_yaxis(); ax.set_xlim(0, 29); ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel('Successful episodes out of 27 selected cases')
    for i, count in enumerate(counts):
        ax.text(count+.3, i, f'{count}/27', va='center', weight='bold')
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    for column, panel in enumerate(('targeted-467', 'clean-200')):
        selected = [r for r in rows if r['panel'] == panel]
        ax = figure.add_subplot(grid[1, column])
        values = np.array([[int(r['outcomes'][a]) for a in arms] for r in selected])
        ax.imshow(values, cmap=ListedColormap(['#efe3dc', '#438e81']), vmin=0, vmax=1, aspect='auto')
        ax.set_xticks([0, 1, 2], ['Fixed8', 'Adaptive', 'After close'])
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(selected)), [f"{r['ordinal']:03d} · "+
            ('fixture' if r['task_group'] == 'fixture_involving' else 'object') for r in selected])
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                ax.text(x, y, '✓' if value else '×', ha='center', va='center',
                        color='white' if value else '#906650', fontsize=14, weight='bold')
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(selected), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=3)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_title(f'{len(selected)} '+('targeted' if column == 0 else 'clean')+' cases', pad=34)
        for spine in ax.spines.values():
            spine.set_visible(False)
    contrast = summary['groups']['all']['contrasts'][0]
    figure.suptitle('Does delaying adaptive routing until after closing help?\n'
        f"Gated vs original: {contrast['left_only']} wins, {contrast['right_only']} losses on paired cases", fontsize=17)
    figure.supxlabel('81 fresh rollouts on a previously selected, outcome-stratified panel. '
                        'Green = success. These counts do not estimate population performance.', fontsize=10)
    for extension in ('png', 'svg'):
        figure.savefig(output/f'paired-outcomes.{extension}', dpi=150)
    plt.close(figure)
    atomic_json(output/'receipt.json', seal_contract({'summary_digest': summary['digest'],
        'source_sha256': sha256_file(Path(__file__)),
        'files_sha256': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
