"""Plot gripper score crossings from the already completed replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from week11_replay_common import read, sha, write

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(base, output):
    s = read(base/'summary.json'); rows = s['closing_cases']
    output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(11.8, 5.5), layout='constrained')
    for i, r in enumerate(rows):
        a, b = r['h8_raw_score'], r['h4_raw_score']
        ax.plot([a, b], [i, i], color='#aeb7c1', lw=3)
        ax.scatter(a, i, color='#2f4858', s=70, label='Horizon 8' if i == 0 else None, zorder=3)
        ax.scatter(b, i, color='#bf6138', marker='D', s=65, label='Horizon 4 (recorded)' if i == 0 else None, zorder=3)
        ax.text(1.05, i, f'Close on action {r["h8_close_action"]} → {r["h4_close_action"]}', va='center', fontsize=10)
    ax.axvline(0, color='#444444', ls=':', lw=1.5)
    ax.set_yticks(np.arange(len(rows)), [r['id'].replace('close-', '')+' · '+r['outcome_class'].replace('_', ' ') for r in rows])
    ax.invert_yaxis(); ax.set_xlim(-1.1, 1.85); ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xlabel('Raw gripper score at the changed action: negative = open; zero or positive = close')
    ax.grid(axis='x', alpha=.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.13), frameon=False, ncol=2)
    fig.suptitle('Horizon changes move the gripper decision across zero\nAll six shift closing by one action (50 ms): three earlier, three later', fontsize=13)
    fig.supxlabel('Identical inputs; original replay tensors. Scores are regression outputs, not calibrated probabilities. Outcomes describe entire recorded episodes.', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(output/f'gripper-score-crossings.{ext}', dpi=160)
    plt.close(fig)
    write(output/'receipt.json', {'summary_digest': s['digest'], 'script_sha256': sha(__file__),
        'files_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    run(a.base, a.output)
