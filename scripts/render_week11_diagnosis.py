"""Render deterministic examples from the completed recording-only diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ('M4-EG-fixed8', 'M4-EG-adaptive')
COLORS = ('#246b91', '#be503b')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def render(root, output, cases_path=None):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    rows = {(r['panel'], r['ordinal']): r for r in map(json.loads, (root/'pairs.jsonl').read_text().splitlines())}
    cases_path = cases_path or root/'selected-cases.json'
    cases = json.loads(cases_path.read_text())['cases']
    inventory = json.loads((root/'inputs.json').read_text())['files']
    selected = []
    for case in cases:
        r = rows[case['panel'], case['ordinal']]
        first = 8*r['divergence']['first_non8_query']
        common_last = min(r['arms'][a]['policy_steps'] for a in ARMS)-1
        fig = plt.figure(figsize=(15, 9.4))
        grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, .95], hspace=.36, wspace=.12)
        horizon_ax = fig.add_subplot(grid[2, :2]); aperture_ax = fig.add_subplot(grid[2, 2:])
        evidence = []
        for ai, arm in enumerate(ARMS):
            path = Path(r['artifacts'][arm])
            relative = str(path).split('/week11-formal-v1/')[1]
            assert sha(path) == inventory[relative]['sha256']
            with h5py.File(path, 'r') as h:
                s = h['steps']; n = len(s['step_index'])
                idxs = [first+10, min(first+32, common_last)+10, common_last+10, n-1]
                for col, (idx, label) in enumerate(zip(idxs,
                        ['First horizon change', '32 frames later', 'Last common frame', 'Own final frame'], strict=True)):
                    ax = fig.add_subplot(grid[ai, col])
                    ax.imshow(np.concatenate([s['agent_rgb'][idx], s['wrist_rgb'][idx]], axis=1))
                    ax.set_title(f'{label} · policy step {idx-10}', fontsize=10)
                    ax.axis('off')
                    if col == 0:
                        success = r['outcomes'][1 if ai == 0 else 2]
                        ax.text(0, -.09, f'{arm} · {"success" if success else "failure"}',
                            transform=ax.transAxes, color=COLORS[ai], fontsize=10, weight='bold')
                p = s['proprioception'][10:]
            evidence.append({'arm': arm, 'recorded_steps': idxs, 'sha256': inventory[relative]['sha256']})
            qs = r['arms'][arm]['queries']
            horizon_ax.step([q['policy_step'] for q in qs]+[n-10],
                [q['used'] for q in qs]+[qs[-1]['used']], where='post', color=COLORS[ai], label=arm)
            aperture_ax.plot(np.arange(len(p)), 1000*np.abs(p[:, 6:8]).sum(axis=1), color=COLORS[ai], label=arm)
        horizon_ax.set_yscale('log', base=2)
        horizon_ax.set_yticks([2, 4, 8, 16, 32], [2, 4, 8, 16, 32])
        horizon_ax.set_ylim(1.7, 39); horizon_ax.set_ylabel('Used future horizon (frames)')
        aperture_ax.set_ylabel('Recorded finger aperture (mm)')
        for ax in (horizon_ax, aperture_ax):
            ax.set_xlabel('Policy step (20 Hz)'); ax.grid(alpha=.18); ax.legend(fontsize=8, loc='best')
            ax.axvline(first, color='#666666', ls=':', alpha=.6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'{case["panel"]} · pair {case["ordinal"]} · {case["outcome_class"].replace("_", " ")}\n'
                     f'{r["task"]["suite"]} / runtime task {r["task"]["runtime_task_index"]}', fontsize=14, y=.98)
        fig.text(.5, .012, 'Existing pre-action frames only · agent camera | wrist camera · last common time and separate episode endings are labeled',
                 ha='center', fontsize=9, color='#555555')
        fig.subplots_adjust(left=.04, right=.985, top=.88, bottom=.09)
        name = f'{case["panel"]}-{case["ordinal"]:04d}-{case["outcome_class"]}.png'
        fig.savefig(output/name, dpi=150); plt.close(fig)
        selected.append({**case, 'file': name, 'evidence': evidence})
    # Exposure is episode weighted; distinguish descriptive associations from causes.
    s = json.loads((root/'summary.json').read_text())['panels']
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    groups = ['adaptive_loss', 'adaptive_win', 'both_success', 'both_failure']
    for ax, panel in zip(axs, ('targeted-467', 'clean-200'), strict=True):
        x = np.arange(4); left = np.zeros(4)
        for key, label, color in [('short_fraction', 'Horizon 2 / 4', '#ca7b38'),
                                  ('h8', 'Horizon 8', '#447f9e'), ('long_fraction', 'Horizon 16 / 32', '#685b9e')]:
            values = []
            for g in groups:
                z = s[panel]['groups'][g]['episode_weighted_full']
                values.append(100*((1-z['non8_fraction']['mean']) if key == 'h8' else z[key]['mean']))
            ax.barh(x, values, left=left, color=color, label=label); left += values
        ax.set_yticks(x, ['Adaptive loss', 'Adaptive win', 'Both succeed', 'Both fail']); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_title(panel); ax.set_xlabel('Mean within-episode query fraction (%)')
        for i, g in enumerate(groups):
            ax.text(102, i, f'n={s[panel]["groups"][g]["episodes"]}', va='center', fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Horizon exposure by paired outcome · descriptive, not a causal attribution', fontsize=13)
    fig.tight_layout(rect=[0, .10, .97, .92])
    fig.savefig(output/'horizon-exposure.png', dpi=180); fig.savefig(output/'horizon-exposure.svg'); plt.close(fig)
    receipt = {'render_seconds': time.monotonic()-started, 'script_sha256': sha(__file__),
        'analysis_completion_sha256': sha(root/'completion.json'), 'cases': selected,
        'case_selection_sha256': sha(cases_path),
        'files_sha256': {p.name: sha(p) for p in sorted(output.iterdir())},
        'no_simulator_steps': True, 'no_model_calls': True}
    (output/'presentation.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'figures': len(receipt['files_sha256']), 'seconds': receipt['render_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=Path)
    a = p.parse_args(); render(a.root, a.output, a.cases)
