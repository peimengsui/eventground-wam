"""Static scientific figures for matched-input horizon sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from week11_replay_common import read, sha, write

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(args):
    s = read(args.base/'summary.json'); rows = read(args.base/'joined-observations.json')['observations']
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    hs = [2, 4, 8, 16, 32]; x = np.arange(5)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9), layout='constrained')
    g = s['groups']['all']['horizons']
    for k, label, color in [('normalized_future_tokens', 'Normalized future tokens', '#127b8c'),
                             ('residual', 'Coupling residual', '#c06a30')]:
        ax[0].plot(x, [100*g[str(h)][k]['relative_l2_delta']['median'] for h in hs], 'o-', label=label, color=color)
    roles = [('shared_first_departure', 'First departure (27)', '#566d9e'),
             ('before_first_closing', 'Before closing (52)', '#bb5e34'),
             ('after_first_closing', 'After closing (52)', '#208378')]
    for role, label, color in roles:
        g = s['groups'][role]['horizons']; n = s['groups'][role]['n']
        ax[1].plot(x, [100*g[str(h)]['translation']['relative_l2_delta']['median'] for h in hs], 'o-', label=label, color=color)
        ax[2].plot(x, [100*g[str(h)]['gripper_changed_chunks']/n for h in hs], 'o-', label=label, color=color)
    for a in ax:
        a.set_xticks(x, hs); a.set_xlabel('Forced future horizon (frames)'); a.set_ylim(bottom=0)
        a.axvline(2, color='#999999', ls=':', lw=1); a.grid(axis='y', alpha=.2); a.legend(fontsize=8, loc='upper left')
    ax[0].set_title('Future representation and coupling shift')
    ax[0].set_ylabel('Median L2 change / horizon-8 L2 (%)')
    ax[1].set_title('Arm translation commands depend on phase')
    ax[1].set_ylabel('Median L2 change / horizon-8 L2 (%)')
    ax[2].set_title('Open/close command changes near closing')
    ax[2].set_ylabel('Chunks with ≥1 different gripper command (%)')
    fig.suptitle('Identical observations, different horizons — 130 observations from 27 selected pairs', fontsize=14)
    fig.supxlabel('Post-hoc case panel; descriptive sensitivity, not a success-rate estimate. All recorded-action and repeatability gates pass.', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(args.output/f'horizon-sensitivity.{ext}', dpi=160)
    plt.close(fig)
    selected = read(args.base/'figure-selection.json')['ids']
    for rid in selected:
        r = next(r for r in rows if r['id'] == rid); p = args.replay/r['tensor_file']
        assert sha(p) == r['tensor_sha256']
        z = np.load(p); a = r['annotation']
        fig = plt.figure(figsize=(12, 5.8), layout='constrained'); grid = fig.add_gridspec(2, 3)
        with h5py.File(r['artifact'], 'r') as f:
            for i, camera in enumerate(('agent_rgb', 'wrist_rgb')):
                v = fig.add_subplot(grid[i, 0]); v.imshow(f['steps'][camera][r['step']]); v.axis('off')
                v.set_title(camera.replace('_rgb', '')+' at recorded query', fontsize=10)
        a1 = fig.add_subplot(grid[0, 1:]); a2 = fig.add_subplot(grid[1, 1:])
        for h, col in [(2, '#9166a8'), (4, '#bb5e34'), (8, '#242424'), (16, '#208378'), (32, '#566d9e')]:
            v = z[f'h{h}_decoded_actions']; ref = z['h8_decoded_actions']
            a1.step(np.arange(1, 9), v[:, 6], where='mid', label=f'h{h}', color=col, alpha=.85, lw=2 if h == 8 else 1.4)
            a2.plot(np.arange(1, 9), np.linalg.norm(v[:, :3]-ref[:, :3], axis=1), 'o-', color=col, ms=3)
        a1.set_yticks([-1, 1], ['Open (−1)', 'Close (+1)']); a1.set_ylim(-1.25, 1.25)
        a1.set_title('Gripper commands from the same recorded observation'); a1.legend(ncol=5, fontsize=9)
        a2.set_ylabel('Translation-command L2 difference vs h8'); a2.set_xlabel('Action in the unchanged 8-action chunk')
        for v in [a1, a2]:
            v.set_xticks(range(1, 9)); v.grid(alpha=.2)
        target = str(a.get('target', a.get('exclusion')))
        pred = r['prediction']
        fig.suptitle(f'{rid}\n{r["outcome_class"]}; recorded horizon {r["used_horizon"]}; raw recorded step {r["step"]}', fontsize=12)
        fig.supxlabel(f'Forecast: {pred["next_event"]} at +{pred["horizon"]}; physical candidate: {target}\nAutomatic candidates are unreviewed; no alternative action chunk was rolled out.', fontsize=9)
        fig.savefig(args.output/f'{rid}.png', dpi=140); plt.close(fig)
    write(args.output/'receipt.json', {'summary_digest': s['digest'], 'example_ids': selected,
        'files_sha256': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
