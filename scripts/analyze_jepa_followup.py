#!/usr/bin/env python3
"""Package the frozen training-only pilot's numerical diagnostic and figure."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file, write_json


def balanced(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row['task_key'], row['group_digest']].append(row[field])
    tasks = defaultdict(list)
    for (task, _), values in groups.items():
        tasks[task].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in tasks.values()]))


def run(args):
    report = read_sealed(args.run/'report.json')
    audit = read_sealed(args.verification)
    assert audit['status'] == 'passed' and audit['report_digest'] == report['digest']
    diagnostics = {}
    for arm in report['arms']:
        rows = read_sealed(args.run/f'{arm}-eval-metrics.json')['rows']
        cells = []
        for motion in ('all', 'low', 'medium', 'high'):
            chosen = [r for r in rows if r['camera'] == CAMERAS[0] and r['horizon'] == 2
                      and (motion == 'all' or r['motion_bin'] == motion)]
            fields = ['cosine', 'copy', 'predicted_change_from_current']
            if arm != 'copy_residual':
                fields += ['head_span_optimistic_error_bound', 'current_head_span_optimistic_error_bound']
            values = {f: balanced(chosen, f) for f in fields}
            cells.append({'motion_bin': motion, 'examples': len(chosen),
                'groups': len({(r['task_key'], r['group_digest']) for r in chosen}),
                'tasks': len({r['task_key'] for r in chosen}), **values,
                'relative_gain_vs_copy': 1-values['cosine']/values['copy'],
                'head_bound_worse_than_copy_count': None if arm == 'copy_residual' else
                    sum(r['head_span_optimistic_error_bound'] > r['copy'] for r in chosen)})
        diagnostics[arm] = {'front_h2': cells, 'final_gradient_probe': report['arms'][arm]['final_gradients']}
    summary = seal_contract({'artifact_type': 'eventground_wam.jepa_followup_analysis',
        'report_digest': report['digest'], 'verification_digest': audit['digest'],
        'analysis_script_sha256': sha256_file(Path(__file__)), 'front_h2_diagnostics': diagnostics,
        'screen': audit['pilot_screen'], 'aggregation': 'equal task, then equal numeric group, then equal rows',
        'reference_limitation': 'Production v1 trained on all selected groups; use only for in-sample diagnosis.'})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'analysis.json', summary)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {'absolute_control': '#b35a2b', 'copy_residual': '#087e8b'}
    names = {'absolute_control': 'Fresh absolute control', 'copy_residual': 'Copy residual'}
    for ci, camera in enumerate(CAMERAS):
        ax = axes[0, ci]
        for arm in ('absolute_control', 'copy_residual'):
            selected = [next(s for s in report['arms'][arm]['summaries'] if s['role'] == 'pilot_eval'
                and s['camera'] == camera and s['horizon'] == h and s['motion_bin'] == 'all') for h in HORIZONS]
            ax.plot(range(5), [s['cosine'] for s in selected], marker='o', color=colors[arm], label=names[arm])
        ax.plot(range(5), [s['copy'] for s in selected], marker='s', linestyle='--', color='#59616a', label='Copy baseline')
        ax.set(xticks=range(5), xticklabels=HORIZONS, xlabel='Horizon (steps)', ylabel='Cosine error (lower is better)',
               title=f"{camera.split('.')[-1].capitalize()} camera: unseen training groups")
        ax.grid(alpha=.18); ax.legend(fontsize=9)
    ax = axes[1, 0]
    for arm in ('absolute_control', 'copy_residual'):
        rows = [json.loads(x) for x in (args.run/f'{arm}-trace.jsonl').read_text().splitlines()]
        ax.plot([r['step'] for r in rows], [r['total'] for r in rows], color=colors[arm], label=names[arm], linewidth=1.5)
    ax.set(xlabel='Optimizer updates', ylabel='Cosine + 10 × variance penalty', title='Matched fitting loss (sampled every 100 updates)')
    ax.grid(alpha=.18); ax.legend(fontsize=9)
    ax = axes[1, 1]; kinds = ('zero-actions', 'reverse-actions', 'zero-horizon', 'swap-tokens')
    for i, arm in enumerate(('absolute_control', 'copy_residual')):
        values = [report['arms'][arm]['ablations'][k]['medium_long_cosine_increase'] for k in kinds]
        ax.bar(np.arange(4)+(i-.5)*.36, values, width=.36, color=colors[arm], label=names[arm])
    ax.axhline(0, color='#444444', linewidth=.8)
    ax.axhline(.001, color='#777777', linestyle=':', label='Action/token screen threshold')
    ax.set(xticks=range(4), xticklabels=['Zero actions', 'Reverse actions', 'Zero horizon', 'Swap tokens'],
           ylabel='Increase in h8/16/32 cosine error', title='Intervention effect: positive means normal input helps')
    ax.tick_params(axis='x', labelsize=8); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.18)
    fig.suptitle('JEPA follow-up • training-only pilot • one seed • 6,000 updates per fresh arm', fontsize=15)
    fig.savefig(args.output/'pilot-results.png', dpi=170)
    fig.savefig(args.output/'pilot-results.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'verification', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
