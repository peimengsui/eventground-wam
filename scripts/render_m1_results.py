#!/usr/bin/env python3
"""Render frozen M1 summaries and deterministic paired rollout contact sheets."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def plot(analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {'B0': '#7b8794', 'M1': '#147d92'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [1, 2]})
    for i, arm in enumerate(('B0', 'M1')):
        row = analysis['summaries'][arm]
        rate = row['success_rate'] * 100
        interval = np.array(row['wilson_95']) * 100
        axes[0].bar(i, rate, color=colors[arm], width=.6)
        axes[0].errorbar(i, rate, yerr=[[rate - interval[0]], [interval[1] - rate]],
                         color='#202830', capsize=5, fmt='none')
        axes[0].text(i, 6, f"{row['successes']}/{row['count']}\n{rate:.1f}%",
                     ha='center', color='white', fontweight='bold')
    axes[0].set(xticks=[0, 1], xticklabels=['B0', 'M1'], ylim=(0, 100), ylabel='Success (%)',
                title='Dev-210 · Wilson 95% intervals')
    field = 'perturbation_category'
    categories = sorted(analysis['summaries']['M1']['breakdowns'][field])
    for offset, arm in ((-.18, 'B0'), (.18, 'M1')):
        rows = [analysis['summaries'][arm]['breakdowns'][field][k] for k in categories]
        rates = [r['successes'] / r['count'] * 100 for r in rows]
        bars = axes[1].barh(np.arange(len(rows)) + offset, rates, height=.32,
                            label=arm, color=colors[arm])
        for bar, row in zip(bars, rows):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                         f"{row['successes']}/{row['count']}", va='center', fontsize=8)
    axes[1].set(yticks=range(len(categories)), yticklabels=categories, xlim=(0, 115),
                xlabel='Success (%)', title='Category counts · descriptive breakdown')
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower right', frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    gap = analysis['M1_minus_B0']
    lo, hi = gap['paired_95_interval_percentage_points']
    fig.suptitle(f"M1 − B0: {gap['gap_percentage_points']:+.1f} percentage points "
                 f"(task bootstrap 95%: {lo:+.1f} to {hi:+.1f})", fontsize=13)
    exact = sum(row['exact_match'] for row in analysis['paired_input_audit'])
    fig.text(.5, .015, f"{exact}/210 exact prefixes; {analysis['pairing']}. Single-seed dev panel; no test-set claim.",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'evaluation-summary.{extension}', dpi=180)
    plt.close(fig)


def previews(package, output):
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    diagnostic = read_sealed(package / 'diagnostics.json')
    m1 = read_sealed(package / 'evaluation/dev-210/M1/evaluation-report.json')
    b0 = read_sealed(package / 'baseline/evaluation-report.json')
    mapping = read_sealed(package / 'baseline-path-map.json')['episodes']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    records = []
    for ordinal in diagnostic['selected_visual_review_ordinals']:
        detail = diagnostic['episodes'][ordinal]
        sheet = Image.new('RGB', (1090, 1010), 'white')
        draw = ImageDraw.Draw(sheet)
        title = f"Task {ordinal:03d} | {detail['perturbation_category']} | difficulty {detail['difficulty_level']}"
        draw.text((10, 5), title, font=font, fill='#101820')
        for i, line in enumerate(textwrap.wrap(detail['task_name'], 125)[:3]):
            draw.text((10, 28+i*17), line, font=small, fill='#101820')
        record = {'ordinal': ordinal, 'task_name': detail['task_name'], 'arms': {}}
        for arm_index, (arm, report) in enumerate((('B0', b0), ('M1', m1))):
            row = report['episodes'][ordinal]
            path = (package / mapping[ordinal]['packaged'] if arm == 'B0' else
                    package / 'evaluation/dev-210/M1' / Path(row['artifact']).name)
            if not path.exists():
                matches = list((package / 'evaluation').rglob(Path(row['artifact']).name))
                if len(matches) != 1:
                    raise ValueError(f'cannot locate copied episode {ordinal}')
                path = matches[0]
            if sha256_file(path) != row['artifact_sha256']:
                raise ValueError(f'episode changed: {path}')
            with h5py.File(path) as f:
                policy = np.flatnonzero(f['steps/phase'][:] == 1)
                samples = policy[np.linspace(0, len(policy)-1, 5).astype(int)]
                y = 92 + arm_index * 450
                draw.text((10, y), f"{arm}: success={row['success']} | {row['termination_reason']}",
                          font=font, fill='#101820')
                for column, step in enumerate(samples):
                    x = 100 + column * 196
                    draw.text((x, y+22), f'step {step}', font=small, fill='#101820')
                    for camera_index, camera in enumerate(('agent_rgb', 'wrist_rgb')):
                        frame = Image.fromarray(f[f'steps/{camera}'][step])
                        sheet.paste(frame.resize((192, 192)), (x, y+42+camera_index*196))
                draw.text((8, y+110), 'Front', font=font, fill='#101820')
                draw.text((8, y+290), 'Wrist', font=font, fill='#101820')
            record['arms'][arm] = {'success': row['success'], 'artifact_sha256': row['artifact_sha256'],
                                   'packaged_path': str(path.relative_to(package)), 'steps': samples.tolist()}
        name = f'paired-task-{ordinal:03d}.png'
        sheet.save(output / name)
        record.update(contact_sheet=name, sha256=sha256_file(output / name))
        records.append(record)
    atomic_json(output / 'preview-manifest.json', seal_contract({
        'artifact_type': 'eventground_wam.week7_m1_paired_previews',
        'diagnostics_digest': diagnostic['digest'], 'selection_rule': diagnostic['selection_rule'],
        'scope': 'Five evenly spaced policy frames per arm. Illustrative, not failure-mode prevalence or causal diagnosis.',
        'previews': records}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plot', 'previews'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'plot':
        plot(read_sealed(args.input), args.output)
    else:
        previews(args.input, args.output)
    print(json.dumps({'rendered': args.mode, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
