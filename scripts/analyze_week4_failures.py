#!/usr/bin/env python3
"""Verify dev-210, summarize clean/dev traces, and render deterministic failure samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from eventground_wam.clean_acceptance import read_sealed
from eventground_wam.clean_comparison import atomic_json, utc_now
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import (
    read_episode_metadata,
    sha256_file,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.failure_analysis import (
    grouped_success,
    inspect_trajectory,
    numeric_summary,
    success_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def base_evidence() -> dict[str, Any]:
    return {'schema_version': 1, 'created_at_utc': utc_now(),
            'analysis_source_revision': subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'analysis_script_sha256': sha256_file(Path(__file__)),
            'motion_flags_are_descriptive_not_causal_labels': True,
            'late_low_motion_definition': 'Last 40 recorded policy steps: XYZ bounding-box diagonal < 0.01 m.',
            'late_low_motion_with_command_definition': 'Low motion plus mean translation-command norm >= 0.1.',
            'motion_comparison_limit': 'Successes stop earlier than timeouts; duration and state-distribution differences confound aggregate trace comparisons.'}


def motion_groups(rows: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[' | '.join(str(row[f]) for f in fields)].append(row)
    return {key: numeric_summary(group) for key, group in sorted(groups.items())}


def analyze_clean(root: Path, output: Path) -> Any:
    report = read_sealed(root / 'comparison-report.json')
    assert report['status'] == 'complete' and report['pair_count'] == 2000
    rows = []
    for policy in ('b0', 'reference'):
        identity = report['policy_identities'][policy]
        for p in sorted((root / policy / 'receipts').glob('*.json')):
            r = read_sealed(p)
            assert r['identity_digest'] == identity['digest'] and r['assessable']
            meta = read_episode_metadata(r['artifact'])
            assert meta['checkpoint_revision'] == identity['checkpoint_revision']
            assert meta['success'] == r['success'] and meta['pair_id'] == r['pair_id']
            rows.append({'policy': policy, **r['pair'], 'ordinal': r['ordinal'],
                         'success': r['success'], 'termination_reason': r['termination_reason'],
                         'artifact': r['artifact'], 'artifact_sha256': r['artifact_sha256'],
                         'receipt_digest': r['digest'], 'motion': inspect_trajectory(r['artifact'])})
    assert len(rows) == 4000
    result = seal_contract({'artifact_type': 'eventground_wam.clean_failure_trace_analysis',
        **base_evidence(), 'comparison_report_digest': report['digest'],
        'validation_scope': 'Sealed receipts and episode identity checked; numeric action/proprioception arrays inspected. Full HDF5 validation was performed by the frozen final comparison and its verification audit.',
        'motion_by_policy_outcome': motion_groups(rows, ('policy', 'success')),
        'motion_by_policy_suite_outcome': motion_groups(rows, ('policy', 'suite', 'success')),
        'episodes': rows})
    atomic_json(output / 'clean-trace-analysis.json', result)
    return result


def analyze_dev(root: Path, clean_root: Path, output: Path) -> Any:
    report = read_sealed(root / 'evaluation-report.json')
    selection = read_sealed(root / 'selection.json')
    manifest = json.loads((ROOT / 'evaluation/splits/libero-plus-v1/dev-210.json').read_text())
    protocol = load_evaluation_protocol()
    assert report['status'] == 'complete' and report['episode_count'] == 210
    assert report['selection_is_complete_manifest'] and selection['task_count'] == 210
    assert len(report['episodes']) == len(selection['tasks']) == 210
    assert report['task_records_sha256'] == selection['task_records_sha256'] == manifest['task_records_sha256']
    assert report['protocol_sha256'] == selection['protocol_sha256'] == protocol_sha256(protocol)
    assert report['checkpoint'] == selection['checkpoint']
    assert report['project_git_revision'] == selection['project_git_revision']
    clean = read_sealed(clean_root / 'comparison-report.json')
    panel = read_sealed(ROOT / 'evaluation/splits/week4-clean-comparison-v1/paired-2000.json')
    canonical = [p for p in panel['pairs'] if p['trial_index'] == 0]
    rows = []
    for ordinal, (entry, selected, frozen) in enumerate(zip(
        report['episodes'], selection['tasks'], manifest['tasks'], strict=True
    )):
        assert all(selected[k] == v for k, v in frozen.items())
        assert entry['artifact'] is not None
        path = Path(entry['artifact'])
        assert sha256_file(path) == entry['artifact_sha256']
        validation = validate_episode_file(path, protocol)
        meta = read_episode_metadata(path)
        for key in ('suite', 'runtime_task_index', 'classification_task_id', 'perturbation_category',
                    'difficulty_level', 'episode_seed', 'instruction', 'bddl_sha256', 'initial_state_sha256'):
            assert meta[key] == selected[key], (ordinal, key)
        assert meta['checkpoint_revision'] == report['checkpoint']['checkpoint_revision']
        assert meta['project_git_revision'] == report['project_git_revision']
        assert meta['success'] == validation['success'] == entry['success']
        assert meta['termination_reason'] == entry['termination_reason']
        matches = [p for p in canonical if p['suite'] == selected['suite'] and
                   (selected['task_name'] == p['task_name'] or
                    selected['task_name'].startswith(p['task_name'] + '_'))]
        base = None
        if len(matches) == 1:
            p = matches[0]; key = f"{p['suite']}/task-{p['runtime_task_index']:02d}"
            base = {'task_key': key, 'task_name': p['task_name'],
                    'clean_b0_rate': clean['per_task'][key]['b0_success'] / 50,
                    'clean_reference_rate': clean['per_task'][key]['reference_success'] / 50,
                    'initial_state_matches_clean_state_zero': selected['initial_state_sha256'] == p['initial_state_sha256']}
        rows.append({**selected, **entry, 'base_task': base, 'motion': inspect_trajectory(path)})
    summary = success_summary(rows)
    assert summary['successes'] == report['success_count']
    all_latency = [v for r in rows for v in r['policy_latencies_ms']]
    baselines = {}
    for category in sorted({r['perturbation_category'] for r in rows}):
        group = [r for r in rows if r['perturbation_category'] == category]
        matched = [r for r in group if r['base_task'] is not None]
        baselines[category] = {'matched_task_count': len(matched), 'task_count': len(group),
            'clean_b0_rate_weighted_by_selected_base_tasks': float(np.mean([
                r['base_task']['clean_b0_rate'] for r in matched])) if matched else None,
            'warning': 'Descriptive composition reference only: different rollout states/seeds and perturbed tasks; not a paired estimate of perturbation effect.'}
    result = seal_contract({'artifact_type': 'eventground_wam.b0_dev210_failure_analysis',
        **base_evidence(), 'evaluation_report_digest': report['digest'], 'selection_digest': selection['digest'],
        'checkpoint': report['checkpoint'], 'protocol_sha256': report['protocol_sha256'],
        'validated_hdf5_count': len(rows), 'summary': summary,
        'breakdowns': {name: grouped_success(rows, fields) for name, fields in {
            'suite': ('suite',), 'category': ('perturbation_category',), 'difficulty': ('difficulty_level',),
            'category_x_difficulty': ('perturbation_category', 'difficulty_level')}.items()},
        'confidence_interval_limit': 'Wilson intervals are descriptive for this fixed 210-task panel; six tasks per category/difficulty cell are not a causal or monotonic difficulty experiment.',
        'operational': {'policy_latency_median_ms': float(np.median(all_latency)),
            'policy_latency_p95_ms': float(np.quantile(all_latency, .95)),
            'episode_wall_seconds_sum': sum(r['episode_wall_time_seconds'] or 0 for r in rows),
            'infrastructure_retries': sum(r['retry_count'] for r in rows),
            'policy_exception_count': sum(r['termination_reason'].startswith('policy_exception') for r in rows)},
        'motion_by_outcome': motion_groups(rows, ('success',)),
        'motion_by_category_outcome': motion_groups(rows, ('perturbation_category', 'success')),
        'composition_references': baselines, 'episodes': rows})
    atomic_json(output / 'dev210-analysis.json', result)
    return result


def render_preview(row: Any, target: Path, label: str) -> dict[str, Any]:
    import av
    from PIL import Image, ImageDraw, ImageFont
    target.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    with h5py.File(row['artifact'], 'r') as f:
        front, wrist = f['steps/agent_rgb'], f['steps/wrist_rgb']
        count = len(front)
        samples = np.unique(np.linspace(min(10, count - 1), count - 1, 8).astype(int))
        sheet = Image.new('RGB', (1024, 4 * 282 + 92), '#ffffff')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), label, font=font, fill='#111111')
        instruction = row.get('instruction', row.get('task_name', ''))
        import textwrap
        for i, line in enumerate(textwrap.wrap(instruction, 135)[:3]):
            draw.text((8, 25 + i * 17), line, font=small, fill='#333333')
        for slot, step in enumerate(samples):
            x, y = (slot % 2) * 512, 92 + (slot // 2) * 282
            pair = Image.fromarray(np.concatenate([front[step], wrist[step]], axis=1))
            sheet.paste(pair, (x, y + 22))
            draw.text((x + 5, y + 3), f'step {step} | policy time {(step-10)/20:.2f}s | front / wrist', font=small, fill='#111111')
        sheet_path = target.with_suffix('.png'); sheet.save(sheet_path)
        with av.open(str(target), 'w') as container:
            stream = container.add_stream('libx264', rate=10)
            stream.width, stream.height, stream.pix_fmt = 512, 288, 'yuv420p'
            stream.options = {'crf': '23', 'preset': 'veryfast', 'threads': '1'}
            for step in range(0, count, 2):
                canvas = Image.new('RGB', (512, 288), '#ffffff')
                canvas.paste(Image.fromarray(np.concatenate([front[step], wrist[step]], axis=1)), (0, 32))
                d = ImageDraw.Draw(canvas)
                d.text((3, 2), label[:72], font=small, fill='#111111')
                d.text((3, 17), f'step {step} | {row["termination_reason"]} | success={row["success"]}', font=small, fill='#111111')
                for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                    container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
    return {'label': label, 'artifact': row['artifact'], 'artifact_sha256': row['artifact_sha256'],
            'suite': row['suite'], 'runtime_task_index': row['runtime_task_index'],
            'success': row['success'], 'instruction': instruction,
            'video': str(target), 'video_sha256': sha256_file(target),
            'contact_sheet': str(sheet_path), 'contact_sheet_sha256': sha256_file(sheet_path),
            'sheet_steps': samples.tolist(), 'video_fps': 10, 'source_step_stride': 2,
            'preview_is_lossy_source_of_record_is_hdf5': True}


def previews(dev: Any, clean: Any, output: Path) -> Any:
    chosen = []
    categories = sorted({r['perturbation_category'] for r in dev['episodes']})
    for index, category in enumerate(categories):
        candidates = sorted([r for r in dev['episodes'] if r['perturbation_category'] == category and not r['success']],
                            key=lambda r: hashlib.sha256(r['artifact_sha256'].encode()).hexdigest())
        if not candidates: continue
        first = candidates[0]
        other = next((r for r in candidates[1:] if r['suite'] != first['suite']),
                     candidates[1] if len(candidates) > 1 else None)
        for j, row in enumerate([r for r in (first, other) if r]):
            chosen.append(render_preview(row, output / 'previews' / f'dev-category-{index+1}-{j+1}.mp4',
                                         f'{category} | {row["suite"]} | difficulty {row["difficulty_level"]}'))
    # Six largest clean deficits, with the first reference-only success pair in each task.
    tasks = defaultdict(lambda: {'b0': 0, 'reference': 0})
    pairs = defaultdict(dict)
    for row in clean['episodes']:
        key = (row['suite'], row['runtime_task_index'])
        tasks[key][row['policy']] += row['success']
        pairs[row['pair_id']][row['policy']] = row
    worst = sorted(tasks, key=lambda k: (tasks[k]['b0'] - tasks[k]['reference'], k))[:6]
    for index, key in enumerate(worst):
        pair = next(p for _, p in sorted(pairs.items()) if
                    (p['b0']['suite'], p['b0']['runtime_task_index']) == key and
                    not p['b0']['success'] and p['reference']['success'])
        for policy in ('b0', 'reference'):
            chosen.append(render_preview(pair[policy], output / 'previews' / f'clean-gap-{index+1}-{policy}.mp4',
                                         f'{policy} | {pair[policy]["pair_id"]}'))
    result = seal_contract({'artifact_type': 'eventground_wam.week4_failure_previews', **base_evidence(),
        'selection_rule': 'Two failure cases per category ranked by SHA256(artifact_sha256), preferring a different suite for the second. Six largest clean per-task deficits, first reference-only successful pair in each. Illustrative selection, not failure-mode prevalence.',
        'dev_analysis_digest': dev['digest'], 'clean_analysis_digest': clean['digest'],
        'previews': chosen})
    atomic_json(output / 'preview-manifest.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('clean', 'dev', 'previews'), required=True)
    parser.add_argument('--clean-root', type=Path, required=True)
    parser.add_argument('--dev-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'clean':
        value = analyze_clean(args.clean_root, args.output_root)
    elif args.mode == 'dev':
        value = analyze_dev(args.dev_root, args.clean_root, args.output_root)
    else:
        value = previews(read_sealed(args.output_root / 'dev210-analysis.json'),
                         read_sealed(args.output_root / 'clean-trace-analysis.json'), args.output_root)
    print(json.dumps({'mode': args.mode, 'digest': value['digest'], 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
