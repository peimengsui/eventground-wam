#!/usr/bin/env python3
"""Back up completed M1 evidence and both sides of the audited dev-210 comparison."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from eventground_wam.auxiliary_evaluation import load_contract, require, verify_endpoint
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.failure_analysis import grouped_success, inspect_trajectory, numeric_summary


def copy_checked(source, destination):
    """A separate physical copy; resume is allowed only when existing bytes agree."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source)
    if not destination.exists():
        temporary = destination.with_name(destination.name + '.copying')
        shutil.copy2(source, temporary)
        require(sha256_file(temporary) == digest, f'backup checksum failed: {source}')
        temporary.replace(destination)
    require(sha256_file(destination) == digest, f'existing backup differs: {destination}')
    return {'source': str(source), 'bytes': source.stat().st_size, 'sha256': digest}


def diagnostics(report, baseline, tasks):
    rows = []
    for i, row in enumerate(report['episodes']):
        rows.append({'ordinal': i, 'task_name': tasks[i]['task_name'],
                     'classification_task_id': tasks[i]['classification_task_id'],
                     **row, 'b0_success': baseline['episodes'][i]['success'],
                     'motion': inspect_trajectory(row['artifact'])})
    selected = []
    for m1, b0, count in [(False, True, 3), (True, False, 3), (False, False, 1), (True, True, 1)]:
        selected.extend(row['ordinal'] for row in
                        [r for r in rows if r['success'] == m1 and r['b0_success'] == b0][:count])
    return seal_contract({'artifact_type': 'eventground_wam.week7_m1_descriptive_diagnostics',
        'selection_rule': 'First 3 M1-only failures, first 3 M1-only successes, first shared failure and shared success, in frozen task order.',
        'selected_visual_review_ordinals': selected, 'episodes': rows,
        'grouped_outcomes': {field: grouped_success(rows, (field,)) for field in
                             ('suite', 'perturbation_category', 'difficulty_level', 'termination_reason')},
        'success_motion_summary': numeric_summary([r for r in rows if r['success']]),
        'failure_motion_summary': numeric_summary([r for r in rows if not r['success']]),
        'interpretation': 'Motion flags describe recorded trajectories; they do not establish failure causes.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-source', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.evaluation_source)
    _, endpoint, completion = verify_endpoint(args.training_root, contract)
    analysis = read_sealed(args.evaluation_root / 'analysis.json')
    require(analysis['contract_digest'] == contract['digest'], 'analysis contract changed')
    require(analysis['status'] == 'analyzed' and len(analysis['paired_input_audit']) == 210,
            'artifact audit is incomplete')
    for unit in ('eventground-m1-auxiliary-v2.service', 'eventground-m1-evaluation-v2.service'):
        state = subprocess.check_output(['systemctl', '--user', 'show', unit,
                                         '-p', 'ActiveState', '-p', 'Result'], text=True)
        require('ActiveState=inactive' in state and 'Result=success' in state, f'job incomplete: {unit}: {state}')
    baseline_root = Path(contract['baseline']['remote_directory'])
    baseline = read_sealed(baseline_root / 'evaluation-report.json')
    require(sha256_file(baseline_root / 'evaluation-report.json') == contract['baseline']['report_sha256'],
            'baseline report changed')
    report = read_sealed(args.evaluation_root / 'dev-210/M1/evaluation-report.json')
    identity = read_sealed(args.evaluation_root / 'dev-210/M1/identity.json')
    source_manifest = read_sealed(args.evaluation_source / 'evaluation-source-manifest.json')
    require(source_manifest['digest'] == identity['evaluation_source_digest'], 'evaluation source identity changed')
    for name, digest in source_manifest['files'].items():
        require(sha256_file(args.evaluation_source / name) == digest, f'evaluation source changed: {name}')
    require(report['digest'] == analysis['report_digests']['M1'], 'M1 report changed')
    require(report['episode_count'] == len(report['episodes']) == 210 and report['all_artifacts_present'],
            'rollout panel incomplete')
    tasks = json.loads((args.evaluation_source / contract['panel']).read_text())['tasks']
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = {}
    required = [(args.training_root, Path('training')),
                (args.evaluation_root, Path('evaluation')),
                (args.evaluation_source, Path('evaluation-source')),
                (args.evidence, Path('supporting-evidence'))]
    required_bytes = sum(p.stat().st_size for source, _ in required for p in source.rglob('*') if p.is_file())
    required_bytes += sum(Path(r['artifact']).stat().st_size for r in baseline['episodes'])
    require(shutil.disk_usage(args.output).free >= required_bytes + 50 * 1024**3, 'backup disk reserve insufficient')
    for source, relative in required:
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                name = relative / path.relative_to(source)
                inventory[str(name)] = copy_checked(path, args.output / name)
        print(json.dumps({'copied': str(relative)}), flush=True)
    for filename in ('identity.json', 'evaluation-report.json', 'task-manifest.json'):
        path = baseline_root / filename
        name = Path('baseline') / filename
        inventory[str(name)] = copy_checked(path, args.output / name)
    mapping = []
    for i, row in enumerate(baseline['episodes']):
        path = Path(row['artifact'])
        require(sha256_file(path) == row['artifact_sha256'], 'baseline episode changed since audit')
        name = Path('baseline/episodes') / f'{i:03d}-{path.name}'
        inventory[str(name)] = copy_checked(path, args.output / name)
        mapping.append({'ordinal': i, 'original': str(path), 'packaged': str(name), 'sha256': row['artifact_sha256']})
    atomic_json(args.output / 'baseline-path-map.json', seal_contract({'episodes': mapping}))
    detail = diagnostics(report, baseline, tasks)
    atomic_json(args.output / 'diagnostics.json', detail)
    script_name = 'package_m1_results.py'
    inventory[script_name] = copy_checked(Path(__file__), args.output / script_name)
    for name in ('baseline-path-map.json', 'diagnostics.json'):
        path = args.output / name
        inventory[name] = {'source': 'generated in package', 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    receipt = seal_contract({'artifact_type': 'eventground_wam.week7_m1_results_backup',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'verified_backup_visual_and_narrative_review_pending',
        'contract_digest': contract['digest'], 'training_completion_digest': completion['digest'],
        'endpoint_report_digest': endpoint['digest'], 'analysis_digest': analysis['digest'],
        'diagnostics_digest': detail['digest'], 'checkpoint_sha256': endpoint['checkpoint_sha256'],
        'copied_file_count': len(inventory), 'copied_bytes': sum(r['bytes'] for r in inventory.values()),
        'copies_are_independent_files': True, 'inventory': inventory,
        'portability': 'Original sealed reports retain original absolute artifact paths. baseline-path-map and inventory map those paths to copied files; frozen base model/assets remain external pinned dependencies.'})
    atomic_json(args.output / 'backup-manifest.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'digest', 'copied_file_count', 'copied_bytes')}), flush=True)


if __name__ == '__main__':
    main()
