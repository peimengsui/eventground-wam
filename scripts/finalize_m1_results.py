#!/usr/bin/env python3
"""Verify the independent M1 backup plus completed review and seal its closeout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from package_m1_results import copy_checked

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    backup = read_sealed(args.package / 'backup-manifest.json')
    analysis = read_sealed(args.package / 'evaluation/analysis.json')
    diagnostic = read_sealed(args.package / 'diagnostics.json')
    review = read_sealed(args.review / 'visual-review.json')
    previews = read_sealed(args.review / 'previews/preview-manifest.json')
    runtime = read_sealed(args.review / 'evaluation-runtime.json')
    require(analysis['digest'] == backup['analysis_digest'], 'analysis differs from backup')
    require(analysis['status'] == 'analyzed', 'analysis incomplete')
    require(review['status'] == 'reviewed', 'visual review incomplete')
    require(review['diagnostics_digest'] == diagnostic['digest'], 'wrong reviewed diagnostics')
    require(review['preview_manifest_digest'] == previews['digest'], 'wrong preview manifest')
    for preview in previews['previews']:
        require(sha256_file(args.review / 'previews' / preview['contact_sheet']) == preview['sha256'],
                f"preview changed: {preview['ordinal']}")
        case = next(r for r in review['cases'] if r['ordinal'] == preview['ordinal'])
        require(case['contact_sheet_sha256'] == preview['sha256'], 'reviewed preview changed')
    require(sorted(r['ordinal'] for r in review['cases']) ==
            sorted(diagnostic['selected_visual_review_ordinals']), 'visual review selection incomplete')
    for unit in ('eventground-m1-auxiliary-v2.service', 'eventground-m1-evaluation-v2.service'):
        require(runtime['services'][unit]['ActiveState'] == 'inactive' and
                runtime['services'][unit]['Result'] == 'success', f'job incomplete: {unit}')
    inventory = {}
    for name, entry in backup['inventory'].items():
        path = args.package / name
        require(path.is_file() and path.stat().st_size == entry['bytes'] and
                sha256_file(path) == entry['sha256'], f'backup changed: {name}')
        inventory[name] = {'bytes': entry['bytes'], 'sha256': entry['sha256']}
    for name in ('WEEK7_M1_RESULTS.md', 'training-summary.json', 'evaluation-runtime.json',
                 'visual-review.json', 'evaluation-startup-disposition.json', 'off-host-backup.json',
                 'evaluation-summary.png', 'training-losses.png', 'previews/preview-manifest.json'):
        require((args.review / name).is_file(), f'missing review evidence: {name}')
    for path in sorted(args.review.rglob('*')):
        if path.is_file():
            name = Path('review') / path.relative_to(args.review)
            entry = copy_checked(path, args.package / name)
            inventory[str(name)] = {'bytes': entry['bytes'], 'sha256': entry['sha256']}
    for name in ('backup-manifest.json',):
        path = args.package / name
        inventory[name] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    receipt = seal_contract({
        'artifact_type': 'eventground_wam.week7_m1_complete',
        'status': 'm1_training_evaluation_and_packaging_complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'backup_manifest_digest': backup['digest'], 'analysis_digest': analysis['digest'],
        'training_completion_digest': backup['training_completion_digest'],
        'checkpoint_sha256': backup['checkpoint_sha256'],
        'visual_review_digest': review['digest'], 'evaluation_runtime_digest': runtime['digest'],
        'summaries': analysis['summaries'], 'M1_minus_B0': analysis['M1_minus_B0'],
        'pairing': analysis['pairing'],
        'reviewed_and_reverified_file_count': len(inventory),
        'reviewed_and_reverified_bytes': sum(r['bytes'] for r in inventory.values()),
        'inventory': inventory,
        'scope': 'M1 fixed-endpoint training, dev-210 evaluation, review and packaging only. Week 7 remains open for M2. Original historical pending-status receipts remain unchanged.',
        'limitations': analysis['limitations'],
    })
    target = args.package / 'completion.json'
    require(not target.exists(), 'completion is immutable; use a new version for a later review')
    atomic_json(target, receipt)
    print({k: receipt[k] for k in ('status', 'digest', 'reviewed_and_reverified_file_count',
                                  'reviewed_and_reverified_bytes')}, flush=True)


if __name__ == '__main__':
    main()
