#!/usr/bin/env python3
"""Preserve every A100-qualified source byte and add the approved M4 launcher."""
from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
from pathlib import Path

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
ADDITIONS = ('scripts/train_coupled_lambda.py', 'scripts/package_coupled_lambda_launch.py',
    'tests/test_coupled_lambda.py', 'configs/training/week8_m4_lambda_budget_v1.json',
    'configs/systemd/eventground-m4-lambda-budget-v1.service')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    require_new = args.output/'source.tar.gz'
    if require_new.exists():
        raise FileExistsError(require_new)
    pilot = ROOT/'docs/evaluation/week8-m4-lambda-qualification-v1/attempt-07'
    parent = read_sealed(pilot/'source-manifest.json')
    amendment = read_sealed(ROOT/'configs/training/week8_m4_lambda_budget_v1.json')
    assert amendment['pilot_source_digest'] == parent['digest']
    package = read_sealed(pilot/'package.json')
    assert sha256_file(pilot/'source.tar.gz') == package['archive_sha256']
    with tarfile.open(pilot/'source.tar.gz') as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    for name,digest in parent['files'].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest,name
    for name in ADDITIONS:
        assert name not in files,name
        files[name] = (ROOT/name).read_bytes()
    receipt = seal_contract({'artifact_type':'eventground_wam.coupled_lambda_launch_source',
        'pilot_source_digest':parent['digest'], 'design_digest':parent['design_digest'],
        'time_budget_amendment_digest':amendment['digest'],
        'all_pilot_source_files_preserved':True, 'original_week7_files_preserved':215,
        'added_files':list(ADDITIONS),
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(files.items())}})
    atomic_json(args.output/'source-manifest.json',receipt)
    files['coupled-launch-source-manifest.json'] = (args.output/'source-manifest.json').read_bytes()
    with tarfile.open(require_new,'w:gz') as archive:
        for name,data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size,info.mode = len(data),0o644
            archive.addfile(info,io.BytesIO(data))
    record = seal_contract({'source_digest':receipt['digest'],
        'archive_sha256':sha256_file(require_new),'pilot_source_digest':parent['digest'],
        'time_budget_amendment_digest':amendment['digest'],'pilot_training_code_unchanged':True})
    atomic_json(args.output/'package.json',record)
    print(record)


if __name__ == '__main__':
    main()
