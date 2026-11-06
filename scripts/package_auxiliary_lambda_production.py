#!/usr/bin/env python3
"""Bind a successful A10 qualification to the separate production launch package."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualified = read_sealed(args.pilot/'run/qualification.json')
    original = read_sealed(args.pilot/'source-manifest.json')
    assert qualified['passed'] and qualified['production_source_digest'] == original['digest']
    assert qualified['recipe_digest'] == original['recipe_digest']
    amendment = read_sealed(ROOT/'configs/training/week7_m2_lambda_budget_v1.json')
    assert qualified['budget_amendment_digest'] == amendment['digest']
    supplement = read_sealed(args.pilot/'run/recovery-supplement.json')
    assert supplement['orchestrator_sha256'] == sha256_file(ROOT/'scripts/qualify_auxiliary_lambda_budget.py')
    args.output.mkdir(parents=True, exist_ok=False)
    # Read the actual sealed pilot archive instead of assuming local files still
    # match it. This also preserves the original Feifei evidence paths verbatim.
    contents = {}
    with tarfile.open(args.pilot/'source.tar.gz', 'r:gz') as archive:
        for name, digest in original['files'].items():
            data = archive.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest() == digest, name
            contents[name] = data
    for name in ('scripts/train_auxiliary_lambda.py', 'scripts/package_auxiliary_lambda_production.py',
                 'tests/test_auxiliary_lambda_production.py',
                 'scripts/qualify_auxiliary_lambda_budget.py',
                 'configs/training/week7_m2_lambda_budget_v1.json',
                 'configs/systemd/eventground-m2-lambda-recovery-v1.service',
                 'configs/systemd/eventground-m2-lambda-v1.service'):
        contents[name] = (ROOT/name).read_bytes()
    contents['lambda-qualification/source-manifest.json'] = (args.pilot/'source-manifest.json').read_bytes()
    for name in ('qualification.json', 'pilot-identity.json', 'correctness.json', 'resources.json',
                 'jepa-resume.json', 'jepa-uninterrupted.json', 'recovery-supplement.json'):
        contents[f'lambda-qualification/{name}'] = (args.pilot/'run'/name).read_bytes()
    contents['lambda-qualification/runtime.json'] = (args.pilot/'runtime.json').read_bytes()
    manifest = seal_contract({'artifact_type': 'eventground_wam.week7_production_source',
        'recipe_digest': original['recipe_digest'], 'verified_source_digest': original['verified_source_digest'],
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())},
        'pilot_source_digest': original['digest'], 'lambda_qualification_digest': qualified['digest'],
        'lambda_budget_amendment_digest': amendment['digest'],
        'scope': 'unchanged qualified M2 implementation plus Lambda production orchestration',
        'pilot_weights_included': False})
    atomic_json(args.output/'source-manifest.json', manifest)
    contents['production-source-manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    path = args.output/'source.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in sorted(contents.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    receipt = seal_contract({'artifact_type': 'eventground_wam.lambda_production_package',
        'source_digest': manifest['digest'], 'archive_sha256': sha256_file(path),
        'recipe_digest': original['recipe_digest'], 'qualification_digest': qualified['digest'],
        'budget_amendment_digest': amendment['digest'],
        'files': len(manifest['files']), 'unchanged_pilot_files': len(original['files'])})
    atomic_json(args.output/'package.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
