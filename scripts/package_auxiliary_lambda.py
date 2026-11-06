#!/usr/bin/env python3
"""Package unchanged accepted M2 inputs plus explicit Lambda qualification code."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from auxiliary_lambda_runtime import verify_recipe
from verify_auxiliary_integration import verify

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe_path = ROOT/'configs/training/week7_m2_auxiliary_v1.json'
    recipe = read_sealed(recipe_path)
    verify_recipe(recipe)
    attempt = ROOT/'docs/evaluation/week7-integration-v1/attempt-04'
    verification = verify(attempt)
    assert verification['digest'] == recipe['verification_digest']
    original = read_sealed(attempt/'source-manifest.json')
    assert original['digest'] == recipe['verified_source_digest']
    files = {name: ROOT/name for name in original['files']}
    for name in ('scripts/train_auxiliary_policy.py', 'scripts/auxiliary_lambda_runtime.py',
                 'scripts/check_auxiliary_lambda.py', 'scripts/package_auxiliary_lambda.py',
                 'tests/test_auxiliary_lambda_runtime.py', 'tests/test_auxiliary_production.py',
                 'configs/training/week7_m2_auxiliary_v1.json'):
        files[name] = ROOT/name
    files['source-manifest.json'] = attempt/'source-manifest.json'
    for name in ('runtime.json', 'initialization-audit.json', 'verification.json'):
        files[f'launch-evidence/{name}'] = attempt.parent/name
    manifest = seal_contract({'artifact_type': 'eventground_wam.week7_production_source',
        'recipe_digest': recipe['digest'], 'verified_source_digest': original['digest'],
        'files': {name: sha256_file(path) for name, path in sorted(files.items())},
        'scope': 'Lambda A10 qualification only; frozen weight; unchanged 206 accepted source files',
        'production_launch_included': False,
        'declared_environment_change': 'Feifei RTX 5090 to user-provisioned Lambda A10',
        'software_identity': 'exact original Python/package/CUDA build; host/GPU/driver pinned separately',
        'path_alias': {'/home/peimengsui/data/eventground-wam': '/home/ubuntu/data/eventground-wam'},
        'pilot_output_policy': 'new persistent directory; no pilot state is a production initialization'})
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'source-manifest.json', manifest)
    archive_path = args.output/'source.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
        data = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
        entry = tarfile.TarInfo('production-source-manifest.json')
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    receipt = seal_contract({'artifact_type': 'eventground_wam.lambda_qualification_package',
        'recipe_digest': recipe['digest'], 'source_digest': manifest['digest'],
        'archive_sha256': sha256_file(archive_path), 'files': len(files),
        'verified_unchanged_files': len(original['files']), 'production_started': False})
    atomic_json(args.output/'package.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
