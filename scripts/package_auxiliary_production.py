#!/usr/bin/env python3
"""Package the exact accepted Week 7 source plus a separate production wrapper."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from verify_auxiliary_integration import verify

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--service', type=Path, required=True)
    args = parser.parse_args()
    recipe_path = args.recipe.resolve()
    recipe = read_sealed(recipe_path)
    attempt = ROOT/'docs/evaluation/week7-integration-v1/attempt-04'
    verified = verify(attempt)
    if verified['digest'] != recipe['verification_digest']:
        raise ValueError('recipe pilot verification mismatch')
    original = read_sealed(attempt/'source-manifest.json')
    if original['digest'] != recipe['verified_source_digest']:
        raise ValueError('recipe source identity mismatch')
    files = {name: ROOT/name for name in original['files']}
    for name in ['scripts/train_auxiliary_policy.py', 'scripts/package_auxiliary_production.py',
                 'tests/test_auxiliary_production.py']:
        files[name] = ROOT/name
    service_path = args.service.resolve()
    files[str(service_path.relative_to(ROOT))] = service_path
    files[str(recipe_path.relative_to(ROOT))] = recipe_path
    files['source-manifest.json'] = attempt/'source-manifest.json'
    for name in ('runtime.json', 'initialization-audit.json', 'verification.json'):
        files[f'launch-evidence/{name}'] = attempt.parent/name
    manifest = seal_contract({'artifact_type': 'eventground_wam.week7_production_source',
        'recipe_digest': recipe['digest'], 'verified_source_digest': original['digest'],
        'files': {name: sha256_file(path) for name, path in sorted(files.items())},
        'scope': 'unchanged accepted implementation plus production orchestration; no pilot weights'})
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
    receipt = seal_contract({'artifact_type': 'eventground_wam.week7_production_package',
        'recipe_digest': recipe['digest'], 'source_digest': manifest['digest'],
        'archive_sha256': sha256_file(archive_path), 'files': len(files),
        'verified_unchanged_files': len(original['files'])})
    atomic_json(args.output/'package.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
