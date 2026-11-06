#!/usr/bin/env python3
"""Create an immutable, hash-listed Week 7 diagnostic source snapshot."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from eventground_wam.b0_training import atomic_json, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root/f'docs/evaluation/week7-integration-v1/attempt-{args.attempt:02d}'
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for folder in ['src', 'scripts', 'configs', 'docs/model', 'environments/b0']:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file()
                     and '.venv' not in p.parts and '__pycache__' not in p.parts
                     and p.suffix in {'.py', '.json', '.jsonl', '.lock', '.toml', '.sh'})
    files.append(root/'docs/evaluation/week6-jepa-targets-v2/report.json')
    manifest = seal_contract({'artifact_type': 'eventground_wam.week7_pilot_source',
        'attempt': args.attempt, 'files': {str(p.relative_to(root)): sha256_file(p) for p in sorted(files)}})
    atomic_json(out/'source-manifest.json', manifest)
    with tarfile.open(out/'source.tar.gz', 'w:gz') as tar:
        for path in sorted(files):
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        data = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
        info = tarfile.TarInfo('source-manifest.json')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(json.dumps({'files': len(files), 'digest': manifest['digest'],
                      'archive_sha256': sha256_file(out/'source.tar.gz')}))


if __name__ == '__main__':
    main()
