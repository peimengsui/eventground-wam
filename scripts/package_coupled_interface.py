#!/usr/bin/env python3
"""Preserve the accepted Week 7 package and add the complete coupling dependency set."""
from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
from pathlib import Path

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.coupled_setup import load_coupled_design
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
ADDITIONS = (
    'src/eventground_wam/future_core.py', 'src/eventground_wam/coupled_policy.py',
    'src/eventground_wam/coupled_checkpoint.py', 'src/eventground_wam/coupled_setup.py',
    'scripts/check_coupled_interface.py', 'scripts/package_coupled_interface.py',
    'tests/test_coupled_policy.py', 'configs/training/week8_coupled_design_v1.json',
    'docs/WEEK8_COUPLED_DESIGN.md',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'source.tar.gz').exists():
        raise FileExistsError(args.output/'source.tar.gz')
    design = load_coupled_design(ROOT)
    parent = ROOT/'docs/evaluation/week7-m1-launch-v2'
    manifest = read_sealed(parent/'source-manifest.json')
    files = {}
    with tarfile.open(parent/'source.tar.gz') as archive:
        for name, digest in manifest['files'].items():
            data = archive.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest() == digest, name
            if (ROOT/name).exists():
                assert sha256_file(ROOT/name) == digest, name
            files[name] = data
        files['source-manifest.json'] = archive.extractfile('source-manifest.json').read()
    additions = sorted(set(ADDITIONS) | set(design['parents']))
    for name in additions:
        data = (ROOT/name).read_bytes()
        if name in files:
            assert data == files[name], name
        files[name] = data
    assert set(design['parents']) <= set(files)
    receipt = seal_contract({'artifact_type': 'eventground_wam.week8_coupling_interface_source',
        'parent_source_digest': manifest['digest'], 'accepted_parent_files_preserved': len(manifest['files']),
        'design_digest': design['digest'],
        'files': {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())},
        'scope': 'Shared interface and bounded smoke only; not production or resource admission.'})
    atomic_json(args.output/'source-manifest.json', receipt)
    files['coupled-source-manifest.json'] = (args.output/'source-manifest.json').read_bytes()
    with tarfile.open(args.output/'source.tar.gz', 'w:gz') as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    record = seal_contract({'source_digest': receipt['digest'],
        'archive_sha256': sha256_file(args.output/'source.tar.gz'),
        'accepted_parent_files_preserved': len(manifest['files']),
        'all_design_parent_configs_included': True, 'production_training_started': False})
    atomic_json(args.output/'package.json', record)
    print(record)


if __name__ == '__main__':
    main()
