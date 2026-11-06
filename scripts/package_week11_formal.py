"""Extend the executed pilot archive without replacing any qualified source byte."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
ADDITIONS = (
    'scripts/run_week11_formal.py', 'scripts/package_week11_formal.py',
    'src/eventground_wam/week11_formal.py', 'tests/test_week11_formal.py',
    'docs/WEEK11_FORMAL_EXECUTION.md',
    'docs/evaluation/week11-integration-v1/analysis.json',
    'docs/evaluation/week11-integration-v1/remote/artifact-inventory.json',
    'docs/evaluation/week11-integration-v1/remote/numerical-qualification.json',
    'docs/evaluation/week11-integration-v1/remote/cohort.json',
)


def build(output):
    parent = ROOT/'docs/evaluation/week11-integration-v1/package-v2'
    accepted = read_sealed(parent/'source-manifest.json')
    receipt = read_sealed(parent/'package.json')
    require(accepted['digest'] == 'ca0d0836a6bedf95672ed06554e3a979fb78b0f4a4555f7bf752fe3de82423a9'
            and sha256_file(parent/'source.tar.gz') == receipt['archive_sha256'], 'Executed pilot package changed')
    with tarfile.open(parent/'source.tar.gz') as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    require(set(files) == set(accepted['files']) | {'week11-source-manifest.json'}, 'Pilot archive inventory changed')
    for name, expected in accepted['files'].items():
        require(hashlib.sha256(files[name]).hexdigest() == expected, 'Qualified source changed: '+name)
    require(json.loads(files['week11-source-manifest.json']) == accepted, 'Pilot source manifest differs')
    for name in ADDITIONS:
        require(name not in files, 'Child would overwrite qualified source: '+name)
        files[name] = (ROOT/name).read_bytes()
    source = seal_contract({'artifact_type': 'eventground_wam.week11_formal_source',
        'base_git_revision': accepted['base_git_revision'], 'qualified_source_digest': accepted['digest'],
        'qualified_archive_sha256': receipt['archive_sha256'],
        'parent_digest': accepted['parent_digest'], 'mapping_digest': accepted['mapping_digest'],
        'authorization': 'User requested continuing and finishing the formal evaluation launch',
        'formal_execution_authorized': True, 'formal_episodes': 2001,
        'qualified_source_replaced': False, 'additions': list(ADDITIONS),
        'files': {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}})
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'source-manifest.json', source)
    files['week11-formal-source-manifest.json'] = (json.dumps(source, sort_keys=True, indent=2)+'\n').encode()
    with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
        for name, content in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(content), 0o644
            archive.addfile(entry, io.BytesIO(content))
    receipt = seal_contract({'source_digest': source['digest'], 'file_count': len(files),
        'archive_sha256': sha256_file(output/'source.tar.gz'),
        'archive_bytes': (output/'source.tar.gz').stat().st_size})
    atomic_json(output/'package.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output)))
