"""Package an additive analysis child of the executed formal source."""
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
ADDITIONS = ('src/eventground_wam/week11_analysis.py', 'scripts/analyze_week11_formal.py',
             'scripts/package_week11_closeout.py', 'tests/test_week11_analysis.py',
             'docs/WEEK11_CLOSEOUT_METHOD.md')


def build(output):
    parent = ROOT/'docs/evaluation/week11-formal-v1/package-v1'
    source = read_sealed(parent/'source-manifest.json')
    package = read_sealed(parent/'package.json')
    require(source['digest'] == '9330bb56d0c31ebaf4c9d90b953948fb45c923e5a1a8ad2e997e763fc7f73328'
            and sha256_file(parent/'source.tar.gz') == package['archive_sha256'], 'Formal package differs')
    with tarfile.open(parent/'source.tar.gz') as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    require(set(files) == set(source['files']) | {'week11-formal-source-manifest.json'}, 'Parent inventory differs')
    for name, expected in source['files'].items():
        require(hashlib.sha256(files[name]).hexdigest() == expected, 'Parent source byte differs: '+name)
    for name in ADDITIONS:
        require(name not in files, 'Analysis would replace executed source: '+name)
        files[name] = (ROOT/name).read_bytes()
    manifest = seal_contract({'artifact_type': 'eventground_wam.week11_closeout_source',
        'formal_source_digest': source['digest'], 'formal_archive_sha256': package['archive_sha256'],
        'files_sha256': {name: hashlib.sha256(files[name]).hexdigest() for name in ADDITIONS},
        'inherited_source_replaced': False, 'authorization': 'User requested completing formal analysis and fixed-control audit',
        'new_training_or_rollouts': 0})
    files['analysis-source-manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'source-manifest.json', manifest)
    with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
        for name, content in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(content), 0o644
            archive.addfile(entry, io.BytesIO(content))
    receipt = seal_contract({'source_digest': manifest['digest'], 'file_count': len(files),
        'archive_sha256': sha256_file(output/'source.tar.gz'), 'archive_bytes': (output/'source.tar.gz').stat().st_size})
    atomic_json(output/'package.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
