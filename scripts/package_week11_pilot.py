"""Build a checksum-bound child source package without editing either freeze."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from verify_week11_horizon import verify

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT)
    parent = ROOT/'docs/evaluation/week8-coupled-evaluator-v1/package-v4'
    accepted = read_sealed(parent/'source-manifest.json')
    with tarfile.open(parent/'source.tar.gz') as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    for name, digest in accepted['files'].items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError('Accepted evaluator package changed: '+name)
    pending = []
    for directory in ('src', 'scripts', 'tests', 'configs', 'evaluation'):
        pending.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for directory in ('docs/evaluation/week11-freeze-v1', 'docs/evaluation/week11-horizon-v1'):
        pending.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*') if p.is_file())
    pending += ['docs/evaluation/week7-m1-final-v1/runtime-environment.json',
                'docs/evaluation/week8-coupled-evaluator-v1/package-v4/source-manifest.json',
                'docs/WEEK11_INTEGRATION.md', 'pyproject.toml']
    pending.extend(files)
    seen = set()

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = ROOT/name
        if path.is_file():
            files[name] = path.read_bytes()
        if name.endswith('.json'):
            value = json.loads(files[name])
            for candidate in strings(value):
                if (candidate.startswith(('src/', 'scripts/', 'tests/', 'configs/', 'docs/',
                                          'evaluation/', 'environments/'))
                        and len(candidate) < 500 and all(len(p) < 255 for p in candidate.split('/'))
                        and (ROOT/candidate).is_file()):
                    pending.append(candidate)
    mapping = read_sealed(ROOT/'configs/evaluation/week11_horizon_mapping_v1.json')
    source = seal_contract({'artifact_type': 'eventground_wam.week11_pilot_source',
        'base_git_revision': accepted['base_git_revision'],
        'parent_digest': mapping['parent']['digest'], 'mapping_digest': mapping['digest'],
        'mapping_package_digest': read_sealed(ROOT/'docs/evaluation/week11-horizon-v1/package-index.json')['digest'],
        'authorization': 'Steps 3 and 4: integration, correctness qualification and 24-episode bounded pilot',
        'formal_execution_authorized': False,
        'files': {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}})
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'source-manifest.json', source)
    files['week11-source-manifest.json'] = (json.dumps(source, sort_keys=True, indent=2)+'\n').encode()
    with tarfile.open(args.output/'source.tar.gz', 'w:gz') as archive:
        for name, content in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(content))
    receipt = seal_contract({'source_digest': source['digest'], 'file_count': len(files),
        'archive_sha256': sha256_file(args.output/'source.tar.gz'),
        'archive_bytes': (args.output/'source.tar.gz').stat().st_size})
    atomic_json(args.output/'package.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
