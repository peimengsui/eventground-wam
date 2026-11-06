#!/usr/bin/env python3
"""Build an isolated evaluator preserving every byte of the accepted M3 qualification source for both coupled arms."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.coupled_evaluation import load_contract
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--git', default='git')
    args = parser.parse_args()
    contract = load_contract(ROOT, 'M3')
    original = ROOT / 'docs/evaluation/week8-m3-feifei-qualification-v1/attempt-02'
    manifest = read_sealed(original / 'source-manifest.json')
    require(manifest['digest'] == contract['production_source_digest'], 'original source changed')
    with tarfile.open(original / 'source.tar.gz') as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    for name, digest in manifest['files'].items():
        require(hashlib.sha256(files[name]).hexdigest() == digest, f'original archive changed: {name}')
    design = read_sealed(ROOT / 'configs/training/week8_coupled_design_v1.json')
    week9 = read_sealed(ROOT / 'configs/evaluation/week9_factorial_v1.json')
    from verify_week9_protocol import verify
    verify(ROOT)
    additions = set(contract['source_files_sha256']) | set(design['parents']) | set(week9['source_files_sha256']) | {
        'configs/evaluation/week8_coupled_dev_v1.json', 'configs/evaluation/week9_factorial_v1.json',
        'docs/WEEK8_COUPLED_EVALUATION.md', 'docs/WEEK9_EVALUATION_PROTOCOL.md',
        'scripts/evaluate_coupled_dev.py', 'scripts/package_coupled_evaluation.py',
        'scripts/coupled_evaluation_feifei.sh', 'scripts/verify_week9_protocol.py',
        'src/eventground_wam/auxiliary_evaluation.py', 'src/eventground_wam/coupled_evaluation.py',
        'src/eventground_wam/coupled_inference.py', 'src/eventground_wam/coupled_rollouts.py',
        'src/eventground_wam/factorial_analysis.py', 'tests/test_coupled_evaluation.py',
        'tests/test_factorial_analysis.py', 'tests/test_auxiliary_evaluation.py',
        'tests/test_b0_oft.py', 'tests/test_reference_policy_evaluator.py', 'tests/test_episode_recorder.py',
        'docs/evaluation/sanity-35-reference-policy-v1.json',
        'configs/evaluation/week7_m1_dev_v1.json',
        'docs/evaluation/week7-m1-final-v1/runtime-environment.json'}
    for arm in ('m3-feifei', 'm4-lambda'):
        additions.update(f'docs/evaluation/week8-{arm}-launch-v1/run/{name}'
                         for name in ('training-manifest.json', 'execution-recipe.json'))
    for name in sorted(additions):
        content = (ROOT / name).read_bytes()
        require(name not in files or files[name] == content, f'cannot overwrite accepted source: {name}')
        files[name] = content
    source = seal_contract({'artifact_type': 'eventground_wam.week8_coupled_evaluation_source',
        'contract_digest': contract['digest'], 'week9_protocol_digest': week9['digest'], 'original_source_digest': manifest['digest'],
        'unchanged_original_file_count': len(manifest['files']),
        'base_git_revision': subprocess.check_output([args.git, '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'base_git_revision_scope': 'base commit only; file hashes identify full executed source',
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}})
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'source-manifest.json', source)
    files['evaluation-source-manifest.json'] = (json.dumps(source, indent=2, sort_keys=True) + '\n').encode()
    with tarfile.open(args.output / 'source.tar.gz', 'w:gz') as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    receipt = seal_contract({'artifact_type': 'eventground_wam.week8_coupled_evaluation_package',
                            'source_digest': source['digest'], 'contract_digest': contract['digest'],
                            'archive_sha256': sha256_file(args.output / 'source.tar.gz'),
                            'status': 'prepared_not_launched', 'file_count': len(files)})
    atomic_json(args.output / 'package.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
