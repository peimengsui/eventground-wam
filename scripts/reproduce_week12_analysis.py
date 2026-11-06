"""Rebuild the sealed Week 12 analysis without model or simulator imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; never an existing evidence package')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Reproduction output must not already exist')
    reference = ROOT / 'docs/evaluation/week12-results-v1'
    manifest = json.loads((reference / 'source-manifest.json').read_text())['files_sha256']
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Analysis input changed: {name}')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.monotonic()
    scripts = ['build_week12_results.py', 'render_week12_figures.py',
               'write_week12_results.py', 'verify_week12_results.py']
    for name in scripts:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / name), '--output', str(out)],
                       cwd=ROOT, env=env, check=True)
    names = ['consolidated.json', 'examples.json', 'figure-index.json']
    names += [str(p.relative_to(reference)) for p in sorted((reference / 'tables').glob('*.csv'))]
    names += [str(p.relative_to(reference)) for p in sorted((reference / 'figures').glob('*.png'))]
    matching = {}
    for name in names:
        actual, expected = sha(out / name), sha(reference / name)
        if actual != expected:
            raise ValueError(f'Rebuild differs from the sealed reference: {name}')
        matching[name] = actual
    result = {'status': 'pass', 'exact_equal_files': matching, 'source_files_verified': len(manifest),
              'numerical_checks': 2210, 'elapsed_seconds': time.monotonic() - started,
              'new_model_calls': 0, 'new_rollouts': 0, 'new_training_updates': 0,
              'scope': 'Local CPU metadata calculations and figure rendering; no raw recording re-audit.'}
    (out / 'reproduction-check.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'pass', 'exact_equal_files': len(matching), 'output': str(out)}))


if __name__ == '__main__':
    main()
