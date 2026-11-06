#!/usr/bin/env python3
"""Complete A10 recovery and record qualification under an explicit time amendment."""
from __future__ import annotations

import argparse
import csv
import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path

from auxiliary_lambda_runtime import require, verify_idle_gpu
from train_auxiliary_policy import immutable_json

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.pilot_root, args.pilot_root/'run'
    amendment = read_sealed(args.amendment)
    identity = read_sealed(output/'pilot-identity.json')
    source = read_sealed(root/'source/production-source-manifest.json')
    require(amendment['pilot_identity_digest'] == identity['digest'], 'pilot identity changed')
    require(amendment['runtime_digest'] == identity['runtime_digest'], 'runtime changed')
    require(amendment['pilot_source_digest'] == source['digest']
            == identity['production_source_digest'], 'source identity changed')
    require(amendment['recipe_digest'] == identity['recipe_digest'], 'recipe changed')
    for name, digest in source['files'].items():
        require(sha256_file(root/'source'/name) == digest, f'pilot file changed: {name}')
    correct = read_sealed(output/'correctness.json')
    resources = read_sealed(output/'resources.json')
    require(correct['passed'] and correct['identity_digest'] == identity['digest'], 'correctness failed')
    require(resources['digest'] == amendment['original_resources_digest'], 'resource evidence changed')
    require(resources['identity_digest'] == identity['digest'], 'resource identity changed')
    require(resources['projected_training_hours'] <= amendment['replacement_limits']['maximum_projected_hours'],
            'amended time limit exceeded')
    verify_idle_gpu()
    started = time.monotonic()
    phases = []
    with (output/'orchestrator.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in ('uninterrupted', 'interrupt', 'resume'):
            command = [sys.executable, str(root/'source/scripts/check_auxiliary_lambda.py'),
                '--phase', phase, '--profile', str(root/'runtime.json'), '--output', str(output)]
            log_path, trace_path = output/f'{phase}.log', output/f'{phase}-device.csv'
            with log_path.open('x') as log, trace_path.open('x') as trace:
                monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                    '--format=csv,noheader,nounits', '-lms', '100'], stdout=trace)
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                        timeout=max(1, 3*3600-(time.monotonic()-started)), check=False)
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=10)
            expected = -signal.SIGKILL if phase == 'interrupt' else 0
            require(result.returncode == expected, f'{phase}: exit {result.returncode}')
            readings = list(csv.reader(trace_path.read_text().splitlines()))
            require(len(readings) > 10 and all(len(row) == 2 for row in readings), 'invalid GPU trace')
            require(all(float(used) <= .9*float(total) for total, used in readings), 'GPU ceiling exceeded')
            require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= 24*1024**3,
                    'pilot output ceiling exceeded')
            phases.append({'phase': phase, 'exit_code': result.returncode,
                'log_sha256': sha256_file(log_path), 'trace_sha256': sha256_file(trace_path),
                'whole_device_peak_mib': max(float(row[1]) for row in readings)})
            print(f'{phase} verified', flush=True)
        resumed = read_sealed(output/'jepa-resume.json')
        uninterrupted = read_sealed(output/'jepa-uninterrupted.json')
        require(resumed['state_digest'] == uninterrupted['state_digest'], 'recovery state differs')
        require(resumed['process_resume_bitwise_equal'] and resumed['damaged_latest_fallback'],
                'recovery failed')
        supplement = seal_contract({'artifact_type': 'eventground_wam.lambda_recovery_supplement',
            'identity_digest': identity['digest'], 'budget_amendment_digest': amendment['digest'],
            'orchestrator_sha256': sha256_file(Path(__file__)), 'phases': phases,
            'recovery_digest': resumed['digest'], 'uninterrupted_digest': uninterrupted['digest'],
            'elapsed_seconds': time.monotonic()-started, 'passed': True})
        immutable_json(output/'recovery-supplement.json', supplement)
        immutable_json(output/'qualification.json', seal_contract({
            'artifact_type': 'eventground_wam.lambda_qualification', 'identity_digest': identity['digest'],
            'recipe_digest': identity['recipe_digest'], 'runtime_digest': identity['runtime_digest'],
            'production_source_digest': source['digest'], 'correctness_digest': correct['digest'],
            'resources_digest': resources['digest'], 'recovery_digest': resumed['digest'],
            'recovery_supplement_digest': supplement['digest'], 'budget_amendment_digest': amendment['digest'],
            'original_resource_gate_passed': resources['passed'],
            'passed': True, 'production_started': False,
            'scope': 'A10 qualification under the user-authorized time amendment; original timing failure retained'}))
        print('Amended qualification complete; production not started.', flush=True)


if __name__ == '__main__':
    main()
