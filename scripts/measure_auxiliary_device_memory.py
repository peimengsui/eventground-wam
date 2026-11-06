#!/usr/bin/env python3
"""Bounded device-level memory supplement using the verified four-update replay."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = read_sealed(args.source/'source-manifest.json')
    plan = seal_contract({'artifact_type': 'eventground_wam.week7_device_memory_plan',
        'source_digest': source['digest'], 'methods': ['pixel', 'jepa'], 'updates_per_method': 4,
        'selection': 'same first four complete logical batches as the frozen recovery test',
        'interval_milliseconds': 100, 'maximum_seconds_per_method': 600,
        'maximum_device_memory_fraction': .9, 'weights_discarded': True,
        'purpose': 'record NVML whole-device memory through forward/backward/Adam, complementing allocator peaks',
        'script_sha256': sha256_file(Path(__file__))})
    atomic_json(args.output/'plan.json', plan)
    results = {}
    for method in plan['methods']:
        shutil.copy2(args.primary/f'{method}-calibration.json', args.output/f'{method}-calibration.json')
        trace = args.output/f'{method}-device.csv'
        with trace.open('x') as stats, (args.output/f'{method}-replay.log').open('x') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.total,memory.used',
                '--format=csv,noheader,nounits', '-lms', str(plan['interval_milliseconds'])], stdout=stats)
            try:
                subprocess.run([sys.executable, str(args.source/'scripts/check_auxiliary_training.py'),
                    '--phase', 'uninterrupted', '--method', method, '--output', str(args.output)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        rows = list(csv.reader(trace.read_text().splitlines()))
        assert len(rows) > 10 and all(len(row) == 3 for row in rows)
        total = int(rows[0][1])
        assert all(int(row[1]) == total for row in rows)
        peak = max(int(row[2]) for row in rows)
        assert peak < .9*total
        replay = read_sealed(args.output/f'{method}-uninterrupted.json')
        reference = read_sealed(args.primary/f'{method}-uninterrupted.json')
        assert replay['state_digest'] == reference['state_digest']
        resource = read_sealed(args.primary/f'{method}-probe-report.json')
        checkpoint = args.primary/f'{method}-resource-recovery'/resource['recovery_record']['checkpoint']
        policy = args.primary/f'{method}-pilot-action-only.pt'
        assert sha256_file(checkpoint) == resource['recovery_record']['sha256']
        assert sha256_file(policy) == resource['action_only_export_sha256']
        results[method] = {'device_total_mib': total, 'peak_device_used_mib': peak,
            'samples': len(rows), 'trace_sha256': sha256_file(trace),
            'state_digest': replay['state_digest'], 'matches_primary_replay': True,
            'checkpoint_bytes': checkpoint.stat().st_size, 'policy_bytes': policy.stat().st_size,
            'three_checkpoints_two_exports_bytes': 3*checkpoint.stat().st_size+2*policy.stat().st_size,
            'remote_checkpoint_and_export_hashes_reverified': True}
        print(json.dumps({'method': method, **results[method]}), flush=True)
    atomic_json(args.output/'report.json', seal_contract({
        'artifact_type': 'eventground_wam.week7_device_memory_report', 'plan_digest': plan['digest'],
        'source_digest': source['digest'], 'results': results, 'status': 'passed',
        'scope': 'NVML whole-device samples at 100ms through four full updates per arm; allocator peaks cover the main 128-update and edge pilots',
        'primary_pilot_storage_bytes': sum(p.stat().st_size for p in args.primary.rglob('*') if p.is_file()),
        'primary_storage_below_24_gib': sum(p.stat().st_size for p in args.primary.rglob('*') if p.is_file()) < 24*1024**3,
        'free_disk_bytes': shutil.disk_usage(args.output).free,
        'free_disk_above_50_gib': shutil.disk_usage(args.output).free > 50*1024**3}))


if __name__ == '__main__':
    main()
