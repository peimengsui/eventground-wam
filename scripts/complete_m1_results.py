#!/usr/bin/env python3
"""Finish the authorized M1 analysis/backup after its running evaluation exits."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/home/peimengsui/data/eventground-wam/libero-plus')
EVALUATION = BASE / 'evaluation/week7-m1-dev-v2'
SOURCE = Path('/home/peimengsui/code/eventground-wam-m1-evaluation-v2')
WORK = Path(__file__).resolve().parent


def status(stage, **details):
    value = {'stage': stage, 'time_unix': time.time(), **details}
    temporary = WORK / 'pipeline-state.json.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(WORK / 'pipeline-state.json')
    print(json.dumps(value), flush=True)


def main():
    status('waiting_for_evaluation_completion')
    start = time.monotonic()
    while True:
        result = subprocess.check_output(['systemctl', '--user', 'show', 'eventground-m1-evaluation-v2.service',
                                           '-p', 'ActiveState', '-p', 'Result'], text=True)
        if 'ActiveState=inactive' in result and 'Result=success' in result:
            break
        if 'ActiveState=failed' in result or time.monotonic() - start > 49 * 3600:
            raise RuntimeError(f'evaluation did not complete: {result}')
        time.sleep(20)
    environment = {**os.environ, **json.loads((EVALUATION / 'runtime-environment.json').read_text())}
    status('auditing_all_210_pairs_and_analyzing')
    subprocess.run([sys.executable, str(SOURCE / 'scripts/evaluate_m1_dev_runtime.py'), 'analyze',
                    '--output-root', str(EVALUATION)], env=environment, check=True)
    status('copying_and_verifying_results_backup')
    subprocess.run([sys.executable, str(WORK / 'package_m1_results.py'),
                    '--evaluation-source', str(SOURCE),
                    '--training-root', str(BASE / 'training/week7-m1-auxiliary-v2'),
                    '--evaluation-root', str(EVALUATION),
                    '--output', str(BASE / 'results/week7-m1-v1'),
                    '--evidence', str(WORK / 'supporting-evidence')], env=environment, check=True)
    status('backup_complete_visual_and_narrative_review_pending')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
