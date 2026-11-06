"""One diagnostic, two disposable qualifications, one pilot and one control."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_preservation_runtime import verify, selected_weight, validate_training_completion


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    base = parser.parse_args().base; p, _, _, _ = verify(base); start = time.monotonic(); done = []
    def save(name, value):
        with (base/name).open('x') as f:
            json.dump(seal_contract(value), f, indent=2, sort_keys=True); f.write('\n')
    save('execution-start.json', dict(protocol_digest=p['digest'], started_at=datetime.now(timezone.utc).isoformat(), automatic_retries=0))
    def command(script, *args, limit):
        remaining = p['queue_wall_seconds']-(time.monotonic()-start)
        if remaining <= 0:
            raise TimeoutError('Aggregate queue ceiling')
        key = script+':'+':'.join(args)
        state = dict(protocol_digest=p['digest'], current=key, completed=done, elapsed_seconds=time.monotonic()-start)
        temp = base/'queue-status.tmp'; temp.write_text(json.dumps(state, indent=2)+'\n'); os.replace(temp, base/'queue-status.json')
        subprocess.run([sys.executable, str(base/'source/scripts'/script), '--base', str(base), *args],
                       check=True, timeout=min(limit, remaining))
        done.append(key)
    status = 'failed_preserved'; error = None
    try:
        command('diagnose_week10_preservation.py', limit=p['diagnostic_resources']['wall_seconds']+120)
        selected_weight(base)
        for stage in ('hybrid-qualification', 'control-qualification', 'hybrid', 'control'):
            command('train_week10_preservation.py', '--stage', stage, limit=p['training_stages'][stage]['wall_seconds']+120)
            validate_training_completion(base, stage)
        for arm in ('hybrid', 'control'):
            command('evaluate_week10_preservation.py', '--arm', arm, '--panel', 'dev', limit=p['inference_resources']['wall_seconds']+120)
        command('analyze_week10_preservation.py', limit=300)
        status = 'complete_pilot_only'
    except Exception as exc:
        error = repr(exc); raise
    finally:
        save('execution-completion.json', dict(protocol_digest=p['digest'], status=status, error=error, completed=done,
             elapsed_seconds=time.monotonic()-start, finished_at=datetime.now(timezone.utc).isoformat(),
             automatic_retries=0, full_training_launched=False, report_panel_opened=False))


if __name__ == '__main__':
    main()
