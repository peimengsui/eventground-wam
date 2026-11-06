"""Verify completed replay/annotation evidence and original source preservation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from week11_replay_common import read, sha, write


def run(output):
    import run_week11_formal as formal
    start = time.monotonic()
    root = Path('/home/peimengsui/data/eventground-wam/libero-plus/evaluation')
    replay_root = root/'week11-horizon-replay-v1'
    annotation_root = root/'week11-horizon-annotations-v4'
    c = read(replay_root/'contract.json'); r = read(replay_root/'replay.json')
    assert read(replay_root/'completion.json')['replay_digest'] == r['digest']
    a = read(annotation_root/'annotations.json')
    assert read(annotation_root/'completion.json')['annotation_digest'] == a['digest']
    assert formal.package_check()['digest'] == c['formal_source_digest']
    for p, digest in c['input_sha256'].items():
        assert sha(p) == digest, p
    for row in r['observations']:
        assert sha(replay_root/row['tensor_file']) == row['tensor_sha256']
    evidence = 0
    for row in a['recordings']:
        d = annotation_root/row['recording_id']
        assert read(d/'candidate.json') == row['candidate']
        for name, digest in row['evidence_sha256'].items():
            assert sha(d/name) == digest; evidence += 1
    cases = []
    for rid in ('targeted-467-0001-M4-EG-fixed8-q001', 'targeted-467-0356-M4-EG-adaptive-q008',
                'clean-200-0151-M4-EG-adaptive-q006'):
        row = next(x for x in r['observations'] if x['id'] == rid)
        z = np.load(replay_root/row['tensor_file'])
        cases.append({'id': rid, 'gripper_commands': {str(h): z[f'h{h}_decoded_actions'][:, 6].tolist() for h in (2, 4, 8, 16, 32)}})
    failures = [read(root/f'week11-horizon-annotations-v{v}/completion.json') for v in (1, 2, 3)]
    result = {'status': 'complete', 'formal_source_unchanged': True,
        'original_input_files_rehashed': len(c['input_sha256']), 'tensors_rehashed': len(r['observations']),
        'annotation_evidence_rehashed': evidence, 'failed_annotation_attempts': failures,
        'example_gripper_commands': cases, 'seconds': time.monotonic()-start}
    write(output, result); print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
