"""Post-hoc matched-time saved camera sheets for three adapter mechanisms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from summarize_week11_motion_inspection import CONDITIONS, sheet
from week11_replay_common import read, sha, write

WINDOWS = {'targeted-467-0231': [90, 122, 146, 162, 174, 180],
           'clean-200-0107': [134, 145, 158, 162, 178, 188],
           'clean-200-0151': [194, 229, 288, 298, 321, 342]}


def run(root):
    cfg = read(root/'inspection-bundle/contract.json')
    ann = read(root/'annotations/annotations.json'); indexed = {r['artifact']: r for r in ann['recordings']}
    output = root/'inspection-focus'; output.mkdir(exist_ok=False)
    for case in cfg['selection']['pairs']:
        loaded = {}; steps = WINDOWS[case['id']]
        for arm, path in case['artifacts'].items():
            assert sha(path) == cfg['input_sha256'][path]
            r = indexed[path]; ev = root/'annotations'/r['recording_id']/'evidence.npz'
            assert sha(ev) == r['evidence_sha256']['evidence.npz']
            with h5py.File(path) as f:
                meta = json.loads(f['episode/metadata_json'][()])
            loaded[arm] = {'meta': meta, 'ev': dict(np.load(ev))}
            assert max(steps) < len(loaded[arm]['ev']['actions'])
        sheet(case, loaded, [steps]*len(CONDITIONS), 'matched-window', output)
    write(output/'completion.json', {'status': 'complete', 'windows': WINDOWS,
        'source_sha256': sha(__file__), 'figures_sha256': {p.name: sha(p) for p in sorted(output.glob('*.png'))},
        'selection': 'Post-hoc physical windows for review; no deployment gate or training targets.',
        'model_calls': 0, 'new_simulator_steps': 0, 'new_render_calls': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); run(p.parse_args().root)
