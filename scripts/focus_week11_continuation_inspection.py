"""Render matched-time saved camera windows selected during assisted review."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from summarize_week11_continuation_inspection import CONDITIONS, sheet
from week11_replay_common import read, sha, write


def run(root):
    start=time.monotonic();cfg=read(root/'bundle/contract.json')
    ann=read(root/'annotations/annotations.json');idx={r['artifact']:r for r in ann['recordings']}
    windows=json.loads((root/'focus-windows.json').read_text());output=root/'focus'
    output.mkdir(exist_ok=False)
    for case in cfg['selection']['pairs']:
        loaded={};steps=windows['windows'][case['id']]
        for c in CONDITIONS:
            path=case['artifacts'][c];receipt=idx[path]
            assert sha(path)==cfg['input_sha256'][path]
            evpath=root/'annotations'/receipt['recording_id']/'evidence.npz'
            assert sha(evpath)==receipt['evidence_sha256']['evidence.npz']
            with h5py.File(path) as f:meta=json.loads(f['episode/metadata_json'][()])
            loaded[c]={'ev':dict(np.load(evpath)),'meta':meta}
            assert max(steps)<len(loaded[c]['ev']['actions'])
        sheet(case,loaded,[steps]*4,'matched-window',output)
    write(output/'completion.json',{'status':'complete','figures_sha256':{p.name:sha(p) for p in sorted(output.glob('*.png'))},
        'source_sha256':sha(__file__),'windows_sha256':sha(root/'focus-windows.json'),'seconds':time.monotonic()-start,
        'model_calls':0,'simulator_steps':0,'new_rollouts':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
