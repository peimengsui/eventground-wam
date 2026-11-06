"""Frame-bounded routing using only the number of successfully executed steps."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from week11_replay_common import read, sha

from eventground_wam.eventground_horizon import select_horizon
from eventground_wam.week11_closing_gate import ADAPTIVE, FIXED, ClosingHistory

ARMS = ('fixed8', 'placement-only', 'late-only', 'adaptive')
WINDOWS = {'placement-only': (138, 218), 'late-only': (218, 298)}


class WindowPolicy:
    def __init__(self, fixed, adaptive, arm):
        if arm not in ARMS:
            raise ValueError('Unknown window arm')
        self.fixed, self.adaptive, self.arm = fixed, adaptive, arm
        self.checkpoint = fixed.checkpoint
        self.reset()

    def reset(self):
        self.history = ClosingHistory()
        self.last_query = None

    def predict_chunk(self, **observation):
        frame = self.history.executed_steps
        if frame < 10 or (frame-10) % 8:
            raise ValueError('Query is not on the frozen eight-action cadence')
        window = WINDOWS.get(self.arm)
        active = self.arm == 'adaptive' or window is not None and window[0] <= frame < window[1]
        chosen = self.adaptive if active else self.fixed
        try:
            return chosen.predict_chunk(**observation)
        finally:
            self.last_query = {**(chosen.last_query or {}), 'window_arm': self.arm,
                              'absolute_frame': frame, 'window_active': active}


def audit_window(row, arm, admission):
    """Reconstruct routing from saved query indices, independent of live history."""
    path = Path(row['artifact'])
    timing = read(path.with_suffix('.timing.json'))
    assert sha(path.with_suffix('.queries.jsonl')) == timing['query_artifact_sha256']
    queries = [json.loads(line) for line in path.with_suffix('.queries.jsonl').read_text().splitlines()]
    assert queries == timing['queries']
    with h5py.File(path) as f:
        n = len(f['steps/executed_action'])
        assert np.isin(f['steps/executed_action'][:, 6], [-1, 1]).all()
    active_frames, changed_frames, horizons = [], [], []
    for i, q in enumerate(queries):
        frame = 10+8*i
        assert q['query_index'] == i and q['chunk_start'] == 8*i and frame < n
        # Intentionally do not use WINDOWS or the live policy's decision here.
        active = (arm == 'adaptive' or arm == 'placement-only' and 138 <= frame < 218
                  or arm == 'late-only' and 218 <= frame < 298)
        p = q['policy']
        assert p['window_arm'] == arm and p['absolute_frame'] == frame and p['window_active'] == active
        assert p['branch_identities'] == admission and p['arm'] == (ADAPTIVE if active else FIXED)
        assert q['exception'] is None and p['exception'] is None
        decision = select_horizon(p['prediction'])
        assert asdict(decision) == p['decision']
        h = decision.selected_horizon if active else 8
        assert p['used_horizon'] == h and p['execute_actions'] == p['proposal_actions'] == 8
        assert p['future_calls'] == 1 and p['zero_padded_actions'] == 24
        for name, v in p['event']['categorical_probabilities'].items():
            assert len(v) == (4 if name == 'event' else 32)
            assert np.isfinite(v).all() and min(v) >= 0 and abs(sum(v)-1) < 1e-5
        horizons.append(h)
        if active:
            active_frames.append(frame)
        if h != 8:
            changed_frames.append(frame)
    return {'valid': True, 'query_count': len(queries), 'active_frames': active_frames,
            'non8_frames': changed_frames, 'horizons': horizons}
