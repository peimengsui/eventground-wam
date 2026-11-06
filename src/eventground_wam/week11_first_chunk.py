"""First-post-close chunk interventions; model inference and mapping stay frozen."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.eventground_horizon import select_horizon
from eventground_wam.week11_closing_gate import ADAPTIVE, FIXED, GATED, ClosingHistory

SKIP = 'M4-EG-skip-first'
ONLY = 'M4-EG-first-only'
ARMS = (FIXED, GATED, SKIP, ONLY)


class FirstChunkPolicy:
    def __init__(self, fixed, adaptive, arm):
        if arm not in ARMS:
            raise ValueError('Unknown first-chunk arm')
        self.fixed, self.adaptive, self.arm = fixed, adaptive, arm
        self.checkpoint = fixed.checkpoint
        self.reset()

    def reset(self):
        self.history = ClosingHistory()
        self.active_queries = 0
        self.last_query = None

    def predict_chunk(self, *, agent_rgb, wrist_rgb, proprioception, instruction, suite):
        active, index = self.history.active, self.active_queries
        adaptive = active and (self.arm == GATED or self.arm == SKIP and index > 0
                               or self.arm == ONLY and index == 0)
        chosen = self.adaptive if adaptive else self.fixed
        history = self.history.snapshot()
        error = None
        try:
            actions = chosen.predict_chunk(agent_rgb=agent_rgb, wrist_rgb=wrist_rgb,
                proprioception=proprioception, instruction=instruction, suite=suite)
            # Frozen cases all forecast h4 at this anchor. Do not silently substitute a different intervention.
            if active and index == 0:
                require(chosen.last_query['decision']['selected_horizon'] == 4,
                        'First post-close forecast no longer selects the frozen horizon 4')
            if active:
                self.active_queries += 1
            return actions
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.last_query = {**(chosen.last_query or {}), 'comparison_arm': self.arm,
                'closing_history': history, 'post_close_query_index': index if active else None}
            if error is not None:
                self.last_query['exception'] = error


def audit_first_chunk(row, arm, admission, settling_steps=10):
    """Independent reconstruction uses the executed command stream, never live counters."""
    if not row['artifact']:
        return {'valid': False, 'reason': 'missing recording'}
    path = Path(row['artifact'])
    timing = read_sealed(path.with_suffix('.timing.json'))
    query_path = path.with_suffix('.queries.jsonl')
    require(sha256_file(query_path) == timing['query_artifact_sha256'], 'Query log changed')
    queries = [json.loads(line) for line in query_path.read_text().splitlines()]
    require(queries == timing['queries'], 'Query/timing mismatch')
    horizons, active_queries = [], []
    with h5py.File(path) as handle:
        actions = handle['steps/executed_action'][:]
        commands = actions[:, 6]
        require(np.isin(commands, [-1, 1]).all(), 'Nonbinary recorded commands')
        transitions = np.flatnonzero((commands[:-1] == -1) & (commands[1:] == 1))+1
        first = int(transitions[0]) if len(transitions) else None
        require(first is None or first >= settling_steps, 'Settling unexpectedly closes')
        first_query = (first-settling_steps)//8+1 if first is not None else None
        for i, query in enumerate(queries):
            step = settling_steps+8*i
            require(query['query_index'] == i and query['chunk_start'] == 8*i,
                    'Query cadence changed')
            require(step <= len(actions), 'Query after end of recording')
            seen = first if first is not None and first < step else None
            expected = {'executed_steps': step, 'previous_command': int(commands[step-1]),
                        'first_closing_step': seen, 'active': seen is not None}
            p = query['policy']
            index = i-first_query if seen is not None else None
            require(p['closing_history'] == expected, 'Gate used unexecuted or stale commands')
            require(p['post_close_query_index'] == index, 'Wrong post-close query index')
            require(p['comparison_arm'] == arm and p['branch_identities'] == admission,
                    'Wrong intervention identities')
            use_adaptive = False
            if index is not None:
                use_adaptive = arm == GATED or (arm == SKIP and index >= 1) or (arm == ONLY and index == 0)
            effective = ADAPTIVE if use_adaptive else FIXED
            require(p['arm'] == effective and p['exception'] == query['exception'],
                    'Wrong effective arm or exception')
            if query['exception'] is not None:
                continue
            decision = select_horizon(p['prediction'])
            require(asdict(decision) == p['decision'], 'Mapping changed')
            if index == 0:
                require(decision.selected_horizon == 4, 'First intervention forecast changed')
            horizon = decision.selected_horizon if use_adaptive else 8
            require(p['used_horizon'] == horizon and p['future_calls'] == 1
                    and p['proposal_actions'] == p['execute_actions'] == 8
                    and p['zero_padded_actions'] == 24, 'Routing/execution contract changed')
            for name, values in p['event']['categorical_probabilities'].items():
                require(len(values) == (4 if name == 'event' else 32) and np.isfinite(values).all()
                        and min(values) >= 0 and abs(sum(values)-1) < 1e-5, 'Invalid event probabilities')
            horizons.append(horizon)
            if seen is not None:
                active_queries.append(i)
    return {'valid': True, 'first_closing_step': first,
            'first_query_after_closing': active_queries[0] if active_queries else None,
            'query_count': len(queries), 'horizons': horizons,
            'query_exceptions': [q['exception'] for q in queries if q['exception']],
            'active_query_count': len(active_queries)}
