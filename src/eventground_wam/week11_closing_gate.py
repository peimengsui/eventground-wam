"""Diagnostic routing from past executed commands; original inference stays intact."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.eventground_horizon import select_horizon

FIXED = 'M4-EG-fixed8'
ADAPTIVE = 'M4-EG-adaptive'
GATED = 'M4-EG-after-close'
ARMS = (FIXED, ADAPTIVE, GATED)


@dataclass
class ClosingHistory:
    """Step indices include settling. A transition is observed only after step returns."""
    executed_steps: int = 0
    previous_command: int | None = None
    first_closing_step: int | None = None

    def observe(self, action):
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all() or action[6] not in (-1, 1):
            raise ValueError('Expected a finite decoded binary-gripper action')
        command = int(action[6])
        if self.first_closing_step is None and self.previous_command == -1 and command == 1:
            self.first_closing_step = self.executed_steps
        self.previous_command = command
        self.executed_steps += 1

    @property
    def active(self):
        return self.first_closing_step is not None

    def snapshot(self):
        return {**asdict(self), 'active': self.active}


class ClosingGatePolicy:
    """Select between original wrappers without changing their prediction or action paths."""
    def __init__(self, fixed, adaptive, arm):
        if arm not in ARMS:
            raise ValueError('Unknown comparison arm')
        self.fixed, self.adaptive, self.arm = fixed, adaptive, arm
        self.checkpoint = fixed.checkpoint
        self.reset()

    def reset(self):
        self.history = ClosingHistory()
        self.last_query = None

    def predict_chunk(self, *, agent_rgb, wrist_rgb, proprioception, instruction, suite):
        active = self.history.active
        chosen = self.adaptive if self.arm == ADAPTIVE or self.arm == GATED and active else self.fixed
        history = self.history.snapshot()
        try:
            return chosen.predict_chunk(agent_rgb=agent_rgb, wrist_rgb=wrist_rgb,
                proprioception=proprioception, instruction=instruction, suite=suite)
        finally:
            self.last_query = {**chosen.last_query, 'comparison_arm': self.arm, 'closing_history': history}


class ExecutedCommandObserver:
    """Do not update history on proposed actions, observations, or failed environment steps."""
    def __init__(self, environment, history):
        self.environment, self.history = environment, history

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def step(self, action):
        result = self.environment.step(action)
        self.history.observe(action)
        return result


def run_gated_episode(*, environment, policy, **kwargs):
    from eventground_wam.week11_recording import run_episode
    policy.reset()  # Including infrastructure retries: no state may leak between attempts.
    return run_episode(environment=ExecutedCommandObserver(environment, policy.history),
                       policy=policy, **kwargs)


def audit_gate(row, arm, admission, settling_steps=10):
    """Reconstruct gate state from recorded executed actions, independently of live history."""
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
        for i, query in enumerate(queries):
            step = settling_steps+8*i
            require(query['query_index'] == i and query['chunk_start'] == 8*i,
                    'Query cadence changed')
            require(step <= len(actions), 'Query after end of recording')
            seen = first if first is not None and first < step else None
            expected = {'executed_steps': step, 'previous_command': int(commands[step-1]),
                        'first_closing_step': seen, 'active': seen is not None}
            p = query['policy']
            require(p['closing_history'] == expected, 'Gate used unexecuted or stale commands')
            require(p['comparison_arm'] == arm and p['branch_identities'] == admission,
                    'Wrong gate identities')
            original_arm = ADAPTIVE if arm == ADAPTIVE or arm == GATED and seen is not None else FIXED
            require(p['arm'] == original_arm and p['exception'] == query['exception'],
                    'Wrong effective arm or exception')
            if query['exception'] is not None:
                continue
            decision = select_horizon(p['prediction'])
            require(asdict(decision) == p['decision'], 'Mapping changed')
            horizon = 8 if original_arm == FIXED else decision.selected_horizon
            require(p['used_horizon'] == horizon and p['future_calls'] == 1
                    and p['proposal_actions'] == p['execute_actions'] == 8
                    and p['zero_padded_actions'] == 24, 'Routing/execution contract changed')
            for name, values in p['event']['categorical_probabilities'].items():
                require(len(values) == (4 if name == 'event' else 32) and np.isfinite(values).all()
                        and min(values) >= 0 and abs(sum(values)-1) < 1e-5,
                        'Invalid event probabilities')
            horizons.append(horizon)
            if seen is not None:
                active_queries.append(i)
    return {'valid': True, 'first_closing_step': first,
            'first_query_after_closing': active_queries[0] if active_queries else None,
            'query_count': len(queries), 'horizons': horizons,
            'query_exceptions': [q['exception'] for q in queries if q['exception']],
            'active_query_count': len(active_queries)}
