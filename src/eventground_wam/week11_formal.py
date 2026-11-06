"""Formal Week 11 scheduling, admission and aggregate/panel resource ledgers."""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_training import sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_comparisons import latency_summary
from eventground_wam.week11_resources import QualificationStopped, ResourceGuard


def schedule(cfg):
    arms = ['M4-fixed', 'M4-EG-fixed8', 'M4-EG-adaptive']
    panels = ['targeted-467', 'clean-200']
    require(cfg['execution']['arm_order'] == arms
            and cfg['execution']['formal_panel_order'] == panels, 'Formal order changed')
    require([cfg['panels'][p]['count'] for p in panels] == [467, 200], 'Formal scope changed')
    return [(arm, panel) for arm in arms for panel in panels]


def relocated_runtime(parent, root, libero):
    """Only the immutable child source path may differ from the pilot runtime."""
    value = copy.deepcopy(parent)
    value.pop('digest')
    value['environment']['PYTHONPATH'] = f'{root}/src:{root}/scripts:{libero}'
    return seal_contract(value)


def projection_admission(cfg, analysis, free_bytes, *, completed=(), active_seconds=0., output_bytes=0):
    require(analysis['engineering_pilot_passed'] and analysis['all_artifacts_valid']
            and analysis['all_cross_arm_inputs_exact'], 'Pilot engineering qualification missing')
    limits = cfg['resources']
    queue = schedule(cfg)
    require(set(completed) <= set(queue), 'Unknown completed panel')
    # Charge the full panel projection until that panel is sealed. Resumption
    # can therefore be more conservative, never more permissive, than launch.
    remaining = [analysis['formal_step_limit_projection'][arm][panel]
                 for arm, panel in queue if (arm, panel) not in completed]
    panel_bytes = sum(p['hdf5_bytes_1p2'] for p in remaining)
    all_panel_bytes = sum(p['hdf5_bytes_1p2'] for ps in analysis['formal_step_limit_projection'].values()
                          for p in ps.values())
    additional = panel_bytes + analysis['formal_additional_bytes_projected_1p2'] - all_panel_bytes
    seconds = (active_seconds + analysis['resources']['active_seconds']
               + sum(p['active_seconds_1p2'] for p in remaining))
    # The original projection also includes fixed load/sidecar overhead.
    all_panel_seconds = sum(p['active_seconds_1p2'] for ps in analysis['formal_step_limit_projection'].values()
                            for p in ps.values())
    seconds += analysis['projected_total_seconds_1p2'] - all_panel_seconds - analysis['resources']['active_seconds']
    total_bytes = analysis['resources']['output_bytes'] + output_bytes + additional
    require(all(p['active_seconds_1p2'] < limits['per_formal_arm_panel_active_seconds'] for p in remaining),
            'Projected panel active time exceeds cap')
    require(seconds < limits['aggregate_active_seconds'], 'Projected aggregate active time exceeds cap')
    require(total_bytes < limits['output_limit_bytes'], 'Projected aggregate output exceeds cap')
    require(free_bytes >= additional + limits['minimum_free_bytes'], 'Insufficient projected disk reserve')
    return seal_contract({'admitted': True, 'analysis_digest': analysis['digest'],
        'remaining_panel_count': len(remaining), 'additional_bytes_projected_1p2': additional,
        'total_output_bytes_projected_1p2': total_bytes, 'total_active_seconds_projected_1p2': seconds,
        'free_bytes': free_bytes, 'reserve_bytes': limits['minimum_free_bytes'],
        'completed_panels': [list(p) for p in completed]})


class FormalResourceGuard(ResourceGuard):
    """Reuse the qualified watchdog; add prior qualification and panel charges."""
    def __init__(self, output, identity, limits, qualification_resources):
        self.formal_limits = limits
        self.qualification_seconds = qualification_resources['active_seconds']
        self.qualification_bytes = qualification_resources['output_bytes']
        self.panel_lock = threading.RLock()
        self.panel_budget = None
        self.panel_key = None
        self.identity = identity
        adjusted = {**limits, 'output_limit_bytes': limits['output_limit_bytes'] - self.qualification_bytes}
        super().__init__(output, identity, adjusted)

    def set_panel(self, arm, panel):
        with self.panel_lock:
            key = f'{arm}/{panel}'
            if key == self.panel_key:
                return
            if self.panel_budget is not None:
                self.panel_budget.close()
            self.panel_key = key
            self.panel_budget = ActiveBudget(self.output/'panel-budgets'/f'{key}.json', self.identity+'/'+key)
        self.check()

    def check(self):
        if self.reason:
            raise QualificationStopped(self.reason)
        if self.budget.seconds + self.qualification_seconds >= self.formal_limits['aggregate_active_seconds']:
            raise QualificationStopped('Aggregate qualification + formal active-time ceiling')
        with self.panel_lock:
            if (self.panel_budget is not None
                    and self.panel_budget.seconds >= self.formal_limits['per_formal_arm_panel_active_seconds']):
                raise QualificationStopped('Formal arm/panel active-time ceiling: '+self.panel_key)

    def snapshot(self):
        value = super().snapshot()
        with self.panel_lock:
            value.update(qualification_active_seconds=self.qualification_seconds,
                total_active_seconds=value['active_seconds']+self.qualification_seconds,
                qualification_output_bytes=self.qualification_bytes,
                total_output_bytes=value['output_bytes']+self.qualification_bytes,
                current_panel=self.panel_key,
                panel_active_seconds=self.panel_budget.seconds if self.panel_budget else None)
        return value

    def close(self):
        # Stop the watchdog before closing either heartbeat ledger.
        result = super().close()
        with self.panel_lock:
            if self.panel_budget is not None:
                self.panel_budget.close()
        return result


def all_attempt_latencies(panel_root, identity):
    """Include first, failed-policy and interrupted/retried attempt query costs."""
    attempts, bindings = [], {}
    for path in sorted(Path(panel_root).glob('episode-*-attempt-*.queries.jsonl')):
        queries = [json.loads(line) for line in path.read_text().splitlines()]
        require(all(q['cohort_digest'] == identity['cohort_digest']
                    and q['admission_digest'] == identity['admission_digest'] for q in queries),
                'Attempt query identity differs')
        require([q['query_index'] for q in queries] == list(range(len(queries))), 'Attempt query order differs')
        attempts.append([q['latency_ms'] for q in queries])
        bindings[path.name] = sha256_file(path)
    return {'latency': latency_summary(attempts), 'attempt_query_files_sha256': bindings,
            'includes_interrupted_and_retried_attempts': True, 'no_queries_excluded': True}
