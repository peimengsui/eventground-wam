"""Derive report tables from the sealed recording-only extraction."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from investigate_week11_adaptive import ARMS, PANELS, describe, read_sealed, sha, write


def summarize(root, output):
    completion = read_sealed(root/'completion.json')
    for name, digest in completion['files_sha256'].items():
        assert sha(root/name) == digest, name
    rows = [json.loads(line) for line in (root/'pairs.jsonl').read_text().splitlines()]
    result = {}
    losses = []
    for panel in PANELS:
        rr = [r for r in rows if r['panel'] == panel]
        groups = {}
        for group in ('adaptive_loss', 'adaptive_win', 'both_success', 'both_failure', 'all'):
            selected = rr if group == 'all' else [r for r in rr if r['outcome_class'] == group]
            comparable = [r for r in selected if (d := r['divergence']['first_non8_at_shared_prefix']) and d['inputs_equal']]
            contexts = [r['divergence']['first_non8_at_shared_prefix'] for r in comparable]
            assert all(d['previous_executed_actions_equal'] and d['simulator_state_equal'] for d in contexts)
            assert all(r['divergence']['first_action_difference']['query'] ==
                       r['divergence']['first_non8_query'] for r in comparable)
            assert all(r['arms'][ARMS[0]]['queries'][d['query']]['event'] == d['event']
                       and r['arms'][ARMS[0]]['queries'][d['query']]['start'] == d['start']
                       for r, d in zip(comparable, contexts, strict=True))
            groups[group] = {'episodes': len(selected), 'shared_input_departures': len(comparable),
                'first_non8_horizon': dict(Counter(str(d['used']) for d in contexts)),
                'first_non8_policy_step': describe([8*d['query'] for d in contexts]),
                'first_non8_translation_rms': describe([d['action']['translation_rms'] for d in contexts]),
                'first_non8_gripper_changes': sum(d['action']['gripper_disagree_actions'] > 0 for d in contexts),
                'event_type_and_onset_equal_across_arms_at_departure': True,
                'simulator_state_and_all_prior_actions_equal_at_departure': True,
                'first_action_change_is_first_horizon_change': True}
            if group == 'all':
                groups[group]['forecast_closing_at_departure'] = {}
                for arm in ARMS:
                    qs = [r['arms'][arm]['queries'][d['query']] for r, d in zip(comparable, contexts, strict=True)]
                    eligible = [q for q in qs if q['event'] == 'gripper_close' and q['closing_proxy']['fully_observed_32_plus_response']]
                    groups[group]['forecast_closing_at_departure'][arm] = {
                        'eligible': len(eligible), 'future_closing_observed': sum(q['closing_proxy']['confirmed_any_in_window'] for q in eligible)}
        result[panel] = {'groups': groups,
            'suite_outcomes': {suite: dict(Counter(r['outcome_class'] for r in rr if r['task']['suite'] == suite))
                for suite in sorted({r['task']['suite'] for r in rr})}}
        for r in rr:
            if r['outcome_class'] == 'adaptive_loss':
                losses.append({'panel': panel, 'ordinal': r['ordinal'], 'task': r['task'],
                    'first_non8': r['divergence']['first_non8_at_shared_prefix'],
                    'arm_policy_steps': {a: r['arms'][a]['policy_steps'] for a in ARMS},
                    'artifacts': r['artifacts']})
    write(output/'report-tables.json', {'panels': result, 'extraction_completion_digest': completion['digest'],
        'source_sha256': sha(__file__), 'scope': 'post-hoc descriptive tables; no horizon-specific causal estimates'})
    write(output/'adaptive-loss-index.json', {'loss_count': len(losses), 'episodes': losses})
    print(json.dumps({'losses': len(losses), 'panels': list(result)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); summarize(a.root, a.output)
