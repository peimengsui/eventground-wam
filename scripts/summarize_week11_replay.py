"""Join frozen observation replay to prediction-independent event candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from week11_replay_common import HORIZONS, read, write

from eventground_wam.eventground_future_onset import target_at
from eventground_wam.eventground_horizon import select_horizon

ROLES = ('shared_first_departure', 'before_first_closing', 'after_first_closing')


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return {'n': len(a), **({k: float(np.quantile(a, q)) for k, q in
        [('min', 0), ('q25', .25), ('median', .5), ('q75', .75), ('max', 1)]} if len(a) else {})}


def annotate(row, candidate):
    result = {'candidate_digest': candidate['digest'], 'quality_validated': False,
              'predicted_horizon': select_horizon(row['prediction']).selected_horizon}
    try:
        target, event, censored = target_at(candidate, row['step'])
    except ValueError as exc:
        return {**result, 'eligible': False, 'exclusion': str(exc)}
    same = target['next_event'] == row['prediction']['next_event']
    mapped = select_horizon(target).selected_horizon
    return {**result, 'eligible': True, 'target': target, 'event': event,
        'end_censored': censored, 'type_match': same, 'target_horizon': mapped,
        'mapped_horizon_match': mapped == result['predicted_horizon'],
        'signed_onset_error_if_type_match': row['prediction']['horizon']-target['horizon']
            if same and target['horizon'] is not None else None}


def metrics(rows):
    return {'n': len(rows), 'horizons': {str(h): {
        **{k: {m: distribution([r['metrics'][str(h)][k][m] for r in rows]) for m in
               ('relative_l2_delta', 'rms_delta', 'cosine')} for k in
           ('future_tokens', 'normalized_future_tokens', 'residual', 'raw_actions', 'translation', 'rotation')},
        'gripper_changed_chunks': sum(r['metrics'][str(h)]['gripper_disagree_actions'] > 0 for r in rows),
        'gripper_changed_actions': sum(r['metrics'][str(h)]['gripper_disagree_actions'] for r in rows),
        'translation_prefix_rms': distribution([r['metrics'][str(h)]['translation_prefix_rms'] for r in rows]),
        'translation_suffix_rms': distribution([r['metrics'][str(h)]['translation_suffix_rms'] for r in rows]),
        } for h in HORIZONS},
        'forecast_selected_gripper_changed_chunks': sum(r['metrics'][str(r['annotation']['predicted_horizon'])]['gripper_disagree_actions'] > 0 for r in rows),
        'actually_used_gripper_changed_chunks': sum(r['metrics'][str(r['used_horizon'])]['gripper_disagree_actions'] > 0 for r in rows)}


def agreement(rows):
    a = [r['annotation'] for r in rows if r['annotation']['eligible']]
    return {'n': len(rows), 'eligible': len(a),
        'exclusions': dict(Counter(r['annotation']['exclusion'] for r in rows if not r['annotation']['eligible'])),
        'type_matches': sum(x['type_match'] for x in a),
        'mapped_horizon_matches': sum(x['mapped_horizon_match'] for x in a),
        'predicted_classes': dict(Counter(r['prediction']['next_event'] for r in rows if r['annotation']['eligible'])),
        'target_classes': dict(Counter(x['target']['next_event'] for x in a)),
        'confusion': dict(Counter(r['prediction']['next_event']+' -> '+r['annotation']['target']['next_event'] for r in rows if r['annotation']['eligible'])),
        'signed_onset_error_if_type_match': distribution([x['signed_onset_error_if_type_match'] for x in a]),
        'absolute_onset_error_if_type_match': distribution([abs(x['signed_onset_error_if_type_match']) for x in a if x['signed_onset_error_if_type_match'] is not None])}


def run(base):
    replay = read(base/'remote-replay/replay.json')
    annotations = read(base/'remote-annotations/annotations.json')
    assert read(base/'remote-replay/completion.json')['status'] == 'complete'
    assert read(base/'remote-annotations/completion.json')['status'] == 'complete'
    by_path = {r['artifact']: r for r in annotations['recordings']}
    rows = [{**r, 'annotation': annotate(r, by_path[r['artifact']]['candidate'])} for r in replay['observations']]
    for r in rows:
        r['paired_adaptive_annotation'] = (annotate(r, by_path[r['paired_adaptive_artifact']]['candidate'])
            if r['paired_adaptive_artifact'] else None)
    groups = {'all': rows, **{role: [r for r in rows if role in r['roles']] for role in ROLES}}
    summary = {'replay_digest': replay['digest'], 'annotations_digest': annotations['digest'],
        'selection_is_posthoc': True, 'population_performance_estimate': False,
        'groups': {k: metrics(v) for k, v in groups.items()},
        'agreement': {k: agreement(v) for k, v in groups.items()},
        'agreement_by_arm': {a: agreement([r for r in rows if r['arm'] == a]) for a in ('M4-EG-fixed8', 'M4-EG-adaptive')},
        'shared_departure_future_classes_fixed8_then_adaptive': dict(Counter(
            r['annotation'].get('target', {}).get('next_event', 'ambiguous')+' -> '+
            r['paired_adaptive_annotation'].get('target', {}).get('next_event', 'ambiguous')
            for r in groups['shared_first_departure'])),
        'first_departure_by_outcome': {o: metrics([r for r in groups['shared_first_departure'] if r['outcome_class'] == o]) for o in ('adaptive_loss', 'adaptive_win', 'both_success', 'both_failure')},
        'reconstruction': {'recordings': len(by_path), **{k: max(r[k] for r in by_path.values()) for k in
            ('max_state_roundtrip_error', 'max_fresh_eef_vs_recorded_m', 'max_fresh_finger_vs_recorded_m')}},
        'parity': {'recorded_actions': sum(r['recorded_action_parity'] for r in rows),
                  'repeat_all_tensors': sum(r['h8_repeat_all_tensors_exact'] for r in rows),
                  'paired_adaptive': sum(r['shared_prefix_adaptive_parity'] is True for r in rows)}}
    write(base/'summary.json', summary); write(base/'joined-observations.json', {'observations': rows})
    print(summary['agreement']); print(summary['reconstruction'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True)
    run(p.parse_args().base)
