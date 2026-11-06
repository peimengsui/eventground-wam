"""Descriptive summaries; never promote partial evidence to certified labels."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from week11_replay_common import read, write


def stats(values):
    a = [v for v in values if v is not None]
    return {'n': len(a), 'missing': len(values)-len(a),
        **({'min': float(min(a)), 'median': float(np.median(a)), 'max': float(max(a))} if a else {})}


def run(base):
    evidence = read(base/'remote/review-evidence.json')
    early = [c for c in evidence['cases'] if c['group'] == 'early_negative']
    late = [c for c in evidence['cases'] if c['group'] == 'preclose_flip']
    rows = []
    for c in early:
        r = c['row']; a, b = c['arms']; p = a['prediction']
        assert p == b['prediction']
        assert a['event_class_probabilities'] == b['event_class_probabilities']
        for arm in c['arms']:
            w = arm['window']
            assert w['commands'] == [-1.]*33 and not any(v['movable_objects'] for v in w['dense_contacts'])
            assert not w['ambiguities'] and not w['candidate_events_overlapping']
        rows.append({'id': c['id'], 'outcome_class': r['outcome_class'], 'predicted_event': p['next_event'],
            'predicted_onset': p['horizon'], 'selected_horizon': r['annotation']['predicted_horizon'],
            'first_candidate_offset_by_arm': {arm['arm']: arm['window']['first_new_candidate_offset'] for arm in c['arms']},
            'excluded_contact_offset_by_arm': {arm['arm']: arm['window']['first_excluded_contact_offset'] for arm in c['arms']},
            'selected_event_probability': a['event_class_probabilities'][['none', 'gripper_close', 'contact', 'grasp'].index(p['next_event'])],
            'none_probability': a['event_class_probabilities'][0],
            'start_probability_mass_29_to_32': sum(a['start_probabilities'][28:32])})
    closing = []
    for c in late:
        h4, h8 = c['horizon_actions']['4'], c['horizon_actions']['8']; w = c['arms'][0]['window']
        flipped = [i for i in range(8) if h4['gripper_commands'][i] != h8['gripper_commands'][i]]
        assert len(flipped) == 1
        i = flipped[0]
        closing.append({'id': c['id'], 'outcome_class': c['row']['outcome_class'], 'forecast': c['row']['prediction'],
            'h4_close_action': h4['first_closed_action_one_based'], 'h8_close_action': h8['first_closed_action_one_based'],
            'timing_delta_frames_h4_minus_h8': h4['first_closed_action_one_based']-h8['first_closed_action_one_based'],
            'changed_action_one_based': i+1, 'h8_raw_score': h8['raw_gripper_scores'][i], 'h4_raw_score': h4['raw_gripper_scores'][i],
            'closing_candidate_offsets': [e['start_offset'] for e in w['candidate_events_overlapping'] if e['event_type'] == 'gripper_close'],
            'grasp_candidate_offsets': [e['start_offset'] for e in w['candidate_events_overlapping'] if e['event_type'] == 'grasp'],
            'first_movable_contact_offset': w['first_movable_contact_offset'],
            'first_excluded_contact_offset': w['first_excluded_contact_offset'],
            'maximum_object_height_increase_mm': max(w['object_max_height_increase_mm'].values()),
            'full_window_eligible': c['row']['annotation']['eligible'],
            'aperture_min_mm': min(w['aperture_mm'])})
    summary = {'evidence_digest': evidence['digest'], 'early_cases': rows, 'closing_cases': closing,
        'early_summary': {'pairs': len(rows), 'recorded_windows': 2*len(rows),
            'all_commands_open': True, 'no_movable_object_contact': True,
            'pairs_with_excluded_contacts': sum(any(v is not None for v in r['excluded_contact_offset_by_arm'].values()) for r in rows),
            'outcomes': dict(Counter(r['outcome_class'] for r in rows)),
            'predicted_events': dict(Counter(r['predicted_event'] for r in rows)),
            'predicted_onsets': dict(Counter(r['predicted_onset'] for r in rows)),
            'first_candidate_offset': stats([v for r in rows for v in r['first_candidate_offset_by_arm'].values()]),
            'selected_event_probability': stats([r['selected_event_probability'] for r in rows]),
            'none_probability': stats([r['none_probability'] for r in rows]),
            'mass_29_to_32_for_boundary_predictions': stats([r['start_probability_mass_29_to_32'] for r in rows if r['predicted_onset'] >= 29])},
        'closing_summary': {'cases': len(closing), 'h4_closes_earlier': sum(r['timing_delta_frames_h4_minus_h8'] < 0 for r in closing),
            'h4_closes_later': sum(r['timing_delta_frames_h4_minus_h8'] > 0 for r in closing),
            'absolute_timing_change_frames': stats([abs(r['timing_delta_frames_h4_minus_h8']) for r in closing]),
            'absolute_raw_score_change': stats([abs(r['h4_raw_score']-r['h8_raw_score']) for r in closing]),
            'minimum_absolute_score_across_both_horizons': min(abs(r[k]) for r in closing for k in ('h4_raw_score', 'h8_raw_score'))},
        'independent_human_qa': False, 'full_window_labels_modified': False, 'population_estimates': False}
    write(base/'summary.json', summary)
    print(summary['early_summary']); print(summary['closing_summary'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True)
    run(p.parse_args().base)
