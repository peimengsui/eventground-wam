"""Dense evidence and recorded-frame sheets for the selected Week 11 windows."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from week11_replay_common import read, sha, write

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def contacts_by_frame(geometry, contacts):
    """Keep excluded contacts visible, while preserving the original event scope."""
    geoms = {g['geom_id']: g for g in geometry['geoms']}
    objects = {o['object_id']: o for o in geometry['objects']}
    rows = []
    for frame in contacts:
        movable, excluded, fingers = set(), set(), {}
        for pair in frame['contacts']:
            if pair['distance_m'] > 0 or pair['efc_address'] < 0:
                continue
            a, b = geoms[pair['geom1']], geoms[pair['geom2']]
            if a['class'] != 'gripper':
                a, b = b, a
            if a['class'] != 'gripper' or b['class'] == 'gripper':
                continue
            if not a['collision_enabled'] or not b['collision_enabled']:
                continue
            obj = objects.get(b['object_id'])
            if obj and obj['movable_free_object']:
                movable.add(obj['object_id'])
                fingers.setdefault(obj['object_id'], set()).update(a['finger_sides'])
            else:
                excluded.add((b['name'], b['class'], b['object_id']))
        rows.append({'frame_index': frame['frame_index'], 'movable_objects': sorted(movable),
            'finger_sides': {k: sorted(v) for k, v in fingers.items()},
            'excluded_counterparts': [list(x) for x in sorted(excluded, key=str)]})
    return rows


def first_after(events, anchor, kind=None):
    eligible = [e for e in events if e['start'] > anchor and not e['left_censored']
                and (kind is None or e['event_type'] == kind)]
    return min(eligible, key=lambda e: e['start']) if eligible else None


def window_summary(candidate, geometry, evidence, contacts, anchor):
    stop = anchor+33
    assert stop <= candidate['frames']
    command = evidence['actions'][:, 6]
    aperture = np.abs(evidence['state'][:, -2:]).sum(axis=1)
    dense = contacts[anchor:stop]
    first = first_after(candidate['events'], anchor)
    positives = np.flatnonzero(command[anchor+1:stop] == 1)
    moving = np.flatnonzero([bool(x['movable_objects']) for x in contacts[anchor+1:stop]])
    excluded = np.flatnonzero([bool(x['excluded_counterparts']) for x in contacts[anchor+1:stop]])
    future_events = [{**e, 'start_offset': e['start']-anchor, 'end_offset': e['end']-anchor}
                     for e in candidate['events'] if e['start'] < stop and e['end'] > anchor]
    ambiguities = [a for a in candidate['ambiguities'] if a['start'] < stop and a['end'] > anchor]
    ids = [o['object_id'] for o in geometry['objects']]
    dp = evidence['object_position'][anchor:stop]-evidence['object_position'][anchor]
    return {'anchor': anchor, 'frames': candidate['frames'], 'window_end': stop-1,
        'first_new_candidate_event': first, 'first_new_candidate_offset': first['start']-anchor if first else None,
        'first_positive_command_offset': int(positives[0])+1 if len(positives) else None,
        'first_movable_contact_offset': int(moving[0])+1 if len(moving) else None,
        'first_excluded_contact_offset': int(excluded[0])+1 if len(excluded) else None,
        'commands': command[anchor:stop].tolist(), 'aperture_mm': (1000*aperture[anchor:stop]).tolist(),
        'dense_contacts': dense, 'candidate_events_overlapping': future_events,
        'ambiguity_reasons': dict(Counter(a['reason'] for a in ambiguities)), 'ambiguities': ambiguities,
        'object_max_displacement_mm': dict(zip(ids, (1000*np.linalg.norm(dp, axis=-1).max(axis=0)).tolist(), strict=True)),
        'object_max_height_increase_mm': dict(zip(ids, (1000*dp[:, :, 2].max(axis=0)).tolist(), strict=True))}


def picture(case, arms, output):
    fig = plt.figure(figsize=(15, 3.25*len(arms)+2.8), layout='constrained')
    grid = fig.add_gridspec(len(arms)+1, 5, height_ratios=[1.6]*len(arms)+[1])
    for ai, arm in enumerate(arms):
        anchor = case['row']['step']; offsets = [0, 8, 16, 24, 32]
        if case['group'] == 'preclose_flip':
            offsets = [0, 3, 8, 16, 32]
        with h5py.File(arm['artifact'], 'r') as f:
            for col, offset in enumerate(offsets):
                ax = fig.add_subplot(grid[ai, col])
                frame = np.concatenate([f['steps/agent_rgb'][anchor+offset], f['steps/wrist_rgb'][anchor+offset]], axis=0)
                ax.imshow(frame); ax.axis('off')
                ax.set_title(f'{arm["arm"].replace("M4-EG-", "")} · +{offset}', fontsize=10)
    a1 = fig.add_subplot(grid[-1, :2]); a2 = fig.add_subplot(grid[-1, 2:])
    for arm, col in zip(arms, ['#246b91', '#bd5838'], strict=False):
        w = arm['window']; x = np.arange(33)
        a1.plot(x, w['aperture_mm'], color=col, label=arm['arm'].replace('M4-EG-', ''))
        m = np.array([bool(c['movable_objects']) for c in w['dense_contacts']])
        e = np.array([bool(c['excluded_counterparts']) for c in w['dense_contacts']])
        y = 1 if arm['arm'].endswith('fixed8') else 2
        a2.scatter(x[m], np.full(m.sum(), y), color=col, marker='o', s=30)
        a2.scatter(x[e], np.full(e.sum(), y+.2), color=col, marker='x', s=30)
    a1.set_ylabel('Physical aperture (mm)'); a1.set_xlabel('Recorded frames after query'); a1.legend(fontsize=8)
    a2.set_xlim(-1, 33); a2.set_ylim(.7, 2.6); a2.set_yticks([1, 2], ['fixed8', 'adaptive'])
    a2.set_xlabel('Recorded frames after query'); a2.set_title('Dense contact: circles = movable objects; × = excluded geometry', fontsize=9)
    for ax in (a1, a2):
        ax.grid(alpha=.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f'{case["id"]} · {case["row"]["outcome_class"]}\n'+case['instruction'], fontsize=12)
    fig.supxlabel('Saved camera frames only; top = agent camera, bottom = wrist camera. Markers use saved-state contact evidence, not image labels.', fontsize=9)
    fig.savefig(output/f'{case["id"]}.png', dpi=120); plt.close(fig)


def run(args):
    start = time.monotonic(); cfg = read(args.bundle/'contract.json')
    for name, digest in cfg['code_sha256'].items():
        assert sha(args.bundle/name) == digest
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'contract.json', cfg)
    annotations = read(Path(cfg['annotation_root'])/'annotations.json')
    assert annotations['digest'] == cfg['annotations_digest']
    by_path = {r['artifact']: r for r in annotations['recordings']}
    cache, verified = {}, {}

    def load_recording(path):
        if path in cache:
            return cache[path]
        r = by_path[path]; d = Path(cfg['annotation_root'])/r['recording_id']
        assert sha(path) == cfg['input_sha256'][path]
        log = str(Path(path).with_suffix('.queries.jsonl'))
        assert sha(log) == cfg['input_sha256'][log]
        for n, digest in r['evidence_sha256'].items():
            assert sha(d/n) == digest
        geometry = read(d/'geometry.json'); candidate = read(d/'candidate.json')
        assert candidate == r['candidate']
        evidence = dict(np.load(d/'evidence.npz'))
        contacts = contacts_by_frame(geometry, [json.loads(l) for l in (d/'contacts.jsonl').read_text().splitlines()])
        logs = [json.loads(l) for l in Path(log).read_text().splitlines()]
        with h5py.File(path, 'r') as f:
            raw = f['episode/metadata_json'][()]
            meta = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        cache[path] = geometry, candidate, evidence, contacts, logs, meta
        verified[path] = {'source_sha256': r['source_sha256'], 'recording_id': r['recording_id'],
                          'evidence_sha256': r['evidence_sha256']}
        return cache[path]

    results = []
    for case in cfg['cases']:
        r = case['row']; paths = [r['artifact']]
        if case['group'] == 'early_negative':
            paths.append(r['paired_adaptive_artifact'])
        arms = []
        for path in paths:
            geometry, candidate, evidence, contacts, logs, meta = load_recording(path)
            w = window_summary(candidate, geometry, evidence, contacts, r['step'])
            q = logs[r['query']]['policy']
            arms.append({'artifact': path, 'arm': q['arm'], 'window': w,
                'prediction': q['prediction'], 'used_horizon': q['used_horizon'],
                'event_class_probabilities': q['event']['categorical_probabilities']['event'],
                'start_probabilities': q['event']['categorical_probabilities']['start']})
        case = {**case, 'instruction': meta['instruction'], 'arms': arms}
        if case['group'] == 'preclose_flip':
            tp = Path(cfg['replay_root'])/r['tensor_file']
            assert sha(tp) == r['tensor_sha256']
            z = np.load(tp); by_horizon = {}
            for h in (2, 4, 8, 16, 32):
                raw = z[f'h{h}_raw_actions'].reshape(8, 7)[:, 6]
                decoded = z[f'h{h}_decoded_actions'][:, 6]
                assert np.array_equal(np.where(raw >= 0, 1, -1), decoded)
                close = np.flatnonzero(decoded == 1)
                by_horizon[str(h)] = {'raw_gripper_scores': raw.tolist(), 'gripper_commands': decoded.tolist(),
                    'first_closed_action_one_based': int(close[0])+1 if len(close) else None}
            case['horizon_actions'] = by_horizon
        results.append(case); picture(case, arms, args.output)
        print(json.dumps({'cases_complete': len(results), 'cases_total': len(cfg['cases'])}), flush=True)
    write(args.output/'review-evidence.json', {'cases': results, 'contract_digest': cfg['digest'],
        'verified_inputs': verified, 'model_calls': 0, 'simulator_steps': 0,
        'interpretation': 'Descriptive evidence for assisted review; predictions and outcomes were already known; not independent human QA.'})
    write(args.output/'completion.json', {'status': 'complete', 'cases': len(results), 'recordings': len(cache),
        'seconds': time.monotonic()-start, 'model_calls': 0, 'simulator_steps': 0,
        'evidence_digest': read(args.output/'review-evidence.json')['digest'],
        'figure_sha256': {p.name: sha(p) for p in args.output.glob('*.png')}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
