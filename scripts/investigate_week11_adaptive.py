"""Post-hoc, recording-only diagnosis. No model loading or simulator stepping."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np

PANELS = ('targeted-467', 'clean-200')
ARMS = ('M4-EG-fixed8', 'M4-EG-adaptive')
HORIZONS = (2, 4, 8, 16, 32)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seal(value):
    value = dict(value)
    value.pop('digest', None)
    value['digest'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return value


def read_sealed(path):
    value = json.loads(Path(path).read_text())
    assert value['digest'] == seal(value)['digest'], str(path)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal(value), sort_keys=True, indent=2, allow_nan=False)+'\n')


def describe(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
            'median': float(np.median(a)) if len(a) else None,
            'p10': float(np.quantile(a, .1)) if len(a) else None,
            'p90': float(np.quantile(a, .9)) if len(a) else None,
            'max': float(a.max()) if len(a) else None}


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def closing_proxy(actions, proprio):
    """Recorded command/aperture rule; no contact/grasp or target-object labels."""
    commands = np.asarray(actions)[:, 6]
    aperture = np.abs(np.asarray(proprio)[:, 6:8]).sum(axis=1)
    assert np.isin(commands, [-1, 1]).all()
    transitions = np.flatnonzero((commands[1:] == 1) & (commands[:-1] == -1))+1
    confirmed, incomplete, weak, short = [], [], [], []
    for t in transitions:
        if t+5 >= len(commands):
            incomplete.append(int(t))
        elif not np.all(commands[t:t+5] == 1):
            short.append(int(t))
        elif aperture[t]-aperture[t+5] < .001:
            weak.append(int(t))
        else:
            confirmed.append(int(t))
    return {'command_onsets': transitions.tolist(), 'confirmed_onsets': confirmed,
            'incomplete_response': incomplete, 'short_command': short,
            'weak_response': weak, 'initial_close_left_censored': bool(commands[0] == 1)}


def proxy_at_query(step, start, n, proxy):
    """Require full future window AND five response frames; don't count censoring as error."""
    observable = step+32+5 < n
    result = {'fully_observed_32_plus_response': observable}
    for name in ('command', 'confirmed'):
        offsets = [t-step for t in proxy[name+'_onsets'] if 1 <= t-step <= 32]
        result[name+'_next_offset'] = min(offsets) if offsets else None
        result[name+'_any_in_window'] = bool(offsets) if observable else None
        result[name+'_nearest_prediction_error'] = (
            min(abs(x-start) for x in offsets) if observable and offsets and start else None)
    return result


def episode_series(path, queries):
    with h5py.File(path, 'r') as h:
        s = h['steps']
        actions, proprio = s['executed_action'][:], s['proprioception'][:]
        qsteps = np.flatnonzero(np.isfinite(s['policy_latency_ms'][:]))
        assert len(qsteps) == len(queries)
        assert np.array_equal(qsteps, 10+8*np.arange(len(queries)))
        assert np.array_equal(s['action_chunk_index'][qsteps], np.zeros(len(qsteps)))
        n = len(actions)
        proxy = closing_proxy(actions, proprio)
        result = []
        for j, (step, q) in enumerate(zip(qsteps, queries, strict=True)):
            p = q['policy']; d = p['decision']; pred = p['prediction']
            probs = p['event']['categorical_probabilities']
            assert q['query_index'] == j and q['chunk_start'] == 8*j
            assert p['execute_actions'] == 8 and p['proposal_actions'] == 8
            assert p['future_calls'] == 1 and q['exception'] is None
            assert p['used_horizon'] == (d['selected_horizon'] if p['arm'] == ARMS[1] else 8)
            result.append({'query': j, 'step': int(step), 'policy_step': 8*j,
                'used': p['used_horizon'], 'selected': d['selected_horizon'],
                'event': pred['next_event'], 'start': d['raw_start'], 'end': d['raw_end'],
                'fallback': d['fallback_reason'],
                'type_confidence': float(max(probs['event'])),
                'start_confidence': float(max(probs['start'])) if d['raw_start'] else None,
                'start_mass_29_32': float(sum(probs['start'][28:32])) if d['raw_start'] else None,
                'command_now': float(actions[step, 6]),
                'close_command_previous_eight': bool(step >= 8 and np.all(actions[step-8:step, 6] == 1)),
                'aperture': float(np.abs(proprio[step, 6:8]).sum()),
                'closing_proxy': proxy_at_query(int(step), d['raw_start'], n, proxy)})
        pa, pp = actions[10:], proprio[10:]
        return {'queries': result, 'recorded_steps': n, 'policy_steps': len(pa),
            'closing_proxy': proxy,
            'motion': {'eef_path_m': float(np.linalg.norm(np.diff(pp[:, :3], axis=0), axis=1).sum()),
                'last32_eef_path_m': float(np.linalg.norm(np.diff(pp[-32:, :3], axis=0), axis=1).sum()),
                'mean_translation_action_step_change': float(np.linalg.norm(np.diff(pa[:, :3], axis=0), axis=1).mean()),
                'gripper_command_switches': int(np.count_nonzero(np.diff(pa[:, 6]))),
                'close_command_fraction': float(np.mean(pa[:, 6] == 1))}}


def action_difference(a, b):
    delta = b-a
    return {'equal': bits(a, b),
        'translation_rms': float(np.sqrt(np.mean(delta[:, :3]**2))),
        'rotation_rms': float(np.sqrt(np.mean(delta[:, 3:6]**2))),
        'translation_max_abs': float(np.max(np.abs(delta[:, :3]))),
        'first_action_translation_l2': float(np.linalg.norm(delta[0, :3])),
        'gripper_disagree_actions': int(np.count_nonzero(a[:, 6] != b[:, 6]))}


def first_divergence(left_path, right_path, adaptive):
    """Only identical input states admit a direct horizon-conditioned action comparison."""
    with h5py.File(left_path, 'r') as lh, h5py.File(right_path, 'r') as rh:
        left, right = lh['steps'], rh['steps']
        ln, rn = len(left['step_index']), len(right['step_index'])
        same_inputs, first_input, first_action, first_non8 = [], None, None, None
        first_non8_query = next((q['query'] for q in adaptive['queries'] if q['used'] != 8), None)
        # Until the first input difference, previous observations/actions can be compared causally.
        # Include that first different-input query for context, then stop (trajectories diverge).
        for q in adaptive['queries']:
            step = q['step']
            if step >= min(ln, rn):
                break
            equal_inputs = all(bits(left[name][step], right[name][step]) for name in ('agent_rgb', 'wrist_rgb'))
            equal_inputs &= bits(left['proprioception'][step].astype(np.float32), right['proprioception'][step].astype(np.float32))
            ac = action_difference(left['predicted_action_chunk'][step], right['predicted_action_chunk'][step])
            context = {'query': q['query'], 'step': step, 'used': q['used'], 'event': q['event'], 'start': q['start'],
                'inputs_equal': bool(equal_inputs), 'action': ac,
                'simulator_state_equal': bits(left['mujoco_state'][step], right['mujoco_state'][step]),
                'previous_executed_actions_equal': bits(left['executed_action'][:step], right['executed_action'][:step])}
            if q['query'] == first_non8_query:
                first_non8 = context
            if not ac['equal'] and first_action is None:
                first_action = context
            if equal_inputs:
                same_inputs.append(context)
            else:
                first_input = context
                break
        return {'identical_input_queries': same_inputs, 'first_input_difference': first_input,
                'first_action_difference': first_action, 'first_non8_at_shared_prefix': first_non8,
                'first_non8_query': first_non8_query}


def outcome_class(outcome):
    fixed, _, adaptive = outcome
    return ('both_success' if adaptive else 'adaptive_loss') if fixed else (
        'adaptive_win' if adaptive else 'both_failure')


def sequence_metrics(queries):
    hs = np.array([q['used'] for q in queries])
    adjacent = [(a, b) for a, b in pairwise(queries)
                if a['event'] == b['event'] != 'none']
    return {'query_count': len(queries), 'horizon_counts': dict(Counter(map(str, hs))),
        'non8_fraction': float(np.mean(hs != 8)), 'short_fraction': float(np.mean(hs < 8)),
        'long_fraction': float(np.mean(hs > 8)), 'h32_fraction': float(np.mean(hs == 32)),
        'switches': int(np.count_nonzero(np.diff(hs))),
        'switch_fraction': float(np.mean(np.diff(hs) != 0)) if len(hs) > 1 else None,
        'short_long_switches': sum((a['used'] < 8 and b['used'] > 8) or
                                   (a['used'] > 8 and b['used'] < 8) for a, b in pairwise(queries)),
        'same_positive_type_adjacent': len(adjacent),
        'same_type_offset_not_decreasing': sum(b['start'] >= a['start'] for a, b in adjacent)}


def summarize(rows):
    summary = {}
    for panel in PANELS:
        rr = [r for r in rows if r['panel'] == panel]
        identical = [q for r in rr for q in r['divergence']['identical_input_queries']]
        groups = {}
        for group in ('adaptive_loss', 'adaptive_win', 'both_success', 'both_failure', 'all'):
            subset = rr if group == 'all' else [r for r in rr if r['outcome_class'] == group]
            metrics = [sequence_metrics(r['arms'][ARMS[1]]['queries']) for r in subset]
            early = [sequence_metrics(r['arms'][ARMS[1]]['queries'][:5]) for r in subset]
            groups[group] = {'episodes': len(subset),
                'first_horizon': dict(Counter(str(r['arms'][ARMS[1]]['queries'][0]['used']) for r in subset)),
                'episode_weighted_full': {k: describe([m[k] for m in metrics if m[k] is not None])
                    for k in ('non8_fraction', 'short_fraction', 'long_fraction', 'h32_fraction', 'switch_fraction')},
                'episode_weighted_first5': {k: describe([m[k] for m in early if m[k] is not None])
                    for k in ('non8_fraction', 'short_fraction', 'long_fraction', 'h32_fraction', 'switch_fraction')},
                'policy_steps': {arm: describe([r['arms'][arm]['policy_steps'] for r in subset]) for arm in ARMS},
                'motion': {arm: {k: describe([r['arms'][arm]['motion'][k] for r in subset])
                    for k in ('last32_eef_path_m', 'gripper_command_switches', 'mean_translation_action_step_change')}
                    for arm in ARMS}}
        first_horizon = {}
        for h in HORIZONS:
            sub = [r for r in rr if r['arms'][ARMS[1]]['queries'][0]['used'] == h]
            first_horizon[str(h)] = {'n': len(sub), 'outcomes': dict(Counter(r['outcome_class'] for r in sub)),
                'net_success_difference_pp': (100*sum(r['outcomes'][2]-r['outcomes'][0] for r in sub)/len(sub)) if sub else None}
        query_summary = {}
        for arm in ARMS:
            qs = [q for r in rr for q in r['arms'][arm]['queries']]
            close = [q for q in qs if q['event'] == 'gripper_close']
            eligible = [q for q in close if q['closing_proxy']['fully_observed_32_plus_response']]
            supported = [q for q in eligible if q['closing_proxy']['confirmed_any_in_window']]
            high_conf = [q for q in eligible if q['type_confidence'] >= .9]
            query_summary[arm] = {'n': len(qs), 'used_horizons': dict(Counter(str(q['used']) for q in qs)),
                'selected_horizons': dict(Counter(str(q['selected']) for q in qs)),
                'events': dict(Counter(q['event'] for q in qs)),
                'raw_starts': dict(Counter(str(q['start']) for q in qs)),
                'start_mass_29_32_positive': describe([q['start_mass_29_32'] for q in qs if q['start']]),
                'gripper_close_proxy': {'all_predictions': len(close), 'eligible_full_windows': len(eligible),
                    'censored': len(close)-len(eligible),
                    'with_future_command_transition': sum(q['closing_proxy']['command_any_in_window'] for q in eligible),
                    'with_future_confirmed_closing': len(supported),
                    'nearest_onset_absolute_error_supported': describe([q['closing_proxy']['confirmed_nearest_prediction_error'] for q in supported]),
                    'close_command_already_held_previous8': sum(q['close_command_previous_eight'] for q in eligible),
                    'high_type_confidence_eligible': len(high_conf),
                    'high_type_confidence_with_confirmed_closing': sum(q['closing_proxy']['confirmed_any_in_window'] for q in high_conf)},
                'sequence_totals': dict(Counter({k: sum(sequence_metrics(r['arms'][arm]['queries'])[k] for r in rr)
                    for k in ('same_positive_type_adjacent', 'same_type_offset_not_decreasing', 'switches', 'short_long_switches')}))}
        summary[panel] = {'groups': groups, 'first_horizon_outcomes': first_horizon,
            'identical_input_comparisons': {str(h): {'n': sum(q['used'] == h for q in identical),
                'different_actions': sum(q['used'] == h and not q['action']['equal'] for q in identical),
                'translation_rms': describe([q['action']['translation_rms'] for q in identical if q['used'] == h]),
                'rotation_rms': describe([q['action']['rotation_rms'] for q in identical if q['used'] == h]),
                'with_gripper_difference': sum(q['used'] == h and q['action']['gripper_disagree_actions'] > 0 for q in identical)} for h in HORIZONS},
            'first_non8_shared_input_episodes': sum(r['divergence']['first_non8_at_shared_prefix'] is not None
                and r['divergence']['first_non8_at_shared_prefix']['inputs_equal'] for r in rr),
            'queries': query_summary}
    return summary


def run(args):
    import psutil
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = read_sealed(args.closeout/'raw-inventory.json')
    matrix = read_sealed(args.closeout/'paired-matrices.json')
    analysis = read_sealed(args.closeout/'analysis.json')
    root = Path(inventory['root'])
    assert args.closeout.resolve() != args.output.resolve()
    source_digest = sha(Path(__file__))
    shutil.copy2(__file__, args.output/Path(__file__).name)
    write(args.output/'scope.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'analysis_digest': analysis['digest'], 'matrix_digest': matrix['digest'],
        'raw_inventory_digest': inventory['digest'], 'source_sha256': source_digest,
        'scope': 'post-hoc recording-only diagnostic; all 667 event-fixed8/adaptive pairs',
        'metrics': ['identical-input action comparison until input divergence',
            'first-horizon outcome strata', 'episode-weighted horizon exposure full and first five queries',
            'horizon switches and repeated-type nondecreasing offsets',
            'command/aperture closing proxy with 32+5 frame censoring', 'movement and action variation',
            'first two losses, first win, first both-success per panel for visual review'],
        'limitations': ['horizons not randomized; no individual-horizon causal effects',
            'post-divergence observations differ; no counterfactual later action comparison',
            'closing proxy does not label contact or grasp, target object, or earliest event type',
            'repeated predictions do not establish same physical event identity',
            'no independent on-policy label QA; no full event accuracy or confidence calibration claim'],
        'rollouts': 0, 'model_inference_calls': 0})
    consumed, stats, rows, peak_rss = {}, {}, [], 0

    def verified(path):
        path = Path(path)
        rel = str(path.relative_to(root))
        assert sha(path) == inventory['files'][rel]['sha256'], rel
        stat = path.stat()
        consumed[rel] = inventory['files'][rel]
        stats[rel] = (stat.st_size, stat.st_mtime_ns)

    with (args.output/'pairs.jsonl').open('x') as stream:
        for panel in PANELS:
            panel_rows = {arm: read_sealed(args.closeout/f'panels/{arm}/{panel}.json')['episodes'] for arm in ARMS}
            assert len(panel_rows[ARMS[0]]) == len(matrix[panel]['tasks']) == len(panel_rows[ARMS[1]])
            for i, task in enumerate(matrix[panel]['tasks']):
                arms, paths = {}, {}
                for arm in ARMS:
                    row = panel_rows[arm][i]
                    path = Path(row['artifact']); qpath = path.with_suffix('.queries.jsonl')
                    verified(path); verified(qpath)
                    queries = [json.loads(line) for line in qpath.read_text().splitlines()]
                    arms[arm] = episode_series(path, queries)
                    paths[arm] = str(path)
                    assert arms[arm]['policy_steps'] == row['policy_step_count']
                result = {'panel': panel, 'ordinal': i, 'task': task,
                    'outcomes': matrix[panel]['outcomes'][i],
                    'outcome_class': outcome_class(matrix[panel]['outcomes'][i]),
                    'artifacts': paths, 'arms': arms,
                    'divergence': first_divergence(paths[ARMS[0]], paths[ARMS[1]], arms[ARMS[1]])}
                rows.append(result)
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False)+'\n'); stream.flush()
                elapsed = time.monotonic()-start
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                assert elapsed < 1800, '30 minute diagnosis cap'
                assert peak_rss < 8*2**30, '8 GiB RSS cap'
                assert shutil.disk_usage(args.output).free > 50*2**30, '50 GiB reserve'
                if i % 50 == 0:
                    print(json.dumps({'panel': panel, 'pairs': i+1, 'seconds': elapsed}), flush=True)
                    write(args.output/'progress.json', {'running': True, 'pairs_completed': len(rows), 'seconds': elapsed})
    summary = summarize(rows)
    write(args.output/'summary.json', {'panels': summary, 'scope': 'exploratory descriptive diagnosis'})
    cases = []
    for panel in PANELS:
        for group, count in [('adaptive_loss', 2), ('adaptive_win', 1), ('both_success', 1)]:
            cases.extend({'panel': r['panel'], 'ordinal': r['ordinal'], 'outcome_class': group,
                'artifacts': r['artifacts']} for r in [r for r in rows if r['panel'] == panel and r['outcome_class'] == group][:count])
    write(args.output/'selected-cases.json', {'selection': 'first manifest ordinals in each specified outcome class', 'cases': cases})
    assert all((root/p).stat().st_size == st[0] and (root/p).stat().st_mtime_ns == st[1] for p, st in stats.items())
    write(args.output/'inputs.json', {'raw_inventory_digest': inventory['digest'], 'files': consumed,
        'sha256_verified_before_read': True, 'size_mtime_unchanged_after_read': True})
    write(args.output/'progress.json', {'running': False, 'pairs_completed': len(rows), 'seconds': time.monotonic()-start})
    write(args.output/'completion.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'pairs': len(rows), 'hdf5_recordings_read': 2*len(rows), 'source_sha256': source_digest,
        'active_seconds': time.monotonic()-start, 'peak_rss_bytes': peak_rss,
        'free_bytes': shutil.disk_usage(args.output).free, 'model_inference_calls': 0, 'rollouts': 0,
        'files_sha256': {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    print(json.dumps({'complete': True, 'pairs': len(rows), 'seconds': time.monotonic()-start}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--closeout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
