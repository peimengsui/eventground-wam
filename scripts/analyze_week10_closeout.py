"""Read-only Week 10 closeout diagnostics from sealed train/dev evidence.

Creates a new bundle. Never runs a model, changes labels, or selects checkpoints.
Report results are linked for closeout only; diagnostics use train/dev exclusively.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_future_onset import audit_row, weak_reward
from eventground_wam.eventground_weak_evaluation import metrics, per_example
from eventground_wam.eventground_weak_supervision import sealed, sha


CLASSES = ('none', 'gripper_close', 'contact', 'grasp')
HORIZONS = (2, 4, 8, 16, 32)
PACKAGES = ('week10-weak-rlvr-v1', 'week10-report-coverage-extension-v1',
            'week10-future-onset-v1', 'week10-preservation-pilot-v1',
            'week10-sft-continuation-report-v1')


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def verify_packages(root):
    result = {}
    for name in PACKAGES:
        package = sealed(root/name/'package-index.json')
        base = root if name == 'week10-report-coverage-extension-v1' else root/name
        for rel, expected in package['files_sha256'].items():
            if sha(base/rel) != expected:
                raise ValueError('Previous package changed: '+str(base/rel))
        result[name] = dict(digest=package['digest'], files=len(package['files_sha256']))
    return result


def timing(rows):
    matched = [r for r in rows if r['target']['next_event'] != 'none' and r['correct_event']]
    signed = [r['prediction']['horizon']-r['target']['horizon'] for r in matched]
    absolute = [abs(x) for x in signed]
    return dict(n=len(matched), source_count=len({r['source_id'] for r in matched}),
                mean_signed_frames=float(np.mean(signed)) if signed else None,
                mae_frames=float(np.mean(absolute)) if signed else None,
                median_absolute_frames=float(np.median(absolute)) if signed else None,
                p90_absolute_frames=float(np.percentile(absolute, 90)) if signed else None,
                within_two_frames=sum(x <= 2 for x in absolute),
                more_than_eight_frames=sum(x > 8 for x in absolute),
                early=sum(x < 0 for x in signed), exact=sum(x == 0 for x in signed), late=sum(x > 0 for x in signed))


def horizon_diagnostic(rows):
    positive = [r for r in rows if r['prediction']['next_event'] != 'none']
    values = [r['prediction']['horizon'] for r in positive]
    sensitivity = {}
    for method in ('nearest_tie_lower', 'floor_clamped', 'ceil_clamped'):
        def map_h(h):
            if method == 'nearest_tie_lower':
                return min(HORIZONS, key=lambda x: (abs(h-x), x))
            if method == 'floor_clamped':
                return max((x for x in HORIZONS if x <= h), default=2)
            return min((x for x in HORIZONS if x >= h), default=32)
        mapped = [8 if r['prediction']['next_event'] == 'none' else map_h(r['prediction']['horizon']) for r in rows]
        sensitivity[method] = dict(counts={str(k):v for k,v in sorted(Counter(mapped).items())}, different_from_eight=sum(h != 8 for h in mapped), n=len(mapped))
    return dict(n=len(rows), predicted_none=len(rows)-len(positive), predicted_positive=len(positive),
                raw_positive_histogram={str(k):v for k,v in sorted(Counter(values).items())},
                raw_positive_different_from_eight=sum(h != 8 for h in values),
                raw_positive_outside_supported_set=sum(h not in HORIZONS for h in values),
                hypothetical_mapping_sensitivity=sensitivity,
                interpretation='Descriptive only. Three mappings and none->8 are hypothetical; no Week 11 mapping, fallback, threshold, or runtime change is selected here.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('docs/evaluation'))
    parser.add_argument('--out', type=Path, default=Path('docs/evaluation/week10-closeout-v1'))
    args = parser.parse_args()
    root, out = args.root, args.out
    if out.exists():
        raise ValueError('Closeout destination already exists; do not overwrite evidence')
    prior = verify_packages(root)
    future, pilot, report = [root/x for x in PACKAGES[2:]]
    manifest = sealed(future/'dev-manifest.json')
    train = sealed(future/'train-manifest.json')
    index = sealed(future/'source-index.json')
    candidates = {r['candidate']['source_id']: r['candidate'] for rows in index['panels'].values() for r in rows}
    dev_rows = {r['prompt_id']: r for r in manifest['rows']}
    train_rows = {r['prompt_id']: r for r in train['rows']}
    assert len(dev_rows) == 396 and len({r['source_id'] for r in dev_rows.values()}) == 103
    assert not ({r['source_id'] for r in dev_rows.values()} & {r['source_id'] for r in train_rows.values()})
    for row in list(dev_rows.values()) + list(train_rows.values()):
        audit_row(row, candidates[row['source_id']])
    paths = {arm: future/'eval/dev'/arm for arm in ('sft', 'primary', 'calibration')}
    paths.update({arm: pilot/'eval/dev'/arm for arm in ('hybrid', 'control')})
    predictions, evaluations, inputs, summaries = {}, {}, {}, {}
    for arm, path in paths.items():
        done = sealed(path/'completion.json')
        file = path/'predictions.jsonl'
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        assert sha(file) == done['predictions_sha256'] and done['manifest_digest'] == manifest['digest']
        assert done['status'] == 'complete' and done['model_unchanged'] and done['training_updates'] == 0
        assert [r['prompt_id'] for r in rows] == list(dev_rows)
        for r in rows:
            recomputed = per_example(dev_rows[r['prompt_id']], r['prediction'])
            assert all(r[k] == v for k, v in recomputed.items())
        assert metrics(rows) == done['metrics']
        predictions[arm] = {r['prompt_id']: r for r in rows}
        evaluations[arm] = done
        inputs[str(file)] = sha(file)
        for suffix in ('completion.json',):
            inputs[str(path/suffix)] = sha(path/suffix)
        matched = [r for r in rows if r['correct_event'] and r['target']['next_event'] != 'none']
        mistakes = [r for r in rows if not r['correct_event']]
        summaries[arm] = dict(metrics=done['metrics'], timing=timing(rows),
            timing_by_class={c: timing([r for r in rows if r['target']['next_event'] == c]) for c in CLASSES[1:]},
            timing_by_lead={b: timing([r for r in rows if r['lead_time_bin'] == b]) for b in ('1-4','5-8','9-16','17-32')},
            high_confidence_type_errors_90=sum(max(r['event_probabilities']) >= .9 for r in mistakes),
            matched_count=len(matched), horizon=horizon_diagnostic(rows))

    # Describe disagreements without changing the earliest-new-onset target semantics.
    temporal_context = {}
    for arm, rows in predictions.items():
        descriptions = []
        for pid, r in rows.items():
            if r['correct_event'] or r['prediction']['next_event'] == 'none':
                continue
            row = dev_rows[pid]; a = row['anchor']; c = candidates[row['source_id']]
            kind = r['prediction']['next_event']
            future_same = [e for e in c['events'] if a < e['start'] <= a+32 and e['event_type'] == kind and not e['left_censored']]
            ongoing_same = [e for e in c['events'] if e['start'] <= a < e['end'] and e['event_type'] == kind]
            descriptions.append(dict(prompt_id=pid, target=r['target']['next_event'], predicted=kind,
                predicted_type_also_has_new_onset_in_window=bool(future_same),
                predicted_type_ongoing_at_anchor=bool(ongoing_same),
                later_onset_offsets=[e['start']-a for e in future_same]))
        temporal_context[arm] = dict(wrong_positive_predictions=descriptions,
            type_present_elsewhere_in_window=sum(r['predicted_type_also_has_new_onset_in_window'] for r in descriptions),
            predicted_type_already_ongoing=sum(r['predicted_type_ongoing_at_anchor'] for r in descriptions))

    reward_summary = {}
    for arm in ('primary', 'calibration', 'hybrid'):
        file = (pilot if arm == 'hybrid' else future)/'runs'/arm/'rewards.json'
        data = json.loads(file.read_text()); inputs[str(file)] = sha(file)
        groups = [g for g in data['groups'] if not g['recovery_replay']]
        assert len(groups) == 500 and len({g['prompt_id'] for g in groups}) == 500
        assert data['group_size'] == 4
        by_class = {}
        for c in CLASSES:
            selected = [g for g in groups if train_rows[g['prompt_id']]['target']['next_event'] == c]
            for g in selected:
                row = train_rows[g['prompt_id']]
                assert len(g['predictions']) == len(g['rewards']) == 4
                recomputed = [weak_reward(p, row, candidates[row['source_id']], tuple(data['weights'])) for p in g['predictions']]
                assert np.allclose(recomputed, g['rewards'], rtol=0, atol=1e-12)
            ties = [g for g in selected if max(g['rewards']) == min(g['rewards'])]
            by_class[c] = dict(n=len(selected), reward_ties=len(ties),
                all_correct_type_ties=sum(all(p['next_event'] == c for p in g['predictions']) for g in ties),
                all_wrong_type_ties=sum(all(p['next_event'] != c for p in g['predictions']) for g in ties),
                mean_reward_std=float(np.mean([np.std(g['rewards']) for g in selected])),
                groups_with_correct_type=sum(any(p['next_event'] == c for p in g['predictions']) for g in selected),
                mean_unique_predictions=float(np.mean([len({json.dumps(p, sort_keys=True) for p in g['predictions']}) for g in selected])))
        reward_summary[arm] = by_class

    paired = {}
    baseline = predictions['sft']; control = predictions['control']
    for arm in ('primary', 'calibration', 'hybrid', 'control'):
        rows = predictions[arm]
        same = [pid for pid in dev_rows if baseline[pid]['correct_event'] and rows[pid]['correct_event'] and baseline[pid]['target']['next_event'] != 'none']
        paired[arm] = dict(correct_to_wrong=[pid for pid in dev_rows if baseline[pid]['correct_event'] and not rows[pid]['correct_event']],
            wrong_to_correct=[pid for pid in dev_rows if not baseline[pid]['correct_event'] and rows[pid]['correct_event']],
            unchanged_positive_type_timing_baseline=timing([baseline[pid] for pid in same]),
            unchanged_positive_type_timing_arm=timing([rows[pid] for pid in same]))

    # Bounded examples: all union-of-arm none FPs; representative confusions,
    # worst onset errors, one interval improvement and one correct positive.
    case_ids = []
    categories = {}
    def add(pid, reason):
        if pid not in categories:
            case_ids.append(pid); categories[pid] = []
        if reason not in categories[pid]: categories[pid].append(reason)
    none_fp = [pid for pid, r in control.items() if r['target']['next_event'] == 'none' and any(p[pid]['prediction']['next_event'] != 'none' for p in predictions.values())]
    for pid in sorted(none_fp): add(pid, 'none false positive in at least one arm')
    for truth, pred in (('contact','grasp'),('grasp','contact')):
        selected = [pid for pid,r in control.items() if r['target']['next_event'] == truth and r['prediction']['next_event'] == pred]
        selected.sort(key=lambda pid: (-max(control[pid]['event_probabilities']), pid))
        used = set()
        for pid in selected:
            source = dev_rows[pid]['source_id']
            if source not in used:
                add(pid, truth+' predicted as '+pred); used.add(source)
            if len(used) == 2: break
    worst = sorted((pid for pid,r in control.items() if r['correct_event'] and r['target']['next_event'] != 'none'),
                   key=lambda pid: (-control[pid]['start_error'],pid))
    used = set()
    for pid in worst:
        source = dev_rows[pid]['source_id']
        if source not in used:
            add(pid, 'large onset error with correct event type'); used.add(source)
        if len(used) == 3: break
    improvements = sorted((pid for pid,r in control.items() if r['correct_event'] and baseline[pid]['correct_event'] and r['target']['next_event'] != 'none'),
                          key=lambda pid: (-(control[pid]['positive_iou']-baseline[pid]['positive_iou']),pid))
    if improvements: add(improvements[0], 'largest interval-IoU improvement with type held fixed')
    good = sorted((pid for pid,r in control.items() if r['target']['next_event'] != 'none' and r['joint_two_frame']),
                  key=lambda pid: (-control[pid]['positive_iou'],pid))
    if good: add(good[0], 'correct event and interval example')
    cases = []
    for i,pid in enumerate(case_ids):
        row=dev_rows[pid]; c=candidates[row['source_id']]; a=row['anchor']
        events=[e for e in c['events'] if e['start'] < a+33 and e['end'] > a]
        cases.append(dict(case_id=f'case-{i+1:02d}', categories=categories[pid], row=row,
            predictions={arm:p[pid] for arm,p in predictions.items()},
            automatic_events_in_window=events,
            ongoing_events_at_anchor=[e for e in events if e['start'] <= a < e['end']],
            label_claim='Automatic predicate agreement only; no independent human or physical-truth validation added.'))

    result = seal_contract(dict(artifact_type='eventground_wam.week10_closeout_diagnostic',
        created_at=datetime.now(timezone.utc).isoformat(), scope='Existing train/dev only; no model calls, training, new recordings or report tuning.',
        dev_manifest_digest=manifest['digest'], train_manifest_digest=train['digest'],
        dev_n=396, dev_sources=103, dev_tasks=len({r['task_key'] for r in dev_rows.values()}),
        evaluations={a:c['digest'] for a,c in evaluations.items()}, arms=summaries, paired_to_sft=paired,
        temporal_context=temporal_context, training_reward_diagnostics=reward_summary,
        none_fp_union_prompt_ids=none_fp, case_count=len(cases),
        example_selection='All union none FPs; top-confidence two/source-distinct confusion examples each direction; top three source-distinct onset errors; top same-type interval improvement; one exact/high-IoU correct positive. Sorting ties by prompt ID. Examples are deliberately diagnostic, not representative sampling.',
        timing_definition='Conditional on correct positive type; onset error in simulator frames. Large-error descriptor >8 frames is not a promotion gate.',
        limitations=['Automatic weak labels; no independent missed-event recall or true-event accuracy established.',
                    'Single training seed and repeatedly used dev panel; no new confirmatory or causal algorithm comparison.',
                    'Current-observation forecasts learned on demonstrated trajectories; deployed-policy validity untested.',
                    'M4 historical source-alias exposure unresolved. No rollout or robot-success claim.',
                    'Horizon frequencies on balanced dev are not estimates of deployment event prevalence.'],
        input_files_sha256=inputs, prior_packages_verified=prior))
    completion = sealed(pilot/'runs/control/completion.json')
    identity = sealed(pilot/'runs/control/identity.json')
    base_done = sealed(future/'runs/sft/completion.json')
    assert completion['identity_digest'] == identity['digest']
    assert identity['sft_completion_digest'] == base_done['digest']
    gates = sealed(pilot/'analysis/results.json')['gates']['control']
    assert gates['promoted'] and all(completion['gates'].values())
    report_done = sealed(report/'completion.json')
    assert report_done['endpoint_sha256'] == completion['endpoint_sha256']
    report_results = sealed(report/'analysis/results.json')
    parent = sealed(Path('configs/training/week10_eventground_experiment_v3.json'))['parent']
    handoff = seal_contract(dict(artifact_type='eventground_wam.week10_checkpoint_handoff',
        status='Week 10 exploratory experiment closed; provisional event-predictor candidate only.',
        selected_arm='SFT-only continuation', selection_basis='Existing dev gate; fixed 32-update endpoint selected before its separate report evaluation.',
        remote_endpoint='/home/peimengsui/data/eventground-wam/libero-plus/results/week10-preservation-pilot-v1/runs/control/endpoint.pt',
        endpoint_sha256=completion['endpoint_sha256'], completion_digest=completion['digest'],
        identity_digest=identity['digest'], inference_header_digest=completion['inference_header']['digest'],
        parent_sft_completion_digest=base_done['digest'], parent_sft_endpoint_sha256=base_done['endpoint_sha256'],
        parent_m4=parent, task_definition=completion['task_definition'],
        dev_metrics=evaluations['control']['metrics'], report_results_digest=report_results['digest'],
        report_package_digest=prior[PACKAGES[-1]]['digest'], diagnostic_digest=result['digest'],
        training_completed=dict(initial_sft_updates=202, initial_sft_presentations=3218,
            continuation_updates=32, continuation_presentations=500),
        model_scope='Child M4 conditioning adapter plus new event head; only inherited language LoRA and event head trained. Action decoder, JEPA future core and coupling not called by event wrapper.',
        deployment_ready=False, independent_qa_passed=False,
        current_rlvr_disposition='Closed at development no-go on revised task; original small-panel pass preserved. Full/main RLVR unlaunched.',
        week11_status='Not started or authorized by this closeout; integration mapping and rollouts remain separate work.',
        limitations=result['limitations']+['Report +1.15 pp interval spans zero; no established reliable continuation improvement.',
            'Original independent-QA gate remains unmet; closure is of the separately approved weak-label study.']))
    out.mkdir()
    save(out/'diagnostic.json', result)
    save(out/'cases.json', seal_contract(dict(artifact_type='eventground_wam.week10_diagnostic_cases',diagnostic_digest=result['digest'],cases=cases)))
    save(out/'handoff.json', handoff)
    save(out/'prior-verification-before.json', seal_contract(dict(packages=prior,verified_at=datetime.now(timezone.utc).isoformat())))
    print(json.dumps(dict(output=str(out),cases=len(cases),prior_files=sum(p['files'] for p in prior.values()),
        control_confusion=summaries['control']['metrics']['confusion'],control_timing=summaries['control']['timing'],
        control_horizon=summaries['control']['horizon'],none_fp_union=len(none_fp),
        control_type_changes={k:len(paired['control'][k]) for k in ('correct_to_wrong','wrong_to_correct')},
        control_context={k:v for k,v in temporal_context['control'].items() if k!='wrong_positive_predictions'})))


if __name__ == '__main__':
    main()
