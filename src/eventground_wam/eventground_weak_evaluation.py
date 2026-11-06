"""Prospective exploratory event evaluation; automatic labels are not independent QA."""
from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
import numpy as np
from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_manifest import CLASSES, _sample, lead_bin, require_sealed
from eventground_wam.eventground_weak_supervision import audit_row
from eventground_wam.instrumented_labeling import prompt_previews
from eventground_wam.instrumented_supervision import labeler_contract


def panel_sources(partitions, metadata, panel):
    require_sealed(partitions); require_sealed(metadata)
    if panel not in ('event-dev', 'event-report') or metadata['panel'] != panel:
        raise ValueError('Explicit held-out panel required')
    rows = [r for r in partitions['rows'] if r['partition'] == panel]
    if metadata['partition_digest'] != partitions['digest'] or metadata['rows'] != rows:
        raise ValueError('Held-out membership/order differs')
    if len(rows) != {'event-dev': 103, 'event-report': 77}[panel]:
        raise ValueError('Incomplete panel')
    if len({r['source_id'] for r in partitions['rows']}) != len(partitions['rows']):
        raise ValueError('Cross-partition source overlap')
    return rows


def validate_panel_contract(contract, partitions, metadata):
    require_sealed(contract)
    rows = panel_sources(partitions, metadata, contract['panel'])
    if contract['source_ids'] != [r['source_id'] for r in rows]:
        raise ValueError('Panel selection changed')
    if contract['independent_qa_passed'] is not False or contract['automatic_retries'] != 0:
        raise ValueError('Weak qualification/retry identity changed')
    if contract['panel'] == 'event-report' and not contract.get('training_decision_digest'):
        raise ValueError('Report is locked until pilot/main/control decisions are sealed')
    return rows


def pilot_rows(dataset):
    require_sealed(dataset)
    rows = _sample(dataset['main_rows'], 125)
    if Counter(r['target']['next_event'] for r in rows) != {
        'contact': 125, 'grasp': 29, 'gripper_close': 125, 'none': 125}:
        raise ValueError('Unexpected frozen nested pilot coverage')
    return sorted(rows, key=lambda r: (hashlib.sha256(f"17:rlvr:{r['prompt_id']}".encode()).hexdigest(), r['prompt_id']))


def heldout_rows(records, partitions, metadata):
    sources = panel_sources(partitions, metadata, metadata['panel'])
    if [r['candidate']['source_id'] for r in records] != [r['source_id'] for r in sources]:
        raise ValueError('Missing/reordered held-out source')
    eligible, exclusions = [], Counter()
    for record, source in zip(records, sources, strict=True):
        c = require_sealed(record['candidate'])
        if (c['labeler_digest'] != labeler_contract()['digest'] or c['frames'] != source['frames']
                or record['task_key'] != source['task_key'] or c['labels_admitted']):
            raise ValueError('Frozen labeler/source mismatch')
        for p in prompt_previews(c)['rows']:
            if p['status'] != 'unreviewed_preview':
                exclusions[p['reason']] += 1
                continue
            row = dict(prompt_id=hashlib.sha256(f"{source['source_id']}:{p['anchor']}".encode()).hexdigest(),
                       source_id=source['source_id'], recording_id=c['recording_id'], task_key=source['task_key'],
                       anchor=p['anchor'], event_id=p['event_id'], lead_time_bin=lead_bin(p['horizon']),
                       recording_path=record['recording_path'], recording_sha256=record['recording_sha256'],
                       instruction=record['instruction'], input_datasets=['observation/front','observation/wrist','observation/state'],
                       target=dict(next_event=p['event_type'],interval=p['interval'],horizon=p['horizon']),
                       label_metadata=dict(end_censored=p['end_censored'], candidate_digest=c['digest']),
                       weak_label_provenance=dict(origin='automatic_physical_predicate', independent_qa_passed=False,
                                                  human_reviewed=False, negative_windows_human_reviewed=False))
            audit_row(row,c)
            eligible.append(row)
    if metadata['panel'] == 'event-dev':
        preliminary = _sample(eligible,250)
        count = min(Counter(r['target']['next_event'] for r in preliminary).get(k,0) for k in CLASSES)
        # Trimming preserves every global cap; never resample to improve model results.
        counts = Counter(); selected=[]
        for row in preliminary:
            kind=row['target']['next_event']
            if counts[kind] < count: selected.append(row);counts[kind]+=1
    else:
        grid=[r for r in eligible if r['anchor']%8==0]
        by_task=defaultdict(lambda:defaultdict(list))
        for r in sorted(grid,key=lambda r:(r['anchor'],r['prompt_id'])):
            by_task[r['task_key']][r['source_id']].append(r)
        selected=[]; events=Counter(); sources_count=Counter()
        while True:
            progress=False
            for task in sorted(by_task):
                for source in sorted(by_task[task]):
                    while by_task[task][source] and sources_count[source]<16:
                        r=by_task[task][source].pop(0)
                        if r['event_id'] and events[r['event_id']]>=4: continue
                        selected.append(r);sources_count[source]+=1
                        if r['event_id']:events[r['event_id']]+=1
                        progress=True;break
            if not progress or len(selected)>=3000:break
    return seal_contract(dict(artifact_type='eventground_wam.weak_heldout_manifest', panel=metadata['panel'],
        partition_digest=partitions['digest'], source_metadata_digest=metadata['digest'], independent_qa_passed=False,
        source_count=len(records), source_candidate_digests={r['candidate']['source_id']:r['candidate']['digest'] for r in records},
        exclusions=dict(exclusions), eligible_count=len(eligible), selected_count=len(selected),
        selected_by_class=dict(Counter(r['target']['next_event'] for r in selected)),
        eligible_by_class=dict(Counter(r['target']['next_event'] for r in eligible)),
        stride_grid_count=sum(r['anchor']%8==0 for r in eligible), rows=selected,
        capped_or_balance_omissions=len(eligible)-len(selected), training_authorized=False,
        label_quality='unreviewed automatic physical predicates; uncertainties excluded; no independent QA'))


def weak_reward(prediction, row, candidate, weights=(.4,.25,.2,.15)):
    """Score extractor agreement, explicitly not independently verified physical truth."""
    if (row['prompt_id'] != hashlib.sha256(f"{row['source_id']}:{row['anchor']}".encode()).hexdigest()
            or row['label_metadata']['candidate_digest'] != candidate['digest']):
        raise ValueError('Weak reward source/anchor/candidate binding differs')
    event=audit_row(row,candidate)
    if tuple(weights) not in ((.4,.25,.2,.15),(.3,.35,.2,.15)):
        raise ValueError('Unfrozen reward weights')
    if not valid_prediction(prediction):return 0.
    target=row['target']; iou=timing=predicate=0.
    if prediction['next_event']==target['next_event']:
        if event is None:iou=timing=predicate=1.
        else:
            a,b=prediction['interval'];c,d=target['interval']
            intersection=max(0,min(b,d)-max(a,c));iou=intersection/(b-a+d-c-intersection)
            timing=max(0.,1-abs(a-c)/32)
            predicate=float(event['start']-row['anchor']<=a<event['end']-row['anchor'])
    return sum(w*v for w,v in zip(weights,(iou,timing,predicate,1.),strict=True))


def valid_prediction(p):
    if not isinstance(p,dict) or set(p)!={'next_event','interval','horizon'} or p['next_event'] not in CLASSES:return False
    if p['next_event']=='none':return p['interval'] is None and p['horizon'] is None
    x=p['interval']
    return (isinstance(x,list) and len(x)==2 and all(type(i)is int for i in x)
            and 1<=x[0]<x[1]<=33 and type(p['horizon'])is int and p['horizon']==x[0])


def per_example(row,pred):
    target=row['target'];correct=valid_prediction(pred) and pred['next_event']==target['next_event']
    positive=target['next_event']!='none';iou=0.;start=end=None
    if correct and positive:
        a,b=pred['interval'];c,d=target['interval'];x=max(0,min(b,d)-max(a,c))
        iou=x/(b-a+d-c-x);start=abs(a-c);end=abs(b-d)
    return dict(prompt_id=row['prompt_id'],source_id=row['source_id'],target=target,prediction=pred,
                valid=valid_prediction(pred),correct_event=correct,positive_iou=iou,
                start_error=start,end_error=end,end_censored=row['label_metadata']['end_censored'],
                lead_time_bin=row['lead_time_bin'],joint_two_frame=bool(correct and (not positive or max(start,end)<=2)))


def metrics(records):
    perclass={}
    for k in CLASSES:
        subset=[r for r in records if r['target']['next_event']==k]
        matched=[r for r in subset if r['correct_event']]
        predicted=sum(r['prediction']['next_event']==k for r in records)
        perclass[k]=dict(n=len(subset),matched=len(matched),recall=len(matched)/len(subset) if subset else None,
            precision=len(matched)/predicted if predicted else None,
            typed_iou=sum(r['positive_iou'] for r in subset)/len(subset) if subset and k!='none' else None)
    positives=[r for r in records if r['target']['next_event']!='none']; matched=[r for r in positives if r['correct_event']]
    none=[r for r in records if r['target']['next_event']=='none']
    ious=[perclass[k]['typed_iou'] for k in CLASSES if k!='none']
    return dict(n=len(records),per_class=perclass,macro_positive_iou=float(np.mean(ious)) if all(x is not None for x in ious) else None,
        none_false_positive_rate=sum(not r['correct_event'] for r in none)/len(none) if none else None,
        matched_positive_start_mae=float(np.mean([r['start_error'] for r in matched])) if matched else None,
        matched_positive_end_mae=float(np.mean([r['end_error'] for r in matched])) if matched else None,
        matched_positive_coverage=len(matched)/len(positives) if positives else None,
        joint_two_frame_accuracy=sum(r['joint_two_frame'] for r in records)/len(records) if records else None,
        format_valid_fraction=sum(r['valid'] for r in records)/len(records) if records else None,
        confusion={a:{b:sum(r['target']['next_event']==a and r['prediction']['next_event']==b for r in records) for b in CLASSES} for a in CLASSES},
        end_censored_count=sum(r['end_censored'] for r in records))


def promotion(sft,pilot,integrity):
    values=(sft['macro_positive_iou'],pilot['macro_positive_iou'],sft['none_false_positive_rate'],
            pilot['none_false_positive_rate'],sft['matched_positive_start_mae'],pilot['matched_positive_start_mae'])
    defined=all(x is not None and np.isfinite(x) for x in values)
    gates=dict(metrics_defined=defined,all_classes_present=all(sft['per_class'][k]['n']>0 for k in CLASSES),
               format_valid=pilot['format_valid_fraction']==1.,integrity=integrity is True,
               iou_gain=bool(defined and values[1]-values[0]>=.01),
               none_fp=bool(defined and values[3]-values[2]<=.02),
               timing=bool(defined and values[5]-values[4]<=1.))
    return dict(promoted=all(gates.values()),gates=gates,interpretation='exploratory weak-label development gate only')
