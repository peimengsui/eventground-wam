"""Counts-only diagnosis of already extracted events; never builds alternate targets."""
from collections import Counter
import argparse
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed, sha
from eventground_wam.instrumented_labeling import prompt_previews


def diagnose(paths):
    stages=Counter();ongoing_types=Counter();ambiguity_types=Counter();inputs={}
    eligible_sources=set();eligible_events=set();raw=Counter();examples=[]
    for p in paths:
        c=sealed(p);inputs[str(p)]=sha(p);raw.update(e['event_type'] for e in c['events'])
        previews=prompt_previews(c)['rows']
        for anchor in range(c['frames']):
            stop=anchor+33
            future=[e for e in c['events'] if anchor<e['start']<stop and not e['left_censored']]
            if not future or future[0]['event_type']!='grasp':continue
            stages['earliest_future_event_is_grasp']+=1
            if stop>c['frames']:
                stages['excluded_incomplete_future']+=1
                continue
            ongoing=[e for e in c['events'] if e['start']<=anchor<e['end']]
            uncertain=[e for e in c['ambiguities'] if e['start']<stop and e['end']>anchor]
            if ongoing:
                stages['excluded_ongoing_event']+=1
                ongoing_types.update({e['event_type'] for e in ongoing})
                if uncertain:stages['ongoing_also_has_ambiguity']+=1
                continue
            if uncertain:
                stages['excluded_ambiguity_after_ongoing_filter']+=1
                ambiguity_types.update({e['event_type'] for e in uncertain})
                continue
            assert previews[anchor]['status']=='unreviewed_preview' and previews[anchor]['event_type']=='grasp'
            stages['eligible_grasp_anchors']+=1
            eligible_sources.add(c['source_id']);eligible_events.add(future[0]['event_id'])
            if anchor%8==0:stages['eligible_grasp_on_stride8']+=1
        if len(examples)<3:
            for e in c['events']:
                if e['event_type']!='grasp':continue
                start=e['start'];anchor=start-1
                if anchor<0:continue
                ongoing=[{k:x[k] for k in ['event_type','start','end']} for x in c['events'] if x['start']<=anchor<x['end']]
                if ongoing:
                    examples.append(dict(recording_id=c['recording_id'],grasp_start=start,anchor=anchor,
                        anchor_status=previews[anchor],ongoing_events=ongoing))
                    break
    assert stages['earliest_future_event_is_grasp']==sum(stages[k] for k in ['excluded_incomplete_future','excluded_ongoing_event','excluded_ambiguity_after_ongoing_filter','eligible_grasp_anchors'])
    return dict(source_count=len(inputs),raw_event_counts=dict(raw),sequential_anchor_counts=dict(stages),
                eligible_grasp_sources=len(eligible_sources),eligible_grasp_events=len(eligible_events),
                excluded_ongoing_types_nonexclusive=dict(ongoing_types),excluded_ambiguity_types_nonexclusive=dict(ambiguity_types),
                examples=examples,input_sha256=inputs,alternate_prompt_rows_generated=0,
                interpretation='Counts of anchors whose earliest known future onset would be grasp, before the unchanged eligibility filters; filters applied in their frozen order. Not corrected labels.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=sorted(a.labels.glob('*/event-candidates.json'))
    if not paths:raise ValueError('No candidates')
    value=seal_contract(dict(artifact_type='eventground_wam.grasp_coverage_causal_count_audit',**diagnose(paths)))
    with a.output.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['input_sha256','examples']}))


if __name__=='__main__':main()
