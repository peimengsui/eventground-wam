from pathlib import Path
import pytest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed
from eventground_wam.eventground_weak_evaluation import heldout_rows
ROOT=Path(__file__).parents[1]

def fake_records(panel):
    from eventground_wam.instrumented_supervision import labeler_contract
    split=sealed(ROOT/'docs/evaluation/week10-source-split-v3/partitions.json')
    meta=sealed(ROOT/f'docs/evaluation/week10-evaluation-preparation-v1/{panel}-source-metadata.json')
    records=[]
    for i,s in enumerate(meta['rows']):
        events=[] if i%4==0 else [dict(event_id=s['source_id']+':event-0',event_type=('contact','grasp','gripper_close')[i%3],
                    object_id='obj',start=20,end=24,left_censored=False,right_censored=False)]
        c=seal_contract(dict(source_id=s['source_id'],recording_id='record-'+s['source_id'],frames=s['frames'],
            labeler_digest=labeler_contract()['digest'],labels_admitted=False,events=events,ambiguities=[]))
        records.append(dict(candidate=c,task_key=s['task_key'],instruction='test',recording_path='/unopened',recording_sha256='0'*64))
    return records,split,meta

@pytest.mark.parametrize('panel',['event-dev','event-report'])
def test_heldout_sampler_deterministic_causal_and_capped(panel):
    from collections import Counter
    records,split,meta=fake_records(panel);a=heldout_rows(records,split,meta);b=heldout_rows(records,split,meta)
    assert a==b and a['selected_count']>0
    assert len({r['prompt_id'] for r in a['rows']})==len(a['rows'])
    assert max(Counter(r['source_id'] for r in a['rows']).values())<=16
    assert max(Counter(r['event_id'] for r in a['rows'] if r['event_id']).values())<=4
    assert all(r['anchor']+33<=next(s['frames'] for s in meta['rows'] if s['source_id']==r['source_id']) for r in a['rows'])
    if panel=='event-dev':assert len(set(a['selected_by_class'].values()))==1
    else:assert all(r['anchor']%8==0 for r in a['rows']) and len(a['rows'])<=1232
