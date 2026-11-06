import copy
from pathlib import Path
import pytest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed
from eventground_wam.eventground_weak_evaluation import (panel_sources,pilot_rows,weak_reward,heldout_rows,per_example,metrics,promotion)
ROOT=Path(__file__).parents[1]

def test_real_frozen_pilot_unique_caps():
    data=sealed(ROOT/'docs/evaluation/week10-exploratory-weak-v4/dataset.json');rows=pilot_rows(data)
    assert len(rows)==404==len({r['prompt_id'] for r in rows})
    assert {r['prompt_id'] for r in rows}<={r['prompt_id'] for r in data['main_rows']}
    from collections import Counter
    assert max(Counter(r['source_id'] for r in rows).values())<=16
    assert max(Counter(r['event_id'] for r in rows if r['event_id']).values())<=4

def test_panel_rejects_cross_partition_and_reordering():
    split=sealed(ROOT/'docs/evaluation/week10-source-split-v3/partitions.json')
    meta=sealed(ROOT/'docs/evaluation/week10-evaluation-preparation-v1/event-dev-source-metadata.json')
    assert len(panel_sources(split,meta,'event-dev'))==103
    bad=copy.deepcopy(meta);bad['rows'][0]=next(r for r in split['rows'] if r['partition']=='train');bad.pop('digest');bad=seal_contract(bad)
    with pytest.raises(ValueError):panel_sources(split,bad,'event-dev')

def test_real_weak_rewards_rederive_targets():
    data=sealed(ROOT/'docs/evaluation/week10-exploratory-weak-v4/dataset.json')
    for kind in ('none','contact','grasp','gripper_close'):
        row=next(r for r in data['main_rows'] if r['target']['next_event']==kind)
        c=sealed(ROOT/f"docs/evaluation/week10-production-labels-v1/run/{row['recording_id']}/event-candidates.json")
        assert weak_reward(row['target'],row,c)==pytest.approx(1.)
        assert weak_reward({},row,c)==0
        wrong=dict(next_event='gripper_close' if kind=='none' else 'none',interval=[1,2] if kind=='none' else None,horizon=1 if kind=='none' else None)
        assert weak_reward(wrong,row,c)==pytest.approx(.15)
        bad=copy.deepcopy(row);bad['anchor']+=1
        with pytest.raises(ValueError):weak_reward(row['target'],bad,c)

def test_gate_missing_metrics_never_passes():
    empty=metrics([]);assert not promotion(empty,empty,True)['promoted']
    data=sealed(ROOT/'docs/evaluation/week10-exploratory-weak-v4/dataset.json')
    rows=[next(r for r in data['main_rows'] if r['target']['next_event']==k) for k in ('contact','grasp','gripper_close','none')]
    results=[per_example(r,r['target']) for r in rows];m=metrics(results)
    assert m['macro_positive_iou']==1 and m['none_false_positive_rate']==0
    assert not promotion(m,m,True)['promoted']
    old=copy.deepcopy(m);old['macro_positive_iou']=.98
    assert promotion(old,m,True)['promoted']
    assert not promotion(old,m,False)['promoted']
