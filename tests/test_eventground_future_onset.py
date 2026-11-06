import copy
import hashlib
import pytest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_future_onset import TASK,target_at,audit_row,weak_reward,build_manifest,schedule
from eventground_wam.eventground_weak_supervision import audit_row as old_audit
from eventground_wam.instrumented_supervision import labeler_contract


def event(kind,start,end,**kw):
    return dict(event_type=kind,start=start,end=end,object_id='object',event_id=f'{kind}:{start}',left_censored=False,right_censored=False,**kw)

def candidate(events,ambiguities=()):
    return seal_contract(dict(source_id='source',recording_id='recording',frames=80,events=events,ambiguities=list(ambiguities),labels_admitted=False,labeler_digest=labeler_contract()['digest']))

def row(c,anchor=10):
    t,e,z=target_at(c,anchor)
    from eventground_wam.eventground_manifest import lead_bin
    return dict(task_definition=TASK,source_id=c['source_id'],recording_id=c['recording_id'],anchor=anchor,
        prompt_id=hashlib.sha256(f"{c['source_id']}:{anchor}".encode()).hexdigest(),target=t,event_id=e['event_id'] if e else None,
        lead_time_bin=lead_bin(t['horizon']),label_metadata=dict(candidate_digest=c['digest'],end_censored=z),
        input_datasets=['observation/front','observation/wrist','observation/state'])


def test_ongoing_contact_next_grasp_and_old_stays_rejected():
    c=candidate([event('contact',5,40),event('grasp',20,45)])
    r=row(c)
    assert r['target']==dict(next_event='grasp',interval=[10,33],horizon=10)
    assert audit_row(r,c)['event_type']=='grasp'
    assert weak_reward(r['target'],r,c)==1
    with pytest.raises(ValueError,match='ongoing'):old_audit(r,c)


def test_none_means_no_new_onset_while_active():
    c=candidate([event('contact',5,60)])
    r=row(c);assert r['target']['next_event']=='none';assert weak_reward(r['target'],r,c)==1


def test_strict_future_horizon_ties_and_censoring():
    c=candidate([event('contact',10,50),event('gripper_close',42,45),event('contact',42,44),event('grasp',42,60)])
    r=row(c);assert r['event_id']=='grasp:42';assert r['target']['interval']==[32,33];assert r['label_metadata']['end_censored']
    assert target_at(candidate([event('grasp',43,60)]),10)[0]['next_event']=='none'


def test_left_censored_is_never_new_and_incomplete_excluded():
    e=event('grasp',20,40);e['left_censored']=True
    assert target_at(candidate([e]),10)[0]['next_event']=='none'
    with pytest.raises(ValueError,match='incomplete'):target_at(candidate([]),48)


@pytest.mark.parametrize('a,b',[(9,11),(10,11),(42,44)])
def test_all_ambiguities_remain_excluded(a,b):
    c=candidate([event('contact',5,30),event('grasp',20,40)],[dict(start=a,end=b)])
    with pytest.raises(ValueError,match='ambiguity'):target_at(c,10)


@pytest.mark.parametrize('change',[lambda r:r.update(task_definition='old'),lambda r:r['target'].update(horizon=9),lambda r:r.update(event_id='other'),lambda r:r.update(prompt_id='other'),lambda r:r['input_datasets'].append('action')])
def test_mutated_reward_binding_rejected(change):
    c=candidate([event('grasp',20,40)]);r=row(c);change(r)
    with pytest.raises(ValueError):weak_reward(dict(next_event='none',interval=None,horizon=None),r,c)


def test_actual_reconstruction_and_matched_exposure():
    from pathlib import Path
    from eventground_wam.eventground_weak_supervision import sealed
    base=Path('docs/evaluation')/TASK
    if not (base/'source-index.json').exists():pytest.skip('Run source preparation first')
    index=sealed(base/'source-index.json')
    for panel in ('train','dev','report'):assert build_manifest(index,panel)==sealed(base/f'{panel}-manifest.json')
    train=sealed(base/'train-manifest.json')
    assert schedule(train,'sft')['batch_sizes']==([16]*100+[9])*2
    assert schedule(train,'primary')['batch_sizes']==[16]*31+[4]
    assert schedule(train,'main')['batches']==schedule(train,'control')['batches']
