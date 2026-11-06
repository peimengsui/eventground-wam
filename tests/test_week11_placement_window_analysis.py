"""Causal prefix acceptance must allow the intervention but reject prior drift."""
import json
import shutil

import h5py
import numpy as np
from analyze_week11_placement_window import compare


def recording(path):
    with h5py.File(path,'w') as f:
        f.create_dataset('episode/metadata_json',data=json.dumps({'success':False}))
        for key in ('agent_rgb','wrist_rgb'):
            f.create_dataset('steps/'+key,data=np.zeros((26,2,2,3),dtype=np.uint8))
        for key in ('proprioception','mujoco_state','executed_action'):
            f.create_dataset('steps/'+key,data=np.zeros((26,7)))
        f.create_dataset('steps/predicted_action_chunk',data=np.zeros((26,8,7)))
    q={'policy':{'prediction':'none','decision':{'horizon':8},
                 'event':{'categorical_probabilities':{'event':[1,0,0,0]}}}}
    path.with_suffix('.queries.jsonl').write_text('\n'.join(json.dumps(q) for _ in range(2)))


def test_intervention_action_may_change_but_its_input_must_match(tmp_path):
    a,b=tmp_path/'a.hdf5',tmp_path/'b.hdf5';recording(a);recording(b)
    with h5py.File(b,'r+') as f:
        f['steps/executed_action'][18,0]=1
        f['steps/predicted_action_chunk'][18,0,0]=1
    result=compare(a,b,18)
    assert result['all_prefix_and_inputs_exact'] and not result['intervention_chunks_equal']
    with h5py.File(b,'r+') as f:f['steps/wrist_rgb'][18,0,0,0]=1
    assert not compare(a,b,18)['all_prefix_and_inputs_exact']


def test_prior_drift_and_forecast_mismatch_are_rejected(tmp_path):
    a,b=tmp_path/'a.hdf5',tmp_path/'b.hdf5';recording(a);recording(b)
    with h5py.File(b,'r+') as f:f['steps/mujoco_state'][17,0]=1e-12
    assert not compare(a,b,18)['all_prefix_and_inputs_exact']
    shutil.copyfile(a,b)
    qs=[json.loads(l) for l in b.with_suffix('.queries.jsonl').read_text().splitlines()]
    qs[1]['policy']['prediction']='changed'
    b.with_suffix('.queries.jsonl').write_text('\n'.join(json.dumps(q) for q in qs))
    assert not compare(a,b,18)['all_prefix_and_inputs_exact']
