import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_week11_first_chunk import prefix


def recording(path):
    with h5py.File(path,'w') as f:
        s=f.create_group('steps')
        for k in ('agent_rgb','wrist_rgb'):s[k]=np.zeros((34,2,2,3),dtype=np.uint8)
        for k in ('proprioception','mujoco_state'):s[k]=np.zeros((34,8),dtype=np.float64)
        s['executed_action']=np.zeros((34,7))
        s['predicted_action_chunk']=np.zeros((34,8,7))


def test_current_action_can_differ_but_inputs_and_prior_actions_must_match(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5';recording(a);recording(b)
    with h5py.File(b,'r+') as f:
        f['steps/predicted_action_chunk'][18,0,0]=.1
        f['steps/executed_action'][18,0]=.1
    check=prefix(a,b,18)
    assert check['all_prefix_and_inputs_exact'] and not check['intervention_chunks_equal']
    assert not prefix(a,b,26)['all_prefix_and_inputs_exact']


def test_single_camera_level_or_prior_state_difference_invalidates_pairing(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5';recording(a);recording(b)
    with h5py.File(b,'r+') as f:f['steps/wrist_rgb'][18,0,0,0]=1
    check=prefix(a,b,18)
    assert not check['all_prefix_and_inputs_exact'] and not check['input_at_intervention']['wrist_rgb']
    with h5py.File(b,'r+') as f:
        f['steps/wrist_rgb'][18,0,0,0]=0
        f['steps/mujoco_state'][12,0]=1e-12
    assert not prefix(a,b,18)['all_prefix_and_inputs_exact']
