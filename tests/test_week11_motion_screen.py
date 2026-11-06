import itertools
import json

import h5py
import numpy as np
from week11_motion_screen import CONDITIONS, decision, prefix_pair


def matrix():
    cases = [{'id': 'retained'}, {'id': 'rescued'}]
    rows = [{'case': c['id'], 'condition': arm, 'repeat': rep,
             'success': c['id'] == 'retained' or arm == CONDITIONS[1]}
            for c in cases for rep in (1, 2, 3) for arm in CONDITIONS]
    pairs = [{'case': c['id'], 'repeat': rep, 'conditions': list(arms), 'causally_matched': True}
             for c in cases for rep in (1, 2, 3) for arms in itertools.combinations(CONDITIONS, 2)]
    return cases, rows, pairs


def test_pass_requires_retention_and_rescue():
    c, r, p = matrix()
    assert decision(c, r, p, integrity_valid=True)['advance_to_27_cases']
    next(x for x in r if x['case'] == 'retained' and x['condition'] == CONDITIONS[1])['success'] = False
    got = decision(c, r, p, integrity_valid=True)
    assert got['status'] == 'failed' and got['h8_success_cases_lost'] == ['retained']


def test_unstable_control_or_unmatched_decisive_pair_is_inconclusive():
    c, r, p = matrix()
    p[0]['causally_matched'] = False
    assert decision(c, r, p, integrity_valid=True)['status'] == 'inconclusive'
    p[0]['causally_matched'] = True
    next(x for x in r if x['case'] == 'rescued' and x['condition'] == CONDITIONS[0])['success'] = True
    assert decision(c, r, p, integrity_valid=True)['status'] == 'inconclusive'


def test_no_rescue_or_integrity_failure_cannot_advance():
    c, r, p = matrix()
    assert decision(c, r, p, integrity_valid=False)['status'] == 'invalid'
    for x in r:
        if x['case'] == 'rescued':
            x['success'] = False
    assert decision(c, r, p, integrity_valid=True)['status'] == 'failed'


def test_unstable_matched_control_is_inconclusive():
    c, r, p = matrix()
    next(x for x in r if x['case'] == 'rescued' and x['condition'] == CONDITIONS[2])['success'] = True
    assert decision(c, r, p, integrity_valid=True)['status'] == 'inconclusive'


def recording(path):
    with h5py.File(path, 'w') as f:
        f['episode/metadata_json'] = json.dumps({'bddl_sha256': 'b', 'initial_state_sha256': 's',
                                                'instruction': 'i', 'episode_seed': 17})
        for name in ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state'):
            f['steps/'+name] = np.zeros((26, 3))
        f['steps/executed_action'] = np.zeros((26, 7))
    p = {'prediction': {}, 'decision': {}, 'event': {'categorical_probabilities': {}}}
    path.with_suffix('.queries.jsonl').write_text('\n'.join(json.dumps({'policy': p}) for _ in range(2)))


def test_first_executed_change_and_consumed_images(tmp_path):
    a, b = tmp_path/'a.hdf5', tmp_path/'b.hdf5'
    recording(a); recording(b)
    with h5py.File(b, 'r+') as f:
        f['steps/executed_action'][20, 0] = 1
        f['steps/wrist_rgb'][17, 0] = 1  # Not consumed by the policy.
    got = prefix_pair(a, b)
    assert got['first_changed_executed_frame'] == 20 and got['query_frames'] == [10, 18]
    assert got['causally_matched']
    with h5py.File(b, 'r+') as f:
        f['steps/wrist_rgb'][18, 0] = 1
    assert not prefix_pair(a, b)['causally_matched']


def test_no_changed_action_still_requires_entire_consumed_prefix(tmp_path):
    a, b = tmp_path/'a.hdf5', tmp_path/'b.hdf5'
    recording(a); recording(b)
    assert prefix_pair(a, b)['causally_matched']
    with h5py.File(b, 'r+') as f:
        f['steps/mujoco_state'][25, 0] = 1
    assert not prefix_pair(a, b)['causally_matched']
