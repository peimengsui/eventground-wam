import copy
import json

import h5py
import numpy as np
import pytest
from week11_h8_validation import CONDITIONS, SUITES, paired_summary, prefix_pair, select_cases


def manifests():
    result = {}
    for panel in ('targeted-467', 'clean-200'):
        tasks = [{'suite': s, 'task_name': f'task{task}_view_{state}'}
                 for s in SUITES for task in range(10) for state in range(5)]
        result[panel] = {'tasks': tasks}
    return result


def test_selection_excludes_old_cases_balances_suites_and_unique_tasks():
    m = manifests(); excluded = {f'{p}-{i:04d}' for p in m for i in (0, 1, 50, 51, 100, 101)}
    cases = select_cases(m, excluded)
    assert len(cases) == 64 and not ({c['id'] for c in cases} & excluded)
    for p in m:
        for suite in SUITES:
            group = [c for c in cases if c['panel'] == p and c['task']['suite'] == suite]
            assert len(group) == len({c['canonical_task'] for c in group}) == 8


def test_selection_does_not_use_outcomes():
    m = manifests(); old = select_cases(m, set()); altered = copy.deepcopy(m)
    for entry in altered.values():
        for i, row in enumerate(entry['tasks']):
            row['success'] = i % 2 == 0
    assert [c['id'] for c in old] == [c['id'] for c in select_cases(altered, set())]


def test_paired_analysis_counts_discordance_and_zero_information():
    rows = [{'suite': 's', CONDITIONS[0]: False, CONDITIONS[1]: True} for _ in range(6)]
    got = paired_summary(rows)
    assert got['h8_only_success'] == 6 and got['vanilla_only_success'] == 0
    assert got['exact_two_sided_mcnemar_p'] == pytest.approx(.03125)
    assert got['paired_95_bootstrap_percentage_points'] == [100., 100.]
    for row in rows:
        row[CONDITIONS[0]] = True
    got = paired_summary(rows)
    assert got['net_successes'] == 0 and got['exact_two_sided_mcnemar_p'] == 1
    assert got['paired_95_bootstrap_percentage_points'] == [0., 0.]


def test_event_free_prefix_excludes_consumed_input_drift(tmp_path):
    paths = [tmp_path/f'{a}.hdf5' for a in ('a', 'b')]
    for p in paths:
        with h5py.File(p, 'w') as f:
            f['episode/metadata_json'] = json.dumps({'bddl_sha256': 'b', 'initial_state_sha256': 's',
                                                    'instruction': 'i', 'episode_seed': 17})
            for k in ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state'):
                f['steps/'+k] = np.zeros((26, 3))
            f['steps/executed_action'] = np.zeros((26, 7))
        p.with_suffix('.queries.jsonl').write_text('\n'.join(json.dumps({'policy': {'event': None}}) for _ in range(2)))
    with h5py.File(paths[1], 'r+') as f:
        f['steps/executed_action'][20, 0] = 1
        f['steps/wrist_rgb'][17, 0] = 1
    assert prefix_pair(*paths)['causally_matched']
    with h5py.File(paths[1], 'r+') as f:
        f['steps/wrist_rgb'][18, 0] = 1
    assert not prefix_pair(*paths)['causally_matched']
