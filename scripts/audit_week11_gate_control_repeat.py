"""Locate the first query difference for controls whose fresh outcome changed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import read_episode_metadata


def run(raw, summary_path, output):
    summary, cfg = read_sealed(summary_path), read_sealed(raw/'contract.json')
    cases = {r['id']: r for r in cfg['cases']}
    results = []
    for row in summary['rows']:
        for arm, reproduced in row['control_outcomes_reproduced'].items():
            if reproduced:
                continue
            old = Path(cases[row['id']]['artifacts'][arm])
            new = Path(row['arms'][arm]['artifact'])
            assert sha256_file(old) == cfg['input_sha256'][str(old)]
            receipt = read_sealed(new.parent/f"result-{row['ordinal']:04d}.json")
            assert sha256_file(new) == receipt['result']['artifact_sha256']
            old_meta, new_meta = read_episode_metadata(old), read_episode_metadata(new)
            for key in ('suite', 'runtime_task_index', 'instruction', 'episode_seed',
                        'initial_state_sha256', 'bddl_sha256'):
                assert old_meta[key] == new_meta[key], key
            oq = [json.loads(line) for line in old.with_suffix('.queries.jsonl').read_text().splitlines()]
            nq = [json.loads(line) for line in new.with_suffix('.queries.jsonl').read_text().splitlines()]
            exact, difference = 0, None
            with h5py.File(old) as oh, h5py.File(new) as nh:
                a, b = oh['steps'], nh['steps']
                for i in range(min(len(oq), len(nq))):
                    step = 10+8*i
                    camera = {}
                    for key in ('agent_rgb', 'wrist_rgb'):
                        delta = b[key][step].astype(np.int16)-a[key][step].astype(np.int16)
                        camera[key] = {'different_channel_values': int(np.count_nonzero(delta)),
                                       'max_absolute_difference': int(np.abs(delta).max())}
                    props_equal = np.array_equal(a['proprioception'][step].astype(np.float32),
                                                 b['proprioception'][step].astype(np.float32))
                    inputs_equal = props_equal and all(v['different_channel_values'] == 0 for v in camera.values())
                    actions_equal = np.array_equal(a['predicted_action_chunk'][step], b['predicted_action_chunk'][step])
                    if inputs_equal:
                        assert actions_equal, 'Same-input control produced different actions'
                        assert oq[i]['policy']['prediction'] == nq[i]['policy']['prediction']
                        exact += 1
                    else:
                        difference = {'query': i, 'step': step, 'cameras': camera,
                            'proprioception_float32_exact': bool(props_equal),
                            'simulator_state_exact': bool(np.array_equal(a['mujoco_state'][step], b['mujoco_state'][step])),
                            'previous_executed_actions_exact': bool(np.array_equal(a['executed_action'][:step], b['executed_action'][:step])),
                            'action_chunk_exact': bool(actions_equal),
                            'old_horizon': oq[i]['policy']['used_horizon'], 'new_horizon': nq[i]['policy']['used_horizon'],
                            'old_prediction': oq[i]['policy']['prediction'], 'new_prediction': nq[i]['policy']['prediction']}
                        break
            results.append({'case': row['id'], 'arm': arm, 'old_success': old_meta['success'],
                'new_success': new_meta['success'], 'matching_initial_query_prefix': exact,
                'first_input_difference': difference, 'old_artifact_sha256': sha256_file(old),
                'new_artifact_sha256': sha256_file(new)})
    atomic_json(output, seal_contract({'summary_digest': summary['digest'], 'cases': results,
        'selection': 'All fresh controls with a changed historical outcome; no episodes excluded.',
        'source_sha256': sha256_file(Path(__file__)), 'model_calls': 0, 'simulator_steps': 0}))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.summary, args.output)
