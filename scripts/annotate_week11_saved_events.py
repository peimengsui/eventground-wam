"""Reconstruct physical event candidates from saved states, without integration."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import signal
import time
from pathlib import Path

import h5py
import numpy as np
from week11_replay_common import read, sha, write


def run(args):
    cfg = read(args.bundle/'annotation-contract.json')
    replay = read(args.bundle/'contract.json')
    assert replay['digest'] == cfg['replay_contract_digest']
    for name, digest in cfg['code_sha256'].items():
        assert sha(args.bundle/name) == digest, name
    for name, digest in cfg['frozen_source_sha256'].items():
        assert sha(Path(cfg['formal_root'])/name) == digest, name
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'contract.json', cfg)
    start = time.monotonic(); stage_calls = 0; results = []; env = None
    result = {'status': 'incomplete', 'simulator_steps': 0, 'render_calls': 0,
              'model_inference_calls': 0, 'independent_human_qa': False}

    def forbidden(*_a, **_kw):
        raise RuntimeError('Saved-state annotation forbids integration and rendering')

    def deadline(*_a):
        raise TimeoutError('Saved-state annotation time ceiling')

    signal.signal(signal.SIGALRM, deadline); signal.alarm(1800)
    try:
        import mujoco

        mujoco.mj_step = forbidden; mujoco.mj_step2 = forbidden
        step1 = mujoco.mj_step1

        def checked(m, d):
            nonlocal stage_calls
            t, v, act, q = d.time, d.qvel.copy(), d.act.copy(), d.qpos.copy()
            mujoco.mj_normalizeQuat(m, q); step1(m, d); stage_calls += 1
            assert stage_calls <= 216 and d.time == t and np.array_equal(d.qvel, v)
            assert np.array_equal(d.act, act) and np.array_equal(d.qpos, q)

        mujoco.mj_step1 = checked
        from libero.libero.envs.env_wrapper import ControlEnv
        from record_week10_production_v2 import contact_rows, geometry_map, native
        from robosuite.utils.binding_utils import MjSim

        from eventground_wam.episode_recorder import read_episode_metadata
        from eventground_wam.instrumented_labeling import RecordingEvidence, extract_events
        from eventground_wam.libero_bddl import resolve_libero_bddl_paths

        MjSim.step = forbidden; MjSim.render = forbidden
        ControlEnv.step = forbidden
        # Deliberately never open prediction logs or use outcomes to construct labels.
        paths = sorted({p for pair in replay['selection']['pairs'] for p in pair['artifacts'].values()})
        for index, path in enumerate(paths):
            assert sha(path) == replay['input_sha256'][path]
            meta = read_episode_metadata(Path(path))
            bddl = resolve_libero_bddl_paths(meta['bddl_path'])
            assert sha(bddl.source) == meta['bddl_sha256']
            with contextlib.redirect_stdout(io.StringIO()):
                np.random.seed(0)
                env = ControlEnv(bddl_file_name=str(bddl.runtime), use_camera_obs=False,
                    has_renderer=False, has_offscreen_renderer=False, control_freq=20)
                env.seed(0); env.reset()
            geometry = native(geometry_map(env))
            ids = [o['body_id'] for o in geometry['objects']]; site = geometry['eef_site_id']
            with h5py.File(path, 'r') as f:
                states = f['steps/mujoco_state'][:]
                proprio = f['steps/proprioception'][:]
                actions = f['steps/executed_action'][:]
            poses = {k: [] for k in ('object_position', 'object_rotation', 'gripper_position', 'gripper_rotation')}
            contacts, fresh_fingers = [], []; roundtrip = eef_error = finger_error = 0.
            for t, state in enumerate(states):
                env.set_state(state); env.sim.forward()
                error = float(np.max(np.abs(env.get_sim_state()-state)))
                assert error < 1e-10, 'Saved state does not round-trip'
                roundtrip = max(roundtrip, error)
                d = env.sim.data._data
                poses['object_position'].append(d.xpos[ids].copy())
                poses['object_rotation'].append(d.xmat[ids].copy().reshape(-1, 3, 3))
                poses['gripper_position'].append(d.site_xpos[site].copy())
                poses['gripper_rotation'].append(d.site_xmat[site].copy().reshape(3, 3))
                fingers = d.qpos[geometry['finger_qpos_addresses']].copy()
                fresh_fingers.append(fingers)
                eef_error = max(eef_error, float(np.max(np.abs(d.site_xpos[site]-proprio[t, :3]))))
                finger_error = max(finger_error, float(np.max(np.abs(fingers-proprio[t, -2:]))))
                contacts.append({'frame_index': t, 'contacts': contact_rows(d)})
            # All geometry/aperture evidence is freshly computed from saved state.
            physical_state = proprio.copy(); physical_state[:, -2:] = np.asarray(fresh_fingers)
            arrays = {k: np.asarray(v) for k, v in poses.items()}
            rid = f'recording-{index:03d}'; dest = args.output/rid; dest.mkdir()
            ev = RecordingEvidence(rid, replay['input_sha256'][path], actions, physical_state,
                geometry=geometry, contacts=contacts, **arrays)
            candidate = extract_events(ev)
            write(dest/'candidate.json', candidate); write(dest/'geometry.json', geometry)
            np.savez_compressed(dest/'evidence.npz', actions=actions, state=physical_state, **arrays)
            with (dest/'contacts.jsonl').open('w') as f:
                for row in contacts:
                    f.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
            row = {'artifact': path, 'source_sha256': replay['input_sha256'][path],
                'candidate': candidate, 'recording_id': rid,
                'max_state_roundtrip_error': roundtrip, 'max_fresh_eef_vs_recorded_m': eef_error,
                'max_fresh_finger_vs_recorded_m': finger_error,
                'reconstructed_model_xml_sha256': hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest(),
                'original_model_xml_available': False,
                'evidence_sha256': {f.name: sha(f) for f in dest.iterdir()}}
            results.append(row); write(dest/'receipt.json', row)
            env.close(); env = None
            print(json.dumps({'completed_recordings': len(results), 'total': len(paths),
                              'seconds': time.monotonic()-start}), flush=True)
        write(args.output/'annotations.json', {'recordings': results, 'contract_digest': cfg['digest'],
            'interpretation': 'Automatic physical candidates independent of predictions; unreviewed, not human gold labels. Saved-state reconstruction lacks original model XML.'})
        result.update(status='complete', recordings=len(results),
                      annotation_digest=read(args.output/'annotations.json')['digest'])
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if env is not None:
            env.close()
        signal.alarm(0)
        result.update(seconds=time.monotonic()-start, checked_position_stage_calls=stage_calls)
        write(args.output/'completion.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
