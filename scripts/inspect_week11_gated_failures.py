"""Pair saved camera views with dense reconstructed motion/contact evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from review_week11_event_windows import contacts_by_frame
from week11_replay_common import read, sha, write

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ('M4-EG-fixed8', 'M4-EG-after-close')
LABELS = ('Fixed8 · success', 'After-close · failure')
COLORS = ('#246b91', '#bd5838')
POLICY_START = 10


def runs(mask):
    m = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1), strict=True)]


def frames(case, loaded, columns, name, output):
    fig, axes = plt.subplots(2, len(columns[0]), figsize=(18, 7.2), layout='constrained')
    for ai, (arm, steps) in enumerate(zip(ARMS, columns, strict=True)):
        with h5py.File(case['artifacts'][arm]) as f:
            for ax, step in zip(axes[ai], steps, strict=True):
                ax.imshow(np.concatenate([f['steps/agent_rgb'][step], f['steps/wrist_rgb'][step]], axis=0))
                ax.axis('off')
                command = loaded[arm]['evidence']['actions'][step, 6]
                ax.set_title(f'{LABELS[ai]}\nstep {step} · {(step-10)/20:.2f}s · '+('close' if command == 1 else 'open'), fontsize=9)
    fig.suptitle(case['id']+' · '+name+'\n'+loaded[ARMS[0]]['metadata']['instruction'], fontsize=13)
    fig.supxlabel('Existing saved frames; top = agent, bottom = wrist. Times exclude settling. Each frame precedes its labeled executed command. Overview rows sample their own episode durations.', fontsize=9)
    fig.savefig(output/f'{case["id"]}-{name}.png', dpi=130)
    plt.close(fig)


def trace(case, loaded, output):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout='constrained', sharex=True)
    objects = loaded[ARMS[0]]['geometry']['objects']
    chosen = [(i, o['object_id']) for i,o in enumerate(objects)
              if any(k in o['object_id'] for k in ('black_bowl', 'alphabet_soup', 'tomato_sauce'))]
    for ai, arm in enumerate(ARMS):
        ev = loaded[arm]['evidence']; x = (np.arange(POLICY_START, len(ev['actions']))-POLICY_START)/20
        axes[0].plot(x, ev['gripper_position'][POLICY_START:, 2], color=COLORS[ai], label=LABELS[ai])
        axes[1].plot(x, 1000*np.abs(ev['state'][POLICY_START:, -2:]).sum(1), color=COLORS[ai])
        for oi, (index, name) in enumerate(chosen):
            axes[2].plot(x, 1000*(ev['object_position'][POLICY_START:, index, 2]-ev['object_position'][POLICY_START, index, 2]),
                         color=COLORS[ai], linestyle=['-', '--', ':'][oi%3], label=LABELS[ai].split(' · ')[0]+' '+name)
    for ax in axes:
        ax.axvline((10+8*case['first_non8_query']-10)/20, color='#444', linestyle=':', label='First non-8 query')
        ax.grid(alpha=.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Gripper z (m)'); axes[0].legend(fontsize=9)
    axes[1].set_ylabel('Finger aperture (mm)')
    axes[2].set_ylabel('Object height change (mm)'); axes[2].legend(fontsize=8)
    axes[2].set_xlabel('Seconds after first policy query')
    fig.suptitle(case['id']+' · recorded-state motion after matched approach', fontsize=13)
    fig.savefig(output/f'{case["id"]}-motion.png', dpi=150); plt.close(fig)


def run(bundle, annotation_root, output):
    started = time.monotonic(); cfg = read(bundle/'contract.json')
    ann = read(annotation_root/'annotations.json')
    assert read(annotation_root/'contract.json')['replay_contract_digest'] == cfg['digest']
    indexed = {r['artifact']:r for r in ann['recordings']}
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cfg['selection']['pairs']:
        loaded = {}; arm_rows = {}
        d = 10+8*case['first_non8_query']
        for arm, path in case['artifacts'].items():
            assert sha(path) == cfg['input_sha256'][path]
            log = str(Path(path).with_suffix('.queries.jsonl'))
            assert sha(log) == cfg['input_sha256'][log]
            row = indexed[path]; root = annotation_root/row['recording_id']
            for name,digest in row['evidence_sha256'].items(): assert sha(root/name) == digest
            geometry = read(root/'geometry.json'); candidate = read(root/'candidate.json')
            ev = dict(np.load(root/'evidence.npz'))
            contacts = contacts_by_frame(geometry, [json.loads(l) for l in (root/'contacts.jsonl').read_text().splitlines()])
            with h5py.File(path) as h:
                meta = json.loads(h['episode/metadata_json'][()])
                current_input = {k:h['steps'][k][d] for k in ('agent_rgb','wrist_rgb','proprioception','mujoco_state')}
                chunk = h['steps/predicted_action_chunk'][d]
            queries = [json.loads(l) for l in Path(log).read_text().splitlines()]
            loaded[arm] = {'geometry':geometry,'candidate':candidate,'evidence':ev,'contacts':contacts,
                           'metadata':meta,'input':current_input,'chunk':chunk,'queries':queries}
            objects = {}
            for oi,obj in enumerate(geometry['objects']):
                name = obj['object_id']; pos = ev['object_position'][:, oi]
                contact = [name in c['movable_objects'] for c in contacts]
                both = [len(c['finger_sides'].get(name, [])) == 2 for c in contacts]
                above = pos[:,2]-pos[POLICY_START,2]>.03
                above[:POLICY_START] = False
                objects[name] = {'initial_position':pos[10].tolist(),'departure_position':pos[d].tolist(),
                    'last_position':pos[-1].tolist(),'max_height_above_start_mm':float(1000*(pos[POLICY_START:,2]-pos[POLICY_START,2]).max()),
                    'max_height_step':int(np.argmax(pos[POLICY_START:,2]))+POLICY_START,
                    'last_height_above_start_mm':float(1000*(pos[-1,2]-pos[10,2])),
                    'gripper_contact_intervals':runs(contact),'two_finger_contact_intervals':runs(both),
                    'above_start_30mm_intervals':runs(above)}
            commands = ev['actions'][:,6]
            arm_rows[arm] = {'artifact':path,'frames':len(commands),'success':meta['success'],
                'departure_query':queries[case['first_non8_query']],
                'closing_steps':(np.flatnonzero((commands[:-1] == -1)&(commands[1:] == 1))+1).tolist(),
                'opening_steps':(np.flatnonzero((commands[:-1] == 1)&(commands[1:] == -1))+1).tolist(),
                'objects':objects,'candidate_events':candidate['events'],
                'candidate_ambiguities':candidate['ambiguities'],
                'last_gripper_position':ev['gripper_position'][-1].tolist(),
                'last_aperture_mm':float(1000*np.abs(ev['state'][-1,-2:]).sum()),
                'input_sha256':cfg['input_sha256'][path]}
        left,right = loaded[ARMS[0]],loaded[ARMS[1]]
        input_matches={k:bool(np.array_equal(left['input'][k], right['input'][k]))
                       for k in ('agent_rgb','wrist_rgb','mujoco_state')}
        input_matches['proprioception_float32']=bool(np.array_equal(
            left['input']['proprioception'].astype(np.float32),right['input']['proprioception'].astype(np.float32)))
        n=min(len(left['evidence']['actions']),len(right['evidence']['actions']))
        a,b=left['evidence']['actions'][:n],right['evidence']['actions'][:n]
        assert np.array_equal(a[:d],b[:d])
        different=np.flatnonzero(np.any(a != b,axis=1)); grip=np.flatnonzero(a[:,6] != b[:,6])
        delta=right['chunk']-left['chunk']
        result={**case,'departure_step':d,'same_inputs_at_departure':all(input_matches.values()),
            'input_matches':input_matches,'same_prior_actions':True,
            'first_action_difference_step':int(different[0]) if len(different) else None,
            'first_gripper_difference_step':int(grip[0]) if len(grip) else None,
            'departure_chunk_fixed8':left['chunk'].tolist(),'departure_chunk_gated':right['chunk'].tolist(),
            'departure_translation_rms_delta':float(np.sqrt(np.mean(delta[:,:3]**2))),
            'departure_rotation_rms_delta':float(np.sqrt(np.mean(delta[:,3:6]**2))),
            'departure_gripper_changed_actions':np.flatnonzero(delta[:,6]).tolist(),
            'arms':arm_rows}
        results.append(result)
        close=[d+i for i in (0,4,8,16,24,32)]
        frames(case,loaded,[close,close],'first-32-frames',output)
        overview=[np.rint(np.linspace(10,len(loaded[a]['evidence']['actions'])-1,12)).astype(int).tolist() for a in ARMS]
        for page in range(2): frames(case,loaded,[x[page*6:(page+1)*6] for x in overview],f'overview-{page+1}',output)
        if case['id'] == 'clean-200-0006':
            frames(case,loaded,[[82,90,98,104],[82,90,98,104]],'placement-matched',output)
            frames(case,loaded,[[90,98,102,104],[104,111,120,229]],'placement-followthrough',output)
        trace(case,loaded,output)
    write(output/'inspection.json',{'contract_digest':cfg['digest'],'annotation_digest':ann['digest'],
        'cases':results,'model_calls':0,'simulator_steps':0,'new_rollouts':0,
        'height_metric_start_step':POLICY_START,'settling_frames_excluded_from_height_metrics_and_plots':True,
        'sampling':'Per arm: 12 uniform overview frames and six matched absolute frames at first non-8 query +0,4,8,16,24,32. Both cameras. Clean 6 has two additional four-column placement sheets. Full dense physical traces retained.'})
    write(output/'completion.json',{'status':'complete','seconds':time.monotonic()-started,
        'inspection_digest':read(output/'inspection.json')['digest'],
        'source_sha256':sha(__file__),'figures_sha256':{p.name:sha(p) for p in output.glob('*.png')},
        'model_calls':0,'simulator_steps':0,'new_rollouts':0})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bundle,a.annotations,a.output)
