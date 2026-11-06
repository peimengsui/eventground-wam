"""Bind staged panel inputs to the previously qualified canonical visual recorder."""
from pathlib import Path
import argparse
import shutil
import numpy as np
from stage_week10_event_panel import save
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import validate_panel_contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);a=p.parse_args();base=a.base
    panel=sealed(base/'panel-contract.json');rows=validate_panel_contract(panel,sealed(base/'partitions.json'),sealed(base/'source-metadata.json'))
    stage=sealed(base/'staging-completion.json')
    if stage['status']!='complete' or stage['contract_digest']!=panel['digest'] or stage['payload_sources']!=len(rows):raise ValueError('Incomplete staging')
    old=sealed(base/'parent-recording-contract.json');by_task={r['task_key']:r for r in old['targets']}
    specs=sealed(base/'visual-scene-specs.json')['tasks'];targets=[]
    inputs={name:h for name,h in old['input_sha256'].items() if not name.startswith('/home/peimengsui/data/eventground-wam/libero-plus/results/')}
    for index,row in enumerate(rows):
        source=base/'staged'/row['task_key'].replace('/','-')/row['source_id'];identity=sealed(source/'identity.json');parent=by_task[row['task_key']]
        for name,h in identity['input_sha256'].items():
            if sha(source/name)!=h:raise ValueError('Changed staged payload')
            inputs[str(source/name)]=h
        # Schema and body/joint ordering must equal the existing task-qualified canonical state.
        old_identity=sealed(Path(parent['initial_state_file']).parent/'input-identity.json')
        if identity['source_joint_names']!=old_identity['source_joint_names'] or identity['initial_state_shape']!=old_identity['initial_state_shape']:raise ValueError('Source schema differs')
        target={**row,'recording_id':f"{panel['panel']}-{index:04d}-{row['source_id'][:12]}",
            'actions_file':str(source/'actions.npz'),'initial_state_file':str(source/'initial-state.npy'),
            'source_model_file':str(source/'source-model.xml'),'input_identity_digest':identity['digest'],
            'action_digest':identity['action_digest'], 'bddl':parent['bddl'],'bddl_sha256':parent['bddl_sha256'],
            'canonical_model_sha256':parent['canonical_model_sha256'],'canonical_schema_digest':parent['canonical_schema_digest'],
            'instruction':parent['instruction']}
        assert specs[row['task_key']]['canonical_reset_sha256']==target['canonical_model_sha256']
        targets.append(target)
    resources=panel['capture_resources'];steps=sum(r['frames']+10 for r in rows)
    c=dict(artifact_type='eventground_wam.weak_event_panel_capture',panel=panel['panel'],panel_contract_digest=panel['digest'],
        parent_recording_contract_digest=old['digest'],partition_digest=panel['partition_digest'],staging_completion_digest=stage['digest'],
        targets=targets,runtime=old['runtime'],input_sha256=inputs,bundle_files_sha256=panel['bundle_sha256'],
        resources={**resources,'remaining_control_steps':steps,'control_steps':steps,'prior_control_steps':0,'prior_seconds_charged':0},
        verification=dict(verifier_sha256=sha(base/'scripts/verify_week10_event_panel.py'),wall_seconds=resources['wall_seconds'],maximum_rerendered_images=6*len(rows)),
        independent_qa_passed=False,training_authorized=False)
    save(base/'recording-contract.json',c);c=sealed(base/'recording-contract.json')
    save(base/'verification-contract.json',dict(artifact_type='eventground_wam.heldout_independent_physical_verification',recording_digest=c['digest'],**c['verification']))
    save(base/'capture-authorization.json',dict(panel_contract_digest=panel['digest'],authorization='user requested steps 1-4; weak held-out panel execution',automatic_retries=0))
    print(c['digest'])
if __name__=='__main__':main()
