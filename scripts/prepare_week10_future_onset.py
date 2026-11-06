"""Freeze the approved task revision against existing, verified recordings only."""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_future_onset import TASK,build_manifest,schedule
from eventground_wam.eventground_report_extension import coverage_gate
from eventground_wam.eventground_weak_supervision import sealed,sha,build_dataset

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/evaluation'/TASK
REMOTE='/home/peimengsui/data/eventground-wam/libero-plus/results/'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    BASE.mkdir(exist_ok=False)
    weak=sealed(ROOT/'configs/training/week10_eventground_exploratory_v4.json')
    prior=sealed(ROOT/'docs/evaluation/week10-exploratory-weak-v4/dataset.json')
    if build_dataset(ROOT,weak)!=prior:raise ValueError('Old training evidence no longer reproduces')
    pins=dict(weak['input_files_sha256'])
    for p in ['configs/training/week10_eventground_experiment_v3.json','configs/training/week10_eventground_numerical_v2.json']:
        d=sealed(ROOT/p)
        for name,h in d.get('source_files_sha256',d.get('code_sha256',{})).items():
            if sha(ROOT/name)!=h:raise ValueError('Original pinned implementation changed '+name)
    blind=sealed(ROOT/weak['inputs']['blind_manifest'])
    records=[]
    for row in blind['episodes']:
        name=f"{weak['inputs']['candidate_root']}/{row['recording_id']}/event-candidates.json"
        c=sealed(ROOT/name);pins[name]=sha(ROOT/name)
        if c['digest']!=prior['source_candidates'][row['source_id']]['candidate_digest']:raise ValueError('Training candidate mismatch')
        records.append(dict(candidate=c,task_key=row['task_key'],instruction=row['instruction'],
            recording_path=weak['capture_base']+'/run/'+row['recording_id']+'/recording.h5',
            recording_sha256=row['evidence_sha256']['recording.h5'],
            weak_label_provenance=dict(origin='automatic_physical_predicate',independent_qa_passed=False,
                assisted_review_revision=177,review_scope='20 episodes / four tasks /113 positive proposals; no missed-event recall review',
                negative_windows_human_reviewed=False)))
    panels={'train':records};evidence={}
    for panel,relative in [('dev','week10-weak-rlvr-v1/event-dev'),('report','week10-report-coverage-extension-startup-repair-v1')]:
        base=ROOT/'docs/evaluation'/relative
        done=sealed(base/'run/completion.json');v=sealed(base/'verification.json');labels=sealed(base/'label-completion.json');capture=sealed(base/'recording-contract.json')
        if (done['status']!='complete' or v['completion_digest']!=done['digest'] or not v['all_panel_visual_states_reconstructed']
            or labels['verification_digest']!=v['digest'] or labels['status']!='complete_weak_labels_not_independent_QA'):
            raise ValueError('Unverified captured panel')
        if any(p['maximum_channel_error']>1 for p in v['pixel_comparisons']):raise ValueError('Camera qualification changed')
        if any(row[k]!=0 for row in v['cases'] for k in ('maximum_contact_error','maximum_object_gripper_pose_error','maximum_proprio_error')):
            raise ValueError('Physical mismatch')
        for name in ['run/completion.json','verification.json','label-completion.json','recording-contract.json','source-metadata.json','manifest.json']:
            pins[str((base/name).relative_to(ROOT))]=sha(base/name)
        records=[]
        for item,result in zip(capture['targets'],done['results'],strict=True):
            name=f"labels/{item['recording_id']}/event-candidates.json";c=sealed(base/name)
            if sha(base/name)!=labels['output_sha256'][name]:raise ValueError('Candidate bytes changed')
            pins[str((base/name).relative_to(ROOT))]=sha(base/name)
            records.append(dict(candidate=c,task_key=item['task_key'],instruction=item['instruction'],
                recording_path=REMOTE+relative+'/run/'+item['recording_id']+'/recording.h5',recording_sha256=result['evidence_sha256']['recording.h5'],
                weak_label_provenance=dict(origin='automatic_physical_predicate',independent_qa_passed=False,human_reviewed=False,
                    negative_windows_human_reviewed=False,label_counts_previously_inspected=True)))
        panels[panel]=records
        evidence[panel]=dict(capture_digest=done['digest'],verification_digest=v['digest'],label_completion_digest=labels['digest'],
            pixel_comparisons=len(v['pixel_comparisons']),bitwise_exact=sum(p['bitwise_equal'] for p in v['pixel_comparisons']),
            maximum_channel_error=max(p['maximum_channel_error'] for p in v['pixel_comparisons']),physical_mismatches=0)
    ids=[{r['candidate']['source_id'] for r in records} for records in panels.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Cross-partition source overlap')
    index=seal_contract(dict(artifact_type='eventground_wam.future_onset_source_index',task_definition=TASK,
        prior_train_digest=prior['digest'],original_input_files_sha256=pins,panels=panels,evidence=evidence,
        authorization='User: Sounds good. Approved. Approval of task revision, rebuilding, fresh SFT, gated RLVR and matched SFT continuation.',
        created_at=datetime.now(timezone.utc).isoformat(),independent_qa_passed=False))
    write(BASE/'source-index.json',index)
    manifests={k:build_manifest(index,k) for k in panels}
    if manifests['train']['summary']['class_counts']!={'contact':385,'grasp':227,'gripper_close':497,'none':500}:raise ValueError('Train count reproduction failed')
    if manifests['report']['summary']['class_counts']!={k:96 for k in ('contact','grasp','gripper_close','none')}:raise ValueError('Report count reproduction failed')
    for panel in ('dev','report'):
        if not coverage_gate(manifests[panel]['rows'])['passed']:raise ValueError('Evaluation coverage floor failed '+panel)
    for k,v in manifests.items():write(BASE/f'{k}-manifest.json',v)
    for stage in ('sft-qualification','sft','rlvr-qualification','primary','calibration','main','control'):
        write(BASE/'schedules'/f'{stage}.json',schedule(manifests['train'],stage))
    write(BASE/'preparation.json',seal_contract(dict(task_definition=TASK,source_index_digest=index['digest'],
        manifest_digests={k:v['digest'] for k,v in manifests.items()},coverage={k:v['summary'] for k,v in manifests.items()},
        coverage_gates={k:coverage_gate(manifests[k]['rows']) for k in ('dev','report')},prior_experiments_preserved=True,
        training_or_model_inference_started=False,independent_qa_passed=False,
        original_report_status='Unchanged zero-grasp original task report; not replaced or pooled',
        limitations=['automatic weak labels','prior label/count inspection disclosed','single seed','unresolved M4 historical transformed alias exposure'],
        physical_qualification=evidence)))
    print(json.dumps({k:v['summary'] for k,v in manifests.items()},indent=2))
if __name__=='__main__':main()
