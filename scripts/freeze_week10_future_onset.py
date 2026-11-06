"""Seal source, targets, schedules, gates and finite resource bounds before execution."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_future_onset import TASK
from eventground_wam.eventground_weak_supervision import sealed,sha

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/evaluation'/TASK

def main():
    source=BASE/'source'
    if source.exists() or (BASE/'protocol.json').exists():raise FileExistsError('Already frozen')
    shutil.copytree(ROOT/'docs/evaluation/week10-weak-rlvr-v1/source',source)
    new=['src/eventground_wam/eventground_future_onset.py','src/eventground_wam/eventground_future_runtime.py',
         'scripts/prepare_week10_future_onset.py','scripts/train_week10_future_onset.py','scripts/evaluate_week10_future_onset.py',
         'scripts/decide_week10_future_onset.py','scripts/run_week10_future_onset.py','scripts/freeze_week10_future_onset.py',
         'tests/test_eventground_future_onset.py','scripts/analyze_week10_weak_pilot.py']
    index=sealed(BASE/'source-index.json')
    for name in list(index['original_input_files_sha256'])+new:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    code={str(x.relative_to(source)):sha(x) for folder in ('src','scripts','tests') for x in (source/folder).rglob('*.py')}
    recipe=sealed(source/'configs/training/week10_eventground_numerical_v2.json')
    stages={}
    for stage in ('sft-qualification','sft','rlvr-qualification','primary','calibration','main','control'):
        sched=sealed(BASE/'schedules'/f'{stage}.json')
        qual=stage.endswith('qualification')
        stages[stage]=dict(wall_seconds=900 if qual else 3600 if stage=='sft' else 1800,
            output_bytes=(2 if qual else 8 if stage=='sft' else 4 if stage in ('primary','calibration') else 6)*1024**3,
            minimum_free_bytes=50*1024**3,maximum_cached_observation_bytes=768*1024**2,
            checkpoint_steps=[1] if qual else [101,202] if stage=='sft' else [16,32] if stage in ('primary','calibration') else [51,101])
    artifacts={str(x.relative_to(BASE)):sha(x) for x in BASE.rglob('*.json') if 'source' not in x.relative_to(BASE).parts}
    p=seal_contract(dict(artifact_type='eventground_wam.approved_future_onset_experiment',version=1,task_definition=TASK,
        user_approved=True,authorization='Sounds good. Approved — revision, rebuild, fresh SFT, gated RLVR and matched SFT continuation.',
        created_at=datetime.now(timezone.utc).isoformat(),allow_ongoing_event_anchors=True,discard_ambiguities=False,
        target='Earliest new onset strictly after anchor within32frames; ties grasp/contact/gripper_close/object; unchanged half-open intervals and censorship.',
        none_definition='No new onset in the complete future window, including while an event is ongoing.',
        physical_labeler_unchanged=True,current_observation_only=True,independent_qa_passed=False,
        recipe_digest=recipe['digest'],code_sha256=code,artifact_sha256=artifacts,
        automatic_retries=0,attempts_per_stage=1,training_stages=stages,
        initialization='SFT fresh M4 + new head, every RLVR/control stage fresh completed revised SFT + empty optimizer; no pilot weight promotion',
        inference_resources=dict(wall_seconds=1800,output_bytes=1024**3,minimum_free_bytes=50*1024**3),
        queue_wall_seconds=27000,
        aggregate_caps=dict(prior_training_gpu_seconds_upper_bound=3600,previous_parent_limit_seconds=144000,
            new_training_seconds=sum(x['wall_seconds'] for x in stages.values()),maximum_inference_seconds=6*1800,
            prior_training_output_bytes_upper_bound=16*1024**3,new_training_output_bytes=sum(x['output_bytes'] for x in stages.values()),parent_output_bytes=100*1024**3),
        reward_weights=dict(primary=[.4,.25,.2,.15],calibration=[.3,.35,.2,.15]),
        promotion=dict(macro_positive_iou_gain_min=.01,none_false_positive_increase_max=.02,matched_start_mae_increase_max=1.,
            format_valid=1.,all_classes_present=True,all_integrity_gates=True),
        calibration='Exactly one fresh-SFT pilot using predefined calibration weights iff primary fails; no other tuning.',
        no_go='If both fixed pilots fail, omit formal RLVR/control and report SFT only; preserve the negative gate result.',
        evaluation=dict(dev='396 balanced prompts from103 sources; fixed before model predictions',
            report='384 balanced prompts from80 probe sources; separate label-informed exploratory panel, not original report',
            source_overlap=False,model_selection='dev only, fixed final update per stage',report_arms_if_pass=['sft','main','control'],
            source_paired_bootstrap=dict(draws=10000,rng='PCG64',seed=10,quantiles=[.025,.975],undefined_limit=.01),
            metrics=['macro typed positive IoU','class precision recall confusion','none FP','matched onset/end MAE and coverage',
                     'joint2frame','lead bins','censoring','event/horizon Brier and fixed10bin ECE','latency and memory'],
            baselines=['always_none','training_only_event_and_timing_prior'],
            latency='synchronized model forward, microbatch1, first5excluded; no preprocessing/calibration diagnostics'),
        limitations=['exploratory weak-label agreement, not independent event truth','labels inspected before task revision',
                    'single seed','M4 historical transformed alias exposure unresolved','no robot policy or rollout improvement claim'],
        original_experiments_preserved=True,original_report_replaced=False,new_recording_capture_authorized=False,
        remote_base='/home/peimengsui/data/eventground-wam/libero-plus/results/'+TASK,
        cache='/home/peimengsui/data/eventground-wam/huggingface-cache'))
    with (BASE/'protocol.json').open('x') as f:json.dump(p,f,indent=2,sort_keys=True);f.write('\n')
    deployment={str(x.relative_to(BASE)):sha(x) for x in BASE.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
    with (BASE/'deployment.json').open('x') as f:json.dump(seal_contract(dict(files_sha256=deployment,protocol_digest=p['digest'])),f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(protocol_digest=p['digest'],files=len(deployment),stages={k:v['wall_seconds'] for k,v in stages.items()})))
if __name__=='__main__':main()
