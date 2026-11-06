"""Freeze prospective weak RLVR/evaluation rules before held-out payload access."""
import json
import shutil
from pathlib import Path
from datetime import datetime,timezone
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import pilot_rows,panel_sources
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/evaluation/week10-weak-rlvr-v1'
REMOTE='/home/peimengsui/data/eventground-wam/libero-plus/results/week10-weak-rlvr-v1'

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(seal_contract(d),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    BASE.mkdir(exist_ok=False)
    dataset=sealed(ROOT/'docs/evaluation/week10-exploratory-weak-v4/dataset.json');pilot=pilot_rows(dataset)
    completion=sealed(ROOT/'docs/evaluation/week10-weak-sft-v1/run/completion.json')
    write(BASE/'pilot-schedule.json',dict(artifact_type='eventground_wam.weak_rlvr_pilot_schedule',dataset_digest=dataset['digest'],
        rows=pilot,batches=[[r['prompt_id'] for r in pilot[i:i+16]] for i in range(0,len(pilot),16)],
        prompt_count=404,updates=26,last_batch_size=4,epochs=1,seed=17,group_size=4))
    code=['src/eventground_wam/eventground_weak_evaluation.py','scripts/prepare_week10_weak_rlvr.py',
          'scripts/stage_week10_event_panel.py','scripts/prepare_week10_event_capture.py','scripts/record_week10_event_panel.py',
          'scripts/verify_week10_event_panel.py','scripts/extract_week10_event_panel.py','tests/test_eventground_weak_evaluation.py']
    protocol=dict(artifact_type='eventground_wam.weak_rlvr_evaluation_protocol',version=1,created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User: Can you do 1-4? Freeze contracts, prepare held-out panels, qualify RLVR and run bounded pilot; stop after pilot decision.',
        exploratory_parent_digest=sealed(ROOT/'configs/training/week10_eventground_exploratory_v4.json')['digest'],
        numerical_recipe_digest=sealed(ROOT/'configs/training/week10_eventground_numerical_v2.json')['digest'],
        dataset_digest=dataset['digest'],sft_completion_digest=completion['digest'],sft_endpoint_sha256=completion['endpoint_sha256'],
        sft_endpoint_tensor_digest='59847f7f75a825f623f4ffaf802d0aa9a18865af74e38be2bafec533ad2b79a8',
        pilot_schedule_digest=sealed(BASE/'pilot-schedule.json')['digest'],code_sha256={p:sha(ROOT/p) for p in code},
        label_evidence='frozen automatic physical predicates; limited assisted training spot-check; no independent QA',independent_qa_passed=False,
        weak_admission_revision='Independent-QA and balanced-2000 prerequisites replaced only for this explicitly approved exploratory child. Original gates remain unchanged.',
        reward=dict(primary=[.4,.25,.2,.15],single_calibration=[.3,.35,.2,.15],components=['typed_interval_IoU','typed_start_timing_1_minus_error_over_32','typed_onset_inside_extracted_raw_event','valid_format'],
            invalid=0,correct_none_components=[1,1,1,1],none_is_human_reviewed=False,format_constant_cancels_group_advantage=True,
            evidence='rederive target from checksum-bound frozen candidate, complete window, no ongoing event or ambiguity; future evidence offline only'),
        training=dict(pilot_prompts=404,per_class=dict(contact=125,grasp=29,gripper_close=125,none=125),updates=26,
            logical_batch=16,microbatch=1,final_batch=4,group_size=4,seed=17,epochs=1,compute_dtype='float32',
            reference='independent frozen completed SFT copy',initialization='fresh completed SFT parameters and empty AdamW; no qualification optimizer/weights reuse',
            primary_reward=[.4,.25,.2,.15],single_calibration_if_primary_gate_fails=True,
            calibration='fresh SFT, same 404 prompts/order, exactly one pass; only reward weights change; no other retry or search',
            numerical_settings='unchanged v2 RLVR head 1e-5, language LoRA 1e-6; temperature1, clip0.2, KL0.02, advantage epsilon1e-4',
            qualification=dict(prompts=16,unique_optimizer_updates=2,recovery_replay_updates=1,maximum_seconds=900,
                checks=['group4 reward validation','finite gradients/parameters','exact trainable allowlist','frozen reference and parent unchanged','bitwise optimizer+RNG recovery','90percentGPU','current_inputs_only']),
            stop_after_pilot_decision=True,main_or_control_execution_authorized=False,
            conditional_future_scope='If either frozen pilot passes, discard pilot weights and use fresh SFT for one pass over all1332 weak prompts plus matched one-pass SFT continuation; otherwise omit both. No report-based revision.',
            endpoint_selection='fixed final update, never best heldout checkpoint'),
        evaluation=dict(dev='103 sources, unchanged conservative eligibility; _sample(250), trim each class to common minimum; no duplicate/class substitution',
            report='77 sources, eligible anchor%8==0; task/source round-robin; no class/outcome balancing; at most4/event,16/source,1232 total',
            report_open_gate='pilot promotion and resulting main/control scope sealed first; no report model predictions in this steps1-4 task',
            qualification='zero-error independent physical reconstruction, same frozen labeler, keep all exclusions/ambiguities; automatic heldout labels not independent truth',
            cameras='strict equality reported; inherited max1 uint8 channel disagreement qualified explicitly; larger/new semantic discrepancy stops admission, no tolerance change',
            inference='deterministic greedy, same complete manifest; no target tokens or future arrays in inputs',
            promotion=dict(macro_positive_IoU_gain_min=.01,none_FP_increase_max=.02,matched_positive_start_MAE_increase_max=1.,format_valid_fraction=1.,
                missing_class_or_undefined_timing='no-go',all_integrity_checks_required=True),
            metrics=['typedIoU wrongtype/misszero','confusion precision recall','noneFP','matched start/end MAE with coverage','joint2frame','horizonbins','censor/exclusions','latency memory costs'],
            bootstrap=dict(unit='physical_source_id',draws=10000,rng='PCG64',seed=10,quantiles=[.025,.975],method='linear',undefined_limit=.01),
            baselines=['always_none','train_only_event_timing_prior'],human_review_seconds=0,
            claim_limits=['weak-label agreement only','sparse grasp, training lead<=8','single seed','M4 historical transformed alias exposure unresolved','no new robot rollout claim']),
        resources=dict(training_total_gpu_seconds=144000,prior_active_gpu_seconds=completion['cumulative_active_gpu_seconds'],
            training_total_output_bytes=100*1024**3,prior_training_output_bytes=completion['output_bytes']+1024**3,
            qualification_seconds=900,pilot_seconds_each=1800,pilot_attempts=2,inference_seconds_each=1800,
            child_training_output_bytes=12*1024**3,minimum_training_free_bytes=50*1024**3,gpu_fraction=.9,
            heldout_budget_scope='newly authorized separate evaluation capture, never consumes/repurposes the frozen train300 reserve',
            dev_capture_verification_seconds=1800,report_capture_verification_seconds=1500,
            dev_capture_bytes=30*1024**3,report_capture_bytes=24*1024**3,staging_seconds_each=600,staging_bytes_each=1024**3,
            extraction_seconds_each=600,extraction_bytes_each=1024**3,minimum_capture_free_bytes=150*1024**3,
            attempts_per_source=1,automatic_retries=0))
    write(ROOT/'configs/training/week10_weak_rlvr_evaluation_v1.json',protocol)
    protocol=sealed(ROOT/'configs/training/week10_weak_rlvr_evaluation_v1.json')
    # Preserve a full standalone source bundle; original pinned snapshots are untouched.
    shutil.copytree(ROOT/'docs/evaluation/week10-weak-sft-v1/source',BASE/'source')
    for name in code+['configs/training/week10_weak_rlvr_evaluation_v1.json']:
        target=BASE/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    for name in ['run/completion.json','run/identity.json','completion-verification.json']:
        target=BASE/'source/docs/evaluation/week10-weak-sft-v1'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'docs/evaluation/week10-weak-sft-v1'/name,target)
    split=sealed(ROOT/'docs/evaluation/week10-source-split-v3/partitions.json')
    for panel in ['event-dev','event-report']:
        out=BASE/panel;out.mkdir();meta=sealed(ROOT/f'docs/evaluation/week10-evaluation-preparation-v1/{panel}-source-metadata.json');rows=panel_sources(split,meta,panel)
        for source,dest in [('docs/evaluation/week10-source-split-v3/partitions.json','partitions.json'),
            (f'docs/evaluation/week10-evaluation-preparation-v1/{panel}-source-metadata.json','source-metadata.json'),
            ('docs/evaluation/week10-production-capture-visual-v2/recording-contract.json','parent-recording-contract.json'),
            ('docs/evaluation/week10-production-capture-visual-v2/visual-scene-specs.json','visual-scene-specs.json')]:shutil.copy2(ROOT/source,out/dest)
        shutil.copytree(ROOT/'src',out/'src',ignore=shutil.ignore_patterns('__pycache__'))
        (out/'scripts').mkdir()
        for name in ['stage_week10_event_panel.py','prepare_week10_event_capture.py','record_week10_event_panel.py','verify_week10_event_panel.py',
                     'extract_week10_event_panel.py','extract_week10_production_candidates.py','run_event_source_pilot.py']:
            shutil.copy2(ROOT/'scripts'/name,out/'scripts'/name)
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        c=dict(artifact_type='eventground_wam.weak_event_panel_execution',panel=panel,protocol_digest=protocol['digest'],
            partition_digest=split['digest'],source_metadata_digest=meta['digest'],source_ids=[r['source_id'] for r in rows],
            independent_qa_passed=False,automatic_retries=0,training_decision_digest=None,report_payload_open_authorized=panel=='event-dev',
            staging_resources=dict(global_seconds=600,per_file_seconds=300,per_file_http_bytes=16*1024**2,per_file_http_requests=2000,
                http_timeout_seconds=10,minimum_free_bytes=150*1024**3,per_task_output_bytes=16*1024**2,output_bytes=1024**3,cumulative_http_bytes=640*1024**2),
            capture_resources=dict(wall_seconds=1800 if panel=='event-dev' else 1500,output_bytes=(30 if panel=='event-dev' else 24)*1024**3,
                minimum_free_bytes=150*1024**3,attempts_per_source=1),bundle_sha256=files)
        write(out/('panel-contract.json' if panel=='event-dev' else 'locked-panel-plan.json'),c)
    write(BASE/'freeze-receipt.json',dict(protocol_digest=protocol['digest'],created_at=datetime.now(timezone.utc).isoformat(),
        event_dev_payload_reads=0,event_report_payload_reads=0,code_tests='4 passed; exact real pilot count and cross-partition/reward/gate failures tested',
        training_or_inference_launched=False,prior_sft_preserved=True))
    print(protocol['digest'])
if __name__=='__main__':main()
