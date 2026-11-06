#!/usr/bin/env python3
"""Admit fixed endpoints and execute fresh, resumable Week 9 panels on Feifei."""
from __future__ import annotations
import argparse, fcntl, json, os, shutil, subprocess, time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json, metadata_for_pair
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.week9_execution import bind_metadata, execute_episode, immutable
from eventground_wam.evaluation_comparisons import audit_panel, summarize_rows, compare_inputs, paired_effect
from verify_week9_protocol import verify
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/peimengsui/data/eventground-wam/libero-plus')
LIBERO=Path('/home/peimengsui/code/vendor/LIBERO-plus')
CACHE=Path('/home/peimengsui/data/eventground-wam/huggingface-cache')
NORMALIZATION=ROOT/'docs/model/week4-fresh-v2/action-statistics.json'
CV2_LIBRARY_DIR='/home/peimengsui/code/eventground-wam-week4-run-v2/environments/b0/.venv/lib/python3.10/site-packages/cv2/../../lib64'
DEFAULT_TRAINING={
    'B0':DATA/'derived/b0/week4-fresh-all-l1-v2/epochs/epoch-02',
    'M1':DATA/'training/week7-m1-auxiliary-v2',
    'M2':DATA/'training/week7-m2-lambda-v1',
    'M3':DATA/'training/week8-m3-coupled-v1',
    'M4':DATA/'training/week8-m4-lambda-v1'}


def contract():
    verify(ROOT)
    return read_sealed(ROOT/'configs/evaluation/week9_factorial_v1.json')


def package_check():
    source=read_sealed(ROOT/'week9-source-manifest.json')
    for name,digest in source['files'].items():
        require(sha256_file(ROOT/name)==digest, f'Week9 source changed: {name}')
    return source


def runtime():
    import torch
    import importlib.metadata
    from evaluate_clean_comparison import runtime_versions
    expected=json.loads((ROOT/'docs/evaluation/week7-m1-final-v1/runtime-environment.json').read_text())
    expected['PYTHONPATH']=f'{ROOT}/src:{ROOT}/scripts:{LIBERO}'
    expected.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
                    TORCHINDUCTOR_CACHE_DIR='/tmp/torchinductor_peimengsui')
    # The pinned OpenCV loader prepends this exact directory on its first import.
    # Initialize it before freezing/comparing the effective simulator runtime.
    loaded_library_path=CV2_LIBRARY_DIR+':'+expected['LD_LIBRARY_PATH']
    require(os.environ.get('LD_LIBRARY_PATH') in (expected['LD_LIBRARY_PATH'], loaded_library_path),
            'unrecognized library search path before OpenCV initialization')
    import cv2
    require(str(Path(cv2.__file__).parent/'../../lib64') == CV2_LIBRARY_DIR, 'OpenCV loader location changed')
    expected['LD_LIBRARY_PATH']=loaded_library_path
    for k,v in expected.items(): require(os.environ.get(k)==v,f'evaluation environment changed: {k}')
    require(torch.cuda.is_available(),'CUDA unavailable')
    torch.cuda.set_device(0)
    versions=runtime_versions()
    require(versions==contract()['cohort']['runtime_versions'],'runtime versions changed')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version',
                                 '--format=csv,noheader'],text=True).strip()
    require(gpu.startswith('NVIDIA GeForce RTX 5090,'),'wrong evaluation GPU')
    return seal_contract({'versions':versions,
        'package_inventory':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())),
        'gpu':gpu,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),
        'backend_flags':{'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision()},
        'environment':expected,'python_executable':os.path.realpath(os.sys.executable)})


def endpoint(arm, training):
    cfg=contract(); pin=cfg['endpoints'][arm]
    if arm=='B0':
        from evaluate_b0_full import verify_training_bundle
        identity,report,checkpoint=verify_training_bundle(training,NORMALIZATION)
        require(report['selected_epoch']==2,'B0 endpoint changed')
        require(identity['recipe']['digest']==cfg['cohort']['action_recipe_digest'],'B0 recipe changed')
        completion=None; arm_contract=None; path=training/'B0.ckpt'
    else:
        identity=read_sealed(training/'training-manifest.json')
        require(identity['digest']==pin['training_identity_digest'],'wrong training identity')
        if arm in ('M3','M4'):
            from eventground_wam.coupled_evaluation import load_contract, verify_endpoint
            arm_contract=load_contract(ROOT,arm)
        else:
            from eventground_wam.auxiliary_evaluation import verify_endpoint
            arm_contract={**cfg['endpoint_selection'], 'arm':arm,
                'training_identity_digest':identity['digest'],'recipe_digest':identity['recipe_digest'],
                'production_source_digest':identity['production_source_digest'], 'checkpoint_name':f'{arm}.ckpt'}
        identity,report,completion=verify_endpoint(training,arm_contract)
        path=training/'epochs/epoch-02'/f'{arm}.ckpt'
    if pin['checkpoint_sha256']: require(report['checkpoint_sha256']==pin['checkpoint_sha256'],'frozen checkpoint changed')
    return seal_contract({'artifact_type':'eventground_wam.week9_endpoint_admission',
        'arm':arm,'week9_protocol_digest':cfg['digest'],'training_root':str(training),
        'training_identity_digest':identity['digest'],'endpoint_report_digest':report['digest'],
        'completion_digest':completion['digest'] if completion else None,
        'checkpoint_path':str(path),'checkpoint_sha256':report['checkpoint_sha256'],
        'checkpoint_id':f'eventground-wam/week9-{arm.lower()}-fixed-epoch-2',
        'arm_contract':arm_contract, 'updates':117132,'selection':'fixed-epoch-2-endpoint'})


def load_policy(admission):
    import torch
    from evaluate_b0_full import FullCheckpointIdentity
    from train_b0 import build_policy
    from eventground_wam.b0_data import load_b0_action_normalizer
    from eventground_wam.b0_inference import B0InferencePolicy
    from eventground_wam.b0_model import load_b0_trainable_state_dict, load_pinned_qwen_processor
    from eventground_wam.b0_oft import OFTInputBuilder
    arm=admission['arm']; training=Path(admission['training_root'])
    if arm in ('M3','M4'):
        from evaluate_coupled_dev import load_policy as load_coupled
        cfg={**admission['arm_contract'],'checkpoint_id':admission['checkpoint_id']}
        return load_coupled(SimpleNamespace(training_root=training,cache_dir=CACHE),cfg,
                            read_sealed(training/'epochs/epoch-02/training-report.json'))
    device=torch.device('cuda:0')
    recipe=read_sealed(ROOT/'configs/training/week4_b0_fresh_all_l1_v2.json')
    model,names=build_policy(recipe,str(CACHE),device)
    state=torch.load(admission['checkpoint_path'],map_location='cpu',weights_only=True)
    from eventground_wam.auxiliary_evaluation import verify_envelope
    verify_envelope(state,{'training_identity_digest':admission['training_identity_digest']})
    require(all(torch.isfinite(t).all() for t in state['state_dict'].values()),'nonfinite policy tensor')
    load_b0_trainable_state_dict(model,state['state_dict'],allowed_exact_names=names)
    model.eval()
    ckpt=FullCheckpointIdentity(checkpoint_id=admission['checkpoint_id'],
        checkpoint_revision=admission['checkpoint_sha256'],training_manifest_digest=admission['training_identity_digest'],
        training_report_digest=admission['endpoint_report_digest'],selected_epoch=2,selected_step=117132)
    return B0InferencePolicy(policy=model,input_builder=OFTInputBuilder(load_pinned_qwen_processor(
        cache_dir=str(CACHE),local_files_only=True)),normalizer=load_b0_action_normalizer(NORMALIZATION),
        device=device,checkpoint=ckpt)


def panel_inventory(panel):
    cfg=contract(); panel_cfg=cfg['panels'][panel]
    protocol=load_evaluation_protocol(ROOT/panel_cfg['protocol'])
    suites={}; states={}
    if panel=='perturbed':
        from evaluate_reference_policy import load_manifest
        tasks=load_manifest(ROOT/panel_cfg['path'],protocol)['tasks']
    else:
        from eventground_wam.clean_evaluation import clean_manifest
        old=load_evaluation_protocol(ROOT/'configs/evaluation/libero_clean_v1.json')
        manifest,suites=clean_manifest(LIBERO,old)
        frozen=read_sealed(ROOT/'evaluation/splits/libero-clean-v1/clean-40.json')
        require(manifest==frozen,'canonical clean inventory changed')
        indexed={(r['suite'],r['runtime_task_index']):r for r in manifest['tasks']}
        tasks=[{**p,'perturbation_category':'Clean','difficulty_level':None}
               for p in read_sealed(ROOT/panel_cfg['path'])['pairs']]
        from eventground_wam.episode_recorder import sha256_array
        for task in tasks:
            key=(task['suite'],task['runtime_task_index'])
            canonical=indexed[key]
            for name in ('task_name','instruction','bddl_sha256','initial_states_file_sha256'):
                require(task[name]==canonical[name],f'clean task changed: {name}')
            if key not in states: states[key]=np.asarray(suites[key].get_task_init_states(key[1]))
            require(len(states[key])==50 and sha256_array(states[key][task['initial_state_index']])
                    ==task['initial_state_sha256'],'clean initial state changed')
    return tasks,protocol,suites,states


def scene(panel, task, protocol, suites, states, identity):
    from evaluate_reference_policy import resolve_libero_bddl_paths, make_episode_metadata
    key=(task['suite'],task['runtime_task_index'])
    if panel=='perturbed':
        from libero.libero import benchmark
        if task['suite'] not in suites: suites[task['suite']]=benchmark.get_benchmark_dict()[task['suite']]()
        suite=suites[task['suite']]
        actual=suite.get_task(task['runtime_task_index'])
        require(actual.name==task['task_name'],'perturbed task changed')
        initial=np.asarray(suite.get_task_init_states(task['runtime_task_index'])[0])
    else:
        suite=suites[key]; actual=suite.get_task(key[1]); initial=states[key][task['initial_state_index']]
    paths=resolve_libero_bddl_paths(suite.get_task_bddl_file_path(task['runtime_task_index']))
    if panel=='clean':
        require(sha256_file(paths.source)==task['bddl_sha256'],'clean BDDL changed')
        metadata=metadata_for_pair(task,protocol,identity,paths.source)
    else:
        metadata=make_episode_metadata(protocol=protocol,manifest_task=task,task=actual,
            bddl_source_path=paths.source,initial_state=initial,project_revision=identity['project_git_revision'],
            checkpoint_id=identity['checkpoint_id'],checkpoint_revision=identity['checkpoint_revision'],
            policy_instruction=actual.language)
    def factory():
        from libero.libero.envs import OffScreenRenderEnv
        resolution=protocol['observation']['capture_resolution']
        return OffScreenRenderEnv(bddl_file_name=str(paths.runtime),camera_heights=resolution[0],
            camera_widths=resolution[1],control_freq=protocol['simulation']['control_frequency_hz'])
    return bind_metadata(metadata,identity),initial,factory


def init_cohort(root, arms):
    source=package_check(); cfg=contract()
    from check_reference_policy import verify_libero
    benchmark=verify_libero(LIBERO)
    admissions={arm:endpoint(arm,DEFAULT_TRAINING[arm]) for arm in arms}
    value=seal_contract({'artifact_type':'eventground_wam.week9_cohort','week9_protocol_digest':cfg['digest'],
        'source_digest':source['digest'],'base_git_revision':source['base_git_revision'],
        'runtime':runtime(),'benchmark':benchmark,'initial_admissions':{k:v['digest'] for k,v in admissions.items()},
        'initial_order':arms,'future_admissions':['M2','M4'], 'fresh_rollouts':True,
        'legacy_profile_and_acceptance_fields_are_format_only':True,
        'resource_limits':{'free_bytes':50*1024**3,'maximum_output_bytes':900*1024**3,
                           'maximum_active_seconds_per_panel':48*3600},
        'authorization':'User requested fresh B0/M1/M3 larger panels on Feifei, adding M2/M4 later under identical protocol.'})
    immutable(root/'cohort.json',value)
    for arm,admission in admissions.items(): immutable(root/'admissions'/f'{arm}.json',admission)
    return value


def verify_cohort(root):
    cfg=contract(); source=package_check(); cohort=read_sealed(root/'cohort.json')
    require(cohort['source_digest']==source['digest'] and cohort['week9_protocol_digest']==cfg['digest'], 'cohort changed')
    require(runtime()==cohort['runtime'],'cohort runtime changed')
    return cohort


def identity_for(cohort, admission, panel, protocol, mode='formal'):
    return seal_contract({'artifact_type':'eventground_wam.week9_panel_run',
        'cohort_digest':cohort['digest'],'week9_protocol_digest':cohort['week9_protocol_digest'],
        'policy_objective':contract()['cohort']['action_objective'],'admission_digest':admission['digest'],
        'arm':admission['arm'],'panel':panel,'mode':mode,
        'project_git_revision':cohort['base_git_revision'],'evaluation_source_digest':cohort['source_digest'],
        'checkpoint_id':admission['checkpoint_id'],'checkpoint_revision':admission['checkpoint_sha256'],
        'protocol_sha256':protocol_sha256(protocol),
        'acceptance_rule_digest':protocol.get('comparison',{}).get('acceptance_rule_digest'),
        'legacy_profile_and_acceptance_fields_are_format_only':True})


def run_panel(root, arm, panel, *, smoke=False):
    import torch
    from evaluate_coupled_dev import require_idle_gpu
    from evaluate_clean_comparison import seed_episode
    from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
    cohort=verify_cohort(root); admission=read_sealed(root/'admissions'/f'{arm}.json')
    require(endpoint(arm,Path(admission['training_root']))==admission,'admitted endpoint changed')
    require_idle_gpu(); install_codec_independent_motion_blur()
    tasks,protocol,suites,states=panel_inventory(panel)
    selected=[0] if smoke else list(range(len(tasks)))
    identity=identity_for(cohort,admission,panel,protocol,'smoke' if smoke else 'formal')
    output=root/('qualification' if smoke else 'panels')/arm/panel
    immutable(output/'identity.json',identity)
    immutable(output/'tasks.json',seal_contract({'tasks':tasks,'selected_ordinals':selected}))
    if not smoke:
        qualification=read_sealed(root/'qualification'/arm/'qualification.json')
        require(qualification['passed_model_and_recorder'] and qualification['admission_digest']==admission['digest']
                and qualification['cohort_digest']==cohort['digest'],'missing qualification')
    from eventground_wam.b0_continuation import ActiveBudget
    budget=ActiveBudget(output/'active-budget.json',identity['digest'])
    started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    model_started=time.monotonic(); policy=load_policy(admission); model_seconds=time.monotonic()-model_started
    require(runtime()==cohort['runtime'],'model loading changed cohort numerical flags')
    records=[]
    try:
        for ordinal in selected:
            limits=cohort['resource_limits']
            require(budget.seconds<limits['maximum_active_seconds_per_panel'],'panel active time cap reached')
            require(shutil.disk_usage(root).free>=limits['free_bytes'],'disk reserve reached')
            if ordinal%25==0:
                require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<limits['maximum_output_bytes'],'cohort output cap reached')
            task=tasks[ordinal]
            metadata,initial,factory=scene(panel,task,protocol,suites,states,identity)
            repetitions=range(2) if smoke else range(1)
            for repeat in repetitions:
                row=execute_episode(root=output/f'repeat-{repeat}' if smoke else output,
                    ordinal=ordinal,task=task,identity=identity,protocol=protocol,metadata=metadata,
                    initial_state=initial,policy=policy,environment_factory=factory,seed_episode=seed_episode)
                records.append(row)
            atomic_json(output/'progress.json',{'completed':len(records),'total':2 if smoke else len(tasks),
                'successes':sum(r['success'] for r in records),'elapsed_current_invocation_seconds':time.monotonic()-started})
            print(json.dumps({'arm':arm,'panel':panel,'completed':len(records),'total':2 if smoke else len(tasks)}),flush=True)
        audits=audit_panel(records,identity,[tasks[0],tasks[0]] if smoke else tasks,protocol)
        report=seal_contract({'artifact_type':'eventground_wam.week9_panel_report','identity_digest':identity['digest'],
            'episodes':records,'summary':summarize_rows(records),'audits':audits,
            'mode':identity['mode'],'model_load_seconds_current_invocation':model_seconds,
            'evaluation_wall_seconds_current_invocation':time.monotonic()-started,
            'active_seconds_all_invocations':budget.seconds,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
            'deployed_parameter_count':sum(p.numel() for p in policy._policy.parameters()),
            'parameters_marked_trainable':sum(p.numel() for p in policy._policy.parameters() if p.requires_grad),
            'status':'audited' if all(a['error'] is None for a in audits) else 'audit_incomplete',
            'no_outcome_selection':True})
        # Timing of analysis/resumed invocation differs; never replace a completed sealed report.
        target=output/'evaluation-report.json'
        if target.exists():
            prior=read_sealed(target)
            require(prior['identity_digest']==identity['digest'] and prior['episodes']==records
                    and prior['audits']==audits,'completed panel changed')
            return prior
        atomic_json(target,report)
        return report
    finally:
        budget.close(); del policy


def qualify(root, arm):
    reports={panel:run_panel(root,arm,panel,smoke=True) for panel in ('perturbed','clean')}
    admission=read_sealed(root/'admissions'/f'{arm}.json'); cohort=read_sealed(root/'cohort.json')
    require(all(all(r['artifact'] and r['policy_step_count']>0 and
        r['termination_reason'] in ('environment_done','policy_step_limit') for r in report['episodes'])
        and all(a['error'] is None for a in report['audits']) for report in reports.values()), 'qualification failed')
    immutable(root/'qualification'/arm/'qualification.json',seal_contract({
        'artifact_type':'eventground_wam.week9_arm_qualification','arm':arm,
        'cohort_digest':cohort['digest'],'admission_digest':admission['digest'],
        'passed_model_and_recorder':True,'reports':{p:r['digest'] for p,r in reports.items()},
        'repeat_inputs_exact':{p:r['audits'][0]['inputs']==r['audits'][1]['inputs'] for p,r in reports.items()},
        'smoke_outcomes_used_for_formal_panel':False,'input_mismatches_retained_and_descriptive':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','admit','qualify','run'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['B0','M1','M2','M3','M4'])
    p.add_argument('--training-root',type=Path)
    p.add_argument('--panel',choices=['perturbed','clean'])
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=True)
    with (DATA/'evaluation/week9-gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='init': result=init_cohort(a.root,['B0','M1','M3'])
        elif a.action=='admit':
            cohort=verify_cohort(a.root)
            require(a.arm in cohort['future_admissions'],'not an authorized later arm')
            result=endpoint(a.arm,a.training_root or DEFAULT_TRAINING[a.arm])
            immutable(a.root/'admissions'/f'{a.arm}.json',result)
        elif a.action=='qualify': result=qualify(a.root,a.arm)
        else: result=run_panel(a.root,a.arm,a.panel)
        print(json.dumps({'action':a.action,'digest':result.get('digest') if result else None}),flush=True)
if __name__=='__main__': main()
