"""Six-condition screen for the two fixed continuation endpoints and original M4."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import run_week11_formal as formal
import run_week11_pilot as pilot
import torch
from evaluate_b0_full import FullCheckpointIdentity
from evaluate_clean_comparison import seed_episode
from evaluate_week9 import runtime, scene
from train_b0 import build_policy
from week11_multihorizon_runtime import (
    CACHE,
    PLAN,
    ROOT,
    StageGuard,
    admission,
    emit,
    require,
    write,
)

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder
from eventground_wam.b0_training import (
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.coupled_checkpoint import load_coupled_policy
from eventground_wam.coupled_inference import CoupledInferencePolicy
from eventground_wam.coupled_policy import CoupledPolicy
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import read_episode_metadata
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, summarize_rows
from eventground_wam.eventground_inference import EventGroundInferencePolicy
from eventground_wam.jepa_predictor import JepaPredictor
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.week11_execution import execute_episode, immutable, validate_query_artifact
from eventground_wam.week11_formal import relocated_runtime
from eventground_wam.week11_multihorizon_analysis import CONDITIONS, MODELS, screen_decision
from eventground_wam.week11_resources import require_idle_gpu

ROUTES={'fixed8':'M4-EG-fixed8','adaptive':'M4-EG-adaptive'}


class Actions:
    """One model allocation, with verified checkpoint switches between episodes."""
    def __init__(self,cfg,run_root,original_endpoint):
        recipe=read_sealed(ROOT/'configs/training/week4_b0_fresh_all_l1_v2.json')
        base,self.names=build_policy(recipe,str(CACHE),torch.device('cuda:0'))
        normalizer=load_b0_action_normalizer(ROOT/'docs/model/week4-fresh-v2/action-statistics.json')
        self.model=CoupledPolicy(base,JepaPredictor(),normalizer,arm='M4').cuda().eval()
        self.flags={n:p.requires_grad for n,p in self.model.named_parameters()}
        self.paths={'original':Path(cfg['starting_endpoints']['m4']['checkpoint_path'])}
        self.headers={}
        self.checkpoints={'original':FullCheckpointIdentity(
            checkpoint_id='eventground-wam/week11-original-m4',
            checkpoint_revision=cfg['starting_endpoints']['m4']['policy_sha256'],
            training_manifest_digest=cfg['starting_endpoints']['m4']['identity_digest'],
            training_report_digest=original_endpoint['digest'],selected_epoch=2,selected_step=117132)}
        for arm in MODELS[1:]:
            path=run_root/'training'/arm/'endpoint'
            end=read_sealed(path/'endpoint.json')
            verification=read_sealed(run_root/'training'/arm/'verify-endpoint.json')
            require(verification['passed'] and verification['fresh_process_endpoint_reload_exact']
                    and verification['endpoint_digest']==end['digest'],'endpoint reload not admitted')
            require(end['design_digest']==cfg['digest'] and end['metadata']['continuation_updates']==2000
                    and end['samples_seen']==58172 and end['future_pairs_seen']==277172,'wrong continuation endpoint')
            require(end['identity']['arm']==arm,'continuation arm mismatch')
            self.paths[arm]=path/'policy.pt';self.headers[arm]=end
            self.checkpoints[arm]=FullCheckpointIdentity(
                checkpoint_id='eventground-wam/week11-'+arm,checkpoint_revision=end['policy_sha256'],
                training_manifest_digest=end['identity']['digest'],training_report_digest=end['digest'],
                selected_epoch=None,selected_step=2000)
        self.policy=CoupledInferencePolicy(policy=self.model,input_builder=OFTInputBuilder(
            load_pinned_qwen_processor(cache_dir=str(CACHE),local_files_only=True)),normalizer=normalizer,
            device=torch.device('cuda:0'),checkpoint=self.checkpoints['original'])
        self.current=None

    def select(self,arm,force=False):
        if self.current==arm and not force:return self.policy
        require(sha256_file(self.paths[arm])==self.checkpoints[arm].checkpoint_revision,'policy checkpoint changed')
        for name,p in self.model.named_parameters():p.requires_grad_(self.flags[name])
        header=load_coupled_policy(self.paths[arm],self.model,self.names)
        if arm!='original':
            require(header['digest']==self.headers[arm]['header_digest']
                    and header['metadata']==self.headers[arm]['metadata'],'continuation export admission differs')
        self.model.requires_grad_(False).eval()
        self.policy.checkpoint=self.checkpoints[arm]
        self.current=arm
        return self.policy


def source_check(cfg,training_source):
    source=read_sealed(ROOT/'week11-multihorizon-screen-source.json')
    require(source['design_digest']==cfg['digest'] and source['training_source_digest']==training_source['digest'],
            'screen source binding changed')
    for name,digest in source['files_sha256'].items():
        require(sha256_file(ROOT/name)==digest,'screen source changed: '+name)
    return source


def projection(screen,inputs,parents):
    seconds,storage=300.,256*2**20
    for c in screen['cases']:
        old=parents[c['id']]['artifacts']
        protocol=inputs[c['panel']][1]
        max_steps=10+protocol['benchmark']['suites'][c['task']['suite']]['max_policy_steps']
        rates=[]
        for path in old.values():
            path=Path(path)
            with h5py.File(path) as f:n=len(f['steps/executed_action'])
            r=read_sealed(path.with_suffix('.timing.json'))
            rates.append((r['episode_wall_time_seconds']/n,path.stat().st_size/n))
        seconds+=6*max_steps*max(r[0] for r in rates)*1.2
        storage+=6*max_steps*max(r[1] for r in rates)*1.2
    return seconds,storage


def qualify(output,actions,event,admissions,guard,parents):
    selected=('targeted-467-0001','targeted-467-0356','clean-200-0006')
    rows=[];queries=0
    for key in selected:
        old=Path(parents[key]['artifacts']['M4-EG-fixed8'])
        meta=read_episode_metadata(old)
        with h5py.File(old) as f:
            obs={k:f['steps'][k][10] for k in ('agent_rgb','wrist_rgb','proprioception')}
            original_chunk=f['steps/predicted_action_chunk'][10]
        obs.update(instruction=meta['instruction'],suite=meta['suite'])
        previous=json.loads(old.with_suffix('.queries.jsonl').read_text().splitlines()[0])['policy']
        reference_forecast=None
        for arm in MODELS:
            action=actions.select(arm)
            outputs=[]
            for route in ('fixed8','adaptive'):
                p=EventGroundInferencePolicy(action,ROUTES[route],event,identities=admissions[arm],guard=guard.check)
                values=p.predict_chunk(**obs);queries+=1;outputs.append(values.copy())
                forecast={'prediction':p.last_query['prediction'],
                          'probabilities':p.last_query['event']['categorical_probabilities']}
                if reference_forecast is None:reference_forecast=forecast
                require(forecast==reference_forecast,'frozen event branch changed on identical input')
                require(p.last_query['prediction']==previous['prediction'],'saved-input event forecast changed')
                if arm=='original' and route=='fixed8':
                    require(np.array_equal(values,original_chunk),'original fixed8 recorded input no longer reproduces')
            actions.select(arm,force=True)
            for i,route in enumerate(('fixed8','adaptive')):
                p=EventGroundInferencePolicy(actions.policy,ROUTES[route],event,identities=admissions[arm],guard=guard.check)
                require(np.array_equal(p.predict_chunk(**obs),outputs[i]),'endpoint switch/reload changed output')
                queries+=1
        rows.append({'case':key,'original_h8_exact':True,'event_branch_exact_across_models':True,
                     'all_routes_reload_exact':True})
    return write(output/'qualification.json',{'passed':True,'cases':rows,'full_policy_queries':queries,
                                              'new_simulator_steps':0})


def run(run_root,preflight_only=False):
    cfg,training_source,_=admission()
    source=source_check(cfg,training_source)
    screen=read_sealed(PLAN/'screen-manifest.json')
    require(screen['digest']==cfg['screen']['manifest_digest'],'screen manifest changed')
    require(screen['conditions']==list(CONDITIONS),'screen conditions changed')
    output=run_root/'screen';output.mkdir(parents=True,exist_ok=True)
    guard=StageGuard(output,cfg['digest'],cfg,'screen')
    started=time.monotonic();result={'status':'incomplete','design_digest':cfg['digest'],'screen_source_digest':source['digest']}
    try:
        os.environ['TOKENIZERS_PARALLELISM']='false'
        os.environ['TORCHINDUCTOR_CACHE_DIR']='/tmp/torchinductor_peimengsui'
        require_idle_gpu();configure_deterministic_cuda();torch.set_num_threads(4);seed_training(17)
        parent_source=formal.package_check()
        integration=read_sealed(ROOT/'configs/evaluation/week11_integration_v1.json')
        original_admission,_contract,original_endpoint,event_recipe=pilot.endpoint_check(integration)
        rt=runtime()
        parent_cohort=read_sealed(pilot.DATA/'results/week11-formal-v1/cohort.json')
        require(rt==relocated_runtime(parent_cohort['runtime'],ROOT,pilot.LIBERO),'evaluation runtime changed')
        from check_reference_policy import verify_libero
        require(verify_libero(pilot.LIBERO)==parent_cohort['benchmark'],'benchmark changed')
        inputs={p:formal.inventory(integration,p) for p in ('targeted-467','clean-200')}
        gate=read_sealed(ROOT/'docs/evaluation/week11-gate-comparison-v1/bundle/week11-gate-contract.json')
        require(gate['digest']==screen['parent_digest'],'historical case binding changed')
        for path,digest in gate['input_sha256'].items():
            require(sha256_file(Path(path))==digest,'historical screen evidence changed: '+path)
        parents={c['id']:c for c in gate['cases']}
        for c in screen['cases']:
            actual=inputs[c['panel']][0][c['ordinal']]
            require(all(actual[k]==v for k,v in c['task'].items()),'screen task/seed changed')
        seconds,storage=projection(screen,inputs,parents)
        require(seconds+guard.budget.seconds<3600,'screen time projection exceeds one hour')
        require(storage<12*2**30 and shutil.disk_usage(output).free>storage+50*2**30,'screen storage projection exceeds cap')
        write(output/'preflight.json',{'passed':True,'runtime':rt,'projected_seconds':seconds,'projected_bytes':storage,
              'source_digest':source['digest'],'original_admission':original_admission})
        if preflight_only:
            result.update(status='preflight_passed');return
        trained=read_sealed(run_root/'training/completion.json')
        require(trained['passed'] and trained['source_digest']==training_source['digest']
                and trained['design_digest']==cfg['digest'],'matched training not complete')
        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        install_codec_independent_motion_blur()
        actions=Actions(cfg,run_root,original_endpoint)
        actions.select('original')
        original_hash=tensor_state_digest(actions.model.state_dict())
        numerical=read_sealed(formal.EVIDENCE/'remote/numerical-qualification.json')
        require(original_hash==numerical['action_state_sha256'],'original action model changed')
        event=pilot.load_event(event_recipe)
        event_hash=tensor_state_digest(event.model.state_dict())
        require(event_hash==numerical['event_state_sha256'],'frozen event model changed')
        admissions={}
        for m in MODELS:
            fields={k:v for k,v in original_admission.items() if k!='digest'}
            if m!='original':
                end=actions.headers[m]
                fields.update(m4_identity=end['identity']['digest'],m4_endpoint=end['digest'],
                    m4_completion=read_sealed(run_root/'training'/m/'train.json')['digest'],
                    m4_sha256=end['policy_sha256'],predictor_sha256=end['predictor_sha256'])
            admissions[m]=seal_contract({**fields,'model':m,
                'parent_original_admission_digest':original_admission['digest'],
                'continuation_endpoint_digest':actions.headers[m]['digest'] if m!='original' else None,
                'training_design_digest':cfg['digest'],'screen_source_digest':source['digest']})
        qualify(output,actions,event,admissions,guard,parents)
        cohort=seal_contract({'artifact_type':'eventground_wam.week11_multihorizon_screen_cohort',
            'parent_digest':cfg['digest'],'source_digest':source['digest'],'runtime':rt,
            'screen_manifest_digest':screen['digest'],'training_completion_digest':trained['digest']})
        immutable(output/'cohort.json',cohort)
        identities={}
        for condition in CONDITIONS:
            m,_route=condition.split('/')
            for panel in inputs:
                identity=formal.identity_for(cohort,admissions[m],condition,panel,inputs[panel][1],parent_source)
                identity.pop('digest')
                identity=seal_contract({**identity,'artifact_type':'eventground_wam.week11_multihorizon_panel',
                    'mode':'selected_diagnostic','evaluation_source_digest':source['digest'],
                    'training_endpoint':actions.checkpoints[m].training_report_digest})
                identities[condition,panel]=identity
                immutable(output/'panels'/condition/panel/'identity.json',identity)
        rows={}
        for number,item in enumerate(screen['schedule']):
            guard.check();c=screen['cases'][item['case_index']];condition=item['condition'];m,route=condition.split('/')
            panel,ordinal=c['panel'],c['ordinal'];tasks,protocol,suites,states=inputs[panel]
            action=actions.select(m)
            policy=EventGroundInferencePolicy(action,ROUTES[route],event,identities=admissions[m],guard=guard.check)
            identity=identities[condition,panel];task=tasks[ordinal]
            metadata,initial,factory=scene('clean' if panel=='clean-200' else 'perturbed',task,protocol,suites,states,identity)
            row=execute_episode(root=output/'panels'/condition/panel,ordinal=ordinal,task=task,identity=identity,
                protocol=protocol,metadata=metadata,initial_state=initial,policy=policy,environment_factory=factory,seed_episode=seed_episode)
            validate_query_artifact(row,ROUTES[route],admissions[m])
            rows[condition,c['id']]=row
            emit(output,'screen',completed=number+1,total=162,condition=condition,case=c['id'],success=row['success'])
        reports={}
        for condition in CONDITIONS:
            for panel in inputs:
                cases=[c for c in screen['cases'] if c['panel']==panel]
                selected=[rows[condition,c['id']] for c in cases]
                tasks=[inputs[panel][0][c['ordinal']] for c in cases]
                audits=audit_panel(selected,identities[condition,panel],tasks,inputs[panel][1])
                reports[condition,panel]=write(output/'panels'/condition/panel/'report.json',{
                    'case_ids':[c['id'] for c in cases],'episodes':selected,'audits':audits,'summary':summarize_rows(selected)})
        comparisons={p:{a+' versus '+b:compare_inputs(reports[a,p]['audits'],reports[b,p]['audits'])
                        for a,b in itertools.combinations(CONDITIONS,2)} for p in inputs}
        integrity=all(a['error'] is None for r in reports.values() for a in r['audits'])
        integrity &= all(c['all_exact'] for panel in comparisons.values() for c in panel.values())
        integrity &= all(not r.get('query_exceptions') or not any(r['query_exceptions']) for r in rows.values())
        actions.select('original',force=True)
        require(tensor_state_digest(actions.model.state_dict())==original_hash,'original frozen tensors changed')
        require(tensor_state_digest(event.model.state_dict())==event_hash,'event model changed')
        require(runtime()==rt,'runtime changed during screen')
        decision=screen_decision(screen['cases'],{a:{c['id']:rows[a,c['id']]['success'] for c in screen['cases']}
                                for a in CONDITIONS},integrity_valid=bool(integrity))
        write(output/'analysis.json',{'decision':decision,'input_comparisons':comparisons,
            'per_case':[{**c,'outcomes':{a:rows[a,c['id']]['success'] for a in CONDITIONS}} for c in screen['cases']],
            'condition_summaries':{a:summarize_rows([rows[a,c['id']] for c in screen['cases']]) for a in CONDITIONS},
            'model_parameters_unchanged_during_screen':True,'event_state_sha256':event_hash})
        result.update(status='complete',episodes=162,integrity_valid=bool(integrity),decision=decision,
                      training_updates_during_screen=0)
    except BaseException as exc:
        result.update(status='stopped',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,resources=guard.close())
        write(output/'completion.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();run(args.root,args.preflight_only)
