from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_episode_recorder import episode_metadata, step_record
from eventground_wam.episode_recorder import EpisodeRecorder, read_episode_metadata
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.clean_comparison import atomic_json, metadata_for_pair
from eventground_wam.b0_training import read_sealed
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.week9_execution import bind_metadata, execute_episode, inspect_timed
from eventground_wam.week9_recording import run_episode
from eventground_wam.evaluation_comparisons import latency_summary, compare_inputs, paired_effect
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def case(tmp_path):
    protocol=load_evaluation_protocol()
    identity=seal_contract({'cohort_digest':'cohort','week9_protocol_digest':'freeze',
        'policy_objective':'all-action-l1-binary-gripper-v2','admission_digest':'admitted'})
    metadata=bind_metadata(episode_metadata(protocol),identity)
    task={k:getattr(metadata,k) for k in ('suite','runtime_task_index','perturbation_category','difficulty_level','episode_seed')}
    return dict(root=tmp_path,ordinal=0,task=task,identity=identity,protocol=protocol,metadata=metadata,
                initial_state=np.zeros(12),policy=None,environment_factory=lambda:SimpleNamespace(close=lambda:None),
                seed_episode=lambda s:None,episode_runner=fixture_episode)

def fixture_episode(*,artifact_path,protocol,metadata,**kwargs):
    with EpisodeRecorder(artifact_path,protocol,metadata) as rec:
        for i in range(11): rec.append(step_record(protocol,i,policy=i==10,done=i==10))
        atomic_json(artifact_path.with_suffix('.timing.json'),seal_contract({
            'checkpoint_revision':metadata.checkpoint_revision,'cohort_digest':metadata.cohort_digest,
            'queries':[{'latency_ms':1.25,'exception':None}], 'episode_wall_time_seconds':2.,
            'gpu_sync_at_both_boundaries':True,'no_queries_excluded':True}))
        rec.finalize(success=True,termination_reason='environment_done')

def test_durable_resume_and_metadata_binding(case):
    row=execute_episode(**case)
    assert execute_episode(**case)==row
    assert read_episode_metadata(Path(row['artifact']))['cohort_digest']=='cohort'
    case['metadata']=replace(case['metadata'],admission_digest='different')
    with pytest.raises(ValueError): execute_episode(**case)

def test_crash_before_environment_consumes_one_attempt(case):
    def interrupt(): raise KeyboardInterrupt
    case['environment_factory']=interrupt
    with pytest.raises(KeyboardInterrupt):execute_episode(**case)
    case['environment_factory']=lambda:SimpleNamespace(close=lambda:None)
    assert execute_episode(**case)['retry_count']==1

def test_two_failures_are_terminal_on_resume(case):
    calls=[]
    def unavailable(): calls.append(1); raise OSError('unavailable')
    case['environment_factory']=unavailable
    r=execute_episode(**case)
    assert not r['success'] and r['artifact'] is None
    assert execute_episode(**case)==r and len(calls)==2

def test_recover_completed_after_interrupt_without_rerun(case):
    def interrupt(**kw): fixture_episode(**kw); raise KeyboardInterrupt
    case['episode_runner']=interrupt
    with pytest.raises(KeyboardInterrupt):execute_episode(**case)
    case['episode_runner']=lambda **kw:pytest.fail('must not rerun')
    assert execute_episode(**case)['success']

def test_clean_storage_preserves_fixed_state_and_adds_binding(case):
    protocol=load_evaluation_protocol(ROOT/'configs/evaluation/week4_clean_comparison_v1.json')
    pair=read_sealed(ROOT/'evaluation/splits/week4-clean-comparison-v1/paired-2000.json')['pairs'][17]
    identity={**case['identity'],'project_git_revision':'a'*40,'checkpoint_id':'M3',
        'checkpoint_revision':'b'*64,'acceptance_rule_digest':protocol['comparison']['acceptance_rule_digest']}
    meta=bind_metadata(metadata_for_pair(pair,protocol,identity,Path('/test.bddl')),identity)
    assert meta.initial_state_index==meta.trial_index==17
    assert meta.legacy_profile_and_acceptance_fields_are_format_only
    fixture_episode(artifact_path=case['root']/'clean.hdf5',protocol=protocol,metadata=meta)
    assert inspect_timed(case['root']/'clean.hdf5',protocol,meta)['success']

def test_latency_all_queries_and_equal_episode_weights():
    s=latency_summary([[1,100],[2],[]])
    assert s['all_queries']['count']==3 and s['first_queries']['count']==2
    assert s['subsequent_queries']['median_ms']==100
    assert s['episode_medians']['median_ms']==26.25 and s['episodes_without_recorded_queries']==1
    with pytest.raises(ValueError):latency_summary([[float('nan')]])

def test_clean_cluster_bootstrap_retains_whole_tasks():
    tasks=[{'suite':'s','runtime_task_index':t} for t in range(40) for _ in range(50)]
    left=[int(t<20) for t in range(40) for _ in range(50)]; right=[0]*2000
    s=paired_effect(left,right,tasks=tasks,clean=True)
    assert s['gap_percentage_points']==50 and s['left_only_success']==1000
    assert s['paired_95_interval_percentage_points']==pytest.approx([35,65])

def test_real_panel_audit_rechecks_timing_sidecar(case):
    from eventground_wam.evaluation_comparisons import audit_panel
    meta=case['metadata']
    case['task']['classification_task_id']=meta.classification_task_id
    case['identity']=seal_contract({**{k:v for k,v in case['identity'].items() if k!='digest'},
        **{k:getattr(meta,k) for k in ('project_git_revision','checkpoint_id','checkpoint_revision')}})
    row=execute_episode(**case)
    assert audit_panel([row],case['identity'],[case['task']],case['protocol'])[0]['error'] is None
    Path(row['timing_artifact']).write_text('{}')
    assert 'timing artifact changed' in audit_panel([row],case['identity'],[case['task']],case['protocol'])[0]['error']

class Environment:
    def __init__(self):self.n=0
    def seed(self,s):pass
    def reset(self):pass
    def set_init_state(self,s):return self.obs()
    def get_sim_state(self):return np.zeros(12)
    def obs(self):return {'agentview_image':np.zeros((256,256,3),dtype=np.uint8),
        'robot0_eye_in_hand_image':np.zeros((256,256,3),dtype=np.uint8),
        'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0.,0.,0.,1.]),'robot0_gripper_qpos':np.zeros(2)}
    def step(self,a):self.n+=1;return self.obs(),0.,self.n==11,{}

@pytest.mark.parametrize('fail',[False,True])
def test_full_query_timer_synchronizes_and_keeps_failed_query(case,fail,monkeypatch):
    monkeypatch.setattr('evaluate_reference_policy.proprioception_from_observation',lambda obs:np.zeros(8))
    monkeypatch.setattr('eventground_wam.week9_recording.proprioception_from_observation',lambda obs:np.zeros(8))
    events=[]
    def predict(**kw):
        events.append('predict')
        if fail:raise ValueError('fixture failure')
        return np.zeros((8,7))
    path=case['root']/'real-loop.hdf5'
    result=run_episode(environment=Environment(),policy=SimpleNamespace(predict_chunk=predict),
        protocol=case['protocol'],metadata=case['metadata'],initial_state=case['initial_state'],
        artifact_path=path,timing_synchronize=lambda:events.append('sync'))
    assert events==['sync','predict','sync']
    timing=inspect_timed(path,case['protocol'],case['metadata'])
    assert len(timing['policy_latencies_ms'])==1
    assert result.success is (not fail)
    assert len(timing['recorded_action_query_latencies_ms'])==(0 if fail else 1)
