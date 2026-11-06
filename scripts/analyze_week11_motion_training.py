"""Independent metadata audit of the two completed adapter training arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from week11_replay_common import read, sha, write

from eventground_wam.b0_training import learning_rate_factor
from eventground_wam.demonstration_contracts import canonical_json_bytes

ARMS=('horizon-motion','h8-information-control')


def analyze(remote, schedule_path, output):
    cfg=read(remote/'bundle/contract.json'); pipeline=read(remote/'completion.json')
    schedule=read(schedule_path)
    assert pipeline['status']=='complete' and pipeline['passed'] and pipeline['production_updates']==4000
    assert pipeline['contract_digest']==cfg['digest'] and pipeline['matched_training_exposure']
    assert schedule['digest']==cfg['schedule_digest']
    init=read(remote/'initialization-audit.json')
    assert init['status']=='pass' and set(init['arms'])==set(ARMS)
    assert all(v['adapter_state_sha256']==init['expected_initial_state']
               and v['zero_updates'] and v['fresh_optimizer'] for v in init['arms'].values())
    qualification=read(remote/'qualification/completion.json')
    assert qualification['status']=='complete' and qualification['passed']
    assert qualification['frozen_reference_unchanged']
    for name,digest in cfg['code_sha256'].items():assert sha(remote/'bundle'/name)==digest
    lat=read(remote/'qualification/latency.json');assert lat['passed']
    timings={}
    for label in ('original-fixed',*ARMS):
        vals=sorted(r['ms'] for r in lat['rows'] if r['condition']==label)
        assert len(vals)==152
        where=(len(vals)-1)*.95; i=math.floor(where)
        p95=vals[i]+(vals[i+1]-vals[i])*(where-i)
        assert math.isclose(p95,lat['summary'][label]['p95_ms'],abs_tol=1e-10)
        timings[label]=p95
    assert all(timings[a]<=2*timings['original-fixed'] for a in ARMS)
    results={}
    for arm in ARMS:
        base=remote/'training'/arm; done=read(base/'completion.json'); end=read(base/'endpoint/endpoint.json')
        assert done['passed'] and done['status']=='complete' and done['updates']==2000
        assert done['seconds']<cfg['training_total_seconds']/2
        assert done['contract_digest']==cfg['digest'] and end['identity']==read(base/'identity.json')
        assert end['reference_sha256']==cfg['h8_endpoint']['policy_sha256']
        assert end['adapter_state_sha256']!=init['expected_initial_state']
        rows=[json.loads(s) for s in (base/'updates.jsonl').read_text().splitlines()]
        assert len(rows)==2000 and [r['global_step'] for r in rows]==list(range(1,2001))
        offset=0; chain='0'*64; counts={h:0 for h in (2,4,16,32)}
        for i,(r,b) in enumerate(zip(rows,schedule['batches'],strict=True)):
            assert b['presentation_start']==offset
            hs=[(2,4,16,32)[(offset+j)%4] for j in range(len(b['tokens']))]
            assert r['candidate_horizons']==hs
            offset+=len(hs); assert r['samples_seen']==offset
            chain=hashlib.sha256(bytes.fromhex(chain)+canonical_json_bytes(hs)).hexdigest()
            assert r['horizon_chain']==chain
            rate=1e-4*learning_rate_factor(i,2000,{'warmup_fraction':.02,'minimum_learning_rate_ratio':.1})
            assert r['learning_rate']==rate
            assert math.isfinite(r['motion_l1']) and math.isfinite(r['gradient_norm'])
            for h in hs:counts[h]+=1
        assert offset==end['samples_seen']==58172 and all(v==14543 for v in counts.values())
        assert chain==end['horizon_chain'] and rows[-1]['sample_chain']==end['sample_chain']
        timing=read(remote/'qualification'/arm/'timing.json')
        for a,b in zip(rows[:64],timing['rows'],strict=True):
            assert all(a[k]==b[k] for k in ('motion_l1','gradient_norm','sample_chain','horizon_chain','learning_rate'))
        records=[remote/'qualification'/arm/'initial-parity.json',remote/'qualification'/arm/'trained-parity.json',
                 remote/'restart'/arm/'parity.json',remote/'endpoint-verification'/arm/'parity.json']
        for path in records:
            parity=read(path); assert parity['passed'] and len(parity['rows'])==456
            assert all(r['h8_exact'] and r['gripper_exact'] for r in parity['rows'])
        restart=read(remote/'restart'/arm/'completion.json');verify=read(remote/'endpoint-verification'/arm/'completion.json')
        assert restart['fresh_process_next_update_exact'] and restart['frozen_reference_unchanged']
        assert verify['fresh_process_endpoint_loaded'] and verify['endpoint_digest']==end['digest']
        assert done['frozen_reference_unchanged'] and verify['frozen_reference_unchanged']
        for record in (done,restart,verify):
            limits=record['resources']
            assert limits['stop_reason'] is None
            assert limits['peak_process_tree_rss_bytes']<28*2**30 and limits['free_bytes']>=50*2**30
            assert limits['peak_reserved_bytes']<.9*limits['gpu_total_bytes']
        results[arm]={'updates':2000,'samples_seen':offset,'horizon_counts':{str(h):v for h,v in counts.items()},
            'endpoint_digest':end['digest'],'adapter_state_sha256':end['adapter_state_sha256'],
            'sample_chain':end['sample_chain'],'horizon_chain':chain,
            'mean_training_step_seconds':sum(r['seconds'] for r in rows)/2000,
            'final_step_motion_l1':rows[-1]['motion_l1'],'first_64_updates_match_qualification':True,
            'frozen_reference_unchanged':True,'all_456_h8_inputs_preserved':True,
            'fresh_process_restart_exact':True,'fresh_process_endpoint_loaded':True}
    assert results[ARMS[0]]['sample_chain']==results[ARMS[1]]['sample_chain']
    assert results[ARMS[0]]['horizon_chain']==results[ARMS[1]]['horizon_chain']
    write(output,{'status':'pass','contract_digest':cfg['digest'],'pipeline_digest':pipeline['digest'],
        'identical_fresh_initialization':True,'initialization_audit_digest':init['digest'],
        'trained_states_distinct':results[ARMS[0]]['adapter_state_sha256']!=results[ARMS[1]]['adapter_state_sha256'],
        'p95_latency_ms':timings,'arms':results,'model_calls':0,'simulator_steps':0,'new_rollouts':0,
        'checkpoint_scope':'Fresh-process loading, tensor hashing, parity and exact metadata audit verified on Feifei; transferred evidence is rehashed locally.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote',type=Path,required=True)
    p.add_argument('--schedule',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.remote,a.schedule,a.output)
