"""Separate early continuous arm motion from the decoded gripper command."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from week11_replay_common import read, sha

from eventground_wam.eventground_horizon import select_horizon
from eventground_wam.week11_closing_gate import ADAPTIVE, FIXED, ClosingHistory

ARMS=('fixed8','early-only','motion-only','gripper-only')


class WindowPolicy:
    def __init__(self,fixed,adaptive,arm):
        if arm not in ARMS:raise ValueError('Unknown channel arm')
        self.fixed,self.adaptive,self.arm=fixed,adaptive,arm
        self.checkpoint=fixed.checkpoint;self.reset()

    def reset(self):
        self.history=ClosingHistory();self.last_query=None

    def predict_chunk(self,**observation):
        frame=self.history.executed_steps
        if frame<10 or (frame-10)%8:raise ValueError('Wrong eight-action cadence')
        active=frame<138
        try:
            fixed=self.fixed.predict_chunk(**observation).copy();fq=dict(self.fixed.last_query)
            adaptive=aq=None
            if active:
                adaptive=self.adaptive.predict_chunk(**observation).copy();aq=dict(self.adaptive.last_query)
                assert fq['prediction']==aq['prediction'] and fq['decision']==aq['decision']
                assert fq['event']['categorical_probabilities']==aq['event']['categorical_probabilities']
            motion=active and self.arm in ('early-only','motion-only')
            gripper=active and self.arm in ('early-only','gripper-only')
            actions=fixed.copy()
            if motion:actions[:,:6]=adaptive[:,:6]
            if gripper:actions[:,6]=adaptive[:,6]
            main=aq if motion else fq
            self.last_query={**main,'window_arm':self.arm,'absolute_frame':frame,'window_active':active,
                'channel_sources':{'motion':'adaptive' if motion else 'fixed8','gripper':'adaptive' if gripper else 'fixed8'},
                'fixed_candidate':{'query':fq,'actions':fixed.tolist()},
                'adaptive_candidate':{'query':aq,'actions':adaptive.tolist()} if active else None,
                'candidate_policy_queries':2 if active else 1}
            assert actions.shape==(8,7) and np.isfinite(actions).all() and np.max(np.abs(actions))<=1
            return actions
        except BaseException as exc:
            self.last_query={'window_arm':self.arm,'absolute_frame':frame,'exception':f'{type(exc).__name__}: {exc}'}
            raise


def audit_window(row,arm,admission):
    path=Path(row['artifact']);timing=read(path.with_suffix('.timing.json'))
    qp=path.with_suffix('.queries.jsonl');assert sha(qp)==timing['query_artifact_sha256']
    queries=[json.loads(l) for l in qp.read_text().splitlines()];assert queries==timing['queries']
    active_frames=[];horizons=[];non8=[];candidate_calls=0;motion_changed=[];gripper_changed=[]
    with h5py.File(path) as f:
        n=len(f['steps/executed_action'])
        for i,q in enumerate(queries):
            frame=10+8*i;p=q['policy'];active=frame<138
            assert frame<n and q['query_index']==i and q['chunk_start']==8*i
            assert p['window_arm']==arm and p['absolute_frame']==frame and p['window_active']==active
            assert p['exception'] is q['exception'] is None
            assert p['candidate_policy_queries']==(2 if active else 1)
            candidate_calls+=p['candidate_policy_queries']
            candidates={'fixed8':p['fixed_candidate']}
            if active:candidates['adaptive']=p['adaptive_candidate']
            else:assert p['adaptive_candidate'] is None
            for route,c in candidates.items():
                v=np.asarray(c['actions']);r=c['query']
                assert v.shape==(8,7) and np.isfinite(v).all() and np.max(np.abs(v))<=1
                assert np.isin(v[:,6],[-1,1]).all() and r['branch_identities']==admission
                assert r['arm']==(FIXED if route=='fixed8' else ADAPTIVE) and r['exception'] is None
                decision=select_horizon(r['prediction']);assert asdict(decision)==r['decision']
                assert r['used_horizon']==(8 if route=='fixed8' else decision.selected_horizon)
                assert r['execute_actions']==r['proposal_actions']==8 and r['zero_padded_actions']==24 and r['future_calls']==1
                for name,prob in r['event']['categorical_probabilities'].items():
                    assert len(prob)==(4 if name=='event' else 32) and np.isfinite(prob).all()
                    assert min(prob)>=0 and abs(sum(prob)-1)<1e-5
            if active:
                x,y=(candidates[k]['query'] for k in ('fixed8','adaptive'))
                assert x['prediction']==y['prediction'] and x['decision']==y['decision']
                assert x['event']['categorical_probabilities']==y['event']['categorical_probabilities']
            sources={'motion':'adaptive' if active and arm in ('early-only','motion-only') else 'fixed8',
                     'gripper':'adaptive' if active and arm in ('early-only','gripper-only') else 'fixed8'}
            assert p['channel_sources']==sources
            expected=np.column_stack([np.asarray(candidates[sources['motion']]['actions'])[:,:6],
                                      np.asarray(candidates[sources['gripper']]['actions'])[:,6]])
            assert np.array_equal(expected,f['steps/predicted_action_chunk'][frame])
            assert np.array_equal(expected[:min(8,n-frame)],f['steps/executed_action'][frame:frame+8])
            fixed=np.asarray(candidates['fixed8']['actions'])
            if not np.array_equal(expected[:,:6],fixed[:,:6]):motion_changed.append(frame)
            if not np.array_equal(expected[:,6],fixed[:,6]):gripper_changed.append(frame)
            if active:active_frames.append(frame)
            horizon=candidates['adaptive']['query']['used_horizon'] if active else 8
            horizons.append(horizon)
            if active and horizon!=8:non8.append(frame)
    return {'valid':True,'query_count':len(queries),'candidate_policy_queries':candidate_calls,
        'active_frames':active_frames,'horizons':horizons,'non8_frames':non8,
        'motion_differs_from_same_input_fixed8_frames':motion_changed,
        'gripper_differs_from_same_input_fixed8_frames':gripper_changed}
