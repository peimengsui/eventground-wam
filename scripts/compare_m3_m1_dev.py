#!/usr/bin/env python3
"""Audit all 210 stored episodes and compare fixed M3/M1 endpoints."""
from pathlib import Path
import argparse, json
from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, paired_effect, summarize_rows

ROOT = Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m3',type=Path,required=True)
    p.add_argument('--m1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    identities={k:read_sealed(v/'identity.json') for k,v in [('M3',a.m3),('M1',a.m1)]}
    reports={k:read_sealed(v/'evaluation-report.json') for k,v in [('M3',a.m3),('M1',a.m1)]}
    require(identities['M1']['digest']=='b601d6099e2231351ee7639b16ff558fbfd88b78577beacb9f26b618c34ae7cb', 'M1 identity changed')
    require(reports['M1']['digest']=='9d0548bb60bc4a4b590877bd4dc65ef7fda131bf71e20d0ac96bbf662f7eb74e', 'M1 report changed')
    require(identities['M3']['checkpoint_revision']=='8d32417789ea8ea41a3f948cb1564bd3dc150b9d00e673be6d7df8f19562e0e2', 'M3 endpoint changed')
    require(identities['M3']['evaluation_source_digest']=='3afa15aabb9b186b26c0212564e4d531c6920d343c4489d7e5350b80e5643304','M3 source changed')
    tasks=json.loads((ROOT/'evaluation/splits/libero-plus-v1/dev-210.json').read_text())['tasks']
    require(len(tasks)==210, 'dev panel changed')
    protocol=load_evaluation_protocol(ROOT/'configs/evaluation/libero_plus_v1.json')
    for k in reports:
        require(reports[k]['identity_digest']==identities[k]['digest'],'report identity changed')
    audits={k:audit_panel(reports[k]['episodes'],identities[k],tasks,protocol) for k in reports}
    comparison=compare_inputs(audits['M3'],audits['M1'])
    result=seal_contract({'artifact_type':'eventground_wam.m3_vs_m1_dev210',
        'status':'analyzed','report_digests':{k:r['digest'] for k,r in reports.items()},
        'identity_digests':{k:r['digest'] for k,r in identities.items()},
        'summary':{k:summarize_rows(r['episodes']) for k,r in reports.items()},
        'M3_minus_M1':paired_effect([r['success'] for r in reports['M3']['episodes']],
                                  [r['success'] for r in reports['M1']['episodes']]),
        'input_comparison':comparison,'artifact_audits':audits,
        'limitations':['single training seed','development panel, not unseen holdout',
            'coupling adds parameters and compute','legacy dev timing; common explicit synchronization starts with Week9',
            comparison['qualification']],
        'all_tasks_retained':True})
    atomic_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('status','digest','M3_minus_M1')}))
if __name__=='__main__': main()
