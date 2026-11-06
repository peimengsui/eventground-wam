#!/usr/bin/env python3
"""Common-resample interim and complete Week 9 reporting; never impute an arm."""
from pathlib import Path
from collections import defaultdict
import argparse, json
import numpy as np
from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, paired_effect, summarize_rows
from eventground_wam.factorial_analysis import summarize_panel, robustness_ratios
from evaluate_week9 import contract, panel_inventory, identity_for, package_check

def samples(rows, tasks, panel):
    values=np.asarray([r['success'] for r in rows],dtype=float)
    if panel=='clean':
        clusters=defaultdict(list)
        for i,t in enumerate(tasks): clusters[(t['suite'],t['runtime_task_index'])].append(i)
        require(len(clusters)==40 and all(len(v)==50 for v in clusters.values()),'clean topology changed')
        values=np.asarray([values[v] for v in clusters.values()])
        rates=values.mean(1); seed=8
    else: rates=values; seed=7
    indices=np.random.Generator(np.random.PCG64(seed)).integers(0,len(rates),(10000,len(rates)))
    return values,float(rates.mean()),rates[indices].mean(1)

def analyze(root):
    cfg=contract(); source=package_check(); cohort=read_sealed(root/'cohort.json')
    require(cohort['source_digest']==source['digest'],'cohort source changed')
    panels={}; matrices={}; rate_draws={}; reports_by_panel={}
    for panel in ('perturbed','clean'):
        tasks,protocol,_,_=panel_inventory(panel)
        reports={}; audits={}; draws={}; points={}; matrices[panel]={}
        for arm in cfg['arms']:
            output=root/'panels'/arm/panel; path=output/'evaluation-report.json'
            if not path.exists(): continue
            report=read_sealed(path); identity=read_sealed(output/'identity.json')
            admission=read_sealed(root/'admissions'/f'{arm}.json')
            require(identity==identity_for(cohort,admission,panel,protocol),'panel identity changed')
            require(report['identity_digest']==identity['digest'] and report['mode']=='formal','wrong panel report')
            current=audit_panel(report['episodes'],identity,tasks,protocol)
            require(current==report['audits'],'artifact audit changed')
            reports[arm]=report; audits[arm]=current
            matrices[panel][arm],points[arm],draws[arm]=samples(report['episodes'],tasks,panel)
        comparisons={}
        for label in cfg['contrasts']['pairwise']:
            left,right=label.split('-')
            if left in reports and right in reports:
                comparisons[label]={'outcomes':paired_effect(
                    [r['success'] for r in reports[left]['episodes']],
                    [r['success'] for r in reports[right]['episodes']],tasks=tasks,clean=panel=='clean'),
                    'inputs':compare_inputs(audits[left],audits[right])}
                if panel=='perturbed':
                    grouped={}
                    for field in ('suite','perturbation_category','difficulty_stratum'):
                        groups=defaultdict(list)
                        for i,t in enumerate(tasks): groups[str(t[field])].append(i)
                        grouped[field]={}
                        for key,indices in groups.items():
                            comparison=paired_effect([reports[left]['episodes'][i]['success'] for i in indices],
                                                     [reports[right]['episodes'][i]['success'] for i in indices])
                            comparison['qualification']=comparisons[label]['inputs']['qualification']
                            grouped[field][key]=comparison
                    comparisons[label]['one_way_subgroups']=grouped
        summaries={arm:summarize_rows(r['episodes']) for arm,r in reports.items()}
        for arm,s in summaries.items():
            s['bootstrap_success_rate_95']=np.quantile(draws[arm],[.025,.975],method='linear').tolist()
            s['primary_interval_unit']='canonical_task_with_50_states' if panel=='clean' else 'variant_task'
            if panel=='perturbed':
                groups=defaultdict(list)
                for suite in protocol['benchmark']['suites']:
                    for category in protocol['benchmark']['classification']['categories']:
                        groups[f'{suite} / {category}']=[]
                dev=json.loads((Path(__file__).resolve().parents[1]/cfg['panels']['development_overlap']['path']).read_text())['tasks']
                devkeys={(t['suite'],t['runtime_task_index']) for t in dev}
                for task,row in zip(tasks,reports[arm]['episodes'],strict=True):
                    groups[f"{task['suite']} / {task['perturbation_category']}"].append(row['success'])
                s['suite_x_category']={key:{'count':len(v),'successes':sum(v),'success_rate':sum(v)/len(v) if v else None} for key,v in groups.items()}
                s['development_overlap']={}
                for name,overlap in [('dev210',True),('additional790',False)]:
                    subset=[row for task,row in zip(tasks,reports[arm]['episodes'],strict=True)
                            if ((task['suite'],task['runtime_task_index']) in devkeys)==overlap]
                    s['development_overlap'][name]=summarize_rows(subset)
        panels[panel]={'available_arms':list(reports),'summaries':summaries,'comparisons':comparisons,
            'factorial':summarize_panel(matrices[panel],panel=panel) if set(reports)==set(cfg['arms'])
                and all(r['status']=='audited' for r in reports.values()) else None,
            'audit_complete':all(r['status']=='audited' for r in reports.values()),
            'report_digests':{a:r['digest'] for a,r in reports.items()}}
        rate_draws[panel]=(points,draws); reports_by_panel[panel]=reports
    ratios={}
    for arm in set(rate_draws['perturbed'][0]) & set(rate_draws['clean'][0]):
        num,den=rate_draws['perturbed'][0][arm],rate_draws['clean'][0][arm]
        x,y=rate_draws['perturbed'][1][arm],rate_draws['clean'][1][arm]
        zero=int((y==0).sum())
        ratios[arm]={'estimate':num/den if den else None,'zero_denominator_draws':zero,
                    'percentile_95':np.quantile(x/y,[.025,.975],method='linear').tolist() if den and not zero else None,
                    'interpretation':'descriptive ratio of distinct panel distributions'}
    result=seal_contract({'artifact_type':'eventground_wam.week9_analysis','cohort_digest':cohort['digest'],
        'status':'complete' if all(set(x['available_arms'])==set(cfg['arms']) and x['audit_complete'] for x in panels.values()) else 'interim',
        'panels':panels,'robustness_ratios':ratios,'limitations':cfg['limitations'],
        'input_qualification':'Whole affected contrasts are descriptive; see every pairwise input audit.',
        'factorial_input_qualification':'Exact only if every contributing pairwise input audit is exact.',
        'missing_arms_imputed':False,'all_queries_including_failed_queries_retained':True,
        'bootstrap':cfg['bootstrap']})
    destination=root/'analysis'/f"analysis-{result['digest'][:16]}.json"
    atomic_json(destination,result)
    atomic_json(root/'analysis/latest.json',seal_contract({'path':str(destination),'analysis_digest':result['digest']}))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); r=analyze(a.root); print(json.dumps({'status':r['status'],'digest':r['digest']}))
