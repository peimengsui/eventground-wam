#!/usr/bin/env python3
"""Independent, checksummed backup of M3 training, dev210, and M1 comparison."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime, timezone
from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed,sha256_file,atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.coupled_evaluation import load_contract,verify_endpoint

def copy_checked(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True); digest=sha256_file(source)
    if not destination.exists():
        temp=destination.with_name(destination.name+'.copying'); shutil.copy2(source,temp)
        require(sha256_file(temp)==digest,'backup copy differs');temp.replace(destination)
    require(sha256_file(destination)==digest,f'backup changed: {destination}')
    return {'source':str(source),'sha256':digest,'bytes':source.stat().st_size}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','evaluation','m1','source','evidence','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); cfg=load_contract(a.source,'M3'); _,endpoint,completion=verify_endpoint(a.training,cfg)
    analysis=read_sealed(a.evaluation/'M3-versus-M1.json')
    require(analysis['status']=='analyzed' and analysis['input_comparison']['count']==210,'analysis incomplete')
    report=read_sealed(a.evaluation/'dev-210/M3/evaluation-report.json')
    require(report['digest']==analysis['report_digests']['M3'] and report['all_artifacts_present'],'M3 report changed or incomplete')
    source=read_sealed(a.source/'evaluation-source-manifest.json')
    for name,digest in source['files'].items():require(sha256_file(a.source/name)==digest,'evaluator changed')
    for arm,base in [('M3',a.evaluation/'dev-210/M3'),('M1',a.m1)]:
        r=read_sealed(base/'evaluation-report.json')
        require(r['digest']==analysis['report_digests'][arm],'comparison report changed')
        for row in r['episodes']:require(sha256_file(Path(row['artifact']))==row['artifact_sha256'],'rollout changed')
    groups=[('training',a.training),('evaluation',a.evaluation),('M1-comparison',a.m1),
            ('evaluation-source',a.source),('qualification',a.evidence)]
    a.output.mkdir(parents=True,exist_ok=True)
    total=sum(p.stat().st_size for _,base in groups for p in base.rglob('*') if p.is_file())
    require(shutil.disk_usage(a.output).free>total+50*1024**3,'backup disk reserve')
    inventory={}
    for label,base in groups:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            relative=Path(label)/path.relative_to(base)
            inventory[str(relative)]=copy_checked(path,a.output/relative)
        print(json.dumps({'copied':label}),flush=True)
    helpers=Path(__file__).parent
    for name in ('package_m3_results.py','compare_m3_m1_dev.py'):
        inventory[name]=copy_checked(helpers/name,a.output/name)
    result=seal_contract({'artifact_type':'eventground_wam.week8_m3_results_package',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'complete_verified_backup',
        'checkpoint_sha256':endpoint['checkpoint_sha256'],'training_completion_digest':completion['digest'],
        'analysis_digest':analysis['digest'],'source_digest':source['digest'],
        'copies_are_independent_files':True,'copied_file_count':len(inventory),
        'copied_bytes':sum(x['bytes'] for x in inventory.values()),'inventory':inventory,
        'input_qualification':analysis['input_comparison']['qualification'],
        'portability':'Inventory maps original absolute paths to independent files. Pinned Qwen base/benchmark assets remain external dependencies.'})
    atomic_json(a.output/'package-manifest.json',result)
    print(json.dumps({k:result[k] for k in ('status','digest','copied_file_count','copied_bytes')}),flush=True)
if __name__=='__main__':main()
