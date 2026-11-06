"""Verify fixed-checkpoint report evidence and every preceding result package."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_sft_report import verify,pilot_base
from eventground_wam.eventground_preservation_runtime import validate_training_completion
from eventground_wam.eventground_runtime import verify_eventground_parent
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    base=parser.parse_args().base;p,_,manifests,_=verify(base)
    report=sealed(base/'analysis/results.json');review=sealed(base/'analysis/retrospective.json');c=sealed(base/'eval/report/control/completion.json')
    done=validate_training_completion(pilot_base(base),'control')
    rows=[json.loads(x) for x in (base/'eval/report/control/predictions.jsonl').read_text().splitlines()]
    if (c['status']!='complete' or c['endpoint_completion_digest']!=done['digest']
        or c['protocol_digest']!=p['digest'] or not c['model_unchanged'] or c['training_updates']!=0
        or not c['targets_reaudited'] or c['future_model_input_reads']!=0
        or c['current_observation_reads']!=384 or metrics(rows)!=c['metrics']
        or sha(base/'eval/report/control/predictions.jsonl')!=c['predictions_sha256']
        or c['elapsed_seconds']>p['inference_resources']['wall_seconds']
        or c['peak_reserved_bytes']>.9*c['gpu_total_bytes']
        or report['evaluations']['control']!=c['digest'] or review['report_results_digest']!=report['digest']):
        raise ValueError('Report or retrospective integrity differs')
    verify_eventground_parent(sealed(base/'source/configs/training/week10_eventground_experiment_v3.json'))
    prior={}
    for name in ('week10-weak-rlvr-v1','week10-report-coverage-extension-v1','week10-future-onset-v1','week10-preservation-pilot-v1'):
        package=sealed(base.parent/name/'package-index.json');root=base.parent if name=='week10-report-coverage-extension-v1' else base.parent/name
        for f,h in package['files_sha256'].items():
            if sha(root/f)!=h:raise ValueError('Earlier evidence changed: '+f)
        prior[name]=dict(digest=package['digest'],files=len(package['files_sha256']))
    proof=seal_contract(dict(status='complete',protocol_digest=p['digest'],evaluation_digest=c['digest'],results_digest=report['digest'],
        retrospective_digest=review['digest'],endpoint_sha256=done['endpoint_sha256'],training_updates=0,
        old_packages_verified=prior,m4_verified=True,report_inputs_reaudited=True,
        verified_at=datetime.now(timezone.utc).isoformat()))
    with (base/'completion.json').open('x') as f:json.dump(proof,f,indent=2,sort_keys=True);f.write('\n')
    files={str(x.relative_to(base)):sha(x) for x in sorted(base.rglob('*')) if x.is_file() and x.suffix not in ('.pt','.pyc') and '__pycache__' not in x.parts}
    package=seal_contract(dict(artifact_type='eventground_wam.sft_report_evidence_package',protocol_digest=p['digest'],completion_digest=proof['digest'],
        files_sha256=files,total_bytes=sum((base/f).stat().st_size for f in files),large_endpoint='Existing SFT-only control checkpoint on Feifei; no new weights.'))
    with (base/'package-index.json').open('x') as f:json.dump(package,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(package_digest=package['digest'],files=len(files),prior_files_verified=sum(x['files'] for x in prior.values()))))


if __name__=='__main__':main()
