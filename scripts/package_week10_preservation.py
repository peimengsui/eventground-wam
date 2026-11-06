"""Verify remote endpoints and prior evidence, then seal a portable metadata package."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from analyze_week10_weak_pilot import read_eval
from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_future_runtime import validate_training_completion as validate_parent_run
from eventground_wam.eventground_preservation_runtime import verify, parent_base, selected_weight, validate_training_completion
from eventground_wam.eventground_runtime import verify_eventground_parent
from eventground_wam.eventground_weak_supervision import sealed, sha
from eventground_wam.pixel_predictor import tensor_state_digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    base=parser.parse_args().base;p,_,_,recipe=verify(base)
    queue=sealed(base/'execution-completion.json');results=sealed(base/'analysis/results.json')
    if queue['status']!='complete_pilot_only' or results['protocol_digest']!=p['digest']:
        raise ValueError('Incomplete queue or result')
    if queue['full_training_launched'] or queue['report_panel_opened']:
        raise ValueError('Exceeded pilot-only scope')
    weight,diagnostic=selected_weight(base)
    checked={}
    for stage in p['training_stages']:
        c=validate_training_completion(base,stage)
        if not stage.endswith('qualification'):
            artifact=torch.load(base/'runs'/stage/'endpoint.pt',map_location='cpu',weights_only=True)
            h=artifact['header'];params=artifact['parameters']
            names=sorted(recipe['exact_allowlist']['head']+recipe['exact_allowlist']['language_lora'])
            if (h!=c['inference_header'] or h['digest']!=contract_digest(h) or h['exact_names']!=names
                or sorted(params)!=names or h['parameter_sha256']!=tensor_state_digest(params)
                or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in params.values())):
                raise ValueError('Endpoint tensor identity changed')
            evaluation,_=read_eval(base/'eval/dev'/stage)
            if evaluation['endpoint_completion_digest']!=c['digest'] or evaluation['digest']!=results['evaluations'][stage]:
                raise ValueError('Evaluation endpoint changed')
        checked[stage]=c['digest']
    completed=validate_parent_run(parent_base(base),'sft')
    if completed['digest']!=p['sft_completion_digest']:
        raise ValueError('Parent SFT changed')
    verify_eventground_parent(sealed(base/'source/configs/training/week10_eventground_experiment_v3.json'))
    preserved={}
    for name in ('week10-weak-rlvr-v1','week10-report-coverage-extension-v1'):
        package=sealed(base.parent/name/'package-index.json')
        root=base.parent/name if name=='week10-weak-rlvr-v1' else base.parent
        for filename,digest in package['files_sha256'].items():
            if sha(root/filename)!=digest:raise ValueError('Older evidence changed '+filename)
        preserved[name]=dict(digest=package['digest'],files=len(package['files_sha256']))
    proof=seal_contract(dict(status='passed',protocol_digest=p['digest'],selected_weight=weight,diagnostic_digest=diagnostic['digest'],
        training_completions=checked,parent_sft_endpoint_sha256=completed['endpoint_sha256'],
        parent_package_digest=p['parent_package_digest'],parent_package_files_verified=len(sealed(parent_base(base)/'package-index.json')['files_sha256']),
        earlier_packages=preserved,m4_parent_verified=True,all_fixed_endpoints_verified=True,
        verified_at=datetime.now(timezone.utc).isoformat()))
    with (base/'final-verification.json').open('x') as f:json.dump(proof,f,indent=2,sort_keys=True);f.write('\n')
    files={str(x.relative_to(base)):sha(x) for x in sorted(base.rglob('*'))
           if x.is_file() and x.suffix not in ('.pt','.pyc') and '__pycache__' not in x.parts and '.pytest_cache' not in x.parts}
    package=seal_contract(dict(artifact_type='eventground_wam.preservation_pilot_evidence_package',protocol_digest=p['digest'],
        results_digest=results['digest'],verification_digest=proof['digest'],files_sha256=files,
        large_weights='Feifei only, bound by training completion checksums; metadata package is portable',
        total_metadata_bytes=sum((base/n).stat().st_size for n in files)))
    with (base/'package-index.json').open('x') as f:json.dump(package,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(package_digest=package['digest'],files=len(files),bytes=package['total_metadata_bytes'],prior_packages=preserved)))


if __name__=='__main__':main()
