"""Final CPU-only integrity receipt for the fixed endpoints and preserved studies."""
import argparse
import json
from pathlib import Path
import torch
from eventground_wam.demonstration_contracts import seal_contract,contract_digest
from eventground_wam.eventground_future_runtime import verify,validate_training_completion
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_runtime import verify_eventground_parent
from eventground_wam.pixel_predictor import tensor_state_digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);a=parser.parse_args();base=a.base
    p,index,manifests,recipe=verify(base);execution=sealed(base/'execution-completion.json')
    if execution['status'] not in ('complete','complete_dev_no_go'):raise ValueError('Queue not complete')
    deployment=sealed(base/'deployment.json')
    for name,h in deployment['files_sha256'].items():
        if sha(base/name)!=h:raise ValueError('Deployment mutation '+name)
    names=sorted(recipe['exact_allowlist']['head']+recipe['exact_allowlist']['language_lora']);runs={}
    for directory in sorted((base/'runs').iterdir()):
        stage=directory.name;c=validate_training_completion(base,stage)
        record=dict(completion_digest=c['digest'],updates=c['global_step'],presentations=c['prompts_seen'],checkpoints_verified=len(c['checkpoints']))
        if 'endpoint_file' in c:
            artifact=torch.load(directory/c['endpoint_file'],map_location='cpu',weights_only=True)
            h=artifact['header'];state=artifact['parameters']
            if h!=c['inference_header'] or h['digest']!=contract_digest(h) or sorted(state)!=names or tensor_state_digest(state)!=h['parameter_sha256'] or not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Invalid exported tensor state')
            record.update(endpoint_sha256=c['endpoint_sha256'],parameter_sha256=h['parameter_sha256'],finite_tensors=True)
            del artifact,state
        runs[stage]=record
    verify_eventground_parent(sealed(base/'source/configs/training/week10_eventground_experiment_v3.json'))
    preserved={}
    for folder,relative in [('week10-weak-rlvr-v1',False),('week10-report-coverage-extension-v1',True)]:
        old=sealed(base.parent/folder/'package-index.json');prefix=base.parent if relative else base.parent/folder
        for name,h in old['files_sha256'].items():
            if sha(prefix/name)!=h:raise ValueError('Previous completed result changed '+name)
        preserved[folder]=dict(index_digest=old['digest'],files_verified=len(old['files_sha256']))
    report=seal_contract(dict(artifact_type='eventground_wam.future_onset_final_integrity',protocol_digest=p['digest'],
        execution_completion_digest=execution['digest'],all_passed=True,verified_deployment_files=len(deployment['files_sha256']),
        runs=runs,stored_M4_parent_unchanged=True,prior_packages_unchanged=preserved,all_targets_reconstructed=True,
        analysis_not_yet_packaged=True))
    with (base/'remote-final-verification.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(all_passed=True,runs=list(runs),preserved=preserved)))
if __name__=='__main__':main()
