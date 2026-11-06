"""Seal completed local analysis plus remote endpoint verification evidence."""
import json
import shutil
from pathlib import Path
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_future_onset import TASK

ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'docs/evaluation'/TASK
    p=sealed(base/'protocol.json');results=sealed(base/'analysis/results.json');remote=sealed(base/'remote-final-verification.json')
    if results['protocol_digest']!=p['digest'] or remote['protocol_digest']!=p['digest'] or not remote['all_passed']:raise ValueError('Incomplete final evidence')
    # Copy analysis code separately, without changing the frozen execution snapshot.
    code=base/'analysis-code';code.mkdir(exist_ok=False)
    for name in ('analyze_week10_future_onset.py','package_week10_future_onset.py','verify_week10_future_onset_completion.py','plot_week10_future_onset.py'):
        shutil.copy2(ROOT/'scripts'/name,code/name)
    shutil.copy2(ROOT/'tests/test_future_onset_analysis.py',code/'test_future_onset_analysis.py')
    complete=seal_contract(dict(artifact_type='eventground_wam.future_onset_package_completion',task_definition=TASK,
        protocol_digest=p['digest'],results_digest=results['digest'],remote_verification_digest=remote['digest'],
        status='complete_promoted_comparison' if results['promoted'] else 'complete_dev_no_go',
        independent_qa_passed=False,original_experiments_preserved=True,
        large_artifacts='Completed endpoints/recovery checkpoints remain on Feifei, verified by checksum and endpoint loading; original recordings remain at their existing evidence paths.',
        limitations=results['limitations']))
    with (base/'completion.json').open('x') as f:json.dump(complete,f,indent=2,sort_keys=True);f.write('\n')
    files={str(x.relative_to(base)):sha(x) for x in base.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.name!='package-index.json'}
    package=seal_contract(dict(artifact_type='eventground_wam.future_onset_evidence_package',completion_digest=complete['digest'],
        files_sha256=files,total_bytes=sum((base/name).stat().st_size for name in files),large_remote_evidence=complete['large_artifacts']))
    with (base/'package-index.json').open('x') as f:json.dump(package,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=complete['status'],files=len(files),package_digest=package['digest'])))
if __name__=='__main__':main()
