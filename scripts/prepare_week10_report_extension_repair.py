"""Preserve a zero-source startup failure; restore the previously qualified environment."""
from pathlib import Path
import json
import math
import shutil

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed, sha

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'docs/evaluation/week10-report-coverage-extension-v1'
NEW=ROOT/'docs/evaluation/week10-report-coverage-extension-startup-repair-v1'


def save(path,value):
    with path.open('x') as f:json.dump(seal_contract({k:v for k,v in value.items() if k!='digest'}),f,indent=2,sort_keys=True);f.write('\n')


def main():
    fail=sealed(OLD/'run/completion.json');execution=sealed(OLD/'execution-completion.json')
    assert fail['control_steps']==fail['heldout_sources_recorded']==0 and fail['results']==[]
    assert not any(p.is_dir() for p in (OLD/'run').iterdir())
    assert 'MagickWand shared library not found' in fail['error']
    assert execution['failure']['phase']=='capture'
    NEW.mkdir(exist_ok=False)
    olddeployment=sealed(OLD/'deployment.json')
    for name,h in olddeployment['files_sha256'].items():
        assert sha(OLD/name)==h
        out=NEW/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,out)
    prior=math.ceil(execution['elapsed_seconds']);capture_prior=math.ceil(fail['elapsed_seconds'])
    recording=sealed(OLD/'recording-contract.json');old_digest=recording['digest']
    recording['resources']['wall_seconds']-=capture_prior
    recording['resources']['prior_seconds_charged']=fail['elapsed_seconds']
    recording['verification']['wall_seconds']-=capture_prior
    recording['startup_repair']=dict(original_recording_digest=old_digest,
        failed_capture_digest=fail['digest'],failed_execution_digest=execution['digest'],
        original_staged_inputs_reused=True,source_capture_retries=0,
        reason='Restore the ImageMagick environment already used by the successful qualified panel launcher')
    save(NEW/'recording-contract.json',recording);recording=sealed(NEW/'recording-contract.json')
    save(NEW/'verification-contract.json',dict(artifact_type='eventground_wam.heldout_independent_physical_verification',recording_digest=recording['digest'],**recording['verification']))
    for name in ['capture-authorization.json','staging-completion.json']:
        shutil.copy2(OLD/name,NEW/name)
    for name in ['run/completion.json','execution-completion.json','logs/capture.log']:
        out=NEW/'startup-failure'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,out)
    original=(OLD/'scripts/run_week10_report_extension.py').read_text()
    replacement=original.replace("remaining=c['child_total_active_seconds']-(time.monotonic()-start)",
        "remaining=c['child_total_active_seconds']-repair['prior_seconds_charged']-(time.monotonic()-start)")
    replacement=replacement.replace("    start=time.monotonic();phase=None;phases=[];failure=None",
        "    repair=sealed(b/'startup-repair.json')\n    assert repair['source_capture_retries']==0\n    assert sha(b/'scripts/run_week10_report_extension_repair.py')==repair['runner_sha256']\n    env.update(repair['restored_environment'])\n    start=time.monotonic();phase=None;phases=[];failure=None")
    replacement=replacement.replace("commands=[('stage','stage_week10_event_panel.py',600),('bind','prepare_week10_event_capture.py',120),\n              ('capture','record_week10_event_panel.py',1500)",
        "commands=[('capture','record_week10_event_panel.py',1499)")
    replacement=replacement.replace("output_bytes=sum(p.stat().st_size for p in b.rglob('*') if p.is_file()),",
        "prior_seconds_charged=repair['prior_seconds_charged'], cumulative_elapsed_seconds=repair['prior_seconds_charged']+time.monotonic()-start,\n        source_capture_retries=0, prior_output_bytes=repair['prior_output_bytes'],\n        output_bytes=sum(p.stat().st_size for p in b.rglob('*') if p.is_file()),")
    replacement=replacement.replace("if sum(p.stat().st_size for p in b.rglob('*') if p.is_file())>c['child_total_output_bytes']:",
        "if repair['prior_output_bytes']+sum(p.stat().st_size for p in b.rglob('*') if p.is_file())>c['child_total_output_bytes']:")
    assert replacement!=original and "('stage','stage_week10_event_panel.py'" not in replacement
    runner=NEW/'scripts/run_week10_report_extension_repair.py';runner.write_text(replacement)
    save(NEW/'startup-repair.json',dict(artifact_type='eventground_wam.zero_source_startup_repair',
        original_contract_digest=sealed(OLD/'panel-contract.json')['digest'],prior_seconds_charged=prior,
        prior_output_bytes=execution['output_bytes'],source_capture_retries=0,
        failed_execution_digest=execution['digest'],failed_capture_digest=fail['digest'],
        unchanged_source_ids=True,unchanged_coverage_and_label_rules=True,source_payloads_reread_from_network=False,
        runner_sha256=sha(runner),original_launcher_sha256=sha(ROOT/'docs/evaluation/week10-weak-rlvr-v1/run-report-panel.sh'),
        restored_environment=dict(MAGICK_HOME='/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/usr',
            MAGICK_CONFIGURE_PATH='/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/etc/ImageMagick-7',
            LD_LIBRARY_PATH='/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/usr/lib/x86_64-linux-gnu'),
        authorization='User requested blocker resolution; routine launcher correction before any source execution, within the original fixed source/time/output limits. No automatic retry mechanism.'))
    save(NEW/'deployment.json',dict(artifact_type='eventground_wam.report_extension_startup_repair_deployment',
        original_deployment_digest=olddeployment['digest'],files_sha256={str(p.relative_to(NEW)):sha(p) for p in NEW.rglob('*') if p.is_file()}))
    print('repaired_bundle',sealed(NEW/'deployment.json')['digest'],'prior_seconds',prior,'remaining_capture_seconds',recording['resources']['wall_seconds'])


if __name__=='__main__':main()
