"""Run a single frozen coverage probe; fail closed and retain all stage evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_report_extension import validate_panel_contract
from eventground_wam.eventground_weak_supervision import sealed, sha


def save(path, value):
    with path.open('x') as f:
        json.dump(seal_contract(value),f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    b=a.base.resolve();c=sealed(b/'panel-contract.json')
    validate_panel_contract(c,sealed(b/'partitions.json'),sealed(b/'source-metadata.json'))
    for rel,h in c['bundle_sha256'].items():
        if sha(b/rel)!=h:raise ValueError('Bundle changed: '+rel)
    deployment=sealed(b/'deployment.json')
    for rel,h in deployment['files_sha256'].items():
        if sha(b/rel)!=h:raise ValueError('Deployment changed: '+rel)
    free=shutil.disk_usage(b).free
    if free < c['capture_resources']['minimum_free_bytes']+c['child_total_output_bytes']:
        raise RuntimeError('Insufficient room for full child allowance and retained free-space floor')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    if any(int(line.split(',')[1].strip())>512 for line in gpu.splitlines() if line.strip()):
        raise RuntimeError('Substantial GPU work already running')
    python=sealed(b/'parent-recording-contract.json')['runtime']['python_executable']
    env={**os.environ,'PYTHONPATH':str(b/'src')+':/home/peimengsui/code/vendor/LIBERO-plus',
         'CUDA_VISIBLE_DEVICES':'0','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','MUJOCO_EGL_DEVICE_ID':'0',
         'LIBERO_CONFIG_PATH':'/home/peimengsui/data/eventground-wam/libero-plus/config',
         'MAGICK_HOME':'/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/usr',
         'MAGICK_CONFIGURE_PATH':'/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/etc/ImageMagick-7',
         'LD_LIBRARY_PATH':'/home/peimengsui/data/eventground-wam/libero-plus/system/imagemagick/usr/lib/x86_64-linux-gnu',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    (b/'logs').mkdir()
    save(b/'execution-start.json',dict(contract_digest=c['digest'],deployment_digest=deployment['digest'],
        started_at=datetime.now(timezone.utc).isoformat(),free_bytes=free,gpu_process_snapshot=gpu,
        training_or_inference_authorized=False,automatic_retries=0))
    start=time.monotonic();phase=None;phases=[];failure=None
    commands=[('stage','stage_week10_event_panel.py',600),('bind','prepare_week10_event_capture.py',120),
              ('capture','record_week10_event_panel.py',1500),('verify','verify_week10_event_panel.py',1500),
              ('extract','extract_week10_event_panel.py',600)]
    try:
        for phase,script,seconds in commands:
            remaining=c['child_total_active_seconds']-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Cumulative child time allowance exhausted')
            args=[python,str(b/'scripts'/script),'--base',str(b)]
            if phase=='capture':args+=['--authorization',str(b/'capture-authorization.json')]
            print(json.dumps(dict(phase=phase,status='started',at=datetime.now(timezone.utc).isoformat())),flush=True)
            t=time.monotonic()
            with (b/'logs'/f'{phase}.log').open('x') as log:
                subprocess.run(args,cwd=b,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(seconds,remaining))
            phases.append(dict(phase=phase,elapsed_seconds=time.monotonic()-t))
            if sum(p.stat().st_size for p in b.rglob('*') if p.is_file())>c['child_total_output_bytes']:
                raise RuntimeError('Total child output limit exceeded')
            if shutil.disk_usage(b).free<c['capture_resources']['minimum_free_bytes']:
                raise RuntimeError('Free space floor violated')
            print(json.dumps(dict(phase=phase,status='complete',elapsed_seconds=phases[-1]['elapsed_seconds'])),flush=True)
    except Exception as e:
        failure=dict(error=repr(e),traceback=traceback.format_exc(),phase=phase)
    manifest=sealed(b/'manifest.json') if (b/'manifest.json').exists() else None
    result=dict(artifact_type='eventground_wam.report_extension_execution',
        status='failed_preserved' if failure else ('coverage_floor_passed' if manifest['coverage_gate']['passed'] else 'coverage_shortfall'),
        contract_digest=c['digest'],started=sealed(b/'execution-start.json')['digest'],
        completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,phases=phases,
        failure=failure,manifest_digest=manifest['digest'] if manifest else None,
        coverage_gate=manifest['coverage_gate'] if manifest else None,
        output_bytes=sum(p.stat().st_size for p in b.rglob('*') if p.is_file()),
        free_bytes=shutil.disk_usage(b).free,training_updates=0,model_predictions=0,automatic_retries=0)
    save(b/'execution-completion.json',result)
    print(json.dumps(result),flush=True)
    if failure:raise RuntimeError('Probe failed; evidence preserved, no retry')


if __name__=='__main__':main()
