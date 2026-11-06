"""One bounded read of exact held-out actions, first state and XML per admitted source."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import shutil
import signal
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
from run_event_source_pilot import BoundedRange
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import validate_panel_contract
from eventground_wam.instrumented_supervision import numeric_digest


def save(p,d):
    with Path(p).open('x') as f:json.dump(seal_contract(d),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def work(rows,contract,base):
    start=time.monotonic();folder=base/'staged'/rows[0]['task_key'].replace('/','-');folder.mkdir()
    first=rows[0]
    url=f"https://huggingface.co/datasets/{first['repository']}/resolve/{first['revision']}/{first['hdf5_path']}"
    reader=BoundedRange(url,folder,contract['staging_resources']);results=[];failure=None
    try:
        reader.fetch(0,1)
        with h5py.File(reader,'r') as h:
            for row in rows:
                if shutil.disk_usage(base).free<contract['staging_resources']['minimum_free_bytes']:raise RuntimeError('Free-space ceiling')
                reader.phase=f"{contract['panel']}:{row['source_id']}:actions_initial_state_xml"
                d=h['data'][row['demo_id']]
                if d['actions'].shape!=(row['frames'],7) or d['states'].shape[0]!=row['frames']:raise ValueError('Wrong source length')
                actions=d['actions'][:];initial=d['states'][0];xml=d.attrs['model_file']
                if isinstance(xml,bytes):xml=xml.decode()
                if not isinstance(xml,str) or len(xml.encode())>2*1024**2:raise ValueError('Invalid XML')
                joints=[n.attrib['name'] for n in ET.fromstring(xml).find('worldbody').iter() if n.tag in ('joint','freejoint')]
                if len(joints)!=len(set(joints)) or not np.isfinite(initial).all() or not np.isfinite(actions).all():raise ValueError('Invalid source')
                out=folder/row['source_id'];out.mkdir()
                np.save(out/'initial-state.npy',initial,allow_pickle=False);np.savez(out/'actions.npz',actions=actions)
                (out/'source-model.xml').write_text(xml)
                result=dict(source_id=row['source_id'],task_key=row['task_key'],frames=row['frames'],demo_id=row['demo_id'],
                    source_joint_names=joints,initial_state_shape=list(initial.shape),action_digest=numeric_digest(actions),
                    input_sha256={p.name:sha(p) for p in out.iterdir()},state_rows_read=[0],original_video_reads=0)
                save(out/'identity.json',result);results.append(result)
                if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>contract['staging_resources']['per_task_output_bytes']:raise RuntimeError('Output ceiling')
    except Exception as e:
        failure=dict(error=repr(e),traceback=traceback.format_exc())
    result=dict(task_key=first['task_key'],status='complete' if failure is None and len(results)==len(rows) else 'failed_preserved',
        inputs=results,failure=failure,http_bytes=reader.downloaded,http_requests=reader.requests,elapsed_seconds=time.monotonic()-start)
    save(folder/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base
    c=sealed(base/'panel-contract.json');split=sealed(base/'partitions.json');meta=sealed(base/'source-metadata.json')
    rows=validate_panel_contract(c,split,meta)
    for name,h in c['bundle_sha256'].items():
        if sha(base/name)!=h:raise ValueError('Changed code/input '+name)
    if shutil.disk_usage(base).free<c['staging_resources']['minimum_free_bytes']:raise RuntimeError('Free-space ceiling')
    (base/'staged').mkdir();start=time.monotonic();results=[];failure=None
    def deadline(*_):raise TimeoutError('Staging deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.alarm(c['staging_resources']['global_seconds'])
    tasks=[[r for r in rows if r['task_key']==k] for k in sorted({r['task_key'] for r in rows})]
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            for f in concurrent.futures.as_completed([pool.submit(work,t,c,base) for t in tasks]):results.append(f.result())
    except Exception as e:failure=repr(e)
    finally:
        signal.alarm(0)
        status='complete' if failure is None and len(results)==40 and all(r['status']=='complete' for r in results) else 'failed_preserved'
        save(base/'staging-completion.json',dict(status=status,contract_digest=c['digest'],results=sorted(results,key=lambda r:r['task_key']),
            failure=failure,elapsed_seconds=time.monotonic()-start,payload_sources=sum(len(r['inputs']) for r in results),
            http_bytes=sum(r['http_bytes'] for r in results),automatic_retries=0))
    if status!='complete':raise RuntimeError('Staging failed; preserve evidence, no retries')

if __name__=='__main__':main()
