"""Bind unchanged physical reconstruction rules to the completed 12 recordings."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from week11_replay_common import read, sha, write


def run(source,summary_path,inventory_path,output):
    summary=read(summary_path);inventory=read(inventory_path)
    output.mkdir(parents=True,exist_ok=False)
    pairs=[];pins={}
    for row in summary['rows']:
        artifacts={a:v['artifact'] for a,v in row['arms'].items()}
        pairs.append({'id':row['id'],'artifacts':artifacts})
        for path in artifacts.values():
            expected=inventory['files'][str(Path(path).relative_to(inventory['root']))]['sha256']
            assert sha(path)==expected;pins[path]=expected
    write(output/'contract.json',{'artifact_type':'eventground_wam.week11_first_chunk_physical_inputs',
        'summary_digest':summary['digest'],'selection':{'pairs':pairs},'input_sha256':pins})
    template=read(source/'annotation-template.json');template.pop('digest')
    template.update(replay_contract_digest=read(output/'contract.json')['digest'],
        scope='All twelve first-chunk recordings. Frozen physical rules; no integration, rendering, model calls or training.')
    for name,h in template['code_sha256'].items():
        assert sha(source/name)==h;dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
    write(output/'annotation-contract.json',template)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','summary','inventory','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.summary,a.inventory,a.output)
