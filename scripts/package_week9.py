#!/usr/bin/env python3
"""Add Week 9 execution to the immutable, accepted coupled evaluator package."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile
from eventground_wam.b0_training import read_sealed,atomic_json,sha256_file
from eventground_wam.auxiliary_evaluation import require
from eventground_wam.demonstration_contracts import seal_contract
from verify_week9_protocol import verify
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
verify(ROOT)
parent=ROOT/'docs/evaluation/week8-coupled-evaluator-v1/package-v4'
manifest=read_sealed(parent/'source-manifest.json')
require(manifest['digest']=='3afa15aabb9b186b26c0212564e4d531c6920d343c4489d7e5350b80e5643304','wrong parent')
with tarfile.open(parent/'source.tar.gz') as archive:
    files={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
for name,digest in manifest['files'].items():require(hashlib.sha256(files[name]).hexdigest()==digest,'parent source changed')
additions=['scripts/evaluate_week9.py','scripts/analyze_week9.py','scripts/run_week9_queue.py',
 'scripts/package_week9.py','scripts/week9_feifei.sh','scripts/compare_m3_m1_dev.py','scripts/package_m3_results.py',
 'src/eventground_wam/week9_execution.py','src/eventground_wam/week9_recording.py',
 'src/eventground_wam/evaluation_comparisons.py','tests/test_week9_execution.py','docs/WEEK9_EVALUATION_RUNNER.md',
 'evaluation/splits/libero-clean-v1/clean-40.json','tests/test_clean_comparison.py']
for name in additions:
    content=(ROOT/name).read_bytes()
    require(name not in files or files[name]==content,f'cannot modify accepted parent: {name}')
    files[name]=content
source=seal_contract({'artifact_type':'eventground_wam.week9_execution_source',
 'parent_source_digest':manifest['digest'],'base_git_revision':manifest['base_git_revision'],
 'week9_protocol_digest':read_sealed(ROOT/'configs/evaluation/week9_factorial_v1.json')['digest'],
 'unchanged_parent_file_count':len(manifest['files']),
 'files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(files.items())}})
a.output.mkdir(parents=True,exist_ok=False);atomic_json(a.output/'source-manifest.json',source)
files['week9-source-manifest.json']=(json.dumps(source,sort_keys=True,indent=2)+'\n').encode()
with tarfile.open(a.output/'source.tar.gz','w:gz') as archive:
    for n,b in sorted(files.items()):
        entry=tarfile.TarInfo(n);entry.size=len(b);entry.mode=0o644;archive.addfile(entry,io.BytesIO(b))
receipt=seal_contract({'source_digest':source['digest'],'archive_sha256':sha256_file(a.output/'source.tar.gz'),
 'file_count':len(files),'parent_source_digest':manifest['digest']})
atomic_json(a.output/'package.json',receipt);print(json.dumps(receipt))
