"""Seal/verify the new closeout while proving prior evidence remains unchanged."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_week10_closeout import verify_packages, save
from render_week10_closeout import sha
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed


def validate(base):
    prior=verify_packages(base.parent)
    before=sealed(base/'prior-verification-before.json')['packages']
    assert before==prior
    d=sealed(base/'diagnostic.json');h=sealed(base/'handoff.json');e=sealed(base/'examples.json');m=sealed(base/'media-verification.json')
    assert h['diagnostic_digest']==d['digest'] and m['handoff_digest']==h['digest'] and m['examples_digest']==e['digest']
    assert m['selected_endpoint_verified'] and m['parent_sft_verified'] and m['parent_m4_verified']
    assert len(e['cases'])==15 and len(m['frame_files_sha256'])==990 and len(m['recording_files_sha256'])==14
    for f,expected in m['frame_files_sha256'].items():
        assert sha(base/f)==expected, f
    rows={r['prompt_id']:r for r in sealed(base.parent/'week10-future-onset-v1/dev-manifest.json')['rows']}
    for case in e['cases']:
        assert case['row']==rows[case['row']['prompt_id']]
        for arm,r in case['predictions'].items():
            assert r['prompt_id']==case['row']['prompt_id'] and r['target']==case['row']['target']
    for f,expected in d['input_files_sha256'].items():
        # Stored paths begin docs/evaluation; relocate to either host's result root.
        relative=Path(f).relative_to('docs/evaluation')
        assert sha(base.parent/relative)==expected, f
    ui=json.loads((base/'ui-verification.json').read_text())
    assert ui['rendered'] and ui['predicted_onset_offset_verified']==4 and ui['target_onset_offset_verified']==32
    tests=json.loads((base/'metric-tests.json').read_text())
    assert tests['exit_code']==0 and tests['passed']==9
    assert (base/'metric-tests.txt').read_text().split()[0]=='.........'
    # Independent aggregate cross-checks used by the prose report.
    c=d['arms']['control'];conf=c['metrics']['confusion']
    assert sum(sum(row.values()) for row in conf.values())==396
    assert sum(conf[k][k] for k in conf if k!='none')==185
    assert conf['contact']['gripper_close']+conf['gripper_close']['contact']==51
    assert conf['contact']['grasp']+conf['grasp']['contact']==16
    assert len(d['temporal_context']['control']['wrong_positive_predictions'])==85
    assert c['timing']['more_than_eight_frames']==25 and c['timing']['median_absolute_frames']==1
    assert c['horizon']['raw_positive_different_from_eight']==255 and c['horizon']['raw_positive_outside_supported_set']==66
    assert e['ongoing_context_counts']['control']['large_errors_with_same_type_ongoing']==15
    assert e['ongoing_context_counts']['control']['none_fp_with_predicted_type_ongoing']==2
    return prior


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('create','verify'));p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();base=a.base;prior=validate(base)
    if a.mode=='create':
        done=seal_contract(dict(artifact_type='eventground_wam.week10_closeout_completion',status='complete',
            finished_at=datetime.now(timezone.utc).isoformat(),diagnostic_digest=sealed(base/'diagnostic.json')['digest'],
            handoff_digest=sealed(base/'handoff.json')['digest'],examples_digest=sealed(base/'examples.json')['digest'],
            remote_checkpoint_media_verification_digest=sealed(base/'media-verification.json')['digest'],
            prior_packages_verified=prior,prior_files_verified=sum(x['files'] for x in prior.values()),
            dev_prompts=396,train_targets_audited=1609,stored_rewards_recomputed=6000,
            model_calls=0,optimizer_updates=0,new_recordings=0,label_changes=0,
            independent_qa_passed=False,week11_started=False,
            study_disposition='Approved exploratory weak-label Week 10 study closed; original independent-QA requirement not claimed satisfied.',
            documentation='Repository documentation snapshots are preserved in repository-docs; navigable bundle summaries are CLOSEOUT.md and DIAGNOSTIC.md.'))
        save(base/'completion.json',done)
        files={str(f.relative_to(base)):sha(f) for f in sorted(base.rglob('*')) if f.is_file() and f.name!='package-index.json' and f.suffix!='.pyc' and '__pycache__' not in f.parts}
        package=seal_contract(dict(artifact_type='eventground_wam.week10_closeout_package',completion_digest=done['digest'],
            files_sha256=files,total_bytes=sum((base/f).stat().st_size for f in files)))
        save(base/'package-index.json',package)
    package=sealed(base/'package-index.json');done=sealed(base/'completion.json')
    assert package['completion_digest']==done['digest']
    for f,expected in package['files_sha256'].items():
        assert sha(base/f)==expected, f
    print(json.dumps(dict(status='verified',package_digest=package['digest'],files=len(package['files_sha256']),
        total_bytes=package['total_bytes'],prior_files_verified=sum(x['files'] for x in prior.values()))))


if __name__=='__main__':main()
