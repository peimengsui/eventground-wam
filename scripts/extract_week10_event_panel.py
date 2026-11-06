"""Apply unchanged physical rules to a verified held-out panel, with explicit weak provenance."""
import argparse
import json
import signal
import time
from pathlib import Path
from extract_week10_production_candidates import load_evidence
from stage_week10_event_panel import save
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import validate_panel_contract,heldout_rows
from eventground_wam.instrumented_labeling import extract_events,prompt_previews


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base;start=time.monotonic()
    panel=sealed(base/'panel-contract.json');split=sealed(base/'partitions.json');meta=sealed(base/'source-metadata.json');validate_panel_contract(panel,split,meta)
    capture=sealed(base/'recording-contract.json');done=sealed(base/'run/completion.json');verification=sealed(base/'verification.json')
    if done['status']!='complete' or verification['completion_digest']!=done['digest'] or not verification['all_panel_visual_states_reconstructed']:raise ValueError('Unverified panel')
    # Retain every equality comparison. A disagreement larger than the inherited one-level qualification stops admission.
    pixels=verification['pixel_comparisons']
    if len(pixels)!=6*len(capture['targets']) or any(p['maximum_channel_error']>1 for p in pixels):raise ValueError('New camera discrepancy requires review')
    for row in verification['cases']:
        if any(row[k]!=0 for k in ('maximum_contact_error','maximum_object_gripper_pose_error','maximum_proprio_error')):raise ValueError('Physical reconstruction mismatch')
    def stop(*_):raise TimeoutError('Extraction deadline')
    signal.signal(signal.SIGALRM,stop);signal.alarm(600)
    out=base/'labels';out.mkdir();records=[]
    for item,result in zip(capture['targets'],done['results'],strict=True):
        folder=base/'run'/item['recording_id'];case=out/item['recording_id'];case.mkdir()
        for rel,h in result['evidence_sha256'].items():
            if sha(folder/rel)!=h:raise ValueError('Changed evidence')
        candidate=extract_events(load_evidence(folder,item,result,capture['digest']))
        # extract_events already returns a sealed unreviewed candidate; avoid nested resealing changes.
        (case/'event-candidates.json').write_text(json.dumps(candidate,indent=2,sort_keys=True)+'\n')
        (case/'prompt-previews.json').write_text(json.dumps(prompt_previews(candidate),indent=2,sort_keys=True)+'\n')
        records.append(dict(candidate=candidate,task_key=item['task_key'],instruction=item['instruction'],
            recording_path=str(folder/'recording.h5'),recording_sha256=result['evidence_sha256']['recording.h5']))
    manifest=heldout_rows(records,split,meta);(base/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    save(base/'label-completion.json',dict(status='complete_weak_labels_not_independent_QA',panel_contract_digest=panel['digest'],
        verification_digest=verification['digest'],manifest_digest=manifest['digest'],independent_qa_passed=False,
        frames=sum(r['candidate']['frames'] for r in records),events=sum(len(r['candidate']['events']) for r in records),
        ambiguities=sum(len(r['candidate']['ambiguities']) for r in records),elapsed_seconds=time.monotonic()-start,
        strict_pixel_equality_claimed=all(p['bitwise_equal'] for p in pixels),pixel_tolerance_changed=False,
        camera_qualification='preserve strict inequality evidence; max-one-channel-level inherited qualification',
        output_sha256={str(p.relative_to(base)):sha(p) for p in out.rglob('*') if p.is_file()}))
    signal.alarm(0);print(json.dumps({k:v for k,v in manifest.items() if k not in ('rows','source_candidate_digests')}))
if __name__=='__main__':main()
