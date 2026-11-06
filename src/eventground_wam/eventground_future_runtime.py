"""Fail-closed execution and endpoint identity for the approved future-onset child."""
from pathlib import Path
import json
import torch
from eventground_wam.demonstration_contracts import seal_contract,contract_digest
from eventground_wam.eventground_future_onset import TASK,build_manifest,schedule
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.pixel_predictor import tensor_state_digest


def verify(base, *, rebuild=True):
    base=Path(base);root=base/'source';p=sealed(base/'protocol.json')
    if (p['task_definition']!=TASK or not p['user_approved'] or not p['allow_ongoing_event_anchors']
        or p['discard_ambiguities'] or p['independent_qa_passed'] or p['automatic_retries']!=0):
        raise ValueError('Invalid approved task identity')
    for name,h in p['code_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Changed frozen code '+name)
    for name,h in p['artifact_sha256'].items():
        if sha(base/name)!=h:raise ValueError('Changed frozen artifact '+name)
    index=sealed(base/'source-index.json')
    for name,h in index['original_input_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Changed original evidence '+name)
    manifests={panel:sealed(base/f'{panel}-manifest.json') for panel in ('train','dev','report')}
    if rebuild:
        for panel,m in manifests.items():
            if build_manifest(index,panel)!=m:raise ValueError('Target reconstruction mismatch '+panel)
    for stage in p['training_stages']:
        if schedule(manifests['train'],stage)!=sealed(base/'schedules'/f'{stage}.json'):raise ValueError('Schedule changed')
    recipe=sealed(root/'configs/training/week10_eventground_numerical_v2.json')
    if recipe['digest']!=p['recipe_digest'] or recipe['compute_dtype']!='float32' or recipe['logical_batch_size']!=16 or recipe['microbatch_size']!=1:
        raise ValueError('Numerical recipe changed')
    return p,index,manifests,recipe


def validate_training_completion(base,stage):
    base=Path(base);p=sealed(base/'protocol.json');out=base/'runs'/stage
    c=sealed(out/'completion.json');s=sealed(base/'schedules'/f'{stage}.json')
    if (c['status']!='complete' or c['protocol_digest']!=p['digest'] or c['schedule_digest']!=s['digest']
        or c['global_step']!=s['updates'] or c['prompts_seen']!=s['presentations']
        or c['partial_update_in_progress'] or not all(c['gates'].values())
        or c['elapsed_seconds']>p['training_stages'][stage]['wall_seconds']
        or c['peak_reserved_bytes']>.9*c['gpu_total_bytes']):raise ValueError('Incomplete/invalid fixed endpoint '+stage)
    records=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    regular=[r for r in records if not r['recovery_replay']]
    if (len(regular)!=s['updates'] or [r['prompt_ids'] for r in regular]!=s['batches']
        or [r['global_step'] for r in regular]!=list(range(1,s['updates']+1))
        or c['executed_updates']!=s['updates']+(1 if stage.endswith('qualification') else 0)):
        raise ValueError('Logged exposure differs from exact schedule')
    for name,h in c['files_sha256'].items():
        if sha(out/name)!=h:raise ValueError('Changed completion file '+name)
    for ck in c['checkpoints']:
        if sha(out/ck['file'])!=ck['sha256']:raise ValueError('Changed recovery checkpoint')
    if stage.endswith('qualification'):
        if c['recovery']['expected_digest']!=c['recovery']['actual_digest'] or not c['recovery']['metrics_identical']:
            raise ValueError('Recovery qualification failed')
    elif sha(out/c['endpoint_file'])!=c['endpoint_sha256']:raise ValueError('Changed endpoint')
    return c


def load_endpoint(model,allowlist,base,stage):
    c=validate_training_completion(base,stage);out=Path(base)/'runs'/stage
    artifact=torch.load(out/c['endpoint_file'],map_location='cpu',weights_only=True)
    h=artifact['header'];state=artifact['parameters'];params=dict(model.named_parameters())
    names=sorted((*allowlist['head'],*allowlist['language_lora']))
    if (h!=c['inference_header'] or h['digest']!=contract_digest(h) or h['task_definition']!=TASK
        or sorted(state)!=names or h['exact_names']!=names or tensor_state_digest(state)!=h['parameter_sha256']):
        raise ValueError('Endpoint tensor identity mismatch')
    with torch.no_grad():
        for n,t in state.items():
            if t.shape!=params[n].shape or t.dtype!=params[n].dtype or not torch.isfinite(t).all():raise ValueError('Invalid endpoint tensor')
            params[n].copy_(t.to(params[n].device))
    return c


def require_stage(base,stage):
    base=Path(base)
    if stage=='sft':validate_training_completion(base,'sft-qualification')
    if stage not in ('sft-qualification','sft'):validate_training_completion(base,'sft')
    if stage in ('primary','calibration','main','control'):validate_training_completion(base,'rlvr-qualification')
    if stage=='calibration':
        d=sealed(base/'decisions/primary.json')
        if d['promoted'] or not d['calibration_permitted']:raise ValueError('Calibration not permitted')
    if stage in ('main','control'):
        d=sealed(base/'decisions/final.json')
        if d['protocol_digest']!=sealed(base/'protocol.json')['digest'] or not d['promoted']:raise ValueError('Development gate did not pass')
