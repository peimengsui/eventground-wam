"""Prepare diagnostic examples and render existing dev observations, CPU only."""
import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def prepare(base):
    root = base.parent; future = root/'week10-future-onset-v1'; pilot=root/'week10-preservation-pilot-v1'
    initial = sealed(base/'cases.json'); cases = initial['cases'].copy()
    manifest = sealed(future/'dev-manifest.json'); rows={r['prompt_id']:r for r in manifest['rows']}
    index = sealed(future/'source-index.json'); candidates={r['candidate']['source_id']:r['candidate'] for r in index['panels']['dev']}
    preds = {}
    for arm in ('sft','primary','calibration','hybrid','control'):
        path=(pilot if arm in ('hybrid','control') else future)/'eval/dev'/arm/'predictions.jsonl'
        preds[arm]={r['prompt_id']:r for r in map(json.loads,path.read_text().splitlines())}
    control=preds['control']
    for truth,pred in (('contact','gripper_close'),('gripper_close','contact')):
        selected=sorted((pid for pid,r in control.items() if r['target']['next_event']==truth and r['prediction']['next_event']==pred),
                        key=lambda pid:(-max(control[pid]['event_probabilities']),pid))
        pid=selected[0];row=rows[pid];a=row['anchor'];c=candidates[row['source_id']]
        events=[e for e in c['events'] if e['start']<a+33 and e['end']>a]
        cases.append(dict(case_id=f'case-{len(cases)+1:02d}',categories=[truth+' predicted as '+pred],row=row,
            predictions={arm:pr[pid] for arm,pr in preds.items()},automatic_events_in_window=events,
            ongoing_events_at_anchor=[e for e in events if e['start']<=a<e['end']],
            label_claim='Automatic predicate agreement only; no independent validation added.'))
    ongoing={}
    for arm,pr in preds.items():
        counts={'matched_positive':0,'large_timing_errors':0,'large_errors_with_same_type_ongoing':0,
                'none_false_positives':0,'none_fp_with_predicted_type_ongoing':0}
        for pid,r in pr.items():
            row=rows[pid];kind=r['prediction']['next_event'];a=row['anchor']
            active=any(e['start']<=a<e['end'] and e['event_type']==kind for e in candidates[row['source_id']]['events'])
            if r['target']['next_event']!='none' and r['correct_event']:
                counts['matched_positive']+=1
                if r['start_error']>8:
                    counts['large_timing_errors']+=1
                    counts['large_errors_with_same_type_ongoing']+=int(active)
            if r['target']['next_event']=='none' and kind!='none':
                counts['none_false_positives']+=1
                counts['none_fp_with_predicted_type_ongoing']+=int(active)
        ongoing[arm]=counts
    save(base/'examples.json',seal_contract(dict(artifact_type='eventground_wam.week10_closeout_examples',
        parent_cases_digest=initial['digest'],cases=cases,
        supplemental_selection='Add top-confidence contact->close and close->contact cases after matrix identifies this as largest confusion pair. Descriptive selection, not prevalence estimation.',
        ongoing_context_counts=ongoing,
        interpretation='Ongoing events do not count as new onsets; correlations here do not prove model mechanism.')))
    print(json.dumps(dict(cases=len(cases),ongoing_context_counts=ongoing)))


def extract(base):
    import h5py
    from PIL import Image
    examples=sealed(base/'examples.json');handoff=sealed(base/'handoff.json')
    # Verify actual endpoint bytes without model inference or optimizer construction.
    endpoint=Path(handoff['remote_endpoint'])
    assert sha(endpoint)==handoff['endpoint_sha256']
    done=sealed(endpoint.parent/'completion.json');identity=sealed(endpoint.parent/'identity.json')
    assert done['digest']==handoff['completion_digest'] and identity['digest']==handoff['identity_digest']
    assert done['inference_header']['digest']==handoff['inference_header_digest']
    start=base.parent/'week10-future-onset-v1/runs/sft/endpoint.pt'
    assert sha(start)==handoff['parent_sft_endpoint_sha256']
    study=sealed(base.parent/'week10-sft-continuation-report-v1/source/configs/training/week10_eventground_experiment_v3.json')
    assert study['parent']==handoff['parent_m4']
    from eventground_wam.eventground_runtime import verify_eventground_parent
    verify_eventground_parent(study)
    media=base/'frames';media.mkdir(exist_ok=False)
    files={}; recordings={}; decoded=0
    for case in examples['cases']:
        row=case['row'];path=Path(row['recording_path'])
        if str(path) not in recordings:
            actual=sha(path)
            assert actual==row['recording_sha256']
            recordings[str(path)]=actual
        elif recordings[str(path)]!=row['recording_sha256']:
            raise ValueError('Inconsistent source hash')
        with h5py.File(path,'r') as f:
            for offset in range(33):
                for camera in ('front','wrist'):
                    raw=f['observation/'+camera][row['anchor']+offset]
                    assert raw.shape==(256,256,3) and str(raw.dtype)=='uint8'
                    output=media/f"{case['case_id']}-{camera}-{offset:02d}.png"
                    Image.fromarray(raw).save(output)
                    files[str(output.relative_to(base))]=sha(output);decoded+=1
    save(base/'media-verification.json',seal_contract(dict(artifact_type='eventground_wam.week10_closeout_media',
        verified_at=datetime.now(timezone.utc).isoformat(),examples_digest=examples['digest'],handoff_digest=handoff['digest'],
        selected_endpoint_verified=True,parent_sft_verified=True,parent_m4_verified=True,
        selected_endpoint_sha256=handoff['endpoint_sha256'],recording_files_sha256=recordings,
        frame_files_sha256=files,decoded_frames=decoded,model_inferences=0,training_updates=0,
        source_access='Only the selected existing event-dev recording windows; later frames displayed for retrospective interpretation, never supplied to a model.',
        image_conversion='Lossless RGB PNG; no rotation, cropping, resizing or pixel annotation. Timeline labels rendered separately in HTML.')))
    print(json.dumps(dict(cases=len(examples['cases']),frames=decoded,recordings=len(recordings),endpoint_verified=True,m4_verified=True)))


def report(base):
    d=sealed(base/'diagnostic.json'); e=sealed(base/'examples.json'); m=sealed(base/'media-verification.json')
    assert m['examples_digest']==e['digest'] and m['selected_endpoint_verified']
    for f,h in m['frame_files_sha256'].items():
        assert sha(base/f)==h
    data=[]
    for case in e['cases']:
        c=dict(case)
        c['images']={camera:[f"frames/{case['case_id']}-{camera}-{i:02d}.png" for i in range(33)] for camera in ('front','wrist')}
        data.append(c)
    # Load only the visible frames; embedding all 990 PNGs made the page 69 MB.
    # Keep this HTML beside its frames directory, or serve the bundle locally.
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week 10 diagnostic examples</title><style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;color:#23313d;background:#f3f6f8;margin:0}main{max-width:1120px;margin:auto;padding:28px}h1{font-size:29px;margin:0}h2{font-size:20px}.muted{color:#566571}.notice{padding:13px 16px;background:#fff4d9;border-left:4px solid #bd8e2d;margin:20px 0}select,button,input{font:inherit}select{width:100%;padding:9px}button{padding:8px 15px;border:1px solid #b7c3cb;border-radius:6px;background:white;cursor:pointer}.card{background:white;border:1px solid #d6dfe5;border-radius:10px;padding:20px;margin-top:18px}.views{display:flex;gap:18px;justify-content:center}.views figure{margin:0;max-width:45%}.views img{width:320px;max-width:100%;image-rendering:auto}.views figcaption{text-align:center;color:#566571}input[type=range]{width:100%}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}table{border-collapse:collapse;width:100%;margin-top:14px}td,th{border-bottom:1px solid #dbe3e9;text-align:left;padding:8px}th{font-size:14px}.track{position:relative;height:23px;background:repeating-linear-gradient(90deg,#eef2f5 0,#eef2f5 calc(100% / 33 - 1px),#d9e1e7 calc(100% / 33 - 1px),#d9e1e7 calc(100% / 33));min-width:180px}.bar{position:absolute;height:15px;top:4px;background:#247a72;border-radius:2px}.pred{background:#597bb0}.cursor{position:absolute;width:2px;top:0;bottom:0;background:#af3429;z-index:1}.tag{font-size:13px;background:#e8eff5;border-radius:4px;padding:3px 6px}.small{font-size:13px}details{margin-top:16px}code{overflow-wrap:anywhere}@media(max-width:650px){main{padding:14px}table{font-size:12px}td,th{padding:5px}.track{min-width:100px}}
</style><main><h1>Week 10 · event forecast diagnostics</h1><p class="muted">15 selected development examples · five fixed endpoints · no new training</p>
<div class="notice"><strong>Labels are automatic, not independent ground truth.</strong> “None” means no <em>new</em> event onset in the next 32 frames; contact or grasp may already be ongoing. These examples explain errors; they are not an unbiased sample or a required QA task.</div>
<label for="choose">Choose an example</label><select id="choose"></select><section class="card"><h2 id="title"></h2><p id="instruction"></p><p id="context"></p><div class="views"><figure><img id="front" alt="Front camera recording"><figcaption>Front camera</figcaption></figure><figure><img id="wrist" alt="Wrist camera recording"><figcaption>Wrist camera</figcaption></figure></div><p id="frameLabel" class="muted"></p><input id="frame" aria-label="Frame offset" type="range" min="0" max="32" value="0"><div class="controls"><button id="play">Play slowly</button><button id="anchor">Current observation</button><button id="target">Target onset</button><button id="predict">SFT continuation onset</button></div><p class="small muted">The model saw only offset 0. Later frames are shown retrospectively. Playback is slowed to 5 frames/second; recordings use 20 frames/second.</p><table><thead><tr><th>Endpoint / label</th><th>Event</th><th>Interval</th><th>Offsets 0 → 32 (end 33 exclusive)</th></tr></thead><tbody id="rows"></tbody></table><details><summary>Source and automatic-event context</summary><pre id="provenance" class="small" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></section><p class="small muted">Timing bands are half-open intervals. Ends at 33 may be clipped; see source metadata. Current selection is exploratory and does not change a checkpoint, label, threshold or Week 11 design.</p></main><script>
const cases=DATA;const choose=document.getElementById('choose'),slider=document.getElementById('frame');let timer=null;
const names={sft:'Starting SFT',primary:'RLVR primary',calibration:'RLVR calibration',hybrid:'RLVR + SFT',control:'SFT continuation'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.case_id+' — '+c.categories.join('; ');choose.append(o)});
function stop(){if(timer)clearInterval(timer);timer=null;document.getElementById('play').textContent='Play slowly'}
function selected(){return cases[Number(choose.value)]}
function showFrame(){let c=selected(),i=Number(slider.value);document.getElementById('front').src=c.images.front[i];document.getElementById('wrist').src=c.images.wrist[i];document.getElementById('frameLabel').textContent='Offset '+i+' · recording frame '+(c.row.anchor+i)+(i===0?' · only model input frame':' · future frame for retrospective review only');document.querySelectorAll('.cursor').forEach(x=>x.style.left=(i/33*100)+'%')}
function tableRow(label,p,gt=false){let interval=p.interval;return '<tr><td>'+esc(label)+'</td><td>'+esc(p.next_event)+'</td><td>'+(interval?'['+interval.join(', ')+')':'—')+'</td><td><div class="track">'+(interval?'<span class="bar '+(gt?'':'pred')+'" style="left:'+(100*interval[0]/33)+'%;width:'+(100*(interval[1]-interval[0])/33)+'%"></span>':'')+'<span class="cursor"></span></div></td></tr>'}
function showCase(){stop();let c=selected();slider.value=0;document.getElementById('title').textContent=c.case_id+' · '+c.row.task_key;document.getElementById('instruction').textContent=c.row.instruction;document.getElementById('context').textContent='Automatic events already ongoing at the anchor: '+(c.ongoing_events_at_anchor.map(x=>x.event_type+(x.object_id?' ('+x.object_id+')':'')).join(', ')||'none')+'.';document.getElementById('rows').innerHTML=tableRow('Automatic target',c.row.target,true)+Object.entries(names).map(([a,n])=>tableRow(n,c.predictions[a].prediction)).join('');document.getElementById('target').disabled=c.row.target.horizon===null;document.getElementById('predict').disabled=c.predictions.control.prediction.horizon===null;document.getElementById('provenance').textContent=JSON.stringify({prompt_id:c.row.prompt_id,source_id:c.row.source_id,recording_id:c.row.recording_id,anchor:c.row.anchor,weak_label_provenance:c.row.weak_label_provenance,end_censored:c.row.label_metadata.end_censored,automatic_events_in_window:c.automatic_events_in_window},null,2);showFrame()}
choose.onchange=showCase;slider.oninput=()=>{stop();showFrame()};document.getElementById('play').onclick=()=>{if(timer){stop();return}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%33;showFrame()},200)};document.getElementById('anchor').onclick=()=>{stop();slider.value=0;showFrame()};document.getElementById('target').onclick=()=>{stop();slider.value=selected().row.target.horizon;showFrame()};document.getElementById('predict').onclick=()=>{stop();slider.value=selected().predictions.control.prediction.horizon;showFrame()};showCase();
</script></html>'''
    payload=json.dumps(data,separators=(',',':')).replace('</','<\/')
    (base/'examples.html').write_text(template.replace('DATA',payload))
    print(json.dumps(dict(html_bytes=(base/'examples.html').stat().st_size,examples=len(data),verified_frames=len(m['frame_files_sha256']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','extract','report'));p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'extract':extract,'report':report}[a.mode](a.base)
