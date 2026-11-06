"""Freeze an 80-source coverage probe before any new source payload access."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_report_extension import GATE, PANEL, validate_panel_contract
from eventground_wam.eventground_weak_supervision import sealed, sha
from eventground_wam.instrumented_supervision import physical_source_id

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/evaluation/week10-report-coverage-extension-v1'
OLD = ROOT/'docs/evaluation/week10-weak-rlvr-v1/event-report'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(seal_contract(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    BASE.mkdir(exist_ok=False)
    prep = ROOT/'docs/evaluation/week10-production-preparation-recovery-v1'
    catalog = sealed(prep/'contract.json')
    original_proposal = sealed(ROOT/'docs/evaluation/week10-production-plan-v2/source-split-v3-proposal.json')
    split = sealed(OLD/'partitions.json')
    excluded = set(original_proposal['excluded_from_new_event_panels_previously_accessed_or_diagnostic_sources'])
    excluded.update(r['source_id'] for r in split['rows'])
    inventory, metadata_hashes = [], {}
    for task in catalog['tasks']:
        p = prep/'run'/task['task_key'].replace('/','-')/'source-lengths.json'
        metadata_hashes[str(p.relative_to(ROOT))] = sha(p)
        lengths = sealed(p)['lengths']
        for demo, frames in lengths.items():
            sid = physical_source_id(catalog['repository'],catalog['revision'],task['hdf5_path'],demo)
            inventory.append(dict(source_id=sid,repository=catalog['repository'],revision=catalog['revision'],
                                  hdf5_path=task['hdf5_path'],demo_id=demo,frames=frames,task_key=task['task_key']))
    inventory_ids = {r['source_id'] for r in inventory}
    # Check actual saved payload identities as well as the old exclusion ledger.
    evidence_hashes, accessed = {}, set()
    for folder in sorted((ROOT/'docs/evaluation').glob('week10-*')):
        if folder == BASE: continue
        for pattern in ('*identity.json', 'event-candidates.json'):
            for p in folder.rglob(pattern):
                value = json.loads(p.read_text())
                sid = value.get('source_id')
                if sid in inventory_ids:
                    accessed.add(sid)
                    evidence_hashes[str(p.relative_to(ROOT))] = sha(p)
    extra = accessed-excluded
    excluded.update(accessed)
    available = [r for r in inventory if r['source_id'] not in excluded]
    rows = []
    for task in sorted({r['task_key'] for r in inventory}):
        pool = [r for r in available if r['task_key'] == task]
        rank = lambda r: hashlib.sha256(('week10-report-coverage-extension-v1:10:'+r['source_id']).encode()).hexdigest()
        assert len(pool) >= 2
        rows.extend({**r, 'partition':PANEL, 'queue_role':'heldout_coverage_probe'} for r in sorted(pool,key=rank)[:2])
    assert len(rows) == 80
    audit = dict(artifact_type='eventground_wam.report_extension_source_audit',metadata_only=True,
        inventory_sources=len(inventory),unused_sources=len(available),excluded_sources=len(excluded),
        additional_accessed_sources=sorted(extra),excluded_source_ids=sorted(excluded),
        unused_counts_by_task=dict(Counter(r['task_key'] for r in available)),
        selected_sources=len(rows),selected_frames=sum(r['frames'] for r in rows),
        selected_control_steps=sum(r['frames']+10 for r in rows),
        original_partition_digest=split['digest'],metadata_sha256=metadata_hashes,
        saved_payload_identity_sha256=evidence_hashes,new_payload_reads=0,
        prior_exclusion_digest=original_proposal['digest'],
        claim='Disjoint canonical source IDs for new event data only; M4 historical alias exposure remains unresolved')
    save(BASE/'source-availability-audit.json',audit)
    audit=sealed(BASE/'source-availability-audit.json')
    save(BASE/'source-metadata.json',dict(artifact_type='eventground_wam.report_extension_source_metadata',
        panel=PANEL,partition_digest=split['digest'],rows=rows,excluded_source_ids=sorted(excluded),
        availability_audit_digest=audit['digest'],source_order='two per task; SHA256 tag+seed10+sourceID; sorted tasks',
        payload_access_authorized=True,labels_admitted=False,training_authorized=False))
    for name in ['partitions.json','parent-recording-contract.json','visual-scene-specs.json','training-decision.json','future-main-control-plan.json']:
        shutil.copy2(OLD/name,BASE/name)
    shutil.copytree(OLD/'src',BASE/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'src/eventground_wam/eventground_report_extension.py',BASE/'src/eventground_wam/eventground_report_extension.py')
    (BASE/'scripts').mkdir()
    transforms={}
    for name in ['stage_week10_event_panel.py','prepare_week10_event_capture.py','record_week10_event_panel.py',
                 'verify_week10_event_panel.py','extract_week10_event_panel.py','extract_week10_production_candidates.py','run_event_source_pilot.py']:
        original=(OLD/'scripts'/name).read_text()
        child=original.replace('from eventground_wam.eventground_weak_evaluation import validate_panel_contract',
                               'from eventground_wam.eventground_report_extension import validate_panel_contract')
        child=child.replace('user requested steps 1-4; weak held-out panel execution',
                            'User requested continuation to resolve report grasp coverage; fixed 80-source child probe')
        (BASE/'scripts'/name).write_text(child)
        transforms[name]=dict(parent_sha256=sha(OLD/'scripts'/name),child_sha256=sha(BASE/'scripts'/name),
                             changes='Only separate panel-validator/manifest import and authorization description; physical capture/verification core unchanged')
    (BASE/'tests').mkdir()
    shutil.copy2(ROOT/'tests/test_eventground_report_extension.py',BASE/'tests/test_eventground_report_extension.py')
    shutil.copy2(Path(__file__),BASE/'scripts'/Path(__file__).name)
    meta=sealed(BASE/'source-metadata.json');old=sealed(OLD/'panel-contract.json')
    save(BASE/'code-lineage.json',dict(files=transforms,old_protocol_and_package_unchanged=True,
         physical_labeler_sha256=sha(BASE/'src/eventground_wam/instrumented_labeling.py')))
    files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()}
    c=dict(artifact_type='eventground_wam.report_coverage_probe',version=1,panel=PANEL,
        created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User: Ok, can you continue to resolve the blocker? Implements the proposed bounded same-definition held-out coverage test; no training or model inference.',
        parent_protocol_digest=old['protocol_digest'],partition_digest=split['digest'],source_metadata_digest=meta['digest'],
        source_ids=[r['source_id'] for r in rows],availability_audit_digest=audit['digest'],
        training_decision_digest=old['training_decision_digest'],future_main_control_plan_digest=old['future_main_control_plan_digest'],
        independent_qa_passed=False,automatic_retries=0,training_authorized=False,report_model_inference_authorized=False,
        coverage_gate=GATE,sampling='Dense eligible anchors; unchanged capped _sample(250), then trim to common balanced count; no model outputs used',
        coverage_floor_reason='Minimum exploratory diversity: at least20/class and10 independent grasp events from10 sources and5 tasks. This is not statistical power or independent QA.',
        scope='Fixed80source probe,2/task, no replacement/extension after observing counts; if failure, preserve evidence and return a scientific/resource decision',
        original_report='Preserved, primary macro remains unestimable; this is a separately identified supplemental panel',
        staging_resources=old['staging_resources'],
        capture_resources=dict(wall_seconds=1500,output_bytes=26*1024**3,minimum_free_bytes=150*1024**3,attempts_per_source=1),
        extraction_resources=dict(wall_seconds=600,output_bytes=1024**3),
        child_total_output_bytes=28*1024**3,child_total_active_seconds=2700,
        inherited_training_budget_unchanged=True,bundle_sha256=files)
    save(BASE/'panel-contract.json',c)
    validate_panel_contract(sealed(BASE/'panel-contract.json'),split,meta)
    print(json.dumps({k:audit[k] for k in ['inventory_sources','unused_sources','excluded_sources','additional_accessed_sources',
                                         'selected_sources','selected_frames','selected_control_steps']}))
    print('contract',sealed(BASE/'panel-contract.json')['digest'])


if __name__=='__main__':main()
