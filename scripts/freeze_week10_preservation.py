"""Freeze an independently approved pilot without changing any parent artifacts."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_manifest import _sample
from eventground_wam.eventground_preservation_runtime import EXPERIMENT
from eventground_wam.eventground_weak_supervision import sealed, sha

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'docs/evaluation/week10-future-onset-v1'
BASE = ROOT/'docs/evaluation'/EXPERIMENT


def main():
    BASE.mkdir(exist_ok=False); source = BASE/'source'
    shutil.copytree(PARENT/'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    new = ['src/eventground_wam/eventground_preservation.py', 'src/eventground_wam/eventground_preservation_runtime.py',
           'tests/test_eventground_preservation.py', 'tests/test_eventground_training.py',
           *['scripts/'+name for name in ('freeze_week10_preservation.py', 'diagnose_week10_preservation.py',
             'train_week10_preservation.py', 'evaluate_week10_preservation.py', 'run_week10_preservation.py', 'analyze_week10_preservation.py')]]
    for name in new:
        shutil.copy2(ROOT/name, source/name)
    for name in ('train-manifest.json', 'dev-manifest.json', 'report-manifest.json', 'source-index.json'):
        shutil.copy2(PARENT/name, BASE/name)
    def save(name, value):
        with (BASE/name).open('x') as f:
            json.dump(seal_contract(value), f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    (BASE/'schedules').mkdir()
    stages = {}
    for stage in ('hybrid-qualification', 'control-qualification', 'hybrid', 'control'):
        qual = stage.endswith('qualification')
        s = sealed(PARENT/'schedules'/('rlvr-qualification.json' if qual else 'primary.json'))
        s.pop('digest'); s.update(experiment=EXPERIMENT, stage=stage)
        save('schedules/'+stage+'.json', s)
        stages[stage] = dict(wall_seconds=900 if qual else 1800, output_bytes=(2 if qual else 4)*1024**3,
                             minimum_free_bytes=50*1024**3, maximum_cached_observation_bytes=768*1024**2,
                             checkpoint_steps=[1] if qual else [16,32])
    train = sealed(BASE/'train-manifest.json'); ids = {pid for batch in sealed(BASE/'schedules/hybrid.json')['batches'] for pid in batch}
    selected = _sample([r for r in train['rows'] if r['prompt_id'] in ids], 8)
    selected.sort(key=lambda r: hashlib.sha256(('17:preservation-diagnostic:'+r['prompt_id']).encode()).hexdigest())
    if len(selected) != 32:
        raise ValueError('Expected 32 diagnostic prompts')
    save('diagnostic-schedule.json', dict(experiment=EXPERIMENT, manifest_digest=train['digest'],
        selection='Existing nested500, _sample8/class, SHA256 17:preservation-diagnostic:prompt_id order',
        batches=[[r['prompt_id'] for r in selected[i:i+16]] for i in (0,16)], optimizer_updates=0, seed=17))
    old = sealed(PARENT/'protocol.json'); recipe = sealed(source/'configs/training/week10_eventground_numerical_v2.json')
    p = dict(artifact_type='eventground_wam.approved_preservation_pilot', version=1, experiment=EXPERIMENT,
        task_definition=old['task_definition'], user_approved=True,
        authorization='User: Ok, can you try — one supervised-preservation RLVR pilot and matched SFT-only control from completed SFT, same gates.',
        created_at=datetime.now(timezone.utc).isoformat(), parent_package_digest=sealed(PARENT/'package-index.json')['digest'],
        parent_protocol_digest=old['digest'], sft_completion_digest=sealed(PARENT/'runs/sft/completion.json')['digest'],
        baseline_evaluation_digest=sealed(PARENT/'eval/dev/sft/completion.json')['digest'],
        recipe_digest=recipe['digest'], code_sha256={str(x.relative_to(source)):sha(x) for folder in ('src','scripts','tests') for x in (source/folder).rglob('*.py')},
        source_files_sha256={str(x.relative_to(source)):sha(x) for x in source.rglob('*') if x.is_file()},
        artifact_sha256={str(x.relative_to(BASE)):sha(x) for x in BASE.rglob('*.json') if 'source' not in x.relative_to(BASE).parts},
        training_stages=stages, diagnostic_resources=dict(wall_seconds=600, output_bytes=1024**3, minimum_free_bytes=50*1024**3),
        inference_resources=dict(wall_seconds=900, output_bytes=1024**3, minimum_free_bytes=50*1024**3), queue_wall_seconds=9000,
        initialization='Every diagnostic/qualification/pilot/control starts from exact completed revised SFT with empty AdamW. No qualification or pilot weights reused.',
        optimizer_rates=recipe['optimization']['rlvr_learning_rates'],
        control='Structured SFT only with identical RLVR learning rates, data/order,32updates,initialization and clipping; wall time and sampled candidates need not match.',
        objective='GRPO + lambda * structured SFT; both terms normalized by actual logical prompt count. Reference remains the completed SFT.',
        weight_rule=dict(formula='clip(0.25*median(two GRPO preclip gradient norms)/median(two SFT preclip gradient norms),0.01,1.0)',
                        training_only=True, optimizer_updates=0, batches=2, prompts_per_batch=16, attempts=1, dev_metrics_used=False),
        reward_weights=dict(primary=old['reward_weights']['primary']), promotion=old['promotion'],
        evaluation=dict(panel='unchanged396 dev; frozen prior SFT baseline reused', bootstrap=old['evaluation']['source_paired_bootstrap'],
                        comparisons=['hybrid-sft','control-sft','hybrid-control'], fixed_final_update_only=True),
        automatic_retries=0, attempts_per_stage=1, calibration_permitted=False, full_training_authorized=False,
        report_evaluation_authorized=False, recording_authorized=False, independent_qa_passed=False,
        limitations=['Adaptive exploratory experiment motivated by prior dev failures; repeated dev use, no untouched confirmation.',
                     'Automatic weak labels, not independently validated event truth; earlier20episode review did not measure missed-event recall.',
                     'Single training seed; source bootstrap captures evaluation sampling, not training variability.',
                     'M4 historical transformed alias exposure unresolved; no robot rollout improvement claim.',
                     'SFT control uses matched RLVR rates, unlike the old unlaunched higher-rate full SFT continuation plan.'],
        remote_base='/home/peimengsui/data/eventground-wam/libero-plus/results/'+EXPERIMENT, cache=old['cache'])
    save('protocol.json', p)
    (BASE/'README.md').write_text('# Supervised-preservation pilot\n\nApproved scope: one no-update training-only weight diagnostic, disposable recovery qualifications, one500prompt hybrid and one matched SFT-only control, same396prompt dev gates. No full training, report evaluation, new data or retries. All prior experiments remain immutable.\n\nThe preservation coefficient is frozen by a prospective gradient-ratio rule before model evaluation. The completed diagnostic receipt binds every downstream identity.\n')
    save('deployment.json', dict(protocol_digest=sealed(BASE/'protocol.json')['digest'],
        files_sha256={str(x.relative_to(BASE)):sha(x) for x in BASE.rglob('*') if x.is_file()}))
    print(json.dumps(dict(protocol_digest=sealed(BASE/'protocol.json')['digest'], diagnostic_prompts=len(selected), stages=list(stages))))


if __name__ == '__main__':
    main()
