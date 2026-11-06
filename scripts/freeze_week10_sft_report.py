"""Seal a new report-evaluation child; immutable parents retain their old scopes."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_sft_report import EXPERIMENT
from eventground_wam.eventground_weak_supervision import sealed,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    pilot=ROOT/'docs/evaluation/week10-preservation-pilot-v1';initial=pilot.parent/'week10-future-onset-v1'
    base=pilot.parent/EXPERIMENT;base.mkdir(exist_ok=False);source=base/'source'
    shutil.copytree(pilot/'source',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    for name in ('src/eventground_wam/eventground_sft_report.py','scripts/evaluate_week10_sft_report.py',
                 'scripts/analyze_week10_sft_report.py','scripts/freeze_week10_sft_report.py','scripts/analyze_week10_future_onset.py'):
        shutil.copy2(ROOT/name,source/name)
    shutil.copy2(initial/'report-manifest.json',base/'report-manifest.json')
    control=sealed(pilot/'runs/control/completion.json');baseline=sealed(initial/'eval/report/sft/completion.json');old=sealed(pilot/'protocol.json')
    p=seal_contract(dict(artifact_type='eventground_wam.approved_sft_report_evaluation',experiment=EXPERIMENT,
        created_at=datetime.now(timezone.utc).isoformat(),user_approved=True,
        authorization='User: Sounds good, can you try the next fix? — accepts the immediately preceding recommendation to evaluate the SFT-only endpoint on the separate report panel and requests an experiment retrospective.',
        arms=['control'],panels=['report'],training_authorized=False,automatic_retries=0,attempts=1,
        pilot_package_digest=sealed(pilot/'package-index.json')['digest'],pilot_protocol_digest=old['digest'],
        selection_decision_digest=sealed(pilot/'analysis/results.json')['digest'],
        endpoint_completion_digest=control['digest'],endpoint_sha256=control['endpoint_sha256'],
        baseline_evaluation_digest=baseline['digest'],recipe_digest=old['recipe_digest'],
        source_files_sha256={str(x.relative_to(source)):sha(x) for x in source.rglob('*') if x.is_file()},
        artifact_sha256={'report-manifest.json':sha(base/'report-manifest.json')},
        inference_resources=dict(wall_seconds=900,output_bytes=1024**3,minimum_free_bytes=50*1024**3),
        bootstrap=dict(draws=10000,rng='PCG64',seed=10,quantiles=[.025,.975],undefined_fraction_limit=.01),
        analysis='Same primary typed macro IoU and source-paired bootstrap; none FP, conditional/common-matched timing, coverage, class/lead-bin metrics, calibration and latency. No new promotion threshold or report tuning.',
        initialization='Exact previously completed SFT-only control endpoint,32updates/500prompts. No optimizer or parameter updates.',
        limitations=['Fixed report sources are disjoint from event train/dev; labels/counts and starting-SFT report results were already inspected, so this is not untouched confirmation.',
                     'Automatic weak-label agreement, not independently validated event truth; limited prior human review does not measure missed-event recall.',
                     'Single training seed; source bootstrap does not measure training variability.',
                     'M4 historical transformed-alias exposure unresolved; no robot-rollout improvement claim.',
                     'Starting-SFT latency was measured earlier and is not a contemporaneous performance benchmark.'],
        remote_base='/home/peimengsui/data/eventground-wam/libero-plus/results/'+EXPERIMENT,cache=old['cache']))
    with (base/'protocol.json').open('x') as f:json.dump(p,f,indent=2,sort_keys=True);f.write('\n')
    deployment=seal_contract(dict(protocol_digest=p['digest'],files_sha256={str(x.relative_to(base)):sha(x) for x in base.rglob('*') if x.is_file()}))
    with (base/'deployment.json').open('x') as f:json.dump(deployment,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(protocol_digest=p['digest'],endpoint_sha256=control['endpoint_sha256'],deployment_files=len(deployment['files_sha256']))))


if __name__=='__main__':main()
