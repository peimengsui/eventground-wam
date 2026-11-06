"""New evaluation-only authorization for the fixed SFT continuation endpoint."""
from pathlib import Path

from eventground_wam.eventground_preservation_runtime import verify as verify_pilot
from eventground_wam.eventground_weak_supervision import sealed, sha

EXPERIMENT = 'week10-sft-continuation-report-v1'


def pilot_base(base):
    return Path(base).parent/'week10-preservation-pilot-v1'


def initial_base(base):
    return Path(base).parent/'week10-future-onset-v1'


def verify(base):
    base=Path(base);p=sealed(base/'protocol.json');pilot=pilot_base(base)
    if (p['experiment']!=EXPERIMENT or not p['user_approved'] or p['training_authorized']
        or p['automatic_retries']!=0 or p['arms']!=['control'] or p['panels']!=['report']):
        raise ValueError('Evaluation-only scope changed')
    package=sealed(pilot/'package-index.json')
    if package['digest']!=p['pilot_package_digest']:
        raise ValueError('Pilot package identity changed')
    for name,digest in package['files_sha256'].items():
        if sha(pilot/name)!=digest:raise ValueError('Pilot evidence changed '+name)
    old,index,manifests,recipe=verify_pilot(pilot)
    if old['digest']!=p['pilot_protocol_digest'] or recipe['digest']!=p['recipe_digest']:
        raise ValueError('Pilot protocol or numerical recipe changed')
    for name,digest in p['source_files_sha256'].items():
        if sha(base/'source'/name)!=digest:raise ValueError('Frozen evaluation code changed '+name)
    for name,digest in p['artifact_sha256'].items():
        if sha(base/name)!=digest:raise ValueError('Frozen evaluation artifact changed '+name)
    report=sealed(base/'report-manifest.json')
    if report!=manifests['report'] or len(report['rows'])!=384:
        raise ValueError('Fixed report panel changed')
    decisions=sealed(pilot/'analysis/results.json')
    if decisions['digest']!=p['selection_decision_digest'] or not decisions['gates']['control']['promoted']:
        raise ValueError('Control dev admission changed')
    c=sealed(pilot/'runs/control/completion.json')
    if (c['digest']!=p['endpoint_completion_digest'] or c['endpoint_sha256']!=p['endpoint_sha256']
        or c['global_step']!=32 or c['prompts_seen']!=500 or c['status']!='complete' or not all(c['gates'].values())):
        raise ValueError('Fixed SFT continuation endpoint changed')
    baseline=sealed(initial_base(base)/'eval/report/sft/completion.json')
    if (baseline['digest']!=p['baseline_evaluation_digest'] or baseline['manifest_digest']!=report['digest']
        or baseline['status']!='complete'
        or sha(initial_base(base)/'eval/report/sft/predictions.jsonl')!=baseline['predictions_sha256']):
        raise ValueError('Existing SFT report baseline changed')
    sources={k:{r['source_id'] for r in m['rows']} for k,m in manifests.items()}
    if sources['report'] & (sources['train'] | sources['dev']):
        raise ValueError('Report canonical source overlap')
    return p,index,manifests,recipe
