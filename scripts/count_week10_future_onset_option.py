"""Counts-only feasibility of a proposed task revision; never admit or persist targets."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_manifest import CLASSES, _sample
from eventground_wam.eventground_report_extension import coverage_gate
from eventground_wam.eventground_weak_supervision import sealed, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'docs/evaluation/week10-coverage-resolution-v1/analyze.py'
    old=sealed(p.with_name('analysis.json'))
    assert sha(p)==old['analysis_script_sha256']
    spec=importlib.util.spec_from_file_location('prior_counts_only_analysis',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result={}
    for name,base in [('original_report',ROOT/'docs/evaluation/week10-weak-rlvr-v1/event-report'),
                      ('coverage_probe',ROOT/'docs/evaluation/week10-report-coverage-extension-startup-repair-v1')]:
        metadata=sealed(base/'source-metadata.json');identities={r['source_id']:r for r in metadata['rows']}
        paths=sorted((base/'labels').glob('*/event-candidates.json'))
        candidates=[sealed(p) for p in paths]
        assert len(candidates)==len(identities) and {c['source_id'] for c in candidates}==set(identities)
        # First reproduce the actual eligible counts to audit the counterfactual implementation.
        current=module.make_rows(candidates,identities,ongoing='all',ambiguity='retain_all',requested=False)
        assert dict(Counter(r['target']['next_event'] for r in current))==sealed(base/'manifest.json')['eligible_by_class']
        proposed=module.make_rows(candidates,identities,ongoing='allow',ambiguity='retain_all',requested=False)
        pre=_sample(proposed,250);count=min(Counter(r['target']['next_event'] for r in pre).get(k,0) for k in CLASSES)
        counts=Counter();selected=[]
        for row in pre:
            kind=row['target']['next_event']
            if counts[kind]<count:selected.append(row);counts[kind]+=1
        result[name]=dict(source_count=len(candidates),current_eligible=module.summarize(current),
            proposed_eligible=module.summarize(proposed),proposed_balanced_capped_counts=module.summarize(selected),
            proposed_coverage_floor=coverage_gate(selected),source_metadata_digest=metadata['digest'],
            candidate_file_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
            admission=False,alternate_rows_persisted=False,model_predictions=0)
    value=seal_contract(dict(artifact_type='eventground_wam.future_onset_counts_only_feasibility',
        status='PROPOSED_TASK_NOT_ACTIVATED',analysis_code_sha256=sha(Path(__file__)),
        previous_counts_code_sha256=sha(p),panels=result,physical_labels_or_uncertainty_changed=False,
        original_report_or_probe_reclassified=False,training_authorized=False,
        claim='Label-informed exploratory task-revision feasibility, not blind confirmation. Future evaluation must disclose this prior label inspection.',
        caution='Counts do not establish model performance or independent label quality. No sampler/target definition is activated by this analysis.'))
    out=ROOT/'docs/evaluation/week10-report-coverage-extension-v1/future-onset-counts-only.json'
    with out.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k!='candidate_file_sha256'} for name,r in result.items()},indent=2))


if __name__=='__main__':main()
