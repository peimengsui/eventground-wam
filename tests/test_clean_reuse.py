from pathlib import Path

import pytest
from test_clean_comparison import case as case  # noqa: PLC0414 - shared pytest fixture

from eventground_wam.clean_acceptance import read_sealed
from eventground_wam.clean_comparison import (
    ComparisonIntegrityError,
    atomic_json,
    compare_pair_receipts,
    execute_pair,
    metadata_for_pair,
    paths_for_pair,
)
from eventground_wam.clean_reuse import (
    audit_cached_comparison,
    bind_reuse_identity,
    comparison_statistics,
    verify_cached_record,
)
from eventground_wam.demonstration_contracts import seal_contract


def test_new_addendum_identity_retains_valid_frozen_episode_metadata(case):
    root = Path(__file__).resolve().parents[1]
    config = read_sealed(root / 'configs/evaluation/week4_oft_clean_reuse_v1.json')
    case['identity'] = bind_reuse_identity(case['identity'], config)
    case['metadata'] = metadata_for_pair(case['pair'], case['protocol'],
                                         case['identity'], Path('/test/task.bddl'))
    assert case['identity']['comparison_config_digest'] == config['digest']
    assert case['identity']['acceptance_rule_digest'] == config['parent_rule_digest']
    assert execute_pair(**case)['success']


def cached(case):
    record = execute_pair(**case)
    return record, compare_pair_receipts(record, record)


def verify(case, pinned):
    return verify_cached_record(case['root'], case['ordinal'], case['pair'],
                                pinned, 'b0', case['identity'])


def test_reuse_preserves_original_source_identity_and_bytes(case):
    record, pinned = cached(case)
    before = {p: p.read_bytes() for p in case['root'].rglob('*') if p.is_file()}
    assert verify(case, pinned) == record
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('changed', ['artifact', 'journal', 'receipt', 'panel'])
def test_reuse_rejects_changed_original_evidence(case, changed):
    record, pinned = cached(case)
    if changed == 'artifact':
        with Path(record['artifact']).open('ab') as stream:
            stream.write(b'changed')
    elif changed == 'journal':
        Path(record['attempt_evidence'][0]['journal']['path']).write_text('{}')
    elif changed == 'receipt':
        record['success'] = False
        record.pop('digest')
        atomic_json(paths_for_pair(case['root'], case['ordinal'], case['pair'])[2],
                    seal_contract(record))
    else:
        case['pair'] = dict(case['pair'], instruction='different instruction')
    with pytest.raises(ComparisonIntegrityError):
        verify(case, pinned)


def test_original_comparison_requires_every_pair_and_its_original_report(case, tmp_path):
    case['ordinal'] = 0
    roots = {p: tmp_path / p for p in ('b0', 'reference')}
    records = {p: execute_pair(**dict(case, root=root)) for p, root in roots.items()}
    final = seal_contract({
        'mode': 'formal', 'panel_digest': 'panel', 'pair_count': 1, 'all_pairs_valid': True,
        'policy_identities': {p: case['identity'] for p in roots},
        'pairs': [compare_pair_receipts(records['b0'], records['reference'])]})
    atomic_json(tmp_path / 'comparison-report.json', final)
    for p, root in roots.items():
        atomic_json(root / 'identity.json', case['identity'])
        atomic_json(root / 'evaluation-report.json', seal_contract({
            'status': 'complete', 'identity_digest': case['identity']['digest'],
            'receipt_digests': [records[p]['digest']]}))
    config = {'reuse': {'comparison_report_digest': final['digest']}}
    panel = {'digest': 'panel', 'pair_count': 1, 'pairs': [case['pair']]}
    result, reused = audit_cached_comparison(tmp_path, config, panel)
    assert result == final and reused['b0'] == [records['b0']]
    bad_panel = dict(panel, pair_count=2)
    with pytest.raises(ComparisonIntegrityError):
        audit_cached_comparison(tmp_path, config, bad_panel)
    atomic_json(roots['reference'] / 'evaluation-report.json', seal_contract({
        'status': 'complete', 'identity_digest': case['identity']['digest'],
        'receipt_digests': []}))
    with pytest.raises(ComparisonIntegrityError):
        audit_cached_comparison(tmp_path, config, panel)


def test_acceptance_only_applies_to_external_reference():
    new = [{'success': i < 1500} for i in range(2000)]
    old = [{'success': i < 1000} for i in range(2000)]
    descriptive = comparison_statistics(new, old, gate=False)
    gate = comparison_statistics(new, old, gate=True)
    assert descriptive['gap_percentage_points'] == 25
    assert 'statistical_decision' not in descriptive
    assert gate['statistical_decision'] == 'PASS'
    assert descriptive['paired_cluster_bootstrap_95_percent_pp'] == (
        gate['paired_cluster_bootstrap_95_percent_pp'])
