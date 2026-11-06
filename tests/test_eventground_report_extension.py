import copy
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_report_extension import GATE, coverage_gate, panel_sources, validate_panel_contract

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/evaluation/week10-report-coverage-extension-v1'


def reseal(d):
    return seal_contract({k:v for k,v in d.items() if k != 'digest'})


def fixture():
    return [json.loads((BASE / p).read_text()) for p in ['panel-contract.json', 'partitions.json', 'source-metadata.json']]


def test_exact_new_sources_and_parent_exclusion():
    c,p,m = fixture()
    assert len(validate_panel_contract(c,p,m)) == 80
    assert not set(c['source_ids']) & {r['source_id'] for r in p['rows']}


@pytest.mark.parametrize('mutation', ['duplicate', 'old_source', 'forged_id', 'remove_exclusion', 'missing_source'])
def test_identity_failures(mutation):
    _,p,m = fixture()
    if mutation == 'duplicate': m['rows'][0] = copy.deepcopy(m['rows'][1])
    elif mutation == 'old_source': m['rows'][0] = {**p['rows'][0], 'partition': m['panel']}
    elif mutation == 'forged_id': m['rows'][0]['source_id'] = 'f'*64
    elif mutation == 'remove_exclusion': m['excluded_source_ids'] = []
    elif mutation == 'missing_source': m['rows'].pop()
    with pytest.raises(ValueError): panel_sources(p,reseal(m))


@pytest.mark.parametrize('field,value', [('automatic_retries',1),('training_authorized',True),
                                        ('report_model_inference_authorized',True),('independent_qa_passed',True)])
def test_no_scope_expansion(field,value):
    c,p,m = fixture();c[field]=value
    with pytest.raises(ValueError):validate_panel_contract(reseal(c),p,m)


def test_repeated_anchors_do_not_replace_independent_grasps():
    rows=[dict(target={'next_event':kind},event_id=f'{kind}-event',source_id='source',task_key='task')
          for kind in ['contact','grasp','gripper_close','none'] for _ in range(20)]
    result=coverage_gate(rows)
    assert result['checks']['prompts_per_class']
    assert not result['passed']
    assert result['actual']['grasp_events'] == result['actual']['grasp_sources'] == 1


def test_distinct_sources_floor_and_empty_panel():
    assert not coverage_gate([])['passed']
    rows=[dict(target={'next_event':kind},event_id=f'{kind}-event-{i//2}',source_id=f'source-{i//2}',task_key=f'task-{i//4}')
          for kind in ['contact','grasp','gripper_close','none'] for i in range(20)]
    assert coverage_gate(rows)['passed']
