from copy import deepcopy
from pathlib import Path

import pytest

from eventground_wam.future_pairs import read_sealed


def test_independent_audit_real_aggregates_and_tampered_cell(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from verify_jepa_candidate import verify_aggregates
    base = root/'docs/evaluation/week6-jepa-baselines-v1/run'
    run = root/'docs/evaluation/week6-jepa-validation-v2/full'
    inputs = [read_sealed(p)[key] for p, key in (
        (base/'episode-metrics.json', 'rows'), (run/'episode-metrics.json', 'rows'),
        (base/'retrieval-panel.json', 'queries'), (run/'retrieval-panel.json', 'queries'))]
    report = read_sealed(run/'report.json')
    result = verify_aggregates(*inputs, report)
    assert result['primary_bootstrap_max_error'] < 1e-10
    assert [c['horizon'] for c in result['camera_horizon_cells'] if not c['passes']] == [2]
    changed = deepcopy(report)
    changed['by_camera_horizon'][0]['prediction']['cosine']['value'] += .001
    with pytest.raises(AssertionError):
        verify_aggregates(*inputs, changed)
    changed = deepcopy(inputs)
    changed[3][0]['pool']['frames'][0] += 1
    with pytest.raises(AssertionError):
        verify_aggregates(*changed, report)
