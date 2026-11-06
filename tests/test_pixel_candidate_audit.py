import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.pixel_baselines import FIELDS, assess_candidate, measurement_rule
from eventground_wam.pixel_evaluation import prediction_summary


def test_independent_audit_recomputes_bootstrap_and_detects_tampering(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("candidate_auditor", root / "scripts/validate_pixel_candidate.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    baseline, predicted = [], []
    for task in range(40):
        for group in range(3):
            ep = task*3+group
            error = .09+.005*group+.0001*task
            for c in CAMERAS:
                for h in HORIZONS:
                    row = {"episode_index": ep, "source_digest": f"source-{ep}", "group_digest": f"g-{ep}",
                           "task_key": f"task-{task:02d}", "camera": c, "horizon": h, "pairs": 37-h}
                    metrics = lambda scale, ssim, error=error: dict(zip(FIELDS, [error*scale, ssim, error*scale*.2, error*scale*.8, .2, .8], strict=True))
                    baseline.append({**row, "copy": metrics(1, .6), "background": metrics(2, .4)})
                    predicted.append({**row, "prediction": metrics(.8, .7)})
    report = {"gate": assess_candidate(baseline, predicted), **prediction_summary(predicted)}
    actual = auditor.independent_aggregate(baseline, predicted, report, measurement_rule()["exit_gate"])
    assert all(actual["independently_recomputed_checks"].values())
    bad = deepcopy(report)
    bad["gate"]["effects"]["l1_vs_copy"]["paired_95"][0] += .01
    with pytest.raises(AssertionError):
        auditor.independent_aggregate(baseline, predicted, bad, measurement_rule()["exit_gate"])
    bad = deepcopy(report)
    bad["by_camera_horizon"][0]["prediction"]["ssim"]["value"] += .01
    with pytest.raises(AssertionError):
        auditor.independent_aggregate(baseline, predicted, bad, measurement_rule()["exit_gate"])
