"""Permanent final-analysis failures must not restart training/evaluation."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fresh_analysis', ROOT / 'scripts/analyze_b0_fresh.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.mark.parametrize('error', [ValueError('inputs differ at ordinal 203'),
                                 FileNotFoundError('missing artifact')])
def test_integrity_failure_records_stop_without_completion_or_mutating_evidence(
    monkeypatch, tmp_path, capsys, error
):
    source = tmp_path / 'frozen-report.json'
    source.write_text('{"successes": 162}')

    def fail(work):
        assert work == tmp_path
        raise error

    monkeypatch.setattr(analysis, 'analyze', fail)
    with pytest.raises(SystemExit) as exc:
        analysis.main(['--work-root', str(tmp_path)])
    assert exc.value.code == 78
    result = analysis.read_sealed(tmp_path / 'analysis-status.json')
    assert result['status'] == 'integrity_stop'
    assert result['acceptance'] == 'NOT_ASSESSED'
    assert result['error'] == str(error)
    assert not (tmp_path / 'completion.json').exists()
    assert json.loads(source.read_text()) == {'successes': 162}
    assert 'Integrity stop:' in capsys.readouterr().err


def test_unexpected_failure_is_not_misclassified_as_integrity(monkeypatch, tmp_path):
    def fail(work):
        raise RuntimeError('unexpected failure')

    monkeypatch.setattr(analysis, 'analyze', fail)
    with pytest.raises(RuntimeError, match='unexpected'):
        analysis.main(['--work-root', str(tmp_path)])
    assert not (tmp_path / 'analysis-status.json').exists()


def test_pipeline_propagates_permanent_stop_to_systemd(tmp_path):
    fake_python = tmp_path / 'python'
    fake_python.write_text('#!/bin/sh\nexit 78\n')
    fake_python.chmod(0o755)
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/run_b0_fresh.sh'), 'pipeline', str(tmp_path)],
        env={**os.environ, 'EVENTGROUND_B0_PYTHON': str(fake_python)},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 78, result.stderr
    service = (ROOT / 'configs/systemd/eventground-b0-fresh-v2.service').read_text()
    assert 'RestartPreventExitStatus=78' in service
