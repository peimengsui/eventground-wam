import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_m1_dev_runtime as runtime


@pytest.mark.parametrize(('processes', 'blocked'), [
    ('999999, /opt/google/chrome/chrome --type=gpu-process --ozone-platform=wayland, 133\n', False),
    ('999999, /opt/google/chrome/chrome --type=gpu-process --ozone-platform=wayland, 300\n', True),
    ('999999, /usr/bin/python, 10\n', True),
    ('999999, /opt/google/chrome/chrome --type=renderer, 133\n', True),
])
def test_evaluator_uses_qualified_gpu_classifier(monkeypatch, processes, blocked):
    monkeypatch.setattr(runtime.subprocess, 'check_output', lambda *args, **kwargs: processes)
    if blocked:
        with pytest.raises(ValueError, match='GPU job'):
            runtime.require_idle_gpu()
    else:
        runtime.require_idle_gpu()
