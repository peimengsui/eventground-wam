"""Evidence opt-in must not hide failures when explicitly requested."""
from pathlib import Path

pytest_plugins = ["pytester"]


def test_default_code_suite_and_explicit_artifact_audit(pytester):
    hook = Path(__file__).with_name("conftest.py").read_text()
    hook += "\nRESEARCH_ARTIFACT_TESTS |= {'test_selection.py::test_registered'}\n"
    pytester.makeconftest(hook)
    pytester.makepyfile(test_selection=(
        "import pytest\n"
        "def test_code():\n    assert 2 + 2 == 4\n"
        "def test_registered():\n    raise FileNotFoundError('restore external evidence')\n"
        "@pytest.mark.research_artifacts\n"
        "def test_marked():\n    raise FileNotFoundError('restore external evidence')\n"
    ))
    pytester.runpytest("-q").assert_outcomes(passed=1, skipped=2)
    pytester.runpytest("-q", "--research-artifacts").assert_outcomes(passed=1, failed=2)
