import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_m1_results import copy_checked


def test_backup_is_independent_and_repeat_verifies_bytes(tmp_path):
    original, copy = tmp_path / 'original', tmp_path / 'backup/copy'
    original.write_bytes(b'checkpoint')
    receipt = copy_checked(original, copy)
    assert original.stat().st_ino != copy.stat().st_ino
    assert copy_checked(original, copy) == receipt
    original.write_bytes(b'changed')
    assert copy.read_bytes() == b'checkpoint'
    with pytest.raises(ValueError, match='existing backup differs'):
        copy_checked(original, copy)


def test_backup_replaces_only_its_incomplete_temporary_copy(tmp_path):
    original, copy = tmp_path / 'original', tmp_path / 'copy'
    original.write_bytes(b'complete bytes')
    copy.with_name('copy.copying').write_bytes(b'incomplete')
    copy_checked(original, copy)
    assert copy.read_bytes() == original.read_bytes()
    assert not copy.with_name('copy.copying').exists()
