"""The portable archive must reject corruption and unsafe extraction paths."""
from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_week12_reproducibility.py'
SPEC = importlib.util.spec_from_file_location('week12_repro_verifier', SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class ReproductionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'bundle.tar.gz'
        self.data = b'preserved evidence\n'
        self.expected = {'src/example.txt': {'bytes': len(self.data),
                         'sha256': hashlib.sha256(self.data).hexdigest()}}

    def archive_with(self, names=('src/example.txt',), kind=tarfile.REGTYPE):
        with tarfile.open(self.archive, 'w:gz') as tar:
            for name in names:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.size = len(self.data) if kind == tarfile.REGTYPE else 0
                member.linkname = '/outside' if kind == tarfile.SYMTYPE else ''
                tar.addfile(member, io.BytesIO(self.data) if member.size else None)

    def test_valid_archive_extracts_exact_bytes(self):
        self.archive_with()
        self.assertEqual(VERIFIER.verify_archive(self.archive, self.expected), 1)
        destination = self.root / 'new'
        VERIFIER.extract_verified(self.archive, destination)
        self.assertEqual((destination / 'src/example.txt').read_bytes(), self.data)

    def test_traversal_absolute_and_noncanonical_paths_rejected(self):
        for name in ('../escape', '/absolute', 'src/../escape', './relative', 'src//file'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                VERIFIER.safe_name(name)

    def test_symlink_rejected(self):
        self.archive_with(kind=tarfile.SYMTYPE)
        with self.assertRaises(ValueError):
            VERIFIER.verify_archive(self.archive, self.expected)

    def test_duplicate_member_rejected(self):
        self.archive_with(names=('src/example.txt', 'src/example.txt'))
        with self.assertRaises(ValueError):
            VERIFIER.verify_archive(self.archive, self.expected)

    def test_unlisted_member_rejected(self):
        self.archive_with(names=('src/example.txt', 'other'))
        with self.assertRaises(ValueError):
            VERIFIER.verify_archive(self.archive, self.expected)

    def test_missing_member_rejected(self):
        self.archive_with(names=())
        with self.assertRaises(ValueError):
            VERIFIER.verify_archive(self.archive, self.expected)

    def test_hash_mismatch_rejected(self):
        self.archive_with()
        self.expected['src/example.txt']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            VERIFIER.verify_archive(self.archive, self.expected)

    def test_size_mismatch_rejected(self):
        self.archive_with()
        self.expected['src/example.txt']['bytes'] += 1
        with self.assertRaises(ValueError):
            VERIFIER.verify_archive(self.archive, self.expected)

    def test_existing_destination_not_overwritten(self):
        self.archive_with()
        destination = self.root / 'existing'
        destination.mkdir()
        sentinel = destination / 'keep'; sentinel.write_text('original')
        with self.assertRaises(ValueError):
            VERIFIER.extract_verified(self.archive, destination)
        self.assertEqual(sentinel.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
