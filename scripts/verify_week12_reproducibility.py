"""Standard-library verification and optional safe extraction of the Step 4 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or not path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    return path


def verify_archive(archive, files):
    seen = set()
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle:
            safe_name(member.name)
            if not member.isfile() or member.name in seen or member.name not in files:
                raise ValueError(f'Unexpected archive member: {member.name}')
            expected = files[member.name]
            if member.size != expected['bytes']:
                raise ValueError(f'Archive size mismatch: {member.name}')
            digest = hashlib.sha256()
            with handle.extractfile(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Archive hash mismatch: {member.name}')
            seen.add(member.name)
    if seen != set(files):
        raise ValueError('Archive is missing listed files')
    return len(seen)


def extract_verified(archive, destination):
    if destination.exists():
        raise ValueError('Extraction destination must not already exist')
    destination.mkdir(parents=True)
    # Only regular, relative, uniquely named files were accepted by verify_archive.
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle:
            target = destination.joinpath(*safe_name(member.name).parts)
            if not member.isfile():
                raise ValueError('Non-regular archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--extract-to', type=Path)
    parser.add_argument('--repository', type=Path, help='Also compare the captured files with an existing checkout')
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / 'bundle-manifest.json').read_text())
    archive = package / manifest['archive']
    if sha_file(archive) != manifest['archive_sha256']:
        raise ValueError('Bundle archive hash mismatch')
    file_count = verify_archive(archive, manifest['files'])
    if args.repository:
        for name, expected in manifest['files'].items():
            if sha_file(args.repository / name) != expected['sha256']:
                raise ValueError(f'Checkout differs from snapshot: {name}')
    sealed_count = None
    index_path = package / 'package-index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())['files_sha256']
        observed = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file() and p != index_path}
        if observed != set(index):
            raise ValueError('Step 4 package inventory differs')
        for name, expected in index.items():
            safe_name(name)
            if sha_file(package / name) != expected:
                raise ValueError(f'Sealed package file changed: {name}')
        sealed_count = len(index)
    if args.extract_to:
        extract_verified(archive, args.extract_to.resolve())
    print(json.dumps({'status': 'pass', 'bundle_files_verified': file_count,
                      'sealed_package_files_verified': sealed_count,
                      'extracted_to': str(args.extract_to.resolve()) if args.extract_to else None}))


if __name__ == '__main__':
    main()
