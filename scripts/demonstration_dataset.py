"""Pinned metadata and common helpers for the LIBERO-Plus demonstrations."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DATASET_REPO_ID = "Sylvest/libero_plus_lerobot"
DATASET_REVISION = "22c57433fef692b5b9ecc0795344daac7fa867a5"
EXPECTED_FILE_COUNT = 43_048
EXPECTED_TOTAL_BYTES = 15_995_527_260
EXPECTED_MANIFEST_SHA256 = "e673a1d3e58a1756879b4f30ce38474aa3407f97232ee5184fd7e1e6ed901510"

EXPECTED_METADATA_SHA256 = {
    ".gitattributes": "56457c49ac198d89670ddec059fed3cf73f9b527237143e6f35b669cdfd84879",
    "README.md": "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17",
    "meta/episodes.jsonl": "8e99bea92dd184a60895237d0d38edcc2fc7d7c4e6eca53eaa5479c0039324f8",
    "meta/info.json": "8e189f2dafbb1d86ca5b2a1fe582a781fe77cd20e08c536247421d541ecd3a7e",
    "meta/tasks.jsonl": "e7c9d0a5f997173326e05ec2640c7b24c3e86a4e306d1e3cc88fa3453d0b4823",
    "norm_stats.json": "6f861c1e0148c622cb51121322975f817ff0db4fa320af4889f965b139dac7f0",
}

EXPECTED_DATASET_INFO = {
    "codebase_version": "v2.1",
    "robot_type": "panda",
    "total_episodes": 14_347,
    "total_frames": 2_238_036,
    "total_tasks": 40,
    "total_videos": 28_694,
    "total_chunks": 15,
    "chunks_size": 1_000,
    "fps": 20,
}


@dataclass(frozen=True)
class SourceFile:
    """One file in the pinned Hugging Face repository snapshot."""

    path: str
    size: int
    sha256: str

    def line(self) -> str:
        return f"{self.path}\t{self.size}\t{self.sha256}\n"


def default_data_root() -> Path:
    return Path(
        os.environ.get(
            "EVENTGROUND_LIBERO_DATA_ROOT",
            "~/data/eventground-wam/libero-plus",
        )
    ).expanduser()


def dataset_base(data_root: Path) -> Path:
    return data_root / "datasets" / "libero-plus-lerobot"


def revision_dir(data_root: Path) -> Path:
    return dataset_base(data_root) / "snapshots" / DATASET_REVISION


def snapshot_dir(data_root: Path) -> Path:
    return revision_dir(data_root) / "repo"


def source_manifest_path(data_root: Path) -> Path:
    return revision_dir(data_root) / "source-manifest.tsv"


def source_record_path(data_root: Path) -> Path:
    return revision_dir(data_root) / "source.json"


def active_snapshot_path(data_root: Path) -> Path:
    return dataset_base(data_root) / "current"


def validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"unsafe repository path: {value!r}")


def canonical_source_files(files: Iterable[SourceFile]) -> bytes:
    ordered = sorted(files, key=lambda item: item.path)
    return "".join(item.line() for item in ordered).encode("utf-8")


def source_manifest_sha256(files: Iterable[SourceFile]) -> str:
    return hashlib.sha256(canonical_source_files(files)).hexdigest()


def source_files_from_repo_info(info: Any) -> list[SourceFile]:
    files: list[SourceFile] = []
    for sibling in info.siblings or []:
        path = sibling.rfilename
        validate_relative_path(path)
        if sibling.size is None:
            raise ValueError(f"Hugging Face omitted the size for {path}")
        lfs = sibling.lfs
        files.append(
            SourceFile(
                path=path,
                size=int(sibling.size),
                sha256=str(lfs.sha256) if lfs is not None else "",
            )
        )
    validate_source_files(files)
    return sorted(files, key=lambda item: item.path)


def validate_source_files(files: list[SourceFile]) -> None:
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("source manifest contains duplicate paths")
    for item in files:
        validate_relative_path(item.path)
        if item.size < 0:
            raise ValueError(f"negative size for {item.path}")
        if item.sha256 and (len(item.sha256) != 64 or not _is_lower_hex(item.sha256)):
            raise ValueError(f"invalid SHA-256 for {item.path}")
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"expected {EXPECTED_FILE_COUNT} files, found {len(files)}")
    total_bytes = sum(item.size for item in files)
    if total_bytes != EXPECTED_TOTAL_BYTES:
        raise ValueError(f"expected {EXPECTED_TOTAL_BYTES} bytes, found {total_bytes}")
    digest = source_manifest_sha256(files)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"expected source-manifest SHA-256 {EXPECTED_MANIFEST_SHA256}, found {digest}"
        )


def load_source_manifest(path: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"invalid source manifest line {line_number}")
            relative_path, size_text, sha256 = parts
            try:
                size = int(size_text)
            except ValueError as error:
                raise ValueError(f"invalid size on source manifest line {line_number}") from error
            files.append(SourceFile(relative_path, size, sha256))
    validate_source_files(files)
    return sorted(files, key=lambda item: item.path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if raw_line.strip():
                try:
                    yield json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path} at line {line_number}") from error


def _is_lower_hex(value: str) -> bool:
    return value == value.lower() and all(character in "0123456789abcdef" for character in value)
