"""Download the pinned official LIBERO-Plus LeRobot demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demonstration_dataset import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    EXPECTED_FILE_COUNT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TOTAL_BYTES,
    active_snapshot_path,
    canonical_source_files,
    default_data_root,
    revision_dir,
    snapshot_dir,
    source_files_from_repo_info,
    source_manifest_path,
    source_record_path,
)


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=expanded_path, default=default_data_root())
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def atomic_write(path: Path, payload: bytes) -> None:
    temporary_path = path.with_name(path.name + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()
    with temporary_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def cached_xet_connection_refresher(
    refresh_connection: Callable[..., Any],
    *,
    clock: Callable[[], float] = time.time,
    expiry_margin_seconds: int = 60,
) -> Callable[..., Any]:
    """Reuse a repository-wide Xet token until it is close to expiration.

    huggingface_hub 0.36 requests the same Xet connection token independently
    for every LFS file. Large snapshots can therefore exhaust the Hub API rate
    limit before their payloads begin downloading. The returned wrapper keeps
    one in-memory token per refresh route and still delegates refreshes to the
    Hub before a token expires.
    """

    connection_info_by_route: dict[str, Any] = {}
    lock = threading.Lock()

    def cached_refresh(*, file_data: Any, headers: dict[str, str]) -> Any:
        route = file_data.refresh_route
        if not route:
            return refresh_connection(file_data=file_data, headers=headers)

        with lock:
            connection_info = connection_info_by_route.get(route)
            expiration = getattr(connection_info, "expiration_unix_epoch", 0)
            if connection_info is not None and expiration > clock() + expiry_margin_seconds:
                return connection_info

            connection_info = refresh_connection(file_data=file_data, headers=headers)
            if connection_info is not None:
                connection_info_by_route[route] = connection_info
            return connection_info

    return cached_refresh


def install_xet_connection_cache() -> None:
    """Install the process-local connection cache used by snapshot_download."""

    from huggingface_hub import file_download

    original = file_download.refresh_xet_connection_info
    if getattr(original, "_eventground_cached", False):
        return
    cached = cached_xet_connection_refresher(original)
    cached._eventground_cached = True  # type: ignore[attr-defined]
    file_download.refresh_xet_connection_info = cached


def activate_snapshot(data_root: Path) -> None:
    link = active_snapshot_path(data_root)
    target = snapshot_dir(data_root)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"active dataset link points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"active dataset path exists and is not a symlink: {link}")
    temporary_link = link.with_name(link.name + ".partial")
    if temporary_link.exists() or temporary_link.is_symlink():
        raise RuntimeError(f"temporary active link already exists: {temporary_link}")
    temporary_link.symlink_to(target.relative_to(link.parent), target_is_directory=True)
    os.replace(temporary_link, link)


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    from huggingface_hub import HfApi, get_token, snapshot_download

    token = get_token()
    if not token:
        print(
            "Hugging Face authentication is not configured; attempting the public "
            "snapshot anonymously. If this is rate-limited, run "
            "environments/demonstrations/.venv/bin/hf auth login on feifei and retry.",
            file=sys.stderr,
            flush=True,
        )
    api = HfApi(token=token)
    print(f"Inspecting {DATASET_REPO_ID}@{DATASET_REVISION}", flush=True)
    info = api.dataset_info(
        repo_id=DATASET_REPO_ID,
        revision=DATASET_REVISION,
        files_metadata=True,
    )
    if info.sha != DATASET_REVISION:
        raise RuntimeError(f"requested {DATASET_REVISION}, Hugging Face resolved {info.sha}")
    source_files = source_files_from_repo_info(info)

    revision_root = revision_dir(args.data_root)
    repository_root = snapshot_dir(args.data_root)
    revision_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = canonical_source_files(source_files)
    manifest_path = source_manifest_path(args.data_root)
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_payload:
        raise RuntimeError(
            f"existing source manifest differs from the pinned manifest: {manifest_path}"
        )
    if not manifest_path.exists():
        atomic_write(manifest_path, manifest_payload)

    print(
        f"Downloading {EXPECTED_FILE_COUNT:,} files "
        f"({EXPECTED_TOTAL_BYTES / 1024**3:.3f} GiB) to {repository_root}",
        flush=True,
    )
    install_xet_connection_cache()
    downloaded_path = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=repository_root,
            max_workers=args.max_workers,
            token=token,
        )
    ).resolve()
    if downloaded_path != repository_root.resolve():
        raise RuntimeError(f"snapshot downloaded to unexpected path: {downloaded_path}")

    source_record = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_total_bytes": EXPECTED_TOTAL_BYTES,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "download_completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(
        source_record_path(args.data_root),
        (json.dumps(source_record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    # Do not publish `current` until the downloaded bytes and representative
    # episode payloads have passed the same checks as the standalone validator.
    from check_demonstrations import (
        validate_episode_samples,
        validate_local_files,
        validate_metadata,
        validate_source_record,
    )

    validate_source_record(args.data_root)
    validate_local_files(args.data_root, full_sha256=False)
    info, episodes = validate_metadata(repository_root)
    validate_episode_samples(repository_root, info, episodes, sample_count=3)
    activate_snapshot(args.data_root)
    print(f"Pinned demonstration snapshot ready: {active_snapshot_path(args.data_root)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Download interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
