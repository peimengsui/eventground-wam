#!/usr/bin/env python3
"""Prepare boundary previews/review files or aggregate completed human QA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from eventground_wam.boundary_qa import (
    aggregate_boundary_reviews,
    build_boundary_review_selection,
    pending_review_output,
    render_boundary_previews,
)
from eventground_wam.boundary_review_page import write_boundary_review_page

DEFAULT_QA_VERSION = "week2-boundary-qa-v1"


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_data_root() -> Path:
    return expanded_path(
        os.environ.get(
            "EVENTGROUND_LIBERO_DATA_ROOT",
            "~/data/eventground-wam/libero-plus",
        )
    )


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="select episodes and write review inputs")
    prepare.add_argument("--trajectory-index", type=expanded_path, required=True)
    prepare.add_argument("--events", type=expanded_path, required=True)
    prepare.add_argument(
        "--output-dir",
        type=expanded_path,
        default=data_root / "derived" / "boundary-qa" / DEFAULT_QA_VERSION,
    )
    prepare.add_argument("--target-episodes", type=int, default=200)
    prepare.add_argument("--context-frames", type=int, default=10)
    prepare.add_argument("--tile-size", type=int, default=128)

    render = subparsers.add_parser(
        "render", help="render a bounded subset of an existing review selection"
    )
    render.add_argument("--repository-root", type=expanded_path, required=True)
    render.add_argument("--review-items", type=expanded_path, required=True)
    render.add_argument("--output-dir", type=expanded_path, required=True)
    render.add_argument("--tile-size", type=int, default=128)
    render.add_argument("--start", type=int, default=0)
    render.add_argument(
        "--limit",
        type=int,
        default=5,
        help="render N episodes from --start (default: 5; use 0 for all remaining)",
    )
    render.add_argument(
        "--manifest-output",
        type=expanded_path,
        help="preview manifest path (default: OUTPUT_DIR/preview-manifest.jsonl)",
    )

    workbook = subparsers.add_parser(
        "workbook",
        help="combine review inputs and embedded previews into one interactive HTML file",
    )
    workbook.add_argument("--review-items", type=expanded_path, required=True)
    workbook.add_argument("--preview-root", type=expanded_path, required=True)
    workbook.add_argument("--output", type=expanded_path, required=True)
    workbook.add_argument(
        "--repository-root",
        type=expanded_path,
        help="optional source root used to show the original videos inside the page",
    )
    workbook.add_argument(
        "--events",
        type=expanded_path,
        help="optional event JSONL used to display extractor confidence and evidence",
    )
    workbook.add_argument("--fps", type=float, default=20.0)
    workbook.add_argument(
        "--reference-previews",
        action="store_true",
        help="reference preview PNGs instead of embedding them for a fast-loading page",
    )

    summarize = subparsers.add_parser(
        "summarize", help="validate reviews and compute completed-review metrics"
    )
    summarize.add_argument("--review-items", type=expanded_path, required=True)
    summarize.add_argument("--reviews", type=expanded_path, required=True)
    summarize.add_argument("--output", type=expanded_path, required=True)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = {
        "selection": args.output_dir / "selection-manifest.json",
        "items": args.output_dir / "review-items.jsonl",
        "templates": args.output_dir / "review-output-template.jsonl",
    }
    conflicts = [path for path in output_paths.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"boundary QA refuses to overwrite outputs: {conflicts}")

    trajectories = _read_jsonl(args.trajectory_index)
    events = _read_jsonl(args.events)
    items, selection = build_boundary_review_selection(
        trajectories,
        events,
        target_episode_count=args.target_episodes,
        context_frames=args.context_frames,
        preview_tile_size=args.tile_size,
    )
    templates = tuple(pending_review_output(item) for item in items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_paths["selection"], selection)
    _write_jsonl(output_paths["items"], items)
    _write_jsonl(output_paths["templates"], templates)
    return {
        "selected_episode_count": len(items),
        "selection_shortfall": selection["selection_shortfall"],
        "output_directory": str(args.output_dir),
        "human_quality_metrics_status": "unavailable_until_reviews_complete",
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    if args.start < 0 or args.limit < 0:
        raise ValueError("render start and limit cannot be negative")
    items = _read_jsonl(args.review_items)
    end = None if args.limit == 0 else args.start + args.limit
    selected_items = items[args.start:end]
    if not selected_items:
        raise ValueError("render selection is empty")
    manifest_output = args.manifest_output or (args.output_dir / "preview-manifest.jsonl")
    if manifest_output.exists():
        raise FileExistsError(f"boundary QA refuses to overwrite {manifest_output}")
    previews = render_boundary_previews(
        selected_items,
        repository_root=args.repository_root,
        output_directory=args.output_dir,
        tile_size=args.tile_size,
    )
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest_output, previews)
    return {
        "start": args.start,
        "selected_episode_count": len(selected_items),
        "rendered_preview_count": len(previews),
        "manifest_output": str(manifest_output),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"boundary QA refuses to overwrite {args.output}")
    report = aggregate_boundary_reviews(
        _read_jsonl(args.review_items),
        _read_jsonl(args.reviews),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    return report


def workbook(args: argparse.Namespace) -> dict[str, Any]:
    return write_boundary_review_page(
        _read_jsonl(args.review_items),
        preview_root=args.preview_root,
        output_path=args.output,
        repository_root=args.repository_root,
        events=None if args.events is None else _read_jsonl(args.events),
        fps=args.fps,
        embed_previews=not args.reference_previews,
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} must contain a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSONL from {path}") from error
    return tuple(records)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    payload = "".join(
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    commands = {
        "prepare": prepare,
        "render": render,
        "workbook": workbook,
        "summarize": summarize,
    }
    result = commands[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
