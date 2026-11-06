"""Self-contained browser workbook for completing boundary human QA."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from eventground_wam.boundary_qa import (
    BOUNDARY_QA_SCHEMA_VERSION,
    BoundaryQAError,
    validate_boundary_review_item,
)
from eventground_wam.demonstration_contracts import validate_contract

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_boundary_review_page(
    items: Iterable[Mapping[str, Any]],
    *,
    preview_root: Path,
    output_path: Path,
    repository_root: Path | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    fps: float = 20.0,
    embed_previews: bool = True,
) -> dict[str, Any]:
    """Write one HTML file containing the queue, previews, rubric, and controls.

    Preview PNGs are embedded as data URLs by default. ``embed_previews=False``
    produces a much smaller page that loads images from the existing preview
    tree. Source videos remain external because embedding hundreds of complete
    episodes would make the workbook impractical; when ``repository_root`` is
    supplied, the page includes synchronized players that reference the original
    local videos.
    """

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    output = output_path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"boundary QA refuses to overwrite {output}")
    previews = preview_root.expanduser().resolve()
    if not previews.is_dir():
        raise FileNotFoundError(f"boundary preview root is missing: {previews}")
    repository = repository_root.expanduser().resolve() if repository_root else None
    if repository is not None and not repository.is_dir():
        raise FileNotFoundError(f"demonstration repository root is missing: {repository}")

    ordered_items = tuple(items)
    if not ordered_items:
        raise BoundaryQAError("cannot build a review workbook from an empty queue")
    for item in ordered_items:
        validate_boundary_review_item(item)

    event_by_digest = _event_details(events)
    required_event_digests = {
        str(prediction["event_digest"])
        for item in ordered_items
        for prediction in item["predictions"]
    }
    if events is not None:
        missing = sorted(required_event_digests - set(event_by_digest))
        if missing:
            raise BoundaryQAError(
                f"event detail stream is missing {len(missing)} selected predictions"
            )

    embedded_preview_count = 0
    referenced_preview_count = 0
    payload_items: list[dict[str, Any]] = []
    for item in ordered_items:
        predictions: list[dict[str, Any]] = []
        for prediction in item["predictions"]:
            preview_path = _safe_relative_path(
                previews,
                prediction["preview_path"],
                "prediction.preview_path",
            )
            if embed_previews:
                try:
                    png = preview_path.read_bytes()
                except OSError as error:
                    raise FileNotFoundError(
                        f"boundary QA preview is unreadable: {preview_path}"
                    ) from error
                if not png.startswith(PNG_SIGNATURE):
                    raise BoundaryQAError(f"preview is not a PNG file: {preview_path}")
                preview_source = (
                    "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                )
                embedded_preview_count += 1
            else:
                preview_source = _page_relative_source(preview_path, output.parent)
                referenced_preview_count += 1
            event_digest = str(prediction["event_digest"])
            detail = event_by_digest.get(event_digest)
            predictions.append(
                {
                    **dict(prediction),
                    "preview_source": preview_source,
                    "event_detail": detail,
                }
            )

        source_references = dict(item["source_references"])
        source_video_urls: dict[str, str | None] = {"front": None, "wrist": None}
        if repository is not None:
            for camera in ("front", "wrist"):
                video_path = _safe_relative_path(
                    repository,
                    source_references[f"{camera}_video_path"],
                    f"source_references.{camera}_video_path",
                )
                source_video_urls[camera] = video_path.as_uri()

        payload_items.append(
            {
                "digest": item["digest"],
                "source_episode_index": item["source_episode_index"],
                "source_task_index": item["source_task_index"],
                "instruction": item["instruction"],
                "episode_length": item["episode_length"],
                "length_stratum": item["length_stratum"],
                "source_references": source_references,
                "source_video_urls": source_video_urls,
                "predictions": predictions,
            }
        )

    page_id = hashlib.sha256(
        "\n".join(str(item["digest"]) for item in ordered_items).encode("ascii")
    ).hexdigest()
    payload = {
        "page_schema_version": 1,
        "review_schema_version": BOUNDARY_QA_SCHEMA_VERSION,
        "page_id": page_id,
        "fps": fps,
        "items": payload_items,
    }
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    serialized = (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = _PAGE_TEMPLATE.replace("__QA_PAYLOAD__", serialized)
    if "__QA_PAYLOAD__" in html:
        raise RuntimeError("boundary review page payload substitution failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output)
    return {
        "output": str(output),
        "episode_count": len(payload_items),
        "embedded_preview_count": embedded_preview_count,
        "referenced_preview_count": referenced_preview_count,
        "external_video_count": 0 if repository is None else len(payload_items) * 2,
        "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _event_details(
    events: Iterable[Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    if events is None:
        return details
    for event in events:
        validate_contract(event)
        digest = str(event["digest"])
        if digest in details:
            raise BoundaryQAError(f"duplicate event detail digest: {digest}")
        details[digest] = {
            "confidence": event["confidence"],
            "evidence": list(event["evidence"]),
            "ontology_tier": event.get("ontology_tier"),
            "review_status": event.get("review_status"),
        }
    return details


def _safe_relative_path(root: Path, value: Any, name: str) -> Path:
    if type(value) is not str or not value:
        raise BoundaryQAError(f"{name} must be a nonempty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value == ".":
        raise BoundaryQAError(f"{name} must be a safe relative path")
    return root.joinpath(*relative.parts)


def _page_relative_source(path: Path, page_directory: Path) -> str:
    try:
        return path.relative_to(page_directory).as_posix()
    except ValueError:
        return path.as_uri()


_PAGE_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EventGround Boundary QA Workbook</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --panel-2: #eef2f5;
  --text: #17212b;
  --muted: #5f6b76;
  --border: #ccd4dc;
  --accent: #1769aa;
  --accent-2: #0c8454;
  --danger: #b42318;
  --warning: #a15c00;
  --shadow: 0 1px 4px rgb(15 23 42 / 12%);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #11161c;
    --panel: #192129;
    --panel-2: #222d36;
    --text: #edf2f7;
    --muted: #aeb8c2;
    --border: #3b4854;
    --accent: #69b5f0;
    --accent-2: #5bd6a0;
    --danger: #ff8d84;
    --warning: #ffc36f;
    --shadow: 0 1px 5px rgb(0 0 0 / 35%);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input, select, textarea { font: inherit; }
button, select, input[type="text"], input[type="number"], textarea {
  color: var(--text);
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
}
button { cursor: pointer; padding: 7px 11px; }
button:hover { border-color: var(--accent); }
button.primary { color: white; background: #1769aa; border-color: #1769aa; }
button.complete { color: white; background: #0c8454; border-color: #0c8454; }
button.danger { color: var(--danger); }
button:disabled { cursor: not-allowed; opacity: .45; }
input[type="text"], input[type="number"], select, textarea { padding: 7px 8px; }
textarea { width: 100%; min-height: 70px; resize: vertical; }
header {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 18px max(18px, calc((100vw - 1500px) / 2));
}
header h1 { margin: 0 0 4px; font-size: 22px; }
header p { margin: 0; color: var(--muted); }
.toolbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: grid;
  grid-template-columns: auto minmax(180px, 1fr) auto auto minmax(170px, 240px) auto;
  gap: 8px;
  align-items: center;
  padding: 10px max(18px, calc((100vw - 1500px) / 2));
  background: color-mix(in srgb, var(--panel) 94%, transparent);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
}
.toolbar .progress { color: var(--muted); white-space: nowrap; }
main { max-width: 1500px; margin: 0 auto; padding: 18px; }
.rubric, .episode, .prediction, .missed-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 9px;
  box-shadow: var(--shadow);
}
.rubric { margin-bottom: 16px; }
.rubric summary { cursor: pointer; font-weight: 700; padding: 12px 14px; }
.rubric-body {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 24px;
  border-top: 1px solid var(--border);
  padding: 12px 16px 15px;
}
.rubric-body h3 { margin: 0 0 5px; font-size: 15px; }
.rubric-body ul { margin: 0; padding-left: 20px; }
.episode { padding: 16px; }
.episode-head { display: flex; justify-content: space-between; gap: 16px; align-items: start; }
.episode h2 { margin: 0 0 4px; font-size: 21px; }
.instruction { margin: 8px 0 12px; font-size: 17px; }
.meta, .small { color: var(--muted); font-size: 13px; }
.status {
  display: inline-block;
  padding: 3px 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  white-space: nowrap;
}
.status.done { color: var(--accent-2); border-color: var(--accent-2); }
.video-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0 8px; }
.video-panel { background: #080a0c; border-radius: 7px; overflow: hidden; }
.video-panel strong { display: block; color: #f8fafc; padding: 5px 8px; }
video { display: block; width: 100%; max-height: 380px; background: black; }
.video-note { color: var(--muted); margin: 7px 0 10px; }
.video-controls { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 15px; }
.predictions { display: grid; gap: 14px; }
.prediction { overflow: hidden; }
.prediction-head { padding: 11px 13px; background: var(--panel-2); }
.prediction-head h3 { margin: 0 0 3px; font-size: 16px; }
.preview-wrap { overflow-x: auto; background: #101418; }
.preview-wrap img { display: block; width: 100%; min-width: 640px; height: auto; }
.frame-labels {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: 1fr;
  min-width: 640px;
  color: #e5edf5;
  background: #202830;
  font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
  text-align: center;
  padding: 5px 0;
}
.legend { padding: 8px 12px; color: var(--muted); border-bottom: 1px solid var(--border); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin: 0 3px 0 10px; }
.dot:first-child { margin-left: 0; }
.green { background: #20d060; } .red { background: #f03030; } .blue { background: #3060f0; }
.evidence { padding: 10px 13px; border-bottom: 1px solid var(--border); }
.evidence ul { margin: 5px 0 0; padding-left: 20px; }
fieldset { border: 0; margin: 0; padding: 12px 13px 14px; }
fieldset legend { font-weight: 700; padding: 0; margin-bottom: 7px; }
.verdicts { display: flex; flex-wrap: wrap; gap: 9px 18px; }
.verdicts label { cursor: pointer; }
.correction { display: none; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
.correction.visible { display: flex; }
.correction input { width: 100px; }
.comment { display: block; width: 100%; margin-top: 10px; }
.missed-panel { margin-top: 14px; padding: 13px; }
.missed-panel h3 { margin: 0 0 5px; font-size: 16px; }
.missed-row {
  display: grid;
  grid-template-columns: 170px 95px 95px minmax(160px, 1fr) auto;
  gap: 7px;
  margin-top: 8px;
}
.episode-final { margin-top: 14px; display: grid; gap: 10px; }
.exhaustive { font-weight: 650; }
.validation { min-height: 22px; color: var(--danger); }
.bottom-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.spacer { flex: 1; }
.file-input { display: none; }
@media (max-width: 850px) {
  .toolbar { grid-template-columns: auto 1fr auto; }
  .toolbar .progress, .toolbar .reviewer-label, .toolbar #reviewer { grid-column: span 3; }
  .rubric-body, .video-grid { grid-template-columns: 1fr; }
  .episode-head { display: block; }
  .status { margin-top: 7px; }
  .missed-row { grid-template-columns: 1fr 1fr; }
  .missed-row .miss-comment { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<header>
  <h1>EventGround boundary QA</h1>
  <p>Review telemetry-derived weak event labels against both camera streams.</p>
</header>
<div class="toolbar">
  <button id="previous" type="button">Previous</button>
  <select id="episode-select" aria-label="Episode"></select>
  <button id="next" type="button">Next</button>
  <span class="progress" id="progress"></span>
  <label class="reviewer-label" for="reviewer">Reviewer ID</label>
  <input id="reviewer" type="text" autocomplete="name" placeholder="required for export">
</div>
<main>
  <details class="rubric">
    <summary>Required review rubric</summary>
    <div class="rubric-body">
      <section>
        <h3>For every prediction</h3>
        <ul>
          <li>Compare front and wrist views around the boundary.</li>
          <li>Confirm a physical gripper-close response follows the indicated onset.</li>
          <li>Use the locked interval meaning: onset plus the five-frame confirmation window, not the entire grasp.</li>
          <li>Judge the event label only; task success and grasp establishment are separate concepts.</li>
        </ul>
      </section>
      <section>
        <h3>Verdicts</h3>
        <ul>
          <li><b>Accepted:</b> event and both interval endpoints are correct.</li>
          <li><b>Corrected:</b> event is real but one or both endpoints need replacement.</li>
          <li><b>Rejected:</b> no corresponding gripper-close occurred.</li>
          <li><b>Ambiguous:</b> available evidence cannot support a confident judgment.</li>
        </ul>
      </section>
      <section>
        <h3>Recall check</h3>
        <ul>
          <li>Watch the complete episode, not only each preview window.</li>
          <li>Add every real event that has no prediction to “Missed events.”</li>
          <li>An empty missed-event list means you explicitly found none.</li>
        </ul>
      </section>
      <section>
        <h3>Preview legend</h3>
        <ul>
          <li>Top row: front camera. Bottom row: wrist camera.</li>
          <li>Green border: inside <code>[start, end)</code>.</li>
          <li>Red left edge: start. Blue right edge: final included frame.</li>
        </ul>
      </section>
    </div>
  </details>
  <article class="episode">
    <div class="episode-head">
      <div>
        <h2 id="episode-title"></h2>
        <div class="meta" id="episode-meta"></div>
      </div>
      <span class="status" id="episode-status"></span>
    </div>
    <p class="instruction" id="instruction"></p>
    <div class="predictions" id="predictions"></div>
    <div id="video-area"></div>
    <section class="missed-panel">
      <h3>Missed events</h3>
      <p class="small">After scanning the full episode, record any event that the extractor did not predict.</p>
      <div id="missed-events"></div>
      <button id="add-missed" type="button">Add missed event</button>
    </section>
    <div class="episode-final">
      <label for="episode-notes"><b>Episode notes</b></label>
      <textarea id="episode-notes" placeholder="Optional overall notes"></textarea>
      <label class="exhaustive">
        <input id="exhaustive" type="checkbox">
        I watched the complete episode and recorded every missed event (including none).
      </label>
      <div class="validation" id="validation" role="alert"></div>
      <div class="bottom-actions">
        <button class="complete" id="mark-complete" type="button">Mark episode complete</button>
        <button id="jump-pending" type="button">Next pending</button>
        <span class="spacer"></span>
        <button id="load-reviews" type="button">Load reviews JSONL</button>
        <input class="file-input" id="review-file" type="file" accept=".jsonl,.json,application/json">
        <button class="primary" id="download" type="button">Download completed-reviews.jsonl</button>
      </div>
    </div>
  </article>
</main>
<script id="qa-data" type="application/json">__QA_PAYLOAD__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("qa-data").textContent);
  const storageKey = "eventground-boundary-qa:" + data.page_id;
  let storageAvailable = true;
  let state = loadState();
  const $ = (id) => document.getElementById(id);
  const videos = () => Array.from(document.querySelectorAll("video[data-camera]"));

  function loadState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(storageKey));
      return parsed && parsed.episodes ? parsed : blankState();
    } catch (_) {
      storageAvailable = false;
      return blankState();
    }
  }

  function blankState() {
    return { reviewerId: "", current: 0, episodes: {} };
  }

  function saveState() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (_) {
      storageAvailable = false;
    }
    updateChrome();
  }

  function episodeState(item) {
    if (!state.episodes[item.digest]) {
      const judgments = {};
      item.predictions.forEach((prediction) => {
        judgments[prediction.event_digest] = {
          verdict: "",
          correctedStart: "",
          correctedEnd: "",
          comment: ""
        };
      });
      state.episodes[item.digest] = {
        completed: false,
        exhaustive: false,
        judgments,
        missed: [],
        notes: ""
      };
    }
    return state.episodes[item.digest];
  }

  function invalidate(item) {
    episodeState(item).completed = false;
    saveState();
  }

  function updateChrome() {
    const completed = data.items.filter((item) => episodeState(item).completed).length;
    $("progress").textContent = `${completed} / ${data.items.length} completed` +
      (storageAvailable ? "" : " · memory only; download often");
    $("previous").disabled = state.current <= 0;
    $("next").disabled = state.current >= data.items.length - 1;
    $("episode-select").value = String(state.current);
    const current = data.items[state.current];
    if (current) {
      const done = episodeState(current).completed;
      $("episode-status").textContent = done ? "Completed" : "Pending";
      $("episode-status").classList.toggle("done", done);
      $("mark-complete").textContent = done ? "Completed — click to revalidate" : "Mark episode complete";
    }
  }

  function option(value, text) {
    const element = document.createElement("option");
    element.value = value;
    element.textContent = text;
    return element;
  }

  function render() {
    const item = data.items[state.current];
    const review = episodeState(item);
    $("episode-title").textContent = `Episode ${item.source_episode_index}`;
    $("episode-meta").textContent =
      `Task ${item.source_task_index} · ${item.episode_length} frames · ${item.length_stratum} stratum · ${item.predictions.length} prediction(s)`;
    $("instruction").textContent = item.instruction;
    $("episode-notes").value = review.notes;
    $("exhaustive").checked = review.exhaustive;
    $("validation").textContent = "";
    renderVideos(item);
    renderPredictions(item, review);
    renderMissed(item, review);
    updateChrome();
  }

  function renderVideos(item) {
    const area = $("video-area");
    area.replaceChildren();
    const available = item.source_video_urls.front && item.source_video_urls.wrist;
    if (!available) {
      const note = document.createElement("p");
      note.className = "video-note";
      note.textContent = "Source videos were not linked when this workbook was generated. Previews remain fully embedded.";
      area.append(note);
      return;
    }
    const grid = document.createElement("div");
    grid.className = "video-grid";
    ["front", "wrist"].forEach((camera) => {
      const panel = document.createElement("div");
      panel.className = "video-panel";
      const label = document.createElement("strong");
      label.textContent = camera === "front" ? "Front camera" : "Wrist camera";
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.dataset.camera = camera;
      video.src = item.source_video_urls[camera];
      panel.append(label, video);
      grid.append(panel);
    });
    const controls = document.createElement("div");
    controls.className = "video-controls";
    const buttons = [
      ["Play both", () => videos().forEach((video) => video.play())],
      ["Pause both", () => videos().forEach((video) => video.pause())],
      ["−1 frame", () => shiftVideos(-1)],
      ["+1 frame", () => shiftVideos(1)]
    ];
    buttons.forEach(([label, action]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.addEventListener("click", action);
      controls.append(button);
    });
    const note = document.createElement("p");
    note.className = "video-note";
    note.textContent = "The players reference the original demonstration files; they are not duplicated inside this workbook.";
    area.append(grid, controls, note);
  }

  function shiftVideos(frameDelta) {
    videos().forEach((video) => {
      video.pause();
      video.currentTime = Math.max(0, video.currentTime + frameDelta / data.fps);
    });
  }

  function seekVideos(frame) {
    videos().forEach((video) => {
      video.pause();
      video.currentTime = frame / data.fps;
    });
  }

  function renderPredictions(item, review) {
    const container = $("predictions");
    container.replaceChildren();
    item.predictions.forEach((prediction, index) => {
      const judgment = review.judgments[prediction.event_digest];
      const card = document.createElement("section");
      card.className = "prediction";

      const head = document.createElement("div");
      head.className = "prediction-head";
      const title = document.createElement("h3");
      const [start, end] = prediction.predicted_frame_range;
      title.textContent = `Prediction ${index + 1}: ${prediction.event_type} [${start}, ${end})`;
      const meta = document.createElement("div");
      meta.className = "small";
      meta.textContent = `Review frames [${prediction.review_frame_range[0]}, ${prediction.review_frame_range[1]}) · onset ${(start / data.fps).toFixed(2)} s`;
      const seek = document.createElement("button");
      seek.type = "button";
      seek.textContent = "Seek videos to onset";
      seek.addEventListener("click", () => seekVideos(start));
      head.append(title, meta, seek);

      const previewWrap = document.createElement("div");
      previewWrap.className = "preview-wrap";
      const image = document.createElement("img");
      image.src = prediction.preview_source;
      image.alt = `${prediction.event_type} boundary preview, front row then wrist row`;
      const frames = document.createElement("div");
      frames.className = "frame-labels";
      prediction.sampled_frame_indices.forEach((frame) => {
        const label = document.createElement("span");
        label.textContent = `f${frame}`;
        frames.append(label);
      });
      previewWrap.append(image, frames);

      const legend = document.createElement("div");
      legend.className = "legend";
      legend.innerHTML = '<span class="dot green"></span>inside interval <span class="dot red"></span>start <span class="dot blue"></span>last included frame';

      const evidence = document.createElement("div");
      evidence.className = "evidence";
      if (prediction.event_detail) {
        const heading = document.createElement("b");
        const detail = prediction.event_detail;
        heading.textContent = `Extractor evidence · confidence ${detail.confidence}`;
        const list = document.createElement("ul");
        detail.evidence.forEach((entry) => {
          const row = document.createElement("li");
          row.textContent = entry;
          list.append(row);
        });
        evidence.append(heading, list);
      } else {
        evidence.textContent = "Extractor evidence was not embedded in this workbook.";
      }

      const fieldset = document.createElement("fieldset");
      const fieldLegend = document.createElement("legend");
      fieldLegend.textContent = "Verdict";
      const verdicts = document.createElement("div");
      verdicts.className = "verdicts";
      ["accepted", "corrected", "rejected", "ambiguous"].forEach((verdict) => {
        const label = document.createElement("label");
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `verdict-${prediction.event_digest}`;
        radio.value = verdict;
        radio.checked = judgment.verdict === verdict;
        radio.addEventListener("change", () => {
          judgment.verdict = verdict;
          if (verdict !== "corrected") {
            judgment.correctedStart = "";
            judgment.correctedEnd = "";
          }
          invalidate(item);
          renderPredictions(item, review);
        });
        label.append(radio, " " + verdict[0].toUpperCase() + verdict.slice(1));
        verdicts.append(label);
      });

      const correction = document.createElement("div");
      correction.className = "correction" + (judgment.verdict === "corrected" ? " visible" : "");
      const correctionLabel = document.createElement("span");
      correctionLabel.textContent = "Corrected [start, end):";
      const correctedStart = numberInput(judgment.correctedStart, 0, item.episode_length - 1);
      const correctedEnd = numberInput(judgment.correctedEnd, 1, item.episode_length);
      correctedStart.addEventListener("input", () => {
        judgment.correctedStart = correctedStart.value;
        invalidate(item);
      });
      correctedEnd.addEventListener("input", () => {
        judgment.correctedEnd = correctedEnd.value;
        invalidate(item);
      });
      correction.append(correctionLabel, correctedStart, correctedEnd);

      const comment = document.createElement("input");
      comment.className = "comment";
      comment.type = "text";
      comment.placeholder = "Optional judgment comment";
      comment.value = judgment.comment;
      comment.addEventListener("input", () => {
        judgment.comment = comment.value;
        invalidate(item);
      });
      fieldset.append(fieldLegend, verdicts, correction, comment);
      card.append(head, previewWrap, legend, evidence, fieldset);
      container.append(card);
    });
  }

  function numberInput(value, min, max) {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "1";
    input.min = String(min);
    input.max = String(max);
    input.value = value;
    return input;
  }

  function renderMissed(item, review) {
    const container = $("missed-events");
    container.replaceChildren();
    const eventTypes = [...new Set(item.predictions.map((prediction) => prediction.event_type))].sort();
    review.missed.forEach((missed, index) => {
      const row = document.createElement("div");
      row.className = "missed-row";
      const type = document.createElement("select");
      eventTypes.forEach((eventType) => type.append(option(eventType, eventType)));
      type.value = missed.eventType;
      const start = numberInput(missed.start, 0, item.episode_length - 1);
      const end = numberInput(missed.end, 1, item.episode_length);
      const comment = document.createElement("input");
      comment.className = "miss-comment";
      comment.type = "text";
      comment.placeholder = "Why this was missed";
      comment.value = missed.comment;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "danger";
      remove.textContent = "Remove";
      type.addEventListener("change", () => { missed.eventType = type.value; invalidate(item); });
      start.addEventListener("input", () => { missed.start = start.value; invalidate(item); });
      end.addEventListener("input", () => { missed.end = end.value; invalidate(item); });
      comment.addEventListener("input", () => { missed.comment = comment.value; invalidate(item); });
      remove.addEventListener("click", () => {
        review.missed.splice(index, 1);
        invalidate(item);
        renderMissed(item, review);
      });
      row.append(type, start, end, comment, remove);
      container.append(row);
    });
  }

  function validationError(item, review) {
    if (!state.reviewerId.trim()) return "Enter a reviewer ID first.";
    for (const prediction of item.predictions) {
      const judgment = review.judgments[prediction.event_digest];
      if (!judgment || !judgment.verdict) {
        return `Choose a verdict for every prediction; ${prediction.event_digest.slice(0, 10)}… is incomplete.`;
      }
      if (judgment.verdict === "corrected") {
        const start = Number(judgment.correctedStart);
        const end = Number(judgment.correctedEnd);
        if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start >= end || end > item.episode_length) {
          return `Corrected range for ${prediction.event_digest.slice(0, 10)}… must be integer [start, end) inside the episode.`;
        }
      }
    }
    for (const missed of review.missed) {
      const start = Number(missed.start);
      const end = Number(missed.end);
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start >= end || end > item.episode_length) {
        return "Every missed event needs an integer [start, end) range inside the episode.";
      }
    }
    if (!review.exhaustive) return "Confirm that you watched the complete episode and checked for missed events.";
    return "";
  }

  function completedRecord(item, review) {
    const reviewedTypes = [...new Set(item.predictions.map((prediction) => prediction.event_type))].sort();
    return {
      artifact_type: "eventground_wam.boundary_review_output",
      schema_version: data.review_schema_version,
      review_item_digest: item.digest,
      status: "completed",
      reviewer_id: state.reviewerId.trim(),
      reviewed_event_types: reviewedTypes,
      prediction_judgments: item.predictions.map((prediction) => {
        const judgment = review.judgments[prediction.event_digest];
        return {
          event_digest: prediction.event_digest,
          verdict: judgment.verdict,
          corrected_frame_range: judgment.verdict === "corrected"
            ? [Number(judgment.correctedStart), Number(judgment.correctedEnd)]
            : null,
          comment: judgment.comment || ""
        };
      }),
      missed_events: review.missed.map((missed) => ({
        event_type: missed.eventType,
        frame_range: [Number(missed.start), Number(missed.end)],
        comment: missed.comment || ""
      })),
      notes: review.notes || ""
    };
  }

  function go(index) {
    state.current = Math.max(0, Math.min(data.items.length - 1, index));
    saveState();
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function nextPending() {
    const total = data.items.length;
    for (let offset = 1; offset <= total; offset += 1) {
      const index = (state.current + offset) % total;
      if (!episodeState(data.items[index]).completed) {
        go(index);
        return;
      }
    }
    $("validation").textContent = "All episodes are complete. Download the JSONL file.";
  }

  function downloadReviews() {
    const records = [];
    for (const item of data.items) {
      const review = episodeState(item);
      if (review.completed) records.push(completedRecord(item, review));
    }
    if (!records.length) {
      $("validation").textContent = "Complete at least one episode before downloading.";
      return;
    }
    const content = records.map((record) => JSON.stringify(record)).join("\n") + "\n";
    const url = URL.createObjectURL(new Blob([content], { type: "application/x-ndjson" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "completed-reviews.jsonl";
    link.click();
    URL.revokeObjectURL(url);
  }

  async function loadReviews(file) {
    const text = await file.text();
    const records = text.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line));
    const itemByDigest = new Map(data.items.map((item) => [item.digest, item]));
    for (const record of records) {
      const item = itemByDigest.get(record.review_item_digest);
      if (!item || record.status !== "completed") continue;
      const review = episodeState(item);
      review.completed = true;
      review.exhaustive = true;
      review.notes = record.notes || "";
      if (record.reviewer_id) state.reviewerId = record.reviewer_id;
      for (const judgment of record.prediction_judgments || []) {
        if (!review.judgments[judgment.event_digest]) continue;
        review.judgments[judgment.event_digest] = {
          verdict: judgment.verdict || "",
          correctedStart: judgment.corrected_frame_range ? String(judgment.corrected_frame_range[0]) : "",
          correctedEnd: judgment.corrected_frame_range ? String(judgment.corrected_frame_range[1]) : "",
          comment: judgment.comment || ""
        };
      }
      review.missed = (record.missed_events || []).map((missed) => ({
        eventType: missed.event_type,
        start: String(missed.frame_range[0]),
        end: String(missed.frame_range[1]),
        comment: missed.comment || ""
      }));
      const error = validationError(item, review);
      if (error) review.completed = false;
    }
    $("reviewer").value = state.reviewerId;
    saveState();
    render();
  }

  data.items.forEach((item, index) => {
    $("episode-select").append(option(String(index), `Episode ${item.source_episode_index}`));
    episodeState(item);
  });
  $("reviewer").value = state.reviewerId;
  $("reviewer").addEventListener("input", () => {
    state.reviewerId = $("reviewer").value;
    saveState();
  });
  $("previous").addEventListener("click", () => go(state.current - 1));
  $("next").addEventListener("click", () => go(state.current + 1));
  $("episode-select").addEventListener("change", () => go(Number($("episode-select").value)));
  $("episode-notes").addEventListener("input", () => {
    const item = data.items[state.current];
    episodeState(item).notes = $("episode-notes").value;
    invalidate(item);
  });
  $("exhaustive").addEventListener("change", () => {
    const item = data.items[state.current];
    episodeState(item).exhaustive = $("exhaustive").checked;
    invalidate(item);
  });
  $("add-missed").addEventListener("click", () => {
    const item = data.items[state.current];
    const review = episodeState(item);
    const eventType = [...new Set(item.predictions.map((prediction) => prediction.event_type))].sort()[0];
    review.missed.push({ eventType, start: "", end: "", comment: "" });
    invalidate(item);
    renderMissed(item, review);
  });
  $("mark-complete").addEventListener("click", () => {
    const item = data.items[state.current];
    const review = episodeState(item);
    const error = validationError(item, review);
    $("validation").textContent = error;
    if (error) return;
    review.completed = true;
    saveState();
    updateChrome();
  });
  $("jump-pending").addEventListener("click", nextPending);
  $("download").addEventListener("click", downloadReviews);
  $("load-reviews").addEventListener("click", () => $("review-file").click());
  $("review-file").addEventListener("change", async () => {
    if (!$("review-file").files.length) return;
    try {
      await loadReviews($("review-file").files[0]);
      $("validation").textContent = "Existing completed reviews loaded.";
    } catch (error) {
      $("validation").textContent = "Could not load that JSONL file: " + error.message;
    }
    $("review-file").value = "";
  });
  saveState();
  render();
})();
</script>
</body>
</html>
'''
