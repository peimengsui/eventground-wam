"""Prepare blind Week 10 review materials; never generate reference annotations."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import html
import json
import struct
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.eventground_qa import (
    LabelReviewError,
    assess_label_review,
    build_review_manifest,
    make_review_template,
)


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path):
    value = read(path)
    if contract_digest(value) != value.get("digest"):
        raise LabelReviewError(f"Changed or unsealed input: {path}")
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_html(manifest):
    options = "".join(
        f'<option value="{html.escape(row["recording_id"])}">'
        f'{index + 1:03d} — {html.escape(row["task_key"])} — '
        f'{row["frames"]} frames</option>'
        for index, row in enumerate(manifest["episodes"])
    )
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Week 10 independent full-episode review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;background:#fafafa;color:#17202a}
button,select,input{font:inherit;margin:5px;padding:6px}img{width:384px;image-rendering:pixelated}
pre{white-space:pre-wrap;font-size:12px;max-height:500px;overflow:auto} .note{padding:12px;background:#fff1ce}
</style><h1>Independent full-episode review</h1>
<p class="note">Unreviewed evidence. No extractor predictions, event counts or task-success
flags are shown. Review both views and physical evidence for every frame. The panel
does not record an annotation or mark anything reviewed. Enter reference intervals,
uncertainties and explicit negative ranges in the separate blank JSON form.</p>
<select id="episode">OPTIONS</select><p id="instruction"></p>
<button id="previous">Previous frame</button><button id="play">Play at 20 Hz</button>
<button id="next">Next frame</button><input id="frame" type="number" min="0" value="0">
<span id="position"></span><br><img id="front" alt="Original front camera">
<img id="wrist" alt="Original wrist camera"><h2>Saved physical evidence at this frame</h2>
<p>Frame intervals are zero-based, half-open, 20 Hz. Commands, contacts and object
poses are offline review evidence and must never become model inputs.</p><pre id="numeric"></pre>
<script>
let manifest, timer=null; const el=id=>document.getElementById(id);
async function show(){if(!manifest)return; const row=manifest.episodes.find(r=>r.recording_id===el('episode').value);
let frame=Math.min(row.frames-1,Math.max(0,Number(el('frame').value)||0)); el('frame').value=frame;
el('frame').max=row.frames-1; el('instruction').textContent=row.instruction;
el('position').textContent=`Frame ${frame}/${row.frames-1}; ${(frame/20).toFixed(2)} s`;
for(const view of ['front','wrist'])el(view).src=`/frame/${row.recording_id}/${frame}/${view}.png`;
const response=await fetch(`/numeric/${row.recording_id}/${frame}.json`);
el('numeric').textContent=JSON.stringify(await response.json(),null,2);}
el('episode').onchange=()=>{el('frame').value=0;show()};el('frame').onchange=show;
el('previous').onclick=()=>{el('frame').value=Number(el('frame').value)-1;show()};
el('next').onclick=()=>{el('frame').value=Number(el('frame').value)+1;show()};
el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='Play at 20 Hz';return;}
el('play').textContent='Pause';timer=setInterval(()=>{el('frame').value=Number(el('frame').value)+1;show()},50)};
fetch('/manifest.json').then(r=>r.json()).then(m=>{manifest=m;show()});
</script></html>""".replace("OPTIONS", options)


def prepare(args):
    start = time.monotonic()
    budget = sealed(args.preparation_contract)
    limits = budget["review_preparation_limits"]
    seconds, maximum = limits["wall_seconds"], limits["output_bytes"]
    if not 0 < seconds <= 1800 or not 0 < maximum <= 1024**3:
        raise LabelReviewError("Review preparation must remain within 30 CPU minutes and 1 GiB")
    completion = sealed(args.capture_base / "run/completion.json")
    if budget["capture_completion_digest"] != completion["digest"]:
        raise LabelReviewError("Preparation budget does not bind this capture")
    manifest = build_review_manifest(
        sealed(args.capture_base / "recording-contract.json"), completion,
        preparation_contract=budget,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", manifest)
    write(args.output / "review-template.json", make_review_template(manifest))
    with (args.output / "index.html").open("x") as stream:
        stream.write(index_html(manifest))
    files = {path.name: sha(path) for path in args.output.iterdir() if path.is_file()}
    total = sum(path.stat().st_size for path in args.output.iterdir())
    if total > maximum or time.monotonic() - start > seconds:
        raise LabelReviewError("Review preparation resource bound exceeded; preserve partial files")
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.week10_review_preparation",
            "manifest_digest": manifest["digest"],
            "preparation_contract_digest": budget["digest"],
            "files_sha256": files,
            "output_bytes_before_completion": total,
            "elapsed_seconds": time.monotonic() - start,
            "episodes": len(manifest["episodes"]),
            "reference_annotations_created": 0,
            "candidate_data_included": False,
            "images_copied_or_rendered": 0,
            "new_physics_steps": 0,
            "labels_admitted": False,
        }
    )
    completion_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if total + len(completion_bytes) > maximum or time.monotonic() - start > seconds:
        raise LabelReviewError("Review completion would exceed its preparation resource bound")
    with (args.output / "completion.json").open("xb") as stream:
        stream.write(completion_bytes)
    print(json.dumps(report, indent=2))


def png(array):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", binascii.crc32(kind + data) & 0xFFFFFFFF
        )

    height, width, channels = array.shape
    if channels != 3 or str(array.dtype) != "uint8":
        raise LabelReviewError("Camera must be original uint8 RGB")
    raw = b"".join(b"\x00" + row.tobytes() for row in array)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


def serve(args):
    """Read-only loopback viewer; no candidates, inference or simulator imports."""
    import h5py

    start = time.monotonic()
    if not 0 < args.seconds <= 1800:
        raise LabelReviewError("A viewer session is bounded to at most 30 minutes")
    manifest = sealed(args.materials / "manifest.json")
    prepared = sealed(args.materials / "completion.json")
    if prepared["manifest_digest"] != manifest["digest"]:
        raise LabelReviewError("Viewer preparation link changed")
    for name, checksum in prepared["files_sha256"].items():
        if name not in {"manifest.json", "review-template.json", "index.html"}:
            raise LabelReviewError("Unexpected blind-review artifact")
        if sha(args.materials / name) != checksum:
            raise LabelReviewError("Blind review artifact changed")
    episodes = {row["recording_id"]: row for row in manifest["episodes"]}
    # Verify all inputs before serving any evidence; do not serve arbitrary paths.
    for row in episodes.values():
        for name, checksum in row["evidence_sha256"].items():
            if sha(args.capture_base / "run" / row["recording_id"] / name) != checksum:
                raise LabelReviewError("Review evidence checksum mismatch")
        if time.monotonic() - start > args.seconds:
            raise LabelReviewError("Viewer input verification exhausted its session budget")
    cache = {}

    class Viewer(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urlsplit(self.path).path.strip("/").split("/")
            try:
                if parts == [""]:
                    payload = (args.materials / "index.html").read_bytes()
                    mime = "text/html; charset=utf-8"
                elif parts == ["manifest.json"]:
                    payload = (args.materials / "manifest.json").read_bytes()
                    mime = "application/json"
                elif len(parts) in (3, 4) and parts[1] in episodes:
                    mode, rid = parts[:2]
                    frame = int(parts[2].removesuffix(".json"))
                    if not 0 <= frame < episodes[rid]["frames"]:
                        raise ValueError("Frame outside recording")
                    folder = args.capture_base / "run" / rid
                    with h5py.File(folder / "recording.h5", "r") as handle:
                        if mode == "frame" and len(parts) == 4 and parts[3] in ("front.png", "wrist.png"):
                            payload = png(handle[f"observation/{parts[3][:-4]}"][frame])
                            mime = "image/png"
                        elif mode == "numeric" and len(parts) == 3:
                            if rid not in cache:
                                cache.clear()
                                cache[rid] = {
                                    "contacts": [json.loads(line) for line in (folder / "contacts.jsonl").read_text().splitlines()],
                                    "geometry": read(folder / "geometry-map.json"),
                                }
                            data = {
                                "frame": frame, "contacts": cache[rid]["contacts"][frame],
                                "geometry": cache[rid]["geometry"],
                                **{key: handle[key][frame].tolist() for key in (
                                    "time", "action", "observation/state", "object_position",
                                    "object_rotation_matrix", "gripper_position", "gripper_rotation_matrix",
                                )},
                            }
                            payload = json.dumps(data, allow_nan=False).encode()
                            mime = "application/json"
                        else:
                            raise ValueError("Unknown evidence route")
                else:
                    raise ValueError("No such blind-review route")
            except (ValueError, KeyError, OSError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", args.port), Viewer)
    server.timeout = 1
    print(f"Read-only blind viewer: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        while time.monotonic() - start < args.seconds:
            server.handle_request()
    finally:
        server.server_close()


def assess(args):
    manifest = sealed(args.materials / "manifest.json")
    review = sealed(args.review)
    candidates = {
        row["recording_id"]: sealed(args.candidates / row["recording_id"] / "event-candidates.json")
        for row in manifest["episodes"]
    }
    report = assess_label_review(
        manifest, review, candidates, trusted_reviewer_ids=args.trusted_reviewer,
        labeler_actor_ids=args.labeler_actor,
    )
    write(args.output, report)
    print(json.dumps({key: report[key] for key in ("digest", "status", "gates", "failures")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--capture-base", type=Path, required=True)
    prep.add_argument("--preparation-contract", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    viewer = sub.add_parser("serve")
    viewer.add_argument("--capture-base", type=Path, required=True)
    viewer.add_argument("--materials", type=Path, required=True)
    viewer.add_argument("--port", type=int, default=8765)
    viewer.add_argument("--seconds", type=int, default=1800)
    check = sub.add_parser("assess")
    check.add_argument("--materials", type=Path, required=True)
    check.add_argument("--review", type=Path, required=True)
    check.add_argument("--candidates", type=Path, required=True)
    check.add_argument("--trusted-reviewer", action="append", required=True)
    check.add_argument("--labeler-actor", action="append", required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "serve": serve, "assess": assess}[args.command](args)


if __name__ == "__main__":
    main()
