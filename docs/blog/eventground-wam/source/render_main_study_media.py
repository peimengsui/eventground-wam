"""Render the blog's main-study media from retained artifacts only.

Dependencies: numpy, Pillow, h5py with HDF5 2.0 support, imageio-ffmpeg.
Raw inputs live in ignored outputs/blog-media-v1, never in the website copy.
No model inference, simulation, or statistical estimation is performed.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
BLOG = ROOT / "docs/blog/eventground-wam"
SITE = BLOG / "site"
INPUT = ROOT / "outputs/blog-media-v1"
IMAGES = SITE / "assets/images/eventground-wam"
VIDEOS = SITE / "assets/videos/eventground-wam"
ARMS = ["B0", "M1", "M3"]
NAMES = {"B0": "Action-only", "M1": "Pixel auxiliary", "M3": "Pixel coupled"}
COLORS = {"B0": "#18344a", "M1": "#117c83", "M3": "#74569b"}
NAVY, GRAY = "#18344a", "#627382"
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def draw_text(draw, xy, label, size=20, color=NAVY, bold=False, anchor=None):
    draw.text(xy, label, font=font(size, bold), fill=color, anchor=anchor)


def asset(path, sources, transformation):
    return {"site_path": str(path.relative_to(SITE)), "source": sources,
            "sha256": sha(path), "bytes": path.stat().st_size,
            "transformation": transformation}


def select_cases():
    reports = {a: json.loads((ROOT / f"outputs/week9-initial-v1-metadata/cohort/panels/{a}/perturbed/evaluation-report.json").read_text()) for a in ARMS}
    selected = []
    for kind, pattern in [("gain", (False, False, True)), ("regression", (True, True, False))]:
        candidates = []
        for ordinal in range(1000):
            if tuple(bool(reports[a]["episodes"][ordinal]["success"]) for a in ARMS) != pattern:
                continue
            inputs = [reports[a]["audits"][ordinal]["inputs"] for a in ARMS]
            if inputs[0] == inputs[1] == inputs[2]:
                candidates.append(ordinal)
        ordinal = candidates[0]
        selected.append({"kind": kind, "ordinal": ordinal,
                         "selection": "First manifest ordinal in this three-policy outcome class with exactly matching recorded scene, settling prefix, and first-query inputs.",
                         "eligible_ordinals": candidates,
                         "arms": {a: reports[a]["episodes"][ordinal] for a in ARMS},
                         "input_audit": reports["B0"]["audits"][ordinal]["inputs"]})
    assert [s["ordinal"] for s in selected] == [35, 34]
    return selected


def load_rollouts(case):
    arrays = {}
    for arm in ARMS:
        path = INPUT / f"{arm}-{case['ordinal']:04d}.hdf5"
        assert sha(path) == case["arms"][arm]["artifact_sha256"]
        with h5py.File(path, "r") as f:
            phase = f["steps/phase"][:]
            indices = np.flatnonzero(phase == 1)
            assert indices[0] == 10
            assert len(indices) == case["arms"][arm]["policy_step_count"]
            assert np.all(np.diff(indices) == 1)
            times = f["steps/sim_time_seconds"][indices]
            assert np.allclose(np.diff(times), 1 / 20)
            metadata = json.loads(f["episode/metadata_json"][()])
            assert metadata["success"] == case["arms"][arm]["success"]
            arrays[arm] = {"front": f["steps/agent_rgb"][indices],
                           "wrist": f["steps/wrist_rgb"][indices],
                           "steps": f["steps/step_index"][indices],
                           "metadata": metadata}
    # Recheck the actual downloaded initial images as well as retained audit hashes.
    for camera in ["front", "wrist"]:
        assert all(np.array_equal(arrays["B0"][camera][0], arrays[a][camera][0]) for a in ARMS)
    return arrays


def video_frame(case, arrays, index):
    canvas = Image.new("RGB", (816, 632), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (16, 12), "Move the black bowl from the table center onto the plate", 24, bold=True)
    draw_text(draw, (16, 45), f"Perturbed robot start · case {case['ordinal']:04d} · simulation time {index / 20:.2f} s", 18, GRAY)
    for col, arm in enumerate(ARMS):
        x = 16 + col * 264
        count = len(arrays[arm]["front"])
        pos = min(index, count - 1)
        draw_text(draw, (x, 83), f"{arm} · {NAMES[arm]}", 20, COLORS[arm], bold=True)
        success = case["arms"][arm]["success"]
        draw_text(draw, (x, 113), "Recorded outcome: " + ("success" if success else "failure"), 18,
                  "#196d53" if success else "#9c4935")
        draw_text(draw, (x, 145), "Front camera", 16, GRAY)
        canvas.paste(Image.fromarray(arrays[arm]["front"][pos]), (x, 170))
        draw_text(draw, (x, 438), "Wrist camera", 16, GRAY)
        canvas.paste(Image.fromarray(arrays[arm]["wrist"][pos]).resize((128, 128), Image.Resampling.LANCZOS), (x, 464))
        if index >= count:
            status = ["Recording ended", "Last frame held"]
        else:
            status = [f"Control step {pos}", f"{pos / 20:.2f} s"]
        for row, line in enumerate(status):
            draw_text(draw, (x + 137, 495 + 25 * row), line, 15, GRAY)
    draw_text(draw, (16, 611), "20 Hz simulation playback · pre-action observations · inference wall time omitted", 16, GRAY)
    return canvas


def render_video(case, arrays):
    path = VIDEOS / f"five-policy-{case['kind']}.mp4"
    frames = max(len(arrays[a]["front"]) for a in ARMS)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "816x632",
               "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for index in range(frames):
        process.stdin.write(video_frame(case, arrays, index).tobytes())
    process.stdin.close()
    errors = process.stderr.read().decode()
    assert process.wait() == 0, errors
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    decoded = sum(1 for _ in reader)
    assert decoded == frames and metadata["size"] == (816, 632)
    assert abs(metadata["fps"] - 20) < 0.001
    assert abs(metadata["duration"] - frames / 20) < 0.01
    poster = IMAGES / f"five-policy-{case['kind']}-poster.png"
    video_frame(case, arrays, 100).save(poster)
    preview = INPUT / f"{case['kind']}-review.png"
    review = Image.new("RGB", (816, 632 * 3), "white")
    for row, index in enumerate([0, 100, frames - 1]):
        review.paste(video_frame(case, arrays, index), (0, row * 632))
    review.save(preview)
    case["export"] = {"frames": frames, "fps": 20, "duration_seconds": frames / 20,
                      "size": [816, 632], "source_policy_frames": {a: len(arrays[a]["front"]) for a in ARMS},
                      "held_frame_after_recording_end": True,
                      "poster_control_step": 100, "decoded_frame_count_verified": decoded}
    sources = [str((INPUT / f"{a}-{case['ordinal']:04d}.hdf5").relative_to(ROOT)) for a in ARMS]
    return [asset(path, sources, "Recorded RGB aligned by control step at 20 Hz; first ten settling steps omitted; shorter recordings hold the last pre-action frame with an explicit label; H.264 presentation export."),
            asset(poster, sources, "Labeled still at comparison control step 100, from the same recording exporter.")]


def render_pixel_panel():
    sheet = ROOT / "outputs/week5-pixel-validation-v1/qualitative/episode-006993.png"
    receipt = json.loads((INPUT / "pixel-006993-receipt.json").read_text())
    assert sha(sheet) == receipt["qualitative_sha256"][sheet.name]
    assert sha(INPUT / "pixel-006993.npz") == receipt["qualitative_sha256"]["episode-006993.npz"]
    predictions = np.load(INPUT / "pixel-006993.npz", allow_pickle=False)["predictions"]
    original = Image.open(sheet).convert("RGB")
    canvas = Image.new("RGB", (948, 864), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (18, 14), "Eight frames into the future", 30, bold=True)
    draw_text(draw, (18, 57), "Saved pixel-predictor validation · conditioned on demonstration actions", 22, GRAY)
    for col, label in enumerate(["Current observation", "Predicted future", "Actual future"]):
        draw_text(draw, (18 + col * 312, 101), label, 24, bold=True)
    for row, source_row in enumerate([2, 3]):
        example = receipt["qualitative_examples"][source_row]
        assert example["anchor"] == 115 and example["target"] == 123 and example["horizon"] == 8
        camera = "Front" if row == 0 else "Wrist"
        y = 144 + row * 344
        draw_text(draw, (18, y), f"{camera} camera · frame 115 → 123 (0.4 s)", 20, GRAY)
        source_y = 100 + source_row * 173 + 19
        cells = [original.crop((12 + col * 171, source_y, 12 + col * 171 + 148, source_y + 148)) for col in [0, 2, 3]]
        rgb = np.round(np.clip(predictions[source_row].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        assert np.array_equal(np.asarray(cells[1]), np.asarray(Image.fromarray(rgb).resize((148, 148), Image.Resampling.NEAREST)))
        for col, cell in enumerate(cells):
            canvas.paste(cell.resize((296, 296), Image.Resampling.NEAREST), (18 + col * 312, y + 30))
    draw_text(draw, (18, 838), "One fixed validation example · original saved pixels, enlarged for reading", 18, GRAY)
    path = IMAGES / "pixel-prediction-example.png"
    canvas.save(path)
    provenance = {"task": "turn on the stove and put the moka pot on it", "episode": 6993,
                  "anchor": 115, "target": 123, "horizon": 8, "cameras": ["front", "wrist"],
                  "selection": "First task in the preselected validation panel, midpoint anchor, horizon eight, both cameras.",
                  "checkpoint_stage": "Accepted pixel predictor before downstream action-policy training",
                  "conditioning": "Current observations, instruction, robot state, recorded intervening demonstration actions, horizon",
                  "sheet_sha256": sha(sheet), "saved_predictions_sha256": sha(INPUT / "pixel-006993.npz"),
                  "prediction_tiles_verified_against_saved_float_arrays": True,
                  "current_and_target_tiles": "Exact crops of the checksum-verified saved panel; nearest-neighbor enlargement only."}
    return asset(path, [str(sheet.relative_to(ROOT)), str((INPUT / "pixel-006993.npz").relative_to(ROOT))], "Recomposed current/prediction/target columns for both cameras from the retained fixed panel; predicted tiles checked against saved float arrays; no new inference."), provenance


def main():
    VIDEOS.mkdir(parents=True, exist_ok=True)
    cases = select_cases()
    assets = []
    for case in cases:
        arrays = load_rollouts(case)
        assets.extend(render_video(case, arrays))
        print(f"Rendered {case['kind']} case {case['ordinal']:04d}", flush=True)
    pixel_asset, pixel_provenance = render_pixel_panel()
    assets.append(pixel_asset)
    # Keep concise selection/identity records; per-query arrays remain in original reports.
    for case in cases:
        case["arms"] = {a: {k: r[k] for k in ["artifact", "artifact_sha256", "success", "policy_step_count", "perturbation_category"]} for a, r in case["arms"].items()}
    provenance = {"date": "2026-11-05", "new_inference_or_rollouts": False, "cases": cases,
                  "pixel_prediction": pixel_provenance, "assets": assets,
                  "renderer_sha256": sha(Path(__file__))}
    (BLOG / "media-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manifest_path = BLOG / "asset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    new_paths = {a["site_path"] for a in assets}
    manifest["assets"] = [a for a in manifest["assets"] if a["site_path"] not in new_paths] + assets
    manifest["media_renderer"] = str(Path(__file__).relative_to(ROOT))
    manifest["media_renderer_sha256"] = sha(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified both complete video decodes and saved prediction pixel identity.")


if __name__ == "__main__":
    main()
