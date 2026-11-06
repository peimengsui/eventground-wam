"""Build publication figures and relabel table metadata from frozen evidence.

Run from any directory with Python, matplotlib, and numpy installed. This does
not rerun inference, statistics, or the sealed research-package renderer.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/robot-policy-blog-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[4]
BLOG = ROOT / "docs/blog/eventground-wam"
SITE = BLOG / "site"
EVIDENCE = ROOT / "docs/evaluation/week12-results-v1"
IMAGES = SITE / "assets/images/eventground-wam"
DATA = SITE / "assets/data/eventground-wam"
NAVY, TEAL, PURPLE, GRAY = "#18344A", "#117C83", "#74569B", "#627382"
RECORDS = []


def record(path, source, transformation):
    RECORDS.append({"site_path": str(path.relative_to(SITE)), "source": source,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size, "transformation": transformation})


def architecture():
    # Coordinates use a top-left origin so the three comparison rows read down.
    fig = plt.figure(figsize=(12, 14.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 120), ylim=(145, 0))
    ax.axis("off")

    def text(x, y, s, size=14, color=NAVY, weight="normal", ha="left"):
        return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center", linespacing=1.45)

    def box(x, y, w, h, label, color=NAVY, fill="#F3F6F8", dashed=False, size=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.3,
                     edgecolor=color, facecolor=fill, linestyle="--" if dashed else "-"))
        text(x+w/2, y+h/2, label, size=size, color=color, ha="center")

    def arrow(a, b, color=GRAY, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                     linewidth=1.6, color=color, linestyle="--" if dashed else "-"))

    def line(points, color=GRAY, dashed=False):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=1.3, ls="--" if dashed else "-")

    text(5, 5.5, "Five policies, three architectures", size=25, weight="bold")
    text(5, 10.5, "What changes: the role of future prediction", size=16, color=GRAY)
    box(5, 15, 110, 8, "Inputs: front + wrist images  ·  instruction  ·  robot state",
        color=GRAY, fill="#F7F9FB", size=15)

    text(5, 29, "1   Action-only", size=19, weight="bold")
    text(115, 29, "B0", size=17, color=GRAY, weight="bold", ha="right")
    box(8, 34, 30, 9, "Policy backbone\nQwen3.5-2B")
    box(49, 34, 28, 9, "Action head")
    box(88, 34, 24, 9, "Execute\n8 actions")
    arrow((38, 38.5), (49, 38.5)); arrow((77, 38.5), (88, 38.5))
    text(8, 47, "Training objective: actions", color=GRAY, size=14)
    line([(5, 51), (115, 51)], "#DDE5EA")

    text(5, 57, "2   Auxiliary future prediction", size=19, weight="bold", color=TEAL)
    text(115, 57, "M1 pixel / M2 latent", size=14, color=TEAL, weight="bold", ha="right")
    box(8, 62, 30, 9, "Policy backbone\nQwen3.5-2B")
    box(49, 62, 28, 9, "Action head")
    box(88, 62, 24, 9, "Execute\n8 actions")
    arrow((38, 66.5), (49, 66.5)); arrow((77, 66.5), (88, 66.5))
    arrow((23, 71), (23, 77), TEAL, dashed=True)
    box(8, 77, 30, 9, "Future predictor", TEAL, "#EDF7F6", dashed=True)
    arrow((38, 81.5), (49, 81.5), TEAL, dashed=True)
    box(49, 77, 28, 9, "Future loss", TEAL, "#EDF7F6", dashed=True)
    text(81, 81.5, "Training only;\nremoved at deployment", size=12.5, color=TEAL)
    text(8, 90, "Future branch uses demo actions + horizon; future targets enter the loss only.", size=12.5, color=GRAY)
    line([(5, 94), (115, 94)], "#DDE5EA")

    text(5, 100, "3   Coupled future prediction", size=19, weight="bold", color=PURPLE)
    text(115, 100, "M3 pixel / M4 latent", size=14, color=PURPLE, weight="bold", ha="right")
    box(5, 106, 24, 10, "Backbone +\naction head", size=13)
    box(34, 106, 19, 10, "8-action\nproposal", size=13)
    box(58, 106, 23, 10, "Future core\nhorizon = 8", PURPLE, "#F4F0F8", size=13)
    box(86, 106, 29, 10, "Attention +\nsame action head", PURPLE, "#F4F0F8", size=13)
    arrow((29, 111), (34, 111)); arrow((53, 111), (58, 111), PURPLE)
    arrow((81, 111), (86, 111), PURPLE)
    text(69.5, 119, "future tokens", size=12, color=PURPLE, ha="center")
    text(100.5, 119, "8 refined actions", size=12, color=PURPLE, ha="center")
    text(5, 124, "Current context also feeds the future core and refinement. One refinement per query.", size=12.5, color=GRAY)
    text(5, 129, "Training: action + auxiliary future losses. Deployment: predict, refine, execute 8 actions.", size=12.5, color=GRAY)
    line([(5, 133), (115, 133)], "#DDE5EA")
    text(5, 138, "Pixel target: future images", size=14, weight="bold")
    text(64, 138, "Latent target: future visual features", size=14, weight="bold")
    for ext in ["svg", "png"]:
        path = IMAGES / f"architecture.{ext}"
        # Embed glyph paths so the browser does not need a locally installed font.
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(path, dpi=200, facecolor="white",
                        metadata={"Creator": "Robot policy blog", "Date": None} if ext == "svg" else None)
        if ext == "svg":
            svg = path.read_text()
            svg = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"',
                         'width="100%" style="height:auto;max-width:1152px"', svg, count=1)
            path.write_text(svg)
        record(path, ["docs/WEEK7_AUXILIARY_INTERFACE.md", "docs/WEEK8_COUPLING_INTERFACE.md"],
               "New model-comparison diagram; training-only auxiliary path distinguished from deployed coupling.")
    plt.close(fig)


def event_horizon_architecture():
    """Show the event branch selecting a horizon, with the action path frozen."""
    fig = plt.figure(figsize=(8, 7.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 800), ylim=(730, 0))
    ax.axis("off")

    def text(x, y, label, size=14, color=NAVY, weight="normal", ha="center"):
        ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", linespacing=1.5)

    def box(x, y, w, h, label, color=NAVY, fill="#F3F6F8", size=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=10", linewidth=1.3,
                     edgecolor=color, facecolor=fill))
        text(x+w/2, y+h/2, label, size=size, color=color)

    def arrow(a, b, color=GRAY):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                     mutation_scale=14, linewidth=1.6, color=color))

    text(30, 29, "Event timing selects the prediction horizon", size=19,
         weight="bold", ha="left")
    box(30, 64, 740, 65, "Current front + wrist images, instruction, robot state", size=14)
    arrow((200, 129), (200, 178), TEAL)
    arrow((600, 129), (600, 178))
    box(30, 180, 340, 90, "Predict next event + timing\nContact / grasp / gripper closure", TEAL, "#EDF7F6", size=13)
    box(430, 180, 340, 90, "Propose 8 actions\nOriginal M4 action path", size=14)
    arrow((200, 270), (200, 324), TEAL)
    arrow((600, 270), (600, 324))
    box(30, 326, 340, 96, "Choose nearest horizon\n2, 4, 8, 16, or 32 frames", TEAL, "#EDF7F6", size=14)
    box(430, 326, 340, 96, "Predict future tokens\nUse proposal + selected horizon", PURPLE, "#F4F0F8", size=13)
    arrow((370, 374), (430, 374), TEAL)
    text(400, 354, "h", size=13, color=TEAL)
    arrow((600, 422), (600, 480), PURPLE)
    box(430, 482, 340, 78, "Refine the 8-action proposal\nUsing the predicted future", PURPLE, "#F4F0F8", size=13)
    text(200, 477, "Illustrative mapping", size=13, color=TEAL, weight="bold")
    text(200, 516, "Contact predicted in 14 frames\n→ request a future 16 frames ahead", size=12, color=TEAL)
    text(200, 571, "No new event → horizon 8", size=12, color=GRAY)
    arrow((600, 560), (600, 606))
    box(30, 608, 740, 64, "Execute 8 actions → observe again → repeat", size=15)
    text(400, 706, "The horizon changes the predicted future. Execution stays at 8 actions.",
         size=12, color=GRAY)
    for ext in ["svg", "png"]:
        path = IMAGES / f"event-horizon-flow.{ext}"
        with plt.rc_context({"svg.fonttype": "path", "svg.hashsalt": "robot-policy-event-horizon"}):
            fig.savefig(path, dpi=180, facecolor="white",
                        metadata={"Creator": "Robot policy blog", "Date": None} if ext == "svg" else None)
        if ext == "svg":
            svg = re.sub(r'width="[\d.]+pt" height="[\d.]+pt"',
                         'width="100%" style="height:auto;max-width:800px"', path.read_text(), count=1)
            path.write_text(svg)
        record(path, ["docs/WEEK11_EXPERIMENT.md", "docs/WEEK11_HORIZON_MAPPING.md"],
               "New schematic of the event-to-horizon rule and unchanged eight-action execution; example timing is illustrative.")
    plt.close(fig)


def result_figures():
    # Reuse the sealed plotting code and data; override presentation text only.
    source = EVIDENCE / "source/render_week12_figures.py"
    spec = importlib.util.spec_from_file_location("frozen_report_renderer", source)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    titles = iter([
        ("Future prediction: five-policy comparison", "15,000 outcomes across five trained policies"),
        ("Adaptive routing reduced success", "Exploratory extension: 2,001 episodes across three conditions"),
    ])
    original_header = renderer.header
    renderer.header = lambda fig, title, subtitle: original_header(fig, *next(titles))
    names = {"fig02-five-arm-comparison": "future-policy-comparison",
             "fig03-adaptive-integration": "adaptive-control-comparison"}

    def save(fig, out, stem, caption, sources):
        if stem == "fig02-five-arm-comparison":
            # Show only the two main effects requested for the blog, using frozen estimates.
            fig.subplots_adjust(wspace=1.0)
            for ax, panel in zip(fig.axes[2:], ["perturbed", "clean"]):
                ax.clear()
                contrasts = data["week9"]["panels"][panel]["factorial"]["contrasts_percentage_points"]
                for y, key in [(1, "latent_minus_pixel_main_effect"), (0, "coupling_main_effect")]:
                    r = contrasts[key]
                    x = r["estimate"]
                    lo, hi = r["percentile_95"]
                    ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=NAVY,
                                capsize=4, lw=1.7, ms=7)
                    ax.text(1.04, y, f"{x:+.2f} pp", transform=ax.get_yaxis_transform(),
                            va="center", fontsize=9)
                ax.axvline(0, color=GRAY, ls="--", lw=1)
                ax.set(xlim=(-6, 8), ylim=(-.5, 1.5))
                ax.set_title("Effects across future models", loc="left", fontweight="bold", pad=12)
                ax.set_yticks([1, 0], labels=["Latent − pixel\n(average)", "Coupled − auxiliary\n(average)"])
                renderer.decorate(ax, "Difference (percentage points)")
            for item in fig.texts:
                if item.get_text().startswith("All factorial intervals"):
                    item.set_text("Average coupling gain: the mean of the pixel and latent gains over auxiliary training.")
                elif item.get_text().startswith("10,000 shared PCG64"):
                    item.set_text("Each gain compares coupled with auxiliary training. B0 is used only as an external reference.")
        # Audit every displayed text element, including labels within axes.
        for item in fig.findobj(matplotlib.text.Text):
            assert not re.search(r"week\s*\d+", item.get_text(), re.I), item.get_text()
            assert "interaction" not in item.get_text().lower(), item.get_text()
        path = IMAGES / f"{names[stem]}.png"
        fig.savefig(path, dpi=220, facecolor="white")
        record(path, "docs/evaluation/week12-results-v1/consolidated.json",
               "Regenerated from frozen plotting code and data; five-policy figure displays two main effects; all displayed estimates and intervals unchanged.")
        plt.close(fig)

    renderer.save = save
    data = json.loads((EVIDENCE / "consolidated.json").read_text())
    renderer.week9(data, EVIDENCE)
    renderer.formal(data, EVIDENCE)


def tables():
    studies = {"Week 9": "Five-policy comparison", "Week 10": "Event forecasting",
               "Week 11 formal": "Adaptive-horizon evaluation",
               "Week 11 fresh H8 follow-up": "Fixed-horizon continuation follow-up",
               "Week 11 continuation screen": "Multi-horizon continuation screen",
               "Week 11 adapter screen": "Motion-adapter screen"}
    # A stable source label preserves provenance without exposing schedule names.
    prefixes = {
        "docs/evaluation/week9-final-v1/": "five-policy/",
        "docs/evaluation/week10-preservation-pilot-v1/": "event-preservation/",
        "docs/evaluation/week10-sft-continuation-report-v1/": "event-continuation-report/",
        "docs/evaluation/week10-future-onset-v1/": "event-forecasting/",
        "docs/evaluation/week11-closeout-v1/": "adaptive-horizon/",
        "docs/evaluation/week11-h8-validation-v1/": "fixed-horizon-follow-up/",
        "docs/evaluation/week11-motion-screen-v1/": "motion-adapter/",
        "docs/evaluation/week11-multihorizon-results-v1/": "multi-horizon-continuation/",
    }
    mapping = {}
    checked = 0
    for name in ["success_rates.csv", "contrasts.csv", "event_forecasting.csv"]:
        source = EVIDENCE / "tables" / name
        with source.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            original = list(reader)
        revised = []
        for row in original:
            r = row.copy()
            if "study" in r:
                r["study"] = studies[r["study"]]
            matches = [p for p in prefixes if r["source"].startswith(p)]
            assert len(matches) == 1, r["source"]
            label = r["source"].replace(matches[0], prefixes[matches[0]], 1)
            mapping[label] = r["source"]
            r["source"] = label
            assert all(r[k] == row[k] for k in fields if k not in {"study", "source"})
            checked += len([k for k in fields if k not in {"study", "source"}])
            revised.append(r)
        path = DATA / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(revised)
        assert not re.search(r"week[\s_-]*\d+", path.read_text(), re.I)
        record(path, str(source.relative_to(ROOT)),
               "Study and source labels renamed for publication; every other cell and row order unchanged.")
    (BLOG / "source-label-map.json").write_text(json.dumps(mapping, indent=2, sort_keys=True)+"\n")
    return checked


def version_article_images():
    """Give edited figures new URLs so browser image caches cannot keep old charts."""
    draft = SITE / "_drafts/building-a-robot-policy-that-looks-ahead.md"

    def version(match):
        url = match.group(1)
        digest = hashlib.sha256((SITE / url.lstrip("/")).read_bytes()).hexdigest()[:12]
        return f"{url}?v={digest}"

    original = draft.read_text()
    revised = re.sub(r'''(/assets/(?:images|videos)/eventground-wam/[^\s)?"<>]+)(?:\?v=[0-9a-f]+)?''', version, original)
    if revised != original:
        draft.write_text(revised)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": NAVY, "axes.labelcolor": NAVY,
                         "xtick.color": GRAY, "ytick.color": NAVY,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "robot-policy-blog"})
    architecture(); event_horizon_architecture(); result_figures()
    checked = tables()
    for name, original in [("adaptive-gain.png", "fig04a-targeted-gain.png"),
                           ("adaptive-loss.png", "fig04b-targeted-loss.png")]:
        path = IMAGES / name
        source = EVIDENCE / "examples" / original
        assert path.read_bytes() == source.read_bytes()
        record(path, str(source.relative_to(ROOT)), "unchanged copy")
    # Media is rendered separately from retained recordings; preserve its records.
    media_provenance = BLOG / "media-provenance.json"
    media_fields = {}
    if media_provenance.exists():
        media = json.loads(media_provenance.read_text())
        for item in media["assets"]:
            assert hashlib.sha256((SITE / item["site_path"]).read_bytes()).hexdigest() == item["sha256"]
            RECORDS.append(item)
        media_fields = {"media_renderer": "docs/blog/eventground-wam/source/render_main_study_media.py",
                        "media_renderer_sha256": media["renderer_sha256"]}
    (BLOG / "asset-manifest.json").write_text(json.dumps({
        "created": "2026-11-03", "updated": "2026-11-05",
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_label_map": "source-label-map.json",
        "article_image_urls": "Image and video references include the first 12 SHA-256 characters as a version query.",
        "table_cells_verified_unchanged": checked, "assets": RECORDS, **media_fields,
    }, indent=2)+"\n")
    version_article_images()
    print(json.dumps({"generated_assets": len(RECORDS), "unchanged_table_cells": checked}))


if __name__ == "__main__":
    main()
