"""Verify and summarize the exploratory pilot; never promote its checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest, seal_contract


def read(path):
    data = json.loads(path.read_text())
    if contract_digest(data) != data["digest"]:
        raise ValueError(f"Unsealed artifact: {path}")
    return data


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle
    run = bundle / "run"
    report = read(run / "completion.json")
    contract = read(bundle / "qualification-contract.json")
    dataset = read(bundle / "dataset.json")
    if (report["contract_digest"] != contract["digest"] or report["dataset_digest"] != dataset["digest"]
            or report["formal_training_authorized"] is not False or report["independent_qa_passed"] is not False):
        raise ValueError("Pilot identity or evidence claim changed")
    for name, expected in report["files_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or checksum(run/name) != expected:
            raise ValueError(f"Completed pilot output changed: {name}")
    metrics = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    scheduled = [r for r in metrics if r["kind"] == "scheduled"]
    replays = [r for r in metrics if r["kind"] == "recovery_replay"]
    if any(not math.isfinite(r[k]) for r in metrics for k in ("loss", "gradient_norm", "seconds")):
        raise ValueError("Nonfinite logged metric")
    limits = {
        "wall": report["wall_seconds"] <= contract["wall_seconds"],
        "qualification_budget": report["cumulative_qualification_seconds"] <= 3*3600,
        "output_bytes": report["output_bytes"] <= contract["resources"]["output_bytes"],
        "gpu_memory": report["peak_reserved_bytes"] <= .9*report["gpu_total_bytes"],
    }
    passed = report["status"] == "passed_exploratory_qualification"
    if passed:
        if (not all(report["gates"].values()) or not all(limits.values())
                or [r["global_step"] for r in scheduled] != list(range(1,65))
                or len(replays) != 1 or replays[0]["global_step"] != 9
                or report["executed_updates"] != 65 or report["logical_updates"] != 64
                or report["heldout_payload_reads"] != 0):
            raise ValueError("Completion claim disagrees with frozen schedule or limits")
        if any(scheduled[8][k] != replays[0][k] for k in ("loss", "gradient_norm", "prompts_seen")):
            raise ValueError("Recovery replay log differs")
    analysis = bundle / "analysis"
    analysis.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    measured = report["evaluations"]
    axes[0].plot([r["global_step"] for r in scheduled], [r["loss"] for r in scheduled],
                 color="#2463a5", label="Loss before each update")
    axes[0].plot([r["global_step"] for r in measured], [r["loss"] for r in measured],
                 "o", color="#142a46", label="Greedy-evaluation checkpoint loss")
    axes[0].set(xlabel="Logical optimizer update", ylabel="Structured SFT loss", title="Fixed 16-example fitting test")
    axes[0].legend(fontsize=9)
    for key, label, color in [("event_accuracy", "Event class", "#2463a5"),
                              ("joint_two_frame_accuracy", "Class + boundaries within 2 frames", "#c36a1c")]:
        axes[1].plot([r["global_step"] for r in measured], [100*r[key] for r in measured],
                     "o-", label=label, color=color)
    axes[1].set(xlabel="Logical optimizer update", ylabel="Agreement with weak targets (%)",
                ylim=(-3,103), title="Same training examples; no held-out evaluation")
    axes[1].legend(fontsize=9, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("Week 10 exploratory qualification — limited human spot-checking", fontsize=13)
    fig.savefig(analysis/"qualification-curve.png", dpi=160)
    fig.savefig(analysis/"qualification-curve.pdf")
    plt.close(fig)
    initial, final = measured[0], measured[-1]
    summary = seal_contract({"artifact_type": "eventground_wam.week10_weak_qualification_summary",
        "completion_digest": report["digest"], "contract_digest": contract["digest"],
        "status": report["status"], "resource_limits_passed": limits,
        "verified_completed_files": len(report["files_sha256"]), "coverage": dataset["coverage"],
        "initial_loss": initial["loss"], "final_loss": final["loss"],
        "loss_ratio": final["loss"]/initial["loss"], "final_event_accuracy": final["event_accuracy"],
        "final_joint_two_frame_accuracy": final["joint_two_frame_accuracy"],
        "final_exact_triple_accuracy": final["exact_triple_accuracy"],
        "physical_updates": report["executed_updates"], "logical_updates": report["logical_updates"],
        "independent_qa_passed": False, "formal_training_launched": False,
        "qualification_checkpoint_promotable": False,
        "heldout_payload_reads": 0, "packaging_script_sha256": checksum(Path(__file__)),
        "figures_sha256": {p.name: checksum(p) for p in analysis.iterdir()}})
    with (analysis/"summary.json").open("x") as stream:
        json.dump(summary,stream,sort_keys=True,indent=2);stream.write("\n")
    text = f"""# Week 10 exploratory qualification result

Status: **{report['status']}**. Fixed final logical update: {report['logical_updates']}.
There were {report['executed_updates']} physical optimizer executions, including one
checkpoint-recovery replay. No formal SFT/RLVR or held-out evaluation was launched.

| Fitting metric on the same 16 examples | Initial | Fixed endpoint |
| --- | ---: | ---: |
| Structured SFT loss | {initial['loss']:.6f} | {final['loss']:.6f} |
| Event-class agreement | {initial['event_accuracy']:.1%} | {final['event_accuracy']:.1%} |
| Class and boundaries within two frames | {initial['joint_two_frame_accuracy']:.1%} | {final['joint_two_frame_accuracy']:.1%} |
| Exact event/start/end agreement | {initial['exact_triple_accuracy']:.1%} | {final['exact_triple_accuracy']:.1%} |

Wall duration: {report['wall_seconds']/60:.2f} minutes. Peak reserved GPU memory:
{report['peak_reserved_bytes']/1024**3:.2f} GiB. Mean optimizer-execution duration:
{report['mean_seconds_per_update']:.3f} seconds. All frozen resource limits passed:
{all(limits.values())}. Bitwise recovery passed: {report['gates'].get('bitwise_checkpoint_recovery', False)}.

![Qualification curve](qualification-curve.png)

These are fitting and implementation checks against automatic targets, not
independent label validation or generalization evidence. The full conservative
pool has 1,332 unique prompts from 198/200 recordings and 40 tasks; only 29 are
grasp, all with lead times at most eight frames. The original 1,816 ambiguity
flags remain excluded from overlapping training windows. Human review covers
113 positive proposals from 20 episodes in four tasks; negative recall is unknown.

The fixed qualification checkpoint remains on Feifei and cannot initialize a
formal run. The next step is a separate weak-label SFT admission/launch contract
using a fresh M4 initialization. Full training is not yet launched. RLVR reward
qualification and held-out evaluation remain separate work.
"""
    (analysis/"RESULTS.md").write_text(text)
    print(json.dumps({k: summary[k] for k in ("status", "initial_loss", "final_loss", "final_event_accuracy",
                                            "final_joint_two_frame_accuracy", "logical_updates")}))


if __name__ == "__main__":
    main()
