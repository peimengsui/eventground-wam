"""Bounded Week 10 original-source matching; no simulator or label generation.

Only complete, equal-length source recordings are considered. Lengths shared
with a held-out group are excluded before either candidate's payload is read.
The exclusion applies to this full-sequence hypothesis; it does not establish
provenance for cropped, filtered, resampled or regenerated demonstrations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.event_supervision import compare_recorded_trajectory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def numeric_digest(*arrays):
    result = hashlib.sha256()
    for a in arrays:
        value = np.ascontiguousarray(a, dtype="<f4")
        result.update(str(value.shape).encode() + b"\0")
        result.update(value.tobytes())
    return result.hexdigest()


def heldout_collision(candidate, groups):
    return any(
        g["split"] != "train"
        and g["task_key"] == candidate["task_key"]
        and g["episode_length"] == candidate["frame_count"]
        for g in groups
    )


def select_source_demos(data, names, n, strategy="dataset_shapes"):
    if strategy == "dataset_shapes":
        schemas = {
            name: {k: list(data[name][k].shape) for k in ["actions", "states"]} for name in names
        }
        matches = [name for name in names if schemas[name]["actions"] == [n, 7]]
    elif strategy == "num_samples_attribute":
        schemas = {
            name: {
                "declared_num_samples": int(data[name].attrs["num_samples"]),
                "length_source": "num_samples_attribute",
            }
            for name in names
        }
        matches = [name for name in names if schemas[name]["declared_num_samples"] == n]
        if any(data[name]["actions"].shape != (n, 7) for name in matches):
            raise ValueError("Declared source length disagrees with actual action shape")
    else:
        raise ValueError("Unsealed source metadata strategy")
    return schemas, matches


class BoundedRange(io.RawIOBase):
    """Exact HTTP reads, logged as offset/length/hash, with no data prefetch."""

    def __init__(self, url, output, limits):
        super().__init__()
        self.url, self.output, self.limits = url, output, limits
        self.position = 0
        self.size = None
        self.downloaded = self.requests = 0
        self.started = time.monotonic()
        self.phase = "metadata"
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Bad seek origin")
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def fetch(self, start, count):
        if time.monotonic() - self.started > self.limits["per_file_seconds"]:
            raise TimeoutError("Per-file wall limit reached")
        if (start, count) in self.cache:
            return self.cache[(start, count)]
        if count < 1 or self.downloaded + count > self.limits["per_file_http_bytes"]:
            raise ValueError("Per-file download bound")
        if self.requests >= self.limits["per_file_http_requests"]:
            raise ValueError("Per-file request bound")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{start + count - 1}",
                "Accept-Encoding": "identity",
            },
        )
        self.requests += 1
        with urllib.request.urlopen(
            request, timeout=self.limits["http_timeout_seconds"]
        ) as response:
            if response.status != 206:
                raise ValueError("Range ignored; full-file download forbidden")
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match or int(match[1]) != start or int(match[2]) != start + count - 1:
                raise ValueError("Wrong HTTP byte range")
            size = int(match[3])
            if self.size is not None and self.size != size:
                raise ValueError("Source size changed")
            self.size = size
            data = response.read(count + 1)
        self.downloaded += len(data)
        if len(data) != count:
            raise ValueError("Truncated/oversized HTTP response")
        with (self.output / "http-ranges.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "phase": self.phase,
                        "offset": start,
                        "length": count,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                + "\n"
            )
        self.cache[(start, count)] = data
        return data

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        data = self.fetch(self.position, size)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def inspect_one(job):
    candidate, contract, groups, output = job
    rank = candidate["queue_rank"]
    output = Path(output) / f"candidate-{rank:02d}"
    output.mkdir(exist_ok=False)
    result = {
        "candidate": candidate,
        "status": "started",
        "comparisons": [],
        "target_payload_reads": 0,
        "original_demo_payload_reads": 0,
        "full_simulator_state_reads": 0,
        "physical_scene_identity_verified": False,
        "labels_admitted": False,
    }
    write_json(
        output / "started.json", {"pilot_digest": contract["digest"], "candidate": candidate}
    )
    reader = None
    try:
        gid = candidate["group_digest"]
        group = next(g for g in groups if g["group_digest"] == gid)
        episode = candidate["source_episode_index"]
        if group["split"] != "train" or episode not in group["episode_indices"]:
            raise ValueError("Candidate is not in the sealed training group")
        if heldout_collision(candidate, groups):
            result["status"] = "excluded_heldout_task_length_collision"
            return result
        if shutil.disk_usage(output).free < contract["resources"]["minimum_free_bytes"]:
            raise ValueError("Free disk floor")
        reader = BoundedRange(
            candidate["candidate_original_hdf5_url"], output, contract["resources"]
        )
        reader.fetch(0, 1)
        with h5py.File(reader, "r") as source:
            data = source["data"]
            names = sorted(data.keys(), key=lambda n: int(n.split("_")[-1]))
            if len(names) > contract["resources"]["maximum_demos_per_file"]:
                raise ValueError("Source demo-count bound")
            n = candidate["frame_count"]
            schemas, matches = select_source_demos(
                data, names, n, contract.get("metadata_length_strategy", "dataset_shapes")
            )
            write_json(output / "source-schema.json", schemas)
            result["source_demo_count"] = len(names)
            result["equal_length_demo_ids"] = matches
            if not matches:
                result["status"] = "no_equal_length_original_demo"
                return result
            if len(matches) > contract["resources"]["maximum_payload_candidates_per_target"]:
                result["status"] = "excluded_excess_equal_length_ambiguity"
                return result
            target_path = Path(contract["target_snapshot"]) / (
                f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
            )
            import pyarrow.parquet as pq

            table = pq.read_table(
                target_path,
                columns=["observation.state", "action", "episode_index", "frame_index"],
                use_threads=False,
            )
            result["target_payload_reads"] = 1
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if (
                len(states) != n
                or set(table["episode_index"].to_pylist()) != {episode}
                or table["frame_index"].to_pylist() != list(range(n))
            ):
                raise ValueError("Pinned episode/frame identity mismatch")
            digest = numeric_digest(states, actions)
            if digest not in group["numeric_digests"]:
                raise ValueError("Pinned numerical identity mismatch")
            result["target_numeric_digest"] = digest
            result["target_parquet_sha256"] = sha(target_path)
            np.savez_compressed(output / "target-numeric.npz", states=states, actions=actions)
            for name in matches:
                demo = data[name]
                if (
                    demo["obs/ee_pos"].shape != (n, 3)
                    or demo["obs/ee_ori"].shape != (n, 3)
                    or demo["obs/gripper_states"].shape != (n, 2)
                ):
                    raise ValueError("Original proprioception schema mismatch")
                reader.phase = f"numeric:{name}"
                result["original_demo_payload_reads"] += 1
                source_actions = demo["actions"][()]
                source_states = np.concatenate(
                    [
                        demo["obs/ee_pos"][()],
                        demo["obs/ee_ori"][()],
                        demo["obs/gripper_states"][()],
                    ],
                    axis=1,
                )
                np.savez_compressed(
                    output / f"{name}-numeric.npz", states=source_states, actions=source_actions
                )
                comparison = compare_recorded_trajectory(
                    states=states,
                    actions=actions,
                    source_states=source_states,
                    source_actions=source_actions,
                    source_frames=np.arange(n),
                    evidence_origin="recorded_simulator_states",
                )
                row = {
                    "demo_id": name,
                    "alignment": comparison,
                    "source_numeric_digest": numeric_digest(source_states, source_actions),
                    "state_channel_max_errors": np.max(
                        np.abs(states - source_states), axis=0
                    ).tolist(),
                    "action_channel_max_errors": np.max(
                        np.abs(actions - source_actions), axis=0
                    ).tolist(),
                }
                if comparison["numeric_alignment_passed"]:
                    reader.phase = f"physical_evidence:{name}"
                    full = demo["states"][()]
                    if full.ndim != 2 or len(full) != n or not np.isfinite(full).all():
                        raise ValueError("Invalid recorded full states")
                    result["full_simulator_state_reads"] += 1
                    np.save(output / f"{name}-simulator-states.npy", full, allow_pickle=False)
                    xml = demo.attrs["model_file"]
                    if isinstance(xml, bytes):
                        xml = xml.decode()
                    (output / f"{name}-model.xml").write_text(xml)
                    attrs = {}
                    for key in [
                        "env_args",
                        "bddl_file_name",
                        "bddl_file_content",
                        "problem_info",
                        "macros_image_convention",
                    ]:
                        if key in data.attrs:
                            value = data.attrs[key]
                            attrs[key] = value.decode() if isinstance(value, bytes) else str(value)
                    write_json(output / f"{name}-environment.json", attrs)
                    row["physical_metadata_recovered"] = True
                    row["physical_scene_identity_verified"] = False
                    row["observation_state_phase"] = "unverified_requires_independent_check"
                result["comparisons"].append(row)
            passed = sum(r["alignment"]["numeric_alignment_passed"] for r in result["comparisons"])
            result["status"] = (
                "unique_numeric_match_scene_unverified"
                if passed == 1
                else "ambiguous_numeric_matches"
                if passed > 1
                else "numeric_mismatch"
            )
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        result["status"] = "access_or_integrity_failure"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if reader is not None:
            result["http"] = {
                "bytes": reader.downloaded,
                "requests": reader.requests,
                "seconds": time.monotonic() - reader.started,
                "source_size": reader.size,
            }
            reader.close()
        result["evidence_sha256"] = {
            p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
        }
        write_json(output / "result.json", seal_contract(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.bundle / "contract.json").read_text())
    if contract_digest(contract) != contract["digest"]:
        raise ValueError("Contract identity")
    for name, expected in contract["bundle_files_sha256"].items():
        if sha(args.bundle / name) != expected:
            raise ValueError(f"Bundle file changed: {name}")
    groups = [json.loads(x) for x in (args.bundle / "groups.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(
        args.output / "started.json",
        {"contract_digest": contract["digest"], "pid": os.getpid(), "started_unix": time.time()},
    )
    jobs = [(c, contract, groups, str(args.output)) for c in contract["candidates"]]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=contract["resources"]["workers"]
    ) as pool:
        futures = [pool.submit(inspect_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "done": len(rows),
                        "rank": row["candidate"]["queue_rank"],
                        "status": row["status"],
                        "http": row.get("http"),
                    }
                ),
                flush=True,
            )
    result = seal_contract(
        {
            "artifact_type": "eventground_wam.source_matching_pilot_results",
            "schema_version": 1,
            "contract_digest": contract["digest"],
            "candidate_results": sorted(rows, key=lambda r: r["candidate"]["queue_rank"]),
            "wall_seconds": time.monotonic() - started,
            "training_launched": False,
            "labels_admitted": False,
            "simulator_steps": 0,
            "heldout_target_payload_reads": 0,
        }
    )
    write_json(args.output / "results.json", result)


if __name__ == "__main__":
    main()
