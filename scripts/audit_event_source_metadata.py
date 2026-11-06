"""Read only public container schemas, with bounded HTTP range requests.

Never read HDF5 episode arrays or RLDS examples. No downloaded code is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import h5py


class RangeReader(io.RawIOBase):
    """Exact-range file object: no speculative prefetch of episode payloads."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.position = 0
        self.downloaded = 0
        self.requests = 0
        self.started = time.monotonic()
        self.size = None
        self._range(0, 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def _range(self, start, count):
        if (
            count < 1
            or self.downloaded + count > 8 * 1024**2
            or self.requests >= 1500
            or time.monotonic() - self.started > 600
        ):
            raise ValueError("Metadata audit resource limit")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{start + count - 1}",
                "Accept-Encoding": "identity",
            },
        )
        self.requests += 1
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise ValueError("Server ignored bounded Range; no full download permitted")
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match or int(match[1]) != start or int(match[2]) != start + count - 1:
                raise ValueError("Unexpected Content-Range")
            size = int(match[3])
            if self.size is not None and size != self.size:
                raise ValueError("Remote source size changed")
            self.size = size
            payload = response.read(count + 1)
        self.downloaded += len(payload)
        if len(payload) != count:
            raise ValueError("Truncated or oversized Range response")
        return payload

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        payload = self._range(self.position, size)
        self.position += len(payload)
        return payload

    def readinto(self, buffer):
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["original_hdf5", "plus_rlds_zip"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    urls = {
        "original_hdf5": "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        "plus_rlds_zip": "https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite/resolve/1a0f5c97a96e4187fbe52331b9a484f21d244bb2/rlds/libero_spatial.zip",
    }
    result = {
        "source_url": urls[args.kind],
        "kind": args.kind,
        "episode_dataset_reads": 0,
        "labels_generated": 0,
    }
    reader = None
    try:
        reader = RangeReader(urls[args.kind])
        if args.kind == "original_hdf5":
            with h5py.File(reader, "r") as handle:
                data = handle["data"]
                names = sorted(data.keys())
                demo = data[names[0]]
                result.update(
                    {
                        "demo_count": len(names),
                        "data_attribute_names": sorted(data.attrs),
                        "sample_demo": names[0],
                        "demo_attribute_names": sorted(demo.attrs),
                        "datasets": {
                            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                            for k, v in demo.items()
                            if isinstance(v, h5py.Dataset)
                        },
                        "observation_datasets": {
                            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                            for k, v in demo.get("obs", {}).items()
                        },
                        "full_state_dataset_present": "states" in demo,
                        "model_xml_attribute_present": "model_file" in demo.attrs,
                        "identity_with_pinned_plus_demonstrations": "not_established",
                    }
                )
        else:
            with zipfile.ZipFile(reader) as archive:
                names = archive.namelist()
                metadata = [n for n in names if n.endswith(("features.json", "dataset_info.json"))]
                result["archive_entry_count"] = len(names)
                result["metadata"] = {}
                for name in metadata:
                    entry = archive.getinfo(name)
                    if entry.file_size > 1024**2:
                        raise ValueError("Metadata member too large")
                    payload = archive.read(name)
                    result["metadata"][name] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "content": json.loads(payload),
                    }
        result["status"] = "metadata_inspected"
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        result["status"] = "metadata_access_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if reader is not None:
            result.update(
                {
                    "container_bytes": reader.size,
                    "http_bytes_read": reader.downloaded,
                    "http_requests": reader.requests,
                    "elapsed_seconds": time.monotonic() - reader.started,
                }
            )
            reader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
