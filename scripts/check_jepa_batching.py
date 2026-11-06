#!/usr/bin/env python3
"""Investigate the retained v1 batch-dependence failure on its saved RGB inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import sha256_file, write_json


def main(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    first = torch.load(args.run/"first-observation-check.pt", map_location="cpu", weights_only=True)
    images = first["rgb"].numpy().reshape(-1, 256, 256, 3)
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True).image_processor
    def process(frames):
        return processor(images=[np.ascontiguousarray(x) for x in frames], do_resize=False,
                         do_rescale=True, do_normalize=True, do_convert_rgb=True, return_tensors="pt")
    all_encoded = process(images)
    singles = [process(images[i:i+1]) for i in range(len(images))]
    preprocessing_error = float((all_encoded["pixel_values"]-
                                 torch.cat([x["pixel_values"] for x in singles])).abs().max())
    model = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True)
    visual = model.model.visual.requires_grad_(False).eval().to("cuda")
    del model
    @torch.no_grad()
    def encode(encoded):
        return visual(encoded["pixel_values"].to("cuda", torch.bfloat16),
                      grid_thw=encoded["image_grid_thw"].to("cuda"), return_dict=True).pooler_output.cpu()
    batched = encode(all_encoded)
    single = torch.cat([encode(x) for x in singles])
    result = seal_contract({"artifact_type": "eventground_wam.jepa_batching_investigation",
                            "script_sha256": sha256_file(Path(__file__)),
                            "input_sha256": sha256_file(args.run/"first-observation-check.pt"),
                            "preprocessing_batch_vs_single_max_error": preprocessing_error,
                            "saved_batched_raw_replay_max_error": float((batched-first["raw"].flatten(0, 1)).abs().max()),
                            "raw_batch_vs_single_max_error": float((batched-single).abs().max()),
                            "single_image_repeat_max_error": float((single[:64]-encode(singles[0])).abs().max()),
                            "interpretation": "Localizes differences to the vision execution path when preprocessing is identical; does not identify the underlying kernel or prove cross-image attention."})
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home()/"data/eventground-wam/huggingface-cache")
    main(parser.parse_args())
