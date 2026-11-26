"""Capture the first Xiaomi vision QKV matrix multiply from a real 448px OCR input."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


class CapturedInput(Exception):
    """Stop the forward pass once the first QKV input has been captured."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True).image_processor
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    vision = model.visual
    qkv = vision.blocks[0].attn.qkv
    source_weight_dtype = str(qkv.weight.dtype)
    vision.float()
    with Image.open(args.image) as image:
        if image.size != (448, 448):
            parser.error("expected the existing 448×448 OCR fixture")
        inputs = processor(images=image.convert("RGB"), do_resize=False, return_tensors="pt")

    captured = None

    def capture(_layer, operands):
        nonlocal captured
        captured = operands[0].detach().to(torch.float16).contiguous()
        raise CapturedInput

    with torch.inference_mode(), qkv.register_forward_pre_hook(capture):
        try:
            vision(inputs["pixel_values"].float(), inputs["image_grid_thw"])
        except CapturedInput:
            pass
        if captured is None or tuple(captured.shape) != (784, 768):
            raise RuntimeError("expected 784 visual patch embeddings with width 768")
        weight = qkv.weight.detach().transpose(0, 1).to(torch.float16).contiguous()
        if tuple(weight.shape) != (768, 2304):
            raise RuntimeError("expected the first QKV weight with 2304 output channels")
        if not torch.isfinite(captured).all() or not torch.isfinite(weight).all():
            raise RuntimeError("input or weight overflows FP16")
        reference = captured.float() @ weight.float()

    args.output.mkdir(parents=True)
    arrays = {
        "x.f16": captured.numpy().astype("<f2", copy=False),
        "w_transposed.f16": weight.numpy().astype("<f2", copy=False),
        "reference.f32": reference.numpy().astype("<f4", copy=False),
    }
    files = {}
    for name, array in arrays.items():
        path = args.output / name
        array.tofile(path)
        files[name] = {
            "shape": list(array.shape), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report = {
        "checkpoint": args.model.name,
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "qkv_layer": "model.visual.blocks.0.attn.qkv",
        "source_weight_dtype": source_weight_dtype,
        "benchmark_operands": "Captured vision activations and actual weight rounded to FP16",
        "reference": "FP32 PyTorch matrix multiply of the same FP16 operands, excluding QKV bias",
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
