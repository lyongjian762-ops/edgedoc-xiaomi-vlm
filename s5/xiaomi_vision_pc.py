"""Export Xiaomi-OCR-0's vision tower at one fixed resolution and compare its outputs on the PC."""

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


class FixedGrid(torch.nn.Module):
    def __init__(self, vision, grid):
        super().__init__()
        self.vision = vision
        self.register_buffer("grid_thw", grid)

    def forward(self, pixel):
        return self.vision(pixel, self.grid_thw)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vendor-export", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=448)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("rockchip_vision_export", args.vendor_export)
    vendor = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(vendor)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True).image_processor
    model = AutoModel.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    config = model.config.vision_config
    multiple = config.patch_size * config.spatial_merge_size
    if min(args.height, args.width) <= 0 or args.height % multiple or args.width % multiple:
        raise ValueError(f"image dimensions must be positive multiples of {multiple}")
    vision = vendor.qwen3_5_vl_vision(model, batch_size=1).to(dtype=torch.float32).eval()
    del model
    gc.collect()

    with Image.open(args.image) as source:
        page = source.convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)
    image_processor_output = processor(images=page, do_resize=False, return_tensors="pt")
    grid = image_processor_output["image_grid_thw"]
    resize_grid = [[1, args.height // config.patch_size, args.width // config.patch_size]]
    if grid.tolist() != resize_grid:
        raise RuntimeError(f"unexpected image grid: {grid.tolist()}, wanted {resize_grid}")
    rgb = np.asarray(page, dtype=np.float32) * processor.rescale_factor
    rgb = (rgb - np.asarray(processor.image_mean, dtype=np.float32)) / np.asarray(processor.image_std, dtype=np.float32)
    pixel = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...]))
    fixed = FixedGrid(vision, grid).eval()

    with torch.no_grad():
        reference = vision.vpm(image_processor_output["pixel_values"].float(), grid).pooler_output.numpy()
        wrapped = fixed(pixel).numpy()
    np.testing.assert_allclose(wrapped, reference, atol=0.002, rtol=0.002)
    output_tokens = (args.height // multiple) * (args.width // multiple)
    expected_shape = (output_tokens, config.out_hidden_size)
    if wrapped.shape != expected_shape:
        raise RuntimeError(f"unexpected vision output: {wrapped.shape}, wanted {expected_shape}")
    print(f"PyTorch matched the vendor wrapper on the page: {wrapped.shape}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "vision.onnx"
    with torch.no_grad():
        torch.onnx.export(
            fixed, (pixel,), destination, input_names=["pixel"], output_names=["image_features"],
            opset_version=17, dynamo=False,
        )
    del fixed, vision
    gc.collect()

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(destination), sess_options=session_options, providers=["CPUExecutionProvider"])
    output, = session.run(None, {"pixel": pixel.numpy()})
    np.testing.assert_allclose(output, reference, atol=0.02, rtol=0.005)
    np.save(args.output_dir / "pixel.npy", pixel.numpy())
    np.save(args.output_dir / "reference.npy", reference)
    report = {
        "checkpoint": args.model_dir.name,
        "input_image": str(args.image), "input_shape": list(pixel.shape), "grid_thw": grid.tolist(),
        "output_shape": list(reference.shape), "output_dtype": str(reference.dtype),
        "vendor_wrapper_max_abs_error": float(np.max(np.abs(wrapped - reference))),
        "onnx_max_abs_error": float(np.max(np.abs(output - reference))),
        "onnx_opset": 17, "onnx_bytes": destination.stat().st_size,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
