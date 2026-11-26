"""Compile the fixed-resolution Xiaomi vision ONNX and compare its PC simulator output."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from rknn.api import RKNN

from rknn_smoke import require_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-input", choices=("float32", "uint8"), default="float32")
    args = parser.parse_args()

    graph = args.vision_dir / "vision.onnx"
    pixel = np.load(args.vision_dir / "pixel.npy", allow_pickle=False)
    reference = np.load(args.vision_dir / "reference.npy", allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "xiaomi_vision_rk3588.rknn"

    input_data = pixel
    layout = "nchw"
    preprocessing = {}
    if args.runtime_input == "uint8":
        # The official Xiaomi processor maps RGB bytes to [-1, 1].
        rgb = np.rint((pixel + 1) * 127.5).astype(np.uint8)
        np.testing.assert_allclose(rgb.astype(np.float32) / 127.5 - 1, pixel, rtol=0, atol=1e-6)
        input_data = np.ascontiguousarray(rgb.transpose(0, 2, 3, 1))
        layout = "nhwc"
        preprocessing = {"mean_values": [[127.5] * 3], "std_values": [[127.5] * 3]}
        import cv2

        if not cv2.imwrite(str(args.output_dir / "vision-input.png"), input_data[0, :, :, ::-1]):
            raise RuntimeError("could not save the RGB fixture")

    runtime = RKNN(verbose=False)
    try:
        require_ok("config", runtime.config(target_platform="rk3588", **preprocessing))
        require_ok("load_onnx", runtime.load_onnx(model=str(graph)))
        require_ok("build", runtime.build(do_quantization=False))
        require_ok("export_rknn", runtime.export_rknn(str(destination)))
        print(f"RKNN model built: {destination.stat().st_size} bytes", flush=True)
        require_ok("init_runtime (PC simulator)", runtime.init_runtime())
        candidate, = runtime.inference(inputs=[input_data], data_format=[layout])
    finally:
        runtime.release()

    if candidate.shape != reference.shape:
        raise RuntimeError(f"shape differs: {candidate.shape} vs {reference.shape}")
    if not np.isfinite(candidate).all():
        raise RuntimeError("simulator output includes non-finite numbers")
    candidate = candidate.astype(np.float64, copy=False)
    reference = reference.astype(np.float64, copy=False)
    error = candidate - reference
    norms = np.linalg.norm(candidate) * np.linalg.norm(reference)
    cosine = float(np.vdot(candidate, reference) / norms) if norms else 0.0
    report = {
        "target_platform": "rk3588", "execution": "PC simulator", "toolkit_version": version("rknn-toolkit2"),
        "input_shape": list(input_data.shape), "input_dtype": args.runtime_input, "input_layout": layout,
        "output_shape": list(candidate.shape),
        "max_abs_error": float(np.max(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "cosine_similarity": cosine, "cosine_threshold": 0.999, "rknn_bytes": destination.stat().st_size,
        "quantization_requested": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if cosine < report["cosine_threshold"]:
        raise RuntimeError("simulator output differs materially from the PyTorch reference")


if __name__ == "__main__":
    main()
