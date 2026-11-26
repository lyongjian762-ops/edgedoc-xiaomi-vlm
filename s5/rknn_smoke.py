"""Build the existing L08 convolution fixture for RK3588 and check it on the PC simulator."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def require_ok(stage, status):
    if status != 0:
        raise RuntimeError(f"{stage} returned {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    reference_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    model_input, = reference_session.get_inputs()
    sample = np.linspace(-1, 1, num=64, dtype=np.float32).reshape(1, 1, 8, 8)
    if model_input.shape != list(sample.shape):
        raise ValueError(f"expected the L08 input shape {sample.shape}, got {model_input.shape}")
    reference, = reference_session.run(None, {model_input.name: sample})

    destination = args.output_dir / "conv_bn_rk3588.rknn"
    runtime = RKNN(verbose=False)
    try:
        require_ok("config", runtime.config(target_platform="rk3588"))
        require_ok("load_onnx", runtime.load_onnx(model=str(args.onnx)))
        require_ok("build", runtime.build(do_quantization=False))
        require_ok("export_rknn", runtime.export_rknn(str(destination)))
        require_ok("init_runtime (PC simulator)", runtime.init_runtime())
        candidate, = runtime.inference(inputs=[sample], data_format=["nchw"])
    finally:
        runtime.release()

    np.testing.assert_allclose(candidate, reference, rtol=0, atol=0.003)
    report = {
        "target_platform": "rk3588",
        "execution": "PC simulator",
        "toolkit_version": version("rknn-toolkit2"),
        "onnxruntime_version": ort.__version__,
        "onnx": str(args.onnx),
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "input_shape_nchw": list(sample.shape),
        "input_values": "linspace(-1, 1, 64), float32",
        "output_shape": list(candidate.shape),
        "max_abs_error": float(np.max(np.abs(candidate - reference))),
        "absolute_error_limit": 0.003,
        "quantization_requested": False,
        "rknn_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rknn_bytes": destination.stat().st_size,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
