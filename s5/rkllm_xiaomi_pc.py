"""Build the Xiaomi-OCR-0 language component with Rockchip's RKLLM toolkit."""

import argparse
import json
from pathlib import Path
import time

import torch
from rkllm.api import RKLLM


def check(stage, status, started):
    seconds = time.monotonic() - started
    print(f"{stage}: status={status}, elapsed={seconds:.1f}s", flush=True)
    if status != 0:
        raise RuntimeError(f"{stage} failed with status {status}")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    runtime = RKLLM()

    started = time.monotonic()
    status = runtime.load_huggingface(model=str(args.model_dir.resolve()), device="cpu", dtype="float32")
    load_seconds = check("load weights", status, started)

    started = time.monotonic()
    status = runtime.build(
        do_quantization=True, optimization_level=1, quantized_dtype="w8a8", quantized_algorithm="normal",
        target_platform="rk3588", num_npu_core=3, dataset=str(args.dataset.resolve()), max_context=4096,
    )
    build_seconds = check("build W8A8", status, started)

    destination = args.output_dir / "xiaomi_ocr_0_w8a8_rk3588.rkllm"
    started = time.monotonic()
    export_seconds = check("export", runtime.export_rkllm(str(destination.resolve())), started)
    report = {
        "checkpoint": args.model_dir.name, "target_platform": "rk3588", "quantization": "w8a8",
        "calibration_samples": len(json.loads(args.dataset.read_text(encoding="utf-8"))),
        "context_limit": 4096, "load_seconds": load_seconds, "build_seconds": build_seconds,
        "export_seconds": export_seconds, "rkllm_bytes": destination.stat().st_size,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
