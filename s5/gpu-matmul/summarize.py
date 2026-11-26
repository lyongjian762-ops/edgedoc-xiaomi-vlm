"""Check the returned board matrices and summarize the QKV benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(actual, expected):
    error = actual - expected
    return {
        "max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(expected)),
    }


def load_output(path):
    values = np.fromfile(path, dtype="<f4").astype(np.float64)
    if values.size != 784 * 2304 or not np.isfinite(values).all():
        raise ValueError(f"expected 784×2304 finite FP32 values in {path}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    operands = args.evidence / "operands"
    board = args.evidence / "board"
    manifest = json.loads((operands / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        file_bytes = (operands / name).read_bytes()
        if hashlib.sha256(file_bytes).hexdigest() != item["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")
    reference = load_output(operands / "reference.f32")
    cpu = load_output(board / "cpu.f32")
    gpu = load_output(board / "gpu.f32")
    errors = {
        "cpu_vs_pytorch": compare(cpu, reference),
        "gpu_vs_pytorch": compare(gpu, reference),
        "gpu_vs_cpu": compare(gpu, cpu),
    }
    thresholds = {"max_abs": 1e-3, "relative_l2": 1e-5}
    passed = all(error[metric] <= limit for error in errors.values() for metric, limit in thresholds.items())

    with (board / "timing.csv").open() as stream:
        timing = list(csv.DictReader(stream))
    if [int(row["iteration"]) for row in timing] != list(range(1, 16)):
        raise ValueError("expected all 15 measured CPU/GPU pairs")
    timings_ms = {}
    for key in timing[0]:
        if key.endswith("_ms"):
            samples = [float(row[key]) for row in timing]
            if not np.isfinite(samples).all() or min(samples) <= 0:
                raise ValueError(f"invalid {key}")
            timings_ms[key] = {
                "median": float(np.median(samples)),
                "min": min(samples),
                "max": max(samples),
                "p95": float(np.percentile(samples, 95)),
            }

    with (board / "resources.csv").open() as stream:
        resources = list(csv.DictReader(stream))
    available = [int(row["mem_available_kib"]) for row in resources]
    report = {
        "qkv_layer": manifest["qkv_layer"],
        "dimensions_mkn": [784, 768, 2304],
        "operation_count": 2 * 784 * 768 * 2304,
        "three_tensor_bytes": (784 * 768 + 768 * 2304) * 2 + 784 * 2304 * 4,
        "measured_pairs": len(timing),
        "timing_ms": timings_ms,
        "gpu_wall_over_cpu": timings_ms["gpu_wall_ms"]["median"] / timings_ms["cpu_ms"]["median"],
        "correctness": {"passed": passed, "thresholds": thresholds, "errors": errors},
        "resources": {
            "before_board_mem_available_kib": available[0],
            "minimum_board_mem_available_kib": min(available),
            "after_board_mem_available_kib": available[-1],
            "max_soc_c": max(int(row["soc_millic"]) for row in resources) / 1000,
            "max_gpu_c": max(int(row["gpu_millic"]) for row in resources) / 1000,
            "process": json.loads((board / "process.json").read_text()),
        },
        "runtime_log": (board / "benchmark.log").read_text().splitlines(),
    }
    report_text = json.dumps(report, indent=2) + "\n"
    (args.evidence / "summary.json").write_text(report_text)
    print(report_text, end="")
    if not passed:
        raise SystemExit("numerical comparison exceeds the configured tolerance")


if __name__ == "__main__":
    main()
