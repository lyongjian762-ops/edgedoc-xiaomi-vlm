"""Validate both board rounds and summarize CPU, Mali and RKNPU QKV timings."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize import compare, load_output


BACKENDS = ("tile16", "tile32", "clblast", "npu-core0")
ROUNDS = ("round-a", "round-b")
THRESHOLDS = {"max_abs": 1e-3, "relative_l2": 1e-5}


def stats(values):
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("timing values must be finite and positive")
    return {
        "median": float(np.median(values)),
        "min": min(values),
        "max": max(values),
        "p95": float(np.percentile(values, 95)),
    }


def read_run(path, reference, backend):
    process = json.loads((path / "process.json").read_text())
    if process["returncode"] != 0:
        raise ValueError(f"incomplete benchmark at {path}")
    with (path / "timing.csv").open() as stream:
        timing = list(csv.DictReader(stream))
    if [int(row["iteration"]) for row in timing] != list(range(1, 16)):
        raise ValueError(f"expected 15 timing rows at {path}")
    samples = {
        key: [float(row[key]) for row in timing]
        for key in timing[0] if key.endswith("_ms")
    }
    expected_files = ("npu.f32",) if backend == "npu-core0" else ("cpu.f32", "gpu.f32")
    errors = {}
    for name in expected_files:
        result = path / name
        error = compare(load_output(result), reference)
        passed = all(error[metric] <= limit for metric, limit in THRESHOLDS.items())
        errors[name] = {
            **error,
            "passed": passed,
            "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        }
    with (path / "resources.csv").open() as stream:
        resources = list(csv.DictReader(stream))
    return samples, {
        "timing_ms": {key: stats(values) for key, values in samples.items()},
        "correctness": errors,
        "benchmark_peak_rss_kib": process["benchmark_peak_rss_kib"],
        "minimum_board_mem_available_kib": min(int(row["mem_available_kib"]) for row in resources),
        "max_soc_c": max(int(row["soc_millic"]) for row in resources) / 1000,
        "max_gpu_c": max(int(row["gpu_millic"]) for row in resources) / 1000,
        "runtime_log": (path / "benchmark.log").read_text().splitlines(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("operands", type=Path, help="prior QKV operand directory including reference")
    args = parser.parse_args()
    manifest = json.loads((args.operands / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        if hashlib.sha256((args.operands / name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"operand checksum mismatch: {name}")
    reference = load_output(args.operands / "reference.f32")
    results = {}
    passed = True
    for backend in BACKENDS:
        samples_a, round_a = read_run(args.evidence / "runs" / ROUNDS[0] / backend, reference, backend)
        samples_b, round_b = read_run(args.evidence / "runs" / ROUNDS[1] / backend, reference, backend)
        errors_a = round_a["correctness"]
        errors_b = round_b["correctness"]
        for errors in (errors_a, errors_b):
            passed = passed and all(error["passed"] for error in errors.values())
        results[backend] = {
            "timing_ms": {key: stats(values + samples_b[key]) for key, values in samples_a.items()},
            "bitwise_equal_between_rounds": {
                key: error["sha256"] == errors_b[key]["sha256"] for key, error in errors_a.items()
            },
            "rounds": dict(zip(ROUNDS, (round_a, round_b))),
        }
    report = {
        "layer": manifest["qkv_layer"],
        "dimensions_mkn": [784, 768, 2304],
        "operand_hashes": {key: value["sha256"] for key, value in manifest["files"].items()},
        "measurements_per_backend": 30,
        "correctness_thresholds": THRESHOLDS,
        "correctness_passed": passed,
        "cpu_baseline": "OpenBLAS four Cortex-A76 cores, paired with tile16",
        "backends": results,
    }
    (args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for backend, result in results.items():
        medians = {key: round(value["median"], 4) for key, value in result["timing_ms"].items()}
        print(backend, medians, result["bitwise_equal_between_rounds"])
    if not passed:
        raise SystemExit("numerical comparison exceeds the configured tolerance")


if __name__ == "__main__":
    main()
