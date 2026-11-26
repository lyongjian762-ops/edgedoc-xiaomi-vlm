"""Check the board outputs and summarize the Q4 up-projection benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROWS = (1, 8, 32, 128, 268)
INPUT_WIDTH = 1024
OUTPUT_WIDTH = 3584


def relative_error(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def read_matrix(path, rows):
    matrix = np.fromfile(path, dtype="<f4").astype(np.float64)
    if matrix.size != rows * OUTPUT_WIDTH or not np.isfinite(matrix).all():
        raise ValueError(f"unexpected output data in {path}")
    return matrix.reshape(rows, OUTPUT_WIDTH)


def reference_from_q4(operands, manifest):
    raw = (operands / "up_proj.mnn.weight").read_bytes()
    weight_bytes, scale_bytes, *_ = manifest["quantized_weights"]["external_sizes"]
    if raw[0] != 2 or raw[5] != 16:
        raise ValueError("expected the two-dimensional, four-bit MNN weight format")
    if tuple(np.frombuffer(raw, dtype="<u2", count=2, offset=1)) != (OUTPUT_WIDTH, INPUT_WIDTH):
        raise ValueError("unexpected packed weight dimensions")
    # The exporter writes two lookup-table indices per byte, high nibble first.
    table = np.frombuffer(raw, dtype=np.int8, count=16, offset=6)
    packed = np.frombuffer(raw, dtype=np.uint8, count=weight_bytes - 22, offset=22)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[::2] = packed >> 4
    codes[1::2] = packed & 15
    group_count = manifest["quantized_weights"]["quant"]["readType"] // OUTPUT_WIDTH
    alpha = np.frombuffer(raw, dtype="<f2", count=scale_bytes // 2, offset=weight_bytes).astype(np.float32)
    alpha = alpha.reshape(OUTPUT_WIDTH, group_count, 2)
    quantized = table[codes].reshape(OUTPUT_WIDTH, group_count, INPUT_WIDTH // group_count)
    # Each group stores an offset and a scale in FP16.
    weight = quantized * alpha[:, :, 1, None] + alpha[:, :, 0, None]
    inputs = np.fromfile(operands / "input.f32", dtype="<f4").reshape(ROWS[-1], INPUT_WIDTH)
    return (inputs @ weight.reshape(OUTPUT_WIDTH, INPUT_WIDTH).T).astype(np.float64)


def read_run(path, row_counts, reference, q4_reference):
    process = json.loads((path / "process.json").read_text())
    if process["returncode"] != 0 or process["timed_out"]:
        raise ValueError(f"benchmark did not finish: {path}")
    if process["opened_mali_device"] != ("opencl" in path.name):
        raise ValueError(f"unexpected Mali device observation in {path}")
    with (path / "samples.csv").open() as stream:
        load_values = (row["gpu_load"].split("@")[0] for row in csv.DictReader(stream))
        peak_gpu_load = max((int(value) for value in load_values if value.isdigit()), default=0)
    with (path / "timing.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    measured = {}
    outputs = {}
    for size in row_counts:
        rows = [item for item in samples if int(item["rows"]) == size]
        if [int(item["iteration"]) for item in rows] != list(range(1, 31)):
            raise ValueError(f"missing timings for {size} rows in {path}")
        values = [float(item["wall_us"]) for item in rows]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(f"invalid timing values in {path}")
        matrix = read_matrix(path / f"rows-{size}.f32", size)
        outputs[size] = matrix
        measured[size] = {
            "samples_us": values,
            "relative_l2_vs_original_bf16_matmul": relative_error(matrix, reference[-size:]),
            "relative_l2_vs_dequantized_q4_matmul": relative_error(matrix, q4_reference[-size:]),
        }
    return measured, outputs, {
        "process_wall_ms": process["process_wall_ms"],
        "peak_process_rss_kib": process["peak_process_rss_kib"],
        "opened_mali_device": process["opened_mali_device"],
        "peak_sampled_gpu_load_percent": peak_gpu_load,
    }


def timing_stats(samples, rows):
    return {
        "median_us": float(np.median(samples)),
        "p95_us": float(np.percentile(samples, 95)),
        "median_us_per_row": float(np.median(samples)) / rows,
        "samples": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    operands = args.evidence / "operands"
    manifest = json.loads((operands / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((operands / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"operand checksum mismatch: {name}")
    reference = read_matrix(operands / "reference-bf16-weights.f32", ROWS[-1])
    q4_reference = reference_from_q4(operands, manifest)
    runs = args.evidence / "runs"
    observed = {}
    for mode in ("cpu", "opencl"):
        observed[mode] = [read_run(runs / f"verified-{round_id}-{mode}", ROWS, reference, q4_reference)
                          for round_id in ("a", "b")]
    by_rows = {}
    for size in ROWS:
        row_report = {}
        for mode, pairs in observed.items():
            samples = [us for measured, _, _ in pairs for us in measured[size]["samples_us"]]
            row_report[mode] = {
                **timing_stats(samples, size),
                "round_medians_us": [float(np.median(t[size]["samples_us"])) for t, _, _ in pairs],
                "max_relative_l2_vs_original_bf16_matmul": max(
                    t[size]["relative_l2_vs_original_bf16_matmul"] for t, _, _ in pairs
                ),
                "max_relative_l2_vs_dequantized_q4_matmul": max(
                    t[size]["relative_l2_vs_dequantized_q4_matmul"] for t, _, _ in pairs
                ),
            }
        row_report["cpu_vs_opencl_relative_l2"] = [
            relative_error(gpu[1][size], cpu[1][size])
            for cpu, gpu in zip(observed["cpu"], observed["opencl"])
        ]
        by_rows[size] = row_report
    tuner = {}
    for mode in ("opencl", "opencl-heavy"):
        paths = (runs / f"single-{mode}", runs / f"verified-single-{mode}")
        pairs = [read_run(path, (1,), reference, q4_reference) for path in paths]
        tuner[mode] = {
            **timing_stats([us for t, _, _ in pairs for us in t[1]["samples_us"]], 1),
            "round_medians_us": [float(np.median(t[1]["samples_us"])) for t, _, _ in pairs],
            "processes": [r[2] for r in pairs],
            "max_relative_l2_vs_default_opencl": max(
                relative_error(output[1], observed["opencl"][0][1][1]) for _, output, _ in pairs
            ),
        }
    report = {
        "fixture": manifest,
        "timed_scope": "MNN onForward through host output readMap, after five warmups for each shape",
        "by_rows": by_rows,
        "board_runs": {mode: [r[2] for r in rounds] for mode, rounds in observed.items()},
        "single_row_tuning": tuner,
    }
    (args.evidence / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for size, row in by_rows.items():
        print(size, *(round(row[mode]["median_us"], 2) for mode in ("cpu", "opencl")),
              "max CPU/OpenCL relative error", round(max(row["cpu_vs_opencl_relative_l2"]), 6))
        if any(row[mode]["max_relative_l2_vs_original_bf16_matmul"] > 0.15 for mode in ("cpu", "opencl")):
            raise ValueError(f"unexpected deviation from original BF16 projection for {size} rows")
        if any(row[mode]["max_relative_l2_vs_dequantized_q4_matmul"] > 0.06 for mode in ("cpu", "opencl")):
            raise ValueError(f"unexpected deviation from dequantized Q4 projection for {size} rows")
        if max(row["cpu_vs_opencl_relative_l2"]) > 0.05:
            raise ValueError(f"CPU and OpenCL disagree for {size} rows")
    print("Single-row tuning", tuner)


if __name__ == "__main__":
    main()
