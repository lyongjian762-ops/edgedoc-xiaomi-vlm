"""Check the isolated vocabulary projection and split timing at the readMap boundary."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


WIDTH = 1024
OUTPUTS = 248320


def reference_q4(operands, manifest, rows=1):
    raw = (operands / "lm_head.mnn.weight").read_bytes()
    weight_bytes, scale_bytes, *_ = manifest["quantized_weights"]["external_sizes"]
    quant = manifest["quantized_weights"]["quant"]
    assert quant["shapeInt32"] and raw[0] == 2 and raw[9] == 16
    assert tuple(np.frombuffer(raw, dtype="<u4", count=2, offset=1)) == (OUTPUTS, WIDTH)
    table = np.frombuffer(raw, dtype=np.int8, count=16, offset=10)
    packed = np.frombuffer(raw, dtype=np.uint8, count=weight_bytes - 26, offset=26).reshape(OUTPUTS, WIDTH // 2)
    group_count = quant["readType"] // OUTPUTS
    alpha = np.frombuffer(raw, dtype="<f2", count=scale_bytes // 2, offset=weight_bytes).reshape(OUTPUTS, group_count, 2)
    last = np.fromfile(operands / "input.f32", dtype="<f4").reshape(268, WIDTH)[-rows:]
    logits = np.empty((rows, OUTPUTS), dtype=np.float32)
    # Decode a small range of output channels at a time instead of allocating a full FP32 matrix.
    for start in range(0, OUTPUTS, 4096):
        end = min(start + 4096, OUTPUTS)
        data = packed[start:end]
        codes = np.empty((end - start, WIDTH), dtype=np.uint8)
        codes[:, ::2] = data >> 4
        codes[:, 1::2] = data & 15
        decoded = table[codes].reshape(end - start, group_count, WIDTH // group_count)
        scales = alpha[start:end].astype(np.float32)
        weight = decoded * scales[:, :, 1, None] + scales[:, :, 0, None]
        logits[:, start:end] = last @ weight.reshape(end - start, WIDTH).T
    return logits[0] if rows == 1 else logits


def relative_l2(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def stats(samples):
    values = np.asarray(samples, dtype=np.float64) / 1000
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "samples": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    operands = args.archive / "operands"
    manifest = json.loads((operands / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((operands / name).read_bytes()).hexdigest() == digest, name
    q4 = reference_q4(operands, manifest)
    bf16 = np.fromfile(operands / "reference-bf16.f32", dtype="<f4")
    assert len(bf16) == OUTPUTS
    np.asarray(q4, dtype="<f4").tofile(operands / "reference-q4.f32")

    result = {"fixture": manifest, "median_scope": "5 warmups, 30 samples per round, one row, 2 rounds per backend",
              "reference_q4_top_token": int(np.argmax(q4)), "reference_bf16_top_token": int(np.argmax(bf16)),
              "backends": {}}
    outputs = {}
    for backend in ("cpu", "opencl", "opencl-wait"):
        rounds = []
        measurements = {key: [] for key in ("wall_us", "forward_us", "readmap_us")}
        if backend == "opencl-wait":
            measurements["wait_us"] = []
        outputs[backend] = []
        for letter in ("a", "b"):
            label = f"wait-{letter}-opencl" if backend == "opencl-wait" else f"{letter}-{backend}"
            path = args.archive / "runs" / label
            process = json.loads((path / "process.json").read_text())
            assert process["returncode"] == 0 and not process["timed_out"], path
            assert process["opened_mali_device"] == (backend != "cpu"), path
            with (path / "timing.csv").open(newline="") as stream:
                records = list(csv.DictReader(stream))
            assert [int(row["iteration"]) for row in records] == list(range(1, 31))
            assert all(int(row["rows"]) == 1 for row in records)
            assert all(abs(float(row["wall_us"]) - float(row["forward_us"]) -
                           float(row.get("wait_us", 0)) - float(row["readmap_us"])) < 0.01 for row in records)
            for key, values in measurements.items():
                values.extend(float(row[key]) for row in records)
            actual = np.fromfile(path / "rows-1.f32", dtype="<f4")
            assert len(actual) == OUTPUTS and np.isfinite(actual).all()
            error = relative_l2(actual, q4)
            assert error < 0.08, (path, error)
            outputs[backend].append(actual)
            with (path / "samples.csv").open(newline="") as stream:
                resources = list(csv.DictReader(stream))
            loads = [int(row["gpu_load"].split("@", 1)[0]) for row in resources if row["gpu_load"]]
            rounds.append({"run": path.name, "wall": stats([float(row["wall_us"]) for row in records]),
                           "relative_l2_vs_q4": error, "relative_l2_vs_bf16": relative_l2(actual, bf16),
                           "top_token": int(np.argmax(actual)), "opened_mali_device": process["opened_mali_device"],
                           "peak_gpu_load_percent": max(loads, default=0),
                           "peak_process_rss_kib": process["peak_process_rss_kib"]})
        result["backends"][backend] = {"timing": {key: stats(values) for key, values in measurements.items()}, "rounds": rounds}
        print(backend, result["backends"][backend])
    result["cpu_vs_gpu_relative_l2"] = [relative_l2(cpu, gpu) for cpu, gpu in zip(outputs["cpu"], outputs["opencl"])]
    assert max(result["cpu_vs_gpu_relative_l2"]) < 0.08
    result["gpu_wait_vs_default_relative_l2"] = [relative_l2(normal, waited)
                                                for normal, waited in zip(outputs["opencl"], outputs["opencl-wait"])]
    assert max(result["gpu_wait_vs_default_relative_l2"]) < 0.08
    (args.archive / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
