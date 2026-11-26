"""Validate every custom GEMV run and compare its kernel and wall times."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


OUTPUTS = 248320


def stats(values):
    assert len(values) == 60 and all(x >= 0 for x in values)
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("operands", type=Path)
    args = parser.parse_args()
    prepared = json.loads((args.archive / "prepared/manifest.json").read_text())
    original = json.loads((args.operands / "manifest.json").read_text())
    for name, digest in prepared["files"].items():
        assert hashlib.sha256((args.archive / "prepared" / name).read_bytes()).hexdigest() == digest, name
    assert prepared["source_weights_sha256"] == original["files"]["lm_head.mnn.weight"]
    assert prepared["source_input_sha256"] == original["files"]["input.f32"]
    reference = np.fromfile(args.operands / "reference-q4.f32", dtype="<f4")
    assert reference.size == OUTPUTS
    expected_top = int(reference.argmax())
    weight_payload = sum(original["quantized_weights"]["external_sizes"][:2]) - 26
    cases = [("mnn-cpu", None), ("mnn-opencl", None), ("row-fp32", 64)]
    cases += [(mode, local) for mode in ("tile-fp32", "tile-fp16") for local in (32, 64, 128)]
    results = {"date_asia_shanghai": "2026-11-26", "weight_payload_bytes": weight_payload,
               "reference_top_token": expected_top, "cases": {}}
    for mode, local in cases:
        name = mode if local is None else f"{mode}-{local}"
        measurement = {"wall_ms": []}
        if local is not None:
            measurement.update({key: [] for key in ("kernel_ms", "upload_ms", "readback_ms")})
        rounds = []
        for letter in ("a", "b"):
            label = f"{letter}-{mode}" if local is None or local == 64 else f"{letter}-{mode}-{local}"
            path = args.archive / "runs" / label
            process = json.loads((path / "process.json").read_text())
            assert process["returncode"] == 0 and not process["timed_out"], label
            assert process["opened_mali_device"] == (mode != "mnn-cpu"), label
            with (path / "timing.csv").open(newline="") as stream:
                records = list(csv.DictReader(stream))
            assert [int(row["iteration"]) for row in records] == list(range(1, 31)), label
            elapsed = [float(row["wall_us"]) / 1000 if local is None else float(row["wall_ms"]) for row in records]
            measurement["wall_ms"].extend(elapsed)
            for key in measurement:
                if key != "wall_ms":
                    measurement[key].extend(float(row[key]) for row in records)
            actual = np.fromfile(path / ("rows-1.f32" if local is None else "output.f32"), dtype="<f4")
            assert actual.size == OUTPUTS and np.isfinite(actual).all(), label
            error = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
            assert error < (0.01 if local is not None else 0.08), (label, error)
            assert int(actual.argmax()) == expected_top, label
            with (path / "samples.csv").open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            loads = [int(row["gpu_load"].split("@", 1)[0]) for row in samples if row["gpu_load"]]
            rounds.append({"run": label, "wall_median_ms": float(np.median(elapsed)),
                           "kernel_median_ms": float(np.median([float(row["kernel_ms"]) for row in records])) if local else None,
                           "relative_l2": error, "max_absolute_error": float(np.max(np.abs(actual - reference))),
                           "top_token": int(actual.argmax()), "peak_gpu_load_percent": max(loads, default=0),
                           "peak_process_rss_kib": process["peak_process_rss_kib"]})
        result = {"timing": {key: stats(values) for key, values in measurement.items()}, "rounds": rounds,
                  "max_relative_l2": max(row["relative_l2"] for row in rounds)}
        if local is not None:
            result["payload_equivalent_gb_per_s"] = weight_payload / (result["timing"]["kernel_ms"]["median_ms"] * 1e6)
        results["cases"][name] = result
        print(name, "wall", round(result["timing"]["wall_ms"]["median_ms"], 3),
              "kernel", round(result["timing"]["kernel_ms"]["median_ms"], 3) if local else "-",
              "max L2", f'{result["max_relative_l2"]:.3g}')
    (args.archive / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
