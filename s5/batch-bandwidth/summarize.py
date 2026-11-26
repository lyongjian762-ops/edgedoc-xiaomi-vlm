"""Validate the head's batched logits and the independent read-bandwidth benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lm-head"))
from summarize import reference_q4, relative_l2


ROWS = (1, 2, 4, 8)
OUTPUTS = 248320


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_process(path, gpu):
    data = json.loads((path / "process.json").read_text())
    assert data["returncode"] == 0 and not data["timed_out"], path
    assert data["opened_mali_device"] == gpu, path
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("operands", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.operands / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((args.operands / name).read_bytes()).hexdigest() == digest, name
    reference = reference_q4(args.operands, manifest, 8)
    np.asarray(reference, dtype="<f4").tofile(args.archive / "reference-8-q4.f32")
    results = {"date_asia_shanghai": "2026-11-26", "rows": ROWS,
               "operands": str(args.operands), "batch": {}, "stream_read": {}}
    checksums = set()
    for backend in ("cpu", "opencl"):
        wall = {rows: [] for rows in ROWS}
        errors = {rows: [] for rows in ROWS}
        round_medians = {rows: [] for rows in ROWS}
        bandwidth_values, bandwidth_host = [], []
        bandwidth_rounds = []
        for letter in ("a", "b"):
            path = args.archive / "batch-runs" / f"{letter}-{backend}"
            validate_process(path, backend == "opencl")
            records = read_csv(path / "timing.csv")
            for rows in ROWS:
                subset = [record for record in records if int(record["rows"]) == rows]
                assert [int(row["iteration"]) for row in subset] == list(range(1, 31)), path
                values = [float(record["wall_us"]) / 1000 for record in subset]
                wall[rows].extend(values)
                round_medians[rows].append(float(np.median(values)))
                actual = np.fromfile(path / f"rows-{rows}.f32", dtype="<f4").reshape(rows, OUTPUTS)
                assert np.isfinite(actual).all(), path
                error = max(relative_l2(actual_row, reference_row) for actual_row, reference_row in zip(actual, reference[-rows:]))
                errors[rows].append(error)
                assert error < 0.08, (path, rows, error)

            path = args.archive / "bandwidth-runs" / f"{letter}-{backend}"
            validate_process(path, backend == "opencl")
            samples = read_csv(path / "bandwidth.csv")
            assert [int(row["iteration"]) for row in samples] == list(range(1, 31))
            assert all(int(row["bytes"]) == 128 * 1024 * 1024 and row["backend"] == backend for row in samples)
            checksums.update(int(row["checksum"]) for row in samples)
            rates = [float(row["read_gb_per_s"]) for row in samples]
            bandwidth_values.extend(rates)
            bandwidth_host.extend(int(row["bytes"]) / (float(row["wall_ms"]) * 1e6) for row in samples)
            bandwidth_rounds.append(float(np.median(rates)))
        results["batch"][backend] = {str(rows): {"median_ms": float(np.median(wall[rows])),
                                                "ms_per_row": float(np.median(wall[rows]) / rows),
                                                "p95_ms": float(np.percentile(wall[rows], 95)),
                                                "round_medians_ms": round_medians[rows],
                                                "max_relative_l2_per_row": max(errors[rows])} for rows in ROWS}
        results["stream_read"][backend] = {"bytes": 128 * 1024 * 1024,
                                          "median_gb_per_s": float(np.median(bandwidth_values)),
                                          "host_wall_median_gb_per_s": float(np.median(bandwidth_host)),
                                          "round_median_gb_per_s": bandwidth_rounds,
                                          "samples": len(bandwidth_values)}
    assert len(checksums) == 1, checksums
    results["stream_checksum"] = checksums.pop()
    (args.archive / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for rows in ROWS:
        print(rows, *(round(results["batch"][backend][str(rows)]["median_ms"], 3) for backend in ("cpu", "opencl")))
    print("stream-read", results["stream_read"])


if __name__ == "__main__":
    main()
