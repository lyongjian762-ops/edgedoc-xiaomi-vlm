"""Reorder the real Q4 weight bytes for adjacent Mali work-items."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


WIDTH = 1024
OUTPUTS = 248320
GROUPS = 16
TILE = 64
HEADER = 26


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operands", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    original = json.loads((args.operands / "manifest.json").read_text())
    for name, digest in original["files"].items():
        assert hashlib.sha256((args.operands / name).read_bytes()).hexdigest() == digest, name
    raw = (args.operands / "lm_head.mnn.weight").read_bytes()
    weight_length, scale_length, *_ = original["quantized_weights"]["external_sizes"]
    assert raw[0] == 2 and raw[9] == 16
    assert tuple(np.frombuffer(raw, dtype="<u4", count=2, offset=1)) == (OUTPUTS, WIDTH)
    assert np.array_equal(np.frombuffer(raw, dtype=np.int8, count=16, offset=10), np.arange(-8, 8))
    weights = np.frombuffer(raw, dtype=np.uint8, count=weight_length - HEADER, offset=HEADER)
    weights = weights.reshape(OUTPUTS // TILE, TILE, GROUPS, 4, 8)
    scales = np.frombuffer(raw, dtype="<f2", count=scale_length // 2, offset=weight_length)
    scales = scales.reshape(OUTPUTS // TILE, TILE, GROUPS, 2)
    with (args.output / "tile64.bin").open("wb") as stream:
        weights.transpose(0, 2, 3, 1, 4).tofile(stream)
        scales.transpose(0, 2, 1, 3).tofile(stream)
    packed = np.memmap(args.output / "tile64.bin", dtype=np.uint8, mode="r")
    assert np.array_equal(packed[:weight_length - HEADER].reshape(OUTPUTS // TILE, GROUPS, 4, TILE, 8)
                          .transpose(0, 3, 1, 2, 4), weights)
    assert np.array_equal(packed[weight_length - HEADER:].view("<u2").reshape(OUTPUTS // TILE, GROUPS, TILE, 2)
                          .transpose(0, 2, 1, 3), scales.view("<u2"))
    del packed

    last = np.fromfile(args.operands / "input.f32", dtype="<f4").reshape(268, WIDTH)[-1]
    input_vector = np.concatenate((last[::2], last[1::2]))
    np.asarray(input_vector, dtype="<f4").tofile(args.output / "input-f32.bin")
    input_half = np.asarray(input_vector, dtype="<f2")
    input_half.tofile(args.output / "input-f16.bin")
    np.asarray(last.reshape(GROUPS, WIDTH // GROUPS).sum(axis=1), dtype="<f4").tofile(args.output / "group-sum.f32")
    files = ("tile64.bin", "input-f32.bin", "input-f16.bin", "group-sum.f32")
    manifest = {"source_weights_sha256": original["files"]["lm_head.mnn.weight"],
                "source_input_sha256": original["files"]["input.f32"], "rows": 1, "tile_outputs": TILE,
                "layout": "tile,group,chunk,output_lane,8 packed bytes; scales tile,group,lane,2 half",
                "input_fp16_relative_l2": float(np.linalg.norm(input_half.astype(np.float32) - input_vector)
                                                / np.linalg.norm(input_vector)),
                "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in files}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
