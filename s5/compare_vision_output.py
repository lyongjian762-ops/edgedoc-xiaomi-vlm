"""Compare the Rockchip imgenc float32 dump with the matching PyTorch reference."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="img_vec.bin produced by Rockchip imgenc")
    parser.add_argument("reference", type=Path, help="reference.npy for the same RGB fixture")
    args = parser.parse_args()

    reference = np.load(args.reference, allow_pickle=False)
    if reference.shape != (196, 1024) or reference.dtype != np.float32:
        parser.error("expected a [196, 1024] float32 PyTorch reference for the 448px fixture")
    if args.dump.stat().st_size != reference.nbytes:
        parser.error(f"expected {reference.nbytes} bytes in the float32 dump")

    candidate = np.fromfile(args.dump, dtype="<f4").reshape(reference.shape).astype(np.float64)
    reference = reference.astype(np.float64)
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        parser.error("the dump or reference contains non-finite features")

    error = candidate - reference
    norms = np.linalg.norm(candidate) * np.linalg.norm(reference)
    cosine = float(np.vdot(candidate, reference) / norms) if norms else 0.0
    threshold = 0.999
    print(json.dumps({
        "output_shape": list(candidate.shape),
        "cosine_similarity": cosine,
        "cosine_threshold": threshold,
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }, indent=2))
    return 0 if cosine >= threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
