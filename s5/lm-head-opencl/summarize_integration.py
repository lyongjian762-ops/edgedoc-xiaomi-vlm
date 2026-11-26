"""Check end-to-end OCR regression and restore comparable Decode timing for the split head."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median, mean
import unicodedata


REFERENCE_SHA = "938ec6517469e808590fca283ab769018001320bf9926f63a2ee7e3e9a53e079"


def load_case(directory):
    metrics = json.loads((directory / "metrics.json").read_text())
    process = json.loads((directory / "process.json").read_text())
    assert process["returncode"] == 0 and not process["timed_out"] and process["opened_mali_device"], directory
    assert metrics["backend"] == "opencl" and metrics["finish_reason"] == "eog" and metrics["prompt_tokens"] == 268, directory
    output = (directory / "page.md").read_bytes()
    visual = (directory / "img_vec.bin").read_bytes()
    assert len(visual) == 196 * 1024 * 4, directory
    log = (directory / "vision.stdout.log").read_text(encoding="utf-8")
    vision_ms = float(re.search(r"Encoder the image cost\s+([0-9.]+) ms", log).group(1))
    result = dict(metrics, run=directory.name, process_wall_ms=process["process_wall_ms"],
                  peak_process_rss_kib=process["peak_process_rss_kib"], vision_ms=vision_ms,
                  text_sha256=hashlib.sha256(output).hexdigest(), visual_sha256=hashlib.sha256(visual).hexdigest())
    if (directory / "custom-head.jsonl").is_file():
        rows = [json.loads(line) for line in (directory / "custom-head.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == list(range(metrics["generated_tokens"] + 1)), directory
        assert metrics["custom_head_calls"] == len(rows), directory
        assert abs(sum(row["head_total_ms"] for row in rows[1:]) - metrics["custom_head_decode_ms"]) < 0.01, directory
        assert abs(metrics["raw_mnn_decode_ms"] + metrics["custom_head_decode_ms"] - metrics["decode_ms"]) < 0.01, directory
        result["decode_head_mean_ms"] = {key: mean(row[key] for row in rows[1:]) for key in (
            "trunk_ms", "head_total_ms", "head_wall_ms", "kernel_ms", "upload_ms", "readback_ms")}
    return result


def averages(rows):
    return {key: median(r[key] for r in rows) for key in ("process_wall_ms", "prefill_ms", "decode_ms", "request_ms",
                                                          "decode_tokens_per_sec", "model_load_ms", "sample_ms")}


def receipt_fields(text, fields):
    text = unicodedata.normalize("NFKC", text)
    # Accept decimal/thousands punctuation and horizontal whitespace between digits.
    text = re.sub(r"(?<=\d)[,. \t](?=\d)", "", text)
    return {name: bool(re.search(r"(?<!\d)" + re.sub(r"\D", "", value) + r"(?!\d)", text))
            for name, value in fields.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("receipt_annotation", type=Path)
    parser.add_argument("head_reference", type=Path)
    parser.add_argument("mixed_reference", type=Path)
    args = parser.parse_args()
    root = args.archive / "runs"
    result = {"date_asia_shanghai": "2026-11-26", "comparison": "live RKNN vision plus OpenCL language inference",
              "head_output_bytes": {"full_logits": 248320 * 4, "group_max": 248320 // 64 * 2 * 4}, "pages": {}}
    settings = {"fixed": ("fixed-a", "fixed-b"), "receipt": ("receipt",), "mixed": ("mixed", "mixed-b")}
    for page, pairs in settings.items():
        modes = {mode: [load_case(root / f"{label}-{mode}") for label in pairs] for mode in ("mnn", "custom", "argmax")}
        all_runs = [row for values in modes.values() for row in values]
        hashes = {row["text_sha256"] for row in all_runs}
        embeddings = {row["visual_sha256"] for row in all_runs}
        tokens = {row["generated_tokens"] for row in all_runs}
        assert len(embeddings) == 1, page
        if page != "mixed":
            assert len(hashes) == 1 and len(tokens) == 1, f"OCR regression: {page}"
        else:
            # The baseline itself has more than one output. Require both custom paths
            # to agree across repetitions and match at least one original MNN run.
            custom_hashes = {row["text_sha256"] for mode in ("custom", "argmax") for row in modes[mode]}
            assert len(custom_hashes) == 1 and custom_hashes <= {row["text_sha256"] for row in modes["mnn"]}, page
        if page == "fixed":
            assert hashes == {REFERENCE_SHA}, page
        metrics = {mode: averages(rows) for mode, rows in modes.items()}
        page_result = {"identical_text": len(hashes) == 1, "identical_visual_embedding": True,
                       "generated_tokens_by_mode": {mode: [row["generated_tokens"] for row in rows] for mode, rows in modes.items()},
                       "metrics": metrics, "runs": modes}
        if len(tokens) == 1:
            page_result["generated_tokens"] = next(iter(tokens))
        if page == "receipt":
            fields = json.loads(args.receipt_annotation.read_text(encoding="utf-8"))["total"]
            page_result["receipt_fields_present"] = receipt_fields((root / "receipt-mnn/page.md").read_text(encoding="utf-8"), fields)
        result["pages"][page] = page_result
        print(page, sorted(tokens), "tokens", {mode: round(median(r["process_wall_ms"] for r in rows) / 1000, 3)
                                                    for mode, rows in modes.items()})
    mixed = result["pages"]["mixed"]
    diagnostics = [load_case(root / label) for label in ("mixed-c-mnn", "mixed-d-mnn")]
    baseline = mixed["runs"]["mnn"] + diagnostics
    assert len({row["visual_sha256"] for row in baseline}) == 1
    mixed["additional_baseline_diagnostics_excluded_from_main_medians"] = diagnostics
    mixed["baseline_text_variants"] = len({row["text_sha256"] for row in baseline})
    # A targeted comparison only. The source page is not scored here with a full-page CER.
    def option_line(text):
        return next(line.strip() for line in text.splitlines() if "laughed" in line and "children's" in line)

    def options_abd(line):
        return {key: re.search(r"\b" + key + r"\.\s*(\w+)", line).group(1) for key in ("A", "B", "D")}

    reference_line = option_line(args.mixed_reference.read_text(encoding="utf-8"))
    expected = options_abd(reference_line)
    observations = {}
    for row in baseline + mixed["runs"]["custom"] + mixed["runs"]["argmax"]:
        line = option_line((root / row["run"] / "page.md").read_text(encoding="utf-8"))
        actual = options_abd(line)
        observations[row["run"]] = {"options_line": line, "matches_reference": {key: val == expected[key] for key, val in actual.items()}}
    mixed["targeted_reference_check"] = {"scope": "options A, B and D only; not full-page CER", "reference_line": reference_line,
                                        "observations": observations}
    extra = load_case(root / "fixed-c-full")
    assert extra["text_sha256"] == REFERENCE_SHA and extra["custom_head_calls"] == 99
    result["refactored_full_logits_check"] = extra
    boundary = args.archive / "boundary"
    assert (boundary / "exit-status.txt").read_text().strip() == "1"
    assert "requires greedy sampling" in (boundary / "stderr.log").read_text()
    assert not (boundary / "page.md").exists()
    result["non_greedy_rejected"] = True
    standalone = root / "standalone-check"
    process = json.loads((standalone / "process.json").read_text())
    assert process["returncode"] == 0 and process["opened_mali_device"]
    actual, reference = array("f"), array("f")
    actual.frombytes((standalone / "output.f32").read_bytes())
    reference.frombytes(args.head_reference.read_bytes())
    assert len(actual) == len(reference) == 248320
    error = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(actual, reference)) / math.fsum(x * x for x in reference))
    assert error < 0.001 and max(range(len(actual)), key=actual.__getitem__) == 550
    result["standalone_refactor_relative_l2"] = error
    (args.archive / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
