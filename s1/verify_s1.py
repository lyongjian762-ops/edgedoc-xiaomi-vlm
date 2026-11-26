"""Run the local S1 acceptance cases and compare against the saved S0 responses."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
S0_RUN = ROOT / "s0/results/pc-cpu-s0"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_case(output_dir, command, name, images, expected_exit, extra=()):
    case_dir = output_dir / name
    args = [*command, "--output-dir", str(case_dir), *extra]
    for image in images:
        args.extend(("--image", str(image)))
    with (output_dir / f"{name}.stdout").open("wb") as stdout, (output_dir / f"{name}.stderr").open("wb") as stderr:
        process = subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=240)
    log = (output_dir / f"{name}.stderr").read_text(encoding="utf-8", errors="replace")
    require(process.returncode == expected_exit, f"{name} exit {process.returncode}. See {name}.stderr.")
    print(f"{name}: exit {process.returncode}", flush=True)
    metadata = {"command": args, "exit_code": process.returncode,
                "position_warning_count": log.count("non-consecutive token position")}
    return case_dir, metadata, log


def compare_page(case_dir, index, sample_id, s0):
    page = read_json(case_dir / f"page-{index}.json")
    output = (case_dir / f"page-{index}.md").read_bytes()
    reference = (S0_RUN / f"{sample_id}.output.md").read_bytes()
    require(page["finish_reason"] == "eog", f"{case_dir.name}/{sample_id}: unexpected stop")
    require(page["memory_pos_max_after_reset"] == -1, f"{case_dir.name}/{sample_id}: request memory not empty")
    require(page["prompt_tokens"] == s0["usage"]["prompt_tokens"], f"{sample_id}: prompt count differs from S0")
    require(output == reference, f"{case_dir.name}/{sample_id}: text differs from S0")
    return page, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    s0_summary = read_json(S0_RUN / "summary.json")
    s0_pages = {page["id"]: page for page in s0_summary["pages"]}
    assets = args.assets.resolve()
    for filename, identity in s0_summary["assets"].items():
        with (assets / filename).open("rb") as source:
            require(hashlib.file_digest(source, "sha256").hexdigest() == identity["sha256"],
                    f"asset differs from S0: {filename}")
    for sample_id, record in s0_pages.items():
        image = (ROOT / "s0/samples" / f"{sample_id}.png").read_bytes()
        require(hashlib.sha256(image).hexdigest() == record["input_sha256"], f"input differs from S0: {sample_id}")

    output_dir = args.run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [str(args.cli.resolve()), "--model", str(assets / "Xiaomi-OCR-0.Q4_K_M.gguf"),
               "--mmproj", str(assets / "Xiaomi-OCR-0.mmproj-bf16.gguf"), "--prompt-file", "s1/ocr_prompt.txt"]
    sample_ids = ["zh-notice", "en-service", "zh-notice", "mixed-inventory"]
    images = [Path("s0/samples") / f"{sample_id}.png" for sample_id in sample_ids]
    expected_prompt = read_json(S0_RUN / "zh-notice.request.json")["messages"][0]["content"][1]["text"]
    cases = {}

    single_dir, cases["single"], _ = run_case(output_dir, command, "single", images[:1], 0)
    _, standalone = compare_page(single_dir, 1, sample_ids[0], s0_pages[sample_ids[0]])
    sequence_dir, cases["sequence"], _ = run_case(output_dir, command, "sequence", images, 0)
    require(read_json(sequence_dir / "run.json")["prompt"] == expected_prompt, "prompt differs from S0")
    sequence = []
    for i, sample_id in enumerate(sample_ids, 1):
        page, output = compare_page(sequence_dir, i, sample_id, s0_pages[sample_id])
        require(page["memory_pos_max_before_reset"] >= 0 if i > 1 else page["memory_pos_max_before_reset"] == -1,
                f"page {i}: unexpected state entering the reset")
        if sample_id == sample_ids[0]:
            require(output == standalone, "A output differs between fresh and reused contexts")
        sequence.append(page)

    limited_dir, cases["limited"], _ = run_case(
        output_dir, command, "limited", images[:1], 2, ("--max-tokens", "2"))
    limited = read_json(limited_dir / "page-1.json")
    require(limited["finish_reason"] == "max_tokens" and limited["generated_content_tokens"] == 2,
            "expected generation to stop at two content tokens")
    partial = (limited_dir / "page-1.md").read_text(encoding="utf-8")
    require(bool(partial) and standalone.startswith(partial.encode("utf-8")), "partial output should prefix the full page")

    broken_image = output_dir / "corrupt.png"
    broken_image.write_bytes(b"This file has a PNG extension and contains plain text.\n")
    bad_dir, cases["corrupt"], bad_log = run_case(output_dir, command, "corrupt", [broken_image], 1)
    require("cannot decode a still image" in bad_log and not (bad_dir / "page-1.md").exists(),
            "corrupt input did not fail cleanly before generating a page")
    require("ffprobe" not in bad_log, "still-image build unexpectedly attempted video probing")

    budget_dir, cases["over_budget"], budget_log = run_case(
        output_dir, command, "over_budget", images[:1], 1, ("--max-tokens", "3500"))
    require("input and requested output exceed context capacity" in budget_log and not (budget_dir / "page-1.md").exists(),
            "context budget violation did not fail before prefill")

    summary = {"cases": cases, "sequence_names": sample_ids, "sequence": sequence,
               "distinct_pages_exact_match_s0": 3, "a_b_a_exact_match_fresh_a": True,
               "matching_s0_key_field_checks": sum(len(page["fields"]) for page in s0_pages.values()),
               "limited_output": partial}
    (output_dir / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"S1 acceptance passed. Evidence: {output_dir / 'acceptance.json'}")


if __name__ == "__main__":
    main()
