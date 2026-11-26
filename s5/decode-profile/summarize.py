"""Check the Decode trace against the OCR reference and summarize CPU operator timings."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def inspect_projections(model_json):
    graph = load_json(model_json)
    operators = graph.get("oplists") or graph["opList"]
    projections = []
    for op in operators:
        if op["type"] == "Convolution":
            main = op["main"]
            common = main["common"]
            quant = main.get("quanParameter", {})
            projections.append({"name": op["name"], "input_count": common["inputCount"],
                                "output_count": common["outputCount"],
                                "kernel": [common["kernelY"], common["kernelX"]], "group": common["group"],
                                "quant_bits": quant.get("aMaxOrBits"), "scale_storage": quant.get("scaleStorage"),
                                "external": main.get("external", [])})
    return {"exported_graph_sha256": hashlib.sha256(model_json.read_bytes()).hexdigest(),
            "graph_operator_count": len(operators), "convolutions": projections}


def profile_run(directory, tokens):
    forwards = read_jsonl(directory / "decode-forwards.jsonl")
    operators = read_jsonl(directory / "decode-ops.jsonl")
    assert [row["step"] for row in forwards] == list(range(1, tokens + 1)), directory
    step_times = {row["step"]: row["duration_us"] for row in forwards}
    step_sums = defaultdict(float)
    step_counts = defaultdict(int)
    types = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    names = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    for op in operators:
        assert op["step"] in step_times and op["duration_us"] >= 0, op
        step_sums[op["step"]] += op["duration_us"]
        step_counts[op["step"]] += 1
        for bucket in (types[op["type"]], names[(op["type"], op["name"])]):
            bucket["calls"] += 1
            bucket["duration_us"] += op["duration_us"]
    assert set(step_counts) == set(step_times), directory
    assert all(step_sums[step] <= step_times[step] for step in step_times), directory
    return {"operator_calls": len(operators), "min_calls_per_step": min(step_counts.values()),
            "max_calls_per_step": max(step_counts.values()),
            "forward_us": sum(step_times.values()), "named_operator_us": sum(step_sums.values()),
            "types": types, "names": names}


def combined_buckets(profiles, field, total_tokens, decode_us):
    totals = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    for profile in profiles:
        for key, bucket in profile[field].items():
            totals[key]["calls"] += bucket["calls"]
            totals[key]["duration_us"] += bucket["duration_us"]
    data = []
    for key, bucket in totals.items():
        result = {"type": key if field == "types" else key[0],
                  "calls_per_token": bucket["calls"] / total_tokens,
                  "ms_per_token": bucket["duration_us"] / total_tokens / 1000,
                  "percent_of_profiled_decode": 100 * bucket["duration_us"] / decode_us}
        if field == "names":
            result["name"] = key[1]
        data.append(result)
    return sorted(data, key=lambda row: row["ms_per_token"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--model-json", type=Path)
    args = parser.parse_args()
    metadata_path = args.archive / "model-projections.json"
    if args.model_json:
        metadata = inspect_projections(args.model_json)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata = load_json(metadata_path)
    expected_sha = "938ec6517469e808590fca283ab769018001320bf9926f63a2ee7e3e9a53e079"
    plain, profiled = [], []
    profile_data = []
    for name in ("plain-a", "profile-a", "profile-b", "plain-b"):
        directory = args.archive / name
        metric = load_json(directory / "metrics.json")
        process = load_json(directory / "process.json")
        assert process["returncode"] == 0 and not process["opened_mali_device"], name
        assert metric["backend"] == "cpu" and metric["finish_reason"] == "eog", name
        assert metric["prompt_tokens"] == 268 and metric["generated_tokens"] == 98, name
        assert hashlib.sha256((directory / "page.md").read_bytes()).hexdigest() == expected_sha, name
        row = dict(metric, run=name, process_wall_ms=process["process_wall_ms"], output_matches_reference=True)
        if name.startswith("profile"):
            profile = profile_run(directory, metric["generated_tokens"])
            assert profile["forward_us"] <= metric["decode_ms"] * 1000, name
            profile_data.append(profile)
            row.update({key: profile[key] for key in ("operator_calls", "min_calls_per_step", "max_calls_per_step",
                                                       "forward_us", "named_operator_us")})
            profiled.append(row)
        else:
            plain.append(row)
    assert (args.archive / "boundary/exit-status.txt").read_text().strip() == "1"
    assert "requires the synchronous CPU backend" in (args.archive / "boundary/stderr.log").read_text()
    assert not (args.archive / "boundary/page.md").exists()
    total_tokens = sum(row["generated_tokens"] for row in profiled)
    decode_us = sum(row["decode_ms"] * 1000 for row in profiled)
    plain_ms = median(row["decode_ms"] / row["generated_tokens"] for row in plain)
    profile_ms = decode_us / total_tokens / 1000
    types = combined_buckets(profile_data, "types", total_tokens, decode_us)
    names = combined_buckets(profile_data, "names", total_tokens, decode_us)
    lm_head = next(row for row in metadata["convolutions"] if row["name"] == "/lm/lm_head/Linear")
    lm_head_kernel = next(row for row in names if row["name"] == lm_head["name"])
    report = {"date_asia_shanghai": "2026-11-26", "runtime_revision": "d407447ed56c4121a11ccbd266dc184ca1ead0c2",
              "plain_runs": plain, "profile_runs": profiled, "plain_decode_ms_per_token": plain_ms,
              "profile_decode_ms_per_token": profile_ms,
              "profile_vs_plain_percent": 100 * (profile_ms / plain_ms - 1),
              "profile_forward_ms_per_token": sum(x["forward_us"] for x in profile_data) / total_tokens / 1000,
              "profile_named_op_ms_per_token": sum(x["named_operator_us"] for x in profile_data) / total_tokens / 1000,
              "types": types, "operators": names,
              "lm_head": dict(lm_head, timing=lm_head_kernel,
                              macs_per_token=lm_head["input_count"] * lm_head["output_count"],
                              serialized_weight_and_scale_bytes=sum(lm_head["external"][1:3])),
              "boundary_test_passed": True}
    (args.archive / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f'Plain Decode: {plain_ms:.3f} ms/token; profiled: {profile_ms:.3f} ms/token')
    for row in types:
        print(row["type"], f'{row["ms_per_token"]:.3f} ms/token', f'{row["percent_of_profiled_decode"]:.1f}%')
    print("LM head", report["lm_head"])


if __name__ == "__main__":
    main()
