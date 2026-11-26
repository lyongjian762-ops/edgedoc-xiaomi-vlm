"""Summarize cooling retests beside the previous fixed-page board measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import median


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def thermal_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def values(rows, key):
    return [row[key] for row in rows if row.get(key) is not None]


def thermal_summary(rows, active=None):
    clock_rows = rows if active is None else active
    peak_keys = [key for key in rows[0] if key.endswith("-thermal")]
    stats = {"peak_temp_c": max(value for key in peak_keys for value in values(rows, key)) / 1000,
             "peak_soc_c": max(values(rows, "soc-thermal")) / 1000,
             "start_soc_c": rows[0]["soc-thermal"] / 1000,
             "fan_pwm_values": sorted(set(values(rows, "fan_pwm"))),
             "cpu_large_core_frequency_mhz": {}, "thermal_cooling_state_max": {}}
    for policy in ("policy4", "policy6"):
        clock = values(clock_rows, f"cpu_{policy}_scaling_cur_freq")
        limit = values(clock_rows, f"cpu_{policy}_scaling_max_freq")
        if clock:
            stats["cpu_large_core_frequency_mhz"][policy] = {
                "min": min(clock) / 1000, "median": median(clock) / 1000, "max": max(clock) / 1000,
                "minimum_policy_max": min(limit) / 1000 if limit else None,
                "samples_at_2304mhz": sum(freq >= 2304000 for freq in clock), "samples": len(clock),
            }
    for key in rows[0]:
        if key.startswith("cooling_") and values(rows, key):
            stats["thermal_cooling_state_max"][key] = max(values(rows, key))
    for device in ("fb000000.gpu", "fdab0000.npu", "dmc"):
        clocks = values(clock_rows, f"devfreq_{device}_cur_freq")
        if clocks:
            stats[device + "_frequency_mhz"] = {"min": min(clocks) / 1e6, "median": median(clocks) / 1e6,
                                                   "max": max(clocks) / 1e6}
    return stats


def parse_trial(directory):
    timing = load_json(directory / "time.json")
    if timing["exit_status"] or not timing["output_matches_reference"]:
        raise ValueError(f"Unsuccessful trial {directory}")
    result = {"run": directory.name, "wall_seconds": timing["wall_seconds"],
              "output_matches_reference": True}
    rows = thermal_rows(directory / "thermal.jsonl")
    active = None
    if directory.name.startswith("cpu-"):
        page = load_json(directory / "result/page-1.json")
        events = load_json(directory / "events.json")
        start = next(x["seconds_from_start"] for x in events if "copying image" in x["message"])
        end = next(x["seconds_from_start"] for x in events if "output embedding shape" in x["message"])
        active = [row for row in rows if start <= row["elapsed_seconds"] <= end]
        result.update(request_seconds=page["request_ms"] / 1000, vision_seconds=end - start,
                      generated_tokens=page["generated_content_tokens"], prompt_tokens=page["prompt_tokens"],
                      frequency_scope="CPU vision log interval")
    elif directory.name.startswith("mnn-"):
        meta = load_json(directory / "metrics.json")
        result.update(vision_seconds=None, request_seconds=meta["request_ms"] / 1000,
                      prefill_seconds=meta["prefill_ms"] / 1000,
                      decode_seconds=meta["decode_ms"] / 1000,
                      decode_tokens_per_second=meta["decode_tokens_per_sec"],
                      generated_tokens=meta["generated_tokens"], prompt_tokens=meta["prompt_tokens"],
                      frequency_scope="whole process")
    else:
        log = (directory / "stdout.log").read_text(encoding="utf-8")
        vision = float(re.search(r"ImgEnc Model inference took\s+([0-9.]+) ms", log).group(1)) / 1000
        prefill_ms, prompt, _, _ = map(float, re.search(
            r"\bPrefill\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)", log).groups())
        decode_ms, generated, _, rate = map(float, re.search(
            r"\bGenerate\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)", log).groups())
        result.update(vision_seconds=vision, prefill_seconds=prefill_ms / 1000,
                      decode_seconds=decode_ms / 1000, decode_tokens_per_second=rate,
                      generated_tokens=int(generated), prompt_tokens=int(prompt), frequency_scope="whole process")
    if result["generated_tokens"] != 98 or result["prompt_tokens"] != 268:
        raise ValueError(f"Unexpected token counts {directory}: {result}")
    return dict(result, **thermal_summary(rows, active))


def combined(runs):
    keys = ("wall_seconds", "vision_seconds", "request_seconds", "prefill_seconds", "decode_seconds",
            "decode_tokens_per_second", "start_soc_c", "peak_temp_c", "peak_soc_c")
    stats = {key: median(values(runs, key)) for key in keys if values(runs, key)}
    for key in ("peak_temp_c", "peak_soc_c"):
        if values(runs, key):
            stats["max_" + key] = max(values(runs, key))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--previous-comparison", type=Path, required=True)
    parser.add_argument("--previous-hybrid", type=Path, required=True)
    args = parser.parse_args()
    previous = load_json(args.previous_comparison / "summary.json")
    old_hybrid = load_json(args.previous_hybrid / "summary.json")
    old = {}
    for backend, vision_key in (("cpu", "vision_log_interval_seconds"), ("npu", "vision_encoder_seconds")):
        old[backend] = []
        for letter, row in zip(("a", "b"), previous[backend + "_runs"]):
            label = backend + "-" + letter
            temps = thermal_rows(args.previous_comparison / label / "thermal.jsonl")
            old[backend].append(dict(row, run=label, vision_seconds=row[vision_key],
                                     start_soc_c=row["initial_soc_millicelsius"] / 1000,
                                     peak_temp_c=row["max_sampled_temp_millicelsius"] / 1000,
                                     peak_soc_c=max(values(temps, "soc-thermal")) / 1000,
                                     **({"decode_seconds": row["generate_seconds"],
                                         "decode_tokens_per_second": row["generate_tokens_per_second"]} if backend == "npu" else {})))
    for backend, old_backend in (("mnn-cpu", "mnn-cpu"), ("mnn-gpu", "mnn-opencl")):
        old[backend] = []
        for row in old_hybrid["pipelines"][old_backend]["runs"]:
            with (args.previous_hybrid / "runs" / row["run"] / "samples.csv").open(newline="") as stream:
                samples = list(csv.DictReader(stream))
            temps = [int(x["soc_millic"]) / 1000 for x in samples if x["soc_millic"]]
            gpu_clocks = [int(x["gpu_hz"]) / 1e6 for x in samples if x["gpu_hz"]]
            old[backend].append(dict(row, wall_seconds=row["process_wall_ms"] / 1000,
                                     request_seconds=row["request_ms"] / 1000,
                                     prefill_seconds=row["prefill_ms"] / 1000, decode_seconds=row["decode_ms"] / 1000,
                                     decode_tokens_per_second=row["decode_tokens_per_sec"],
                                     start_soc_c=temps[0], peak_soc_c=max(temps),
                                     peak_gpu_frequency_mhz=max(gpu_clocks) if gpu_clocks else None))
    results = {"baseline_date": "2026-11-23/24", "cooling_retest_date_asia_shanghai": "2026-11-26",
               "sample_shape": [448, 448], "generated_tokens": 98, "prompt_tokens": 268, "backends": {}}
    for backend in ("cpu", "npu", "mnn-cpu", "mnn-gpu"):
        runs = [parse_trial(args.archive / "baseline" / (backend + "-" + letter)) for letter in ("a", "b")]
        before, after = combined(old[backend]), combined(runs)
        metrics = {"old": before, "new": after,
                   "wall_reduction_percent": 100 * (1 - after["wall_seconds"] / before["wall_seconds"]),
                   "wall_speedup": before["wall_seconds"] / after["wall_seconds"], "old_runs": old[backend], "runs": runs}
        results["backends"][backend] = metrics
        print(backend, f'{before["wall_seconds"]:.3f}s -> {after["wall_seconds"]:.3f}s',
              f'{metrics["wall_reduction_percent"]:+.2f}% reduction',
              f'new temp peak {max(x["peak_temp_c"] for x in runs):.2f}C')
    if (args.archive / "sustained").is_dir():
        results["sustained_cpu"] = [parse_trial(directory) for directory in sorted((args.archive / "sustained").iterdir())]
        for trial in results["sustained_cpu"]:
            print(trial["run"], f'{trial["wall_seconds"]:.3f}s', f'{trial["peak_temp_c"]:.2f}C peak')
    diagnostic = args.archive / "gpu-confirm"
    if diagnostic.is_dir():
        process = load_json(diagnostic / "process.json")
        meta = load_json(diagnostic / "metrics.json")
        digest = hashlib.sha256((diagnostic / "page.md").read_bytes()).hexdigest()
        if process["returncode"] != 0 or digest != old_hybrid["generated_ocr_sha256"]:
            raise ValueError("Unsuccessful GPU device confirmation")
        with (diagnostic / "samples.csv").open(newline="") as stream:
            samples = list(csv.DictReader(stream))
        loads = [int(row["gpu_load"].split("@", 1)[0]) for row in samples if row["gpu_load"]]
        temperatures = [int(row["soc_millic"]) for row in samples if row["soc_millic"]]
        results["gpu_confirmation"] = {"output_matches_reference": True,
                                       "process_wall_seconds": process["process_wall_ms"] / 1000,
                                       "opened_mali_device": process["opened_mali_device"],
                                       "peak_gpu_load_percent": max(loads) if loads else None,
                                       "peak_soc_c": max(temperatures) / 1000,
                                       "decode_tokens_per_second": meta["decode_tokens_per_sec"]}
        print("GPU device confirmation", results["gpu_confirmation"])
    (args.archive / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
