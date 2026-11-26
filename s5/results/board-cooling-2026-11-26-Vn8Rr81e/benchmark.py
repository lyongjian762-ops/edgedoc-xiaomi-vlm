"""Repeat fixed-page OCR while recording temperature, clocks and cooling state."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def read_sensor(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def collect_sensors():
    sensors = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        sensors[(zone / "type").read_text().strip()] = zone / "temp"
    for device in sorted(Path("/sys/class/thermal").glob("cooling_device*")):
        sensors["cooling_" + (device / "type").read_text().strip()] = device / "cur_state"
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        for field in ("scaling_cur_freq", "scaling_max_freq"):
            sensors[f"cpu_{policy.name}_{field}"] = policy / field
    for device in sorted(Path("/sys/class/devfreq").glob("*")):
        for field in ("cur_freq", "max_freq", "load"):
            path = device / field
            if path.is_file():
                sensors[f"devfreq_{device.name}_{field}"] = path
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        if (hwmon / "name").read_text().strip() == "pwmfan":
            if (hwmon / "pwm1").exists():
                sensors["fan_pwm"] = hwmon / "pwm1"
            if (hwmon / "fan1_input").exists():
                sensors["fan_rpm"] = hwmon / "fan1_input"
    return sensors


def wait_cool(soc):
    deadline = time.monotonic() + 120
    while read_sensor(soc) > 45000:
        if time.monotonic() >= deadline:
            raise RuntimeError("SoC did not cool to 45 C in 120 seconds")
        time.sleep(2)


def run_job(root, name, command, sensors, expected, temperature_limit):
    directory = root / name
    directory.mkdir()
    if temperature_limit:
        wait_cool(sensors["soc-thermal"])
    initial = read_sensor(sensors["soc-thermal"])
    if initial is None or initial >= 85000:
        raise RuntimeError(f"Unexpected starting SoC temperature: {initial}")
    markers = (b"clip_encode: copying image", b"clip_encode: output embedding shape")
    env = dict(os.environ, RKLLM_LOG_LEVEL="1")
    events = []
    print(f"START {name} soc_millic={initial}", flush=True)
    with (directory / "stdout.log").open("wb") as stdout, (
        directory / "stderr.log"
    ).open("wb") as stderr, (directory / "thermal.jsonl").open("w") as thermal:
        start = time.monotonic()
        with subprocess.Popen(
            ["timeout", "-k", "5s", "210s", *command],
            cwd=directory, stdout=stdout, stderr=subprocess.PIPE, env=env,
        ) as process:
            def record_stderr():
                for line in process.stderr:
                    if any(marker in line for marker in markers):
                        events.append({"seconds_from_start": time.monotonic() - start,
                                       "message": line.decode(errors="replace").strip()})
                    stderr.write(line)

            reader = threading.Thread(target=record_stderr)
            reader.start()
            try:
                while True:
                    sample = {"elapsed_seconds": round(time.monotonic() - start, 3)}
                    sample.update((key, read_sensor(path)) for key, path in sensors.items())
                    thermal.write(json.dumps(sample) + "\n")
                    thermal.flush()
                    try:
                        status = process.wait(timeout=0.5)
                        finished = time.monotonic()
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    process.terminate()
                reader.join()

    report = {"command": command, "initial_soc_millicelsius": initial,
              "wall_seconds": finished - start, "exit_status": status,
              "cooldown_to_45c": temperature_limit}
    (directory / "time.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    if status != 0:
        raise RuntimeError(f"{name}: exit status {status}")
    if name.startswith("cpu-"):
        output_ok = (directory / "result/page-1.md").read_bytes() == expected
    elif name.startswith("mnn-"):
        output_ok = (directory / "page.md").read_bytes() == expected
    else:
        output_ok = expected in (directory / "stdout.log").read_bytes()
    report["output_matches_reference"] = output_ok
    (directory / "time.json").write_text(json.dumps(report, indent=2) + "\n")
    if not output_ok:
        raise RuntimeError(f"{name}: OCR output differs from the reference")
    print(f"DONE {name} {report['wall_seconds']:.3f}s output_ok={output_ok}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("hybrid", type=Path)
    parser.add_argument("--suite", choices=("baseline", "sustained"), default="baseline")
    args = parser.parse_args()
    root, original, hybrid = (path.resolve() for path in (args.output, args.original, args.hybrid))
    root.mkdir()
    vision = original / "npu"
    expected = (vision / "expected.md").read_bytes()
    cpu = [str(original / "edgedoc-s5/edgedoc_cli"),
           "--model", str(original / "models/Xiaomi-OCR-0.Q4_K_M.gguf"),
           "--mmproj", str(original / "models/Xiaomi-OCR-0.mmproj-bf16.gguf"),
           "--prompt-file", str(vision / "ocr_prompt.txt"),
           "--image", str(vision / "vision-input.png"), "--max-tokens", "512",
           "--output-dir", "result"]
    npu = ["sh", str(vision / "run_xiaomi_npu_once.sh")]
    mnn = ["sh", str(hybrid / "run_pipeline.sh"), str(hybrid), str(vision)]
    mnn_cpu = mnn + ["cpu"]
    mnn_gpu = mnn + ["opencl", str(hybrid / "fixed-opencl/mnn_cachefile.bin")]
    sensors = collect_sensors()
    if args.suite == "baseline":
        runs = [("npu-a", npu), ("cpu-a", cpu), ("cpu-b", cpu), ("npu-b", npu),
                ("mnn-cpu-a", mnn_cpu), ("mnn-gpu-a", mnn_gpu),
                ("mnn-gpu-b", mnn_gpu), ("mnn-cpu-b", mnn_cpu)]
        for name, command in runs:
            run_job(root, name, command, sensors, expected, True)
    else:
        for index in range(1, 4):
            run_job(root, f"cpu-sustained-{index}", cpu, sensors, expected, index == 1)


if __name__ == "__main__":
    main()
