from pathlib import Path
import json
import os
import resource
import subprocess
import threading
import time

workspace = Path.cwd().parent
sensors = [(p, (p / "type").read_text().strip()) for p in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
           if (p / "type").is_file() and (p / "temp").is_file()]
fan = next((p / "pwm1" for p in Path("/sys/class/hwmon").glob("hwmon*")
            if (p / "name").is_file() and (p / "name").read_text().strip() == "pwmfan"), None)
soc = next(path / "temp" for path, name in sensors if name == "soc-thermal")
cpu = [str(workspace / "edgedoc-s5/edgedoc_cli"),
       "--model", str(workspace / "models/Xiaomi-OCR-0.Q4_K_M.gguf"),
       "--mmproj", str(workspace / "models/Xiaomi-OCR-0.mmproj-bf16.gguf"),
       "--prompt-file", str(workspace / "npu/ocr_prompt.txt"),
       "--image", str(workspace / "npu/vision-input.png"),
       "--max-tokens", "512"]
npu = ["sh", str(workspace / "npu/run_xiaomi_npu_once.sh")]
markers = (b"clip_encode: copying image", b"clip_encode: output embedding shape")
expected = (workspace / "npu/expected.md").read_bytes()

def run(name, command):
    until = time.monotonic() + 120
    while int(soc.read_text()) > 45000:
        if time.monotonic() >= until:
            raise RuntimeError("SoC did not cool to 45 C within 120 s")
        time.sleep(2)
    directory = Path(name)
    directory.mkdir()
    if name.startswith("cpu"):
        command = command + ["--output-dir", str(directory / "result")]
    env = dict(os.environ, RKLLM_LOG_LEVEL="1")

    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr, (directory / "thermal.jsonl").open("w") as thermal:
        start_temperature = int(soc.read_text())
        started = time.monotonic()
        proc = subprocess.Popen(["timeout", "-k", "5s", "210s", *command], stdout=stdout, stderr=subprocess.PIPE, env=env)
        events = []
        def record_stderr():
            for line in proc.stderr:
                if any(marker in line for marker in markers):
                    events.append({"seconds_from_start": time.monotonic() - started,
                                   "message": line.decode(errors="replace").strip()})
                stderr.write(line)
        reader = threading.Thread(target=record_stderr)
        reader.start()
        while True:
            sample = {"elapsed_seconds": round(time.monotonic() - started, 3)}
            for path, name in sensors:
                sample[name] = int((path / "temp").read_text())
            if fan is not None:
                sample["fan_pwm"] = int(fan.read_text())
            thermal.write(json.dumps(sample) + "\n")
            thermal.flush()
            try:
                status = proc.wait(timeout=1)
                finished = time.monotonic()
                break
            except subprocess.TimeoutExpired:
                continue
        reader.join()
    data = {"wall_seconds": finished - started, "exit_status": status, "initial_soc_millicelsius": start_temperature}
    (directory / "time.json").write_text(json.dumps(data, indent=2) + "\n")
    (directory / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    if status != 0:
        raise RuntimeError(f"{directory}: child exited {status}")
    if directory.name.startswith("cpu"):
        assert (directory / "result/page-1.md").read_bytes() == expected
    else:
        assert expected in (directory / "stdout.log").read_bytes()
    print(directory.name, json.dumps(data), flush=True)

for label, command in (("npu-a", npu), ("cpu-a", cpu), ("cpu-b", cpu), ("npu-b", npu)):
    run(label, command)
