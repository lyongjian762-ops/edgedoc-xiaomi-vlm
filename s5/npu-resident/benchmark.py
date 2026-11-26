"""Measure per-page process startup and one resident process on the same dev inputs."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time


THERMAL = Path("/sys/class/thermal")
SOC_TEMP = THERMAL / "thermal_zone0/temp"


def wait_until_cool():
    deadline = time.monotonic() + 90
    while int(SOC_TEMP.read_text()) > 45000:
        if time.monotonic() >= deadline:
            raise RuntimeError("SoC did not cool to 45 C within 90 seconds")
        time.sleep(2)


def run_case(root, name, command, images, limit, expected_status=0):
    wait_until_cool()
    case = root / name
    case.mkdir()
    args = [*command, str(case / "result"), *map(str, images)]
    with (case / "stdout.log").open("w") as out, (case / "stderr.log").open("w") as err, (
        case / "thermal.jsonl"
    ).open("w") as thermal:
        started = time.monotonic()
        proc = subprocess.Popen(["timeout", "-k", "5s", limit, *args], stdout=out, stderr=err)
        while True:
            sample = {"elapsed_seconds": time.monotonic() - started}
            for zone in THERMAL.glob("thermal_zone*"):
                sample[(zone / "type").read_text().strip()] = int((zone / "temp").read_text())
            thermal.write(json.dumps(sample) + "\n")
            thermal.flush()
            try:
                status = proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
    timing = {
        "wall_seconds": time.monotonic() - started,
        "exit_status": status,
        "peak_child_rss_kib_so_far": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "input_files": [p.name for p in images],
    }
    (case / "time.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(name, timing, flush=True)
    if status != expected_status:
        raise RuntimeError(f"{name}: expected exit {expected_status}, got {status}. See {case / 'stderr.log'}")


def verify_outputs(output, page_count):
    references = []
    for i in range(1, page_count + 1):
        reference = (output / f"fresh-{i}/result/page-1.md").read_bytes()
        resident = (output / f"resident/result/page-{i}.md").read_bytes()
        if not reference or resident != reference:
            raise RuntimeError(f"page {i} differs from its independent run")
        references.append(reference)

    state = output / "state-check/result"
    for i, reference in ((1, references[0]), (2, references[-1]), (4, references[0])):
        if (state / f"page-{i}.md").read_bytes() != reference:
            raise RuntimeError(f"state check page {i} differs from its independent run")
    if json.loads((state / "page-3.json").read_text()) != {"status": "invalid_image"} or (
        state / "page-3.md"
    ).exists():
        raise RuntimeError("invalid image was not rejected cleanly")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("executable", type=Path)
    cli.add_argument("models", type=Path)
    cli.add_argument("inputs", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir()
    samples = json.loads((args.inputs / "inputs.json").read_text())
    images = [args.inputs / item["image"] for item in samples]
    if len(images) < 2:
        cli.error("need at least two different images")
    command = [
        str(args.executable.resolve()),
        str(args.models / "xiaomi_vision_rk3588.rknn"),
        str(args.models / "xiaomi_ocr_0_w8a8_rk3588.rkllm"),
        str(args.inputs / "ocr_prompt.txt"),
    ]
    os.environ["RKLLM_LOG_LEVEL"] = "1"
    for i, image in enumerate(images, 1):
        run_case(args.output, f"fresh-{i}", command, [image], "115s")
    run_case(args.output, "resident", command, images, "360s")
    invalid = args.output / "invalid.png"
    invalid.write_text("deliberately invalid image\n")
    # Check that a failed read leaves the process able to accept another page.
    second = images[-1]
    run_case(args.output, "state-check", command, [images[0], second, invalid, images[0]],
             "220s", expected_status=3)
    verify_outputs(args.output, len(images))
    print("resident text and state checks passed", flush=True)


if __name__ == "__main__":
    main()
