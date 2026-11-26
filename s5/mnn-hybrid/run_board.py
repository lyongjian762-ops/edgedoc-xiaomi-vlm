"""Run one board stage and record process, memory and Mali device observations."""

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def optional_read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def has_mali_fd(pid):
    try:
        for fd in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                if os.readlink(fd) == "/dev/mali0":
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if not command:
        parser.error("expected a command with absolute paths")
    args.output.mkdir()
    soc = Path("/sys/class/thermal/thermal_zone0/temp")
    gpu = Path("/sys/class/devfreq/fb000000.gpu")
    available_samples = []
    saw_mali = False
    status = None
    start = time.monotonic()

    with (args.output / "stdout.log").open("w") as stdout, (
        args.output / "stderr.log"
    ).open("w") as stderr, (args.output / "samples.csv").open("w", newline="") as samples:
        writer = csv.writer(samples)
        writer.writerow(("elapsed_ms", "mem_available_kib", "soc_millic", "mali_fd", "gpu_hz", "gpu_load"))
        with subprocess.Popen(command, cwd=args.output, stdout=stdout, stderr=stderr) as child:
            try:
                while (status := child.poll()) is None:
                    with open("/proc/meminfo") as info:
                        ram = next(int(row.split()[1]) for row in info if row.startswith("MemAvailable:"))
                    available_samples.append(ram)
                    active = has_mali_fd(child.pid)
                    saw_mali |= active
                    elapsed = time.monotonic() - start
                    writer.writerow((round(elapsed * 1000, 1), ram, optional_read(soc), int(active),
                                     optional_read(gpu / "cur_freq"), optional_read(gpu / "load")))
                    if elapsed > args.timeout:
                        break
                    time.sleep(0.1)
            finally:
                if child.poll() is None:
                    child.kill()
        report = {
            "command": command,
            "returncode": child.returncode,
            "timed_out": status is None,
            "process_wall_ms": (time.monotonic() - start) * 1000,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "minimum_board_mem_available_kib": min(available_samples, default=None),
            "opened_mali_device": saw_mali,
        }
    (args.output / "process.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(124 if status is None else 1 if status < 0 else status)


if __name__ == "__main__":
    main()
