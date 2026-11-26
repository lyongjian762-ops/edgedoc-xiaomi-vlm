"""Run the QKV benchmark and sample the board's available RAM and temperatures."""

import argparse
import csv
import json
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpus", default="4-7", help="CPU list for taskset")
    parser.add_argument("--backend", choices=("tile16", "tile32", "clblast", "npu", "npu-core0"),
                        default="tile16")
    args = parser.parse_args()
    args.output.mkdir()
    zones = Path("/sys/class/thermal").glob("thermal_zone*")
    thermals = {
        (zone / "type").read_text().strip(): zone / "temp"
        for zone in zones
    }
    soc = thermals["soc-thermal"]
    gpu = thermals["gpu-thermal"]
    kernels = {"tile16": "qkv.cl", "tile32": "qkv_tile32.cl", "clblast": "--clblast"}
    if args.backend in kernels:
        benchmark = ["./compare_qkv", kernels[args.backend], "operands", str(args.output)]
    else:
        core_mask = "7" if args.backend == "npu" else "1"
        benchmark = ["./compare_npu", "operands", str(args.output), core_mask]
    command = ["taskset", "-c", args.cpus, *benchmark]
    timeout = 90 if args.backend == "clblast" else 40

    with (args.output / "resources.csv").open("w", newline="") as samples, (
        args.output / "benchmark.log"
    ).open("w") as log:
        writer = csv.writer(samples)
        writer.writerow(("elapsed_ms", "mem_available_kib", "soc_millic", "gpu_millic"))
        start = time.monotonic()

        def sample():
            with open("/proc/meminfo") as info:
                available = next(int(line.split()[1]) for line in info if line.startswith("MemAvailable:"))
            writer.writerow((
                round((time.monotonic() - start) * 1000, 3),
                available,
                int(soc.read_text()),
                int(gpu.read_text()),
            ))

        sample()
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                while process.poll() is None:
                    sample()
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"benchmark took more than {timeout} seconds")
                    time.sleep(0.05)
            finally:
                if process.poll() is None:
                    process.kill()
        sample()
        usage = {
            "returncode": process.returncode,
            "command": command,
            "benchmark_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        }
        (args.output / "process.json").write_text(json.dumps(usage, indent=2) + "\n")
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == "__main__":
    main()
