#!/usr/bin/env python3
"""Run the S0 document smoke test against a pinned local CPU build."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
import unicodedata
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


SAMPLES = Path(__file__).resolve().parent / "samples"
COMMIT = "8642541"
ASSETS = {
    "Xiaomi-OCR-0.Q4_K_M.gguf": "1e497ff06b91a3d822c543066df3e9ef69c62c5423be09d739dfa1db55630d3a",
    "Xiaomi-OCR-0.mmproj-bf16.gguf": "634d6bc95781ebf4a3f06570a59859fa5082d488db540fd474bc6b4061aec90e",
}
PROMPT = (
    "Extract all information from the main body of the document image and represent it in markdown format, "
    "ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should "
    "be represented using LATEX format, and the parsing should be organized according to the reading order."
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^ {0,3}#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "")
    return "".join(text.split())


def wait_for_model(client, base_url, server):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and server.poll() is None:
        try:
            with client.open(base_url + "/health", timeout=1) as response:
                if json.load(response).get("status") == "ok":
                    return
        except HTTPError as error:
            error.close()
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("Model startup failed or timed out. See server.log.")


def run_page(client, base_url, sample, run_dir):
    name = sample["id"]
    image = (SAMPLES / f"{name}.png").read_bytes()
    truth = (SAMPLES / f"{name}.txt").read_text(encoding="utf-8")
    image_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    payload = {
        "model": "xiaomi-ocr-0",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 512,
        "cache_prompt": False,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload_log = dict(payload, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"samples/{name}.png (PNG data URL at runtime)"}},
        {"type": "text", "text": PROMPT},
    ]}])
    write_json(run_dir / f"{name}.request.json", payload_log)
    request = Request(base_url + "/v1/chat/completions", json.dumps(payload).encode("utf-8"),
                      {"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with client.open(request, timeout=180) as response:
            raw = response.read()
    except HTTPError as error:
        with error:
            (run_dir / f"{name}.response.json").write_bytes(error.read())
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    (run_dir / f"{name}.response.json").write_bytes(raw)
    body = json.loads(raw)
    choice = body["choices"][0]
    output = choice["message"]["content"] or ""
    (run_dir / f"{name}.output.md").write_text(output, encoding="utf-8")
    normalized = normalize(output)
    fields = {field: normalize(field) in normalized for field in sample["fields"]}
    return {
        "id": name,
        "input_sha256": hashlib.sha256(image).hexdigest(),
        "request_ms": elapsed_ms,
        "finish_reason": choice["finish_reason"],
        "usage": body.get("usage"),
        "server_timings": body.get("timings"),
        "normalized_text_match": normalized == normalize(truth),
        "fields": fields,
        "passed": bool(normalized) and choice["finish_reason"] == "stop" and all(fields.values()),
    }


def main():
    samples = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory for the run")
    parser.add_argument("--sample", choices=[sample["id"] for sample in samples])
    args = parser.parse_args()

    version = subprocess.check_output([str(args.server.resolve()), "--version"],
                                      stderr=subprocess.STDOUT, text=True).strip()
    if f"commit {COMMIT})" not in version:
        raise SystemExit(f"Expected llama.cpp {COMMIT}. Got: {version}")
    assets = {}
    for name, expected in ASSETS.items():
        path = args.assets.resolve() / name
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        assets[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]
    command = [str(args.server.resolve()), "--host", "127.0.0.1", "--port", str(port),
               "--alias", "xiaomi-ocr-0", "--model", assets["Xiaomi-OCR-0.Q4_K_M.gguf"]["path"],
               "--mmproj", assets["Xiaomi-OCR-0.mmproj-bf16.gguf"]["path"],
               "--device", "none", "--n-gpu-layers", "0", "--no-mmproj-offload",
               "--threads", "4", "--threads-batch", "4", "--ctx-size", "4096",
               "--batch-size", "256", "--ubatch-size", "128", "--parallel", "1",
               "--jinja", "--no-cache-prompt", "--no-context-shift"]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"server_version": version, "server_command": command, "assets": assets, "pages": []}
    print(shlex.join(command), flush=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
    client = build_opener(ProxyHandler({}))
    base_url = f"http://127.0.0.1:{port}"
    try:
        with (run_dir / "server.log").open("wb") as log:
            started = time.perf_counter()
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                wait_for_model(client, base_url, server)
                summary["startup_until_ready_ms"] = round((time.perf_counter() - started) * 1000, 1)
                for sample in samples:
                    if args.sample is None or sample["id"] == args.sample:
                        result = run_page(client, base_url, sample, run_dir)
                        summary["pages"].append(result)
                        print(json.dumps(result, ensure_ascii=False), flush=True)
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        summary["error"] = str(error)
        raise
    finally:
        write_json(run_dir / "summary.json", summary)

    return 0 if all(page["passed"] for page in summary["pages"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
