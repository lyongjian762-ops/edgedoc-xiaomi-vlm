"""Isolate the first Q4 up-projection and capture its real input from the OCR image."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import site
import sys

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


LAYER = "/layers.0/mlp/up_proj/Linear"
INPUT_WIDTH = 1024
OUTPUT_WIDTH = 3584


class CapturedInput(Exception):
    pass


def capture(args):
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    projection = model.model.language_model.layers[0].mlp.up_proj
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    with Image.open(args.image) as image:
        if image.size != (448, 448):
            raise ValueError("expected the fixed 448×448 OCR image")
        inputs = processor(text=[rendered], images=[image.convert("RGB")], return_tensors="pt")

    captured = None

    def capture_input(_module, operands):
        nonlocal captured
        captured = operands[0].detach().squeeze(0).float().contiguous()
        raise CapturedInput

    with torch.inference_mode(), projection.register_forward_pre_hook(capture_input):
        try:
            model(**inputs, use_cache=False)
        except CapturedInput:
            pass
        if captured is None or tuple(captured.shape) != (268, INPUT_WIDTH):
            raise ValueError("expected 268×1024 activations at the first language up-projection")
        if projection.bias is not None or tuple(projection.weight.shape) != (OUTPUT_WIDTH, INPUT_WIDTH):
            raise ValueError("unexpected up-projection geometry")
        reference = captured @ projection.weight.float().T
    for name, data in (("input.f32", captured), ("reference-bf16-weights.f32", reference)):
        array = np.asarray(data, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite values in {name}")
        array.tofile(args.output / name)


def extract_mnn(args):
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    projection, = (op for op in graph["oplists"] if op["name"] == LAYER)
    projection = copy.deepcopy(projection)
    common = projection["main"]["common"]
    quant = projection["main"]["quanParameter"]
    if (projection["type"] != "Convolution" or common["kernelX"] != 1 or common["kernelY"] != 1
            or (common["inputCount"], common["outputCount"]) != (INPUT_WIDTH, OUTPUT_WIDTH)
            or quant["aMaxOrBits"] != 4):
        raise ValueError("unexpected MNN up-projection")
    offset, *sizes = projection["main"]["external"]
    with args.weights.open("rb") as weight_file:
        weight_file.seek(offset)
        packed = weight_file.read(sum(sizes))
    if len(packed) != sum(sizes):
        raise ValueError("incomplete MNN packed weight range")
    (args.output / "up_proj.mnn.weight").write_bytes(packed)
    projection["main"]["external"][0] = 0
    projection["inputIndexes"] = [0]
    projection["outputIndexes"] = [1]
    network = {
        "bizCode": graph["bizCode"],
        "sourceType": graph["sourceType"],
        "usage": graph["usage"],
        "tensorNumber": 2,
        "tensorName": ["input", "output"],
        "outputName": ["output"],
        "oplists": [
            {
                "type": "Input", "main_type": "Input", "name": "input", "outputIndexes": [0],
                "main": {"dims": [1, INPUT_WIDTH, 1, 1], "dtype": "DT_FLOAT", "dformat": "NC4HW4"},
            },
            projection,
        ],
    }
    json_path = args.output / "up_proj.json"
    graph_path = args.output / "up_proj.mnn"
    json_path.write_text(json.dumps(network, indent=2) + "\n", encoding="utf-8")
    site.addsitedir(str(args.mnn_site))
    from MNN.tools import mnnconvert

    sys.argv = ["mnnconvert", "-f", "JSON", "--modelFile", str(json_path), "--MNNModel", str(graph_path)]
    mnnconvert.main()
    if not graph_path.is_file() or not graph_path.stat().st_size:
        raise RuntimeError("MNN did not produce the isolated graph")
    return {"external_offset": offset, "external_sizes": sizes, "quant": quant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "image", "prompt", "graph", "weights", "mnn_site", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    quant = extract_mnn(args)
    capture(args)
    files = ("up_proj.mnn", "up_proj.mnn.weight", "input.f32", "reference-bf16-weights.f32")
    report = {
        "layer": LAYER,
        "input_shape": [268, INPUT_WIDTH],
        "output_width": OUTPUT_WIDTH,
        "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "quantized_weights": quant,
        "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in files},
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
