"""Extract the real Q4 vocabulary projection and one real last-position input."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mnn-linear"))
from prepare import CapturedInput, extract_mnn


LAYER = "/lm/lm_head/Linear"
INPUT_WIDTH = 1024
OUTPUT_WIDTH = 248320


def capture_input(args):
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    with Image.open(args.image) as image:
        if image.size != (448, 448):
            raise ValueError("expected the fixed 448x448 image")
        inputs = processor(text=[rendered], images=[image.convert("RGB")], return_tensors="pt")

    hidden = None

    def capture(_module, operands):
        nonlocal hidden
        hidden = operands[0].detach().squeeze(0).float().contiguous()
        raise CapturedInput

    with torch.inference_mode(), model.lm_head.register_forward_pre_hook(capture):
        try:
            model(**inputs, use_cache=False, logits_to_keep=0)
        except CapturedInput:
            pass
    if hidden is None or tuple(hidden.shape) != (268, INPUT_WIDTH):
        raise ValueError(f"unexpected lm_head input shape: {None if hidden is None else hidden.shape}")
    if model.lm_head.bias is not None or tuple(model.lm_head.weight.shape) != (OUTPUT_WIDTH, INPUT_WIDTH):
        raise ValueError("unexpected vocabulary projection geometry")
    with torch.inference_mode():
        last = hidden[-1:]
        reference = torch.cat([last @ block.float().T for block in model.lm_head.weight.split(8192, dim=0)], dim=1)
    for name, values in (("input.f32", hidden), ("reference-bf16.f32", reference)):
        array = np.asarray(values, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite data in {name}")
        array.tofile(args.output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "image", "prompt", "graph", "weights", "mnn_site", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    quant = extract_mnn(args, LAYER, OUTPUT_WIDTH, "lm_head")
    capture_input(args)
    files = ("lm_head.mnn", "lm_head.mnn.weight", "input.f32", "reference-bf16.f32")
    report = {"layer": LAYER, "source": "Last hidden row of the original BF16 model's fixed OCR prefill",
              "input_shape": [268, INPUT_WIDTH], "measured_rows": 1, "output_width": OUTPUT_WIDTH,
              "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "quantized_weights": quant,
              "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in files}}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
