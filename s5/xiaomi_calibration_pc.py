"""Capture real Xiaomi language-model inputs in Rockchip's multimodal calibration format."""

import argparse
import json
from pathlib import Path
import pickle

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class CapturedInputs(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    formatted = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "llm_inputs").mkdir()

    captured = {}

    def capture_language_input(_module, _args, kwargs):
        captured.update(kwargs)
        raise CapturedInputs

    handle = model.model.language_model.register_forward_pre_hook(capture_language_input, with_kwargs=True)
    dataset = []
    report = []
    try:
        for index, filename in enumerate(args.image):
            with Image.open(filename) as source:
                page = source.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
            inputs = processor(text=[formatted], images=[page], padding=True, return_tensors="pt")
            captured.clear()
            try:
                with torch.no_grad():
                    model(**inputs, use_cache=False)
            except CapturedInputs:
                pass
            captured["inputs_embeds"] = captured["inputs_embeds"].float().contiguous()
            tokens = captured["inputs_embeds"].shape[1]
            sample = f"llm_inputs/sample_{index}"
            with (args.output_dir / sample).open("wb") as file:
                pickle.dump(captured, file)
            dataset.append({"sample": sample, "token_nums": tokens})
            report.append({"image": str(filename), "embeddings": list(captured["inputs_embeds"].shape),
                           "position_ids": list(captured["position_ids"].shape),
                           "grid_thw": inputs["image_grid_thw"].tolist()})
    finally:
        handle.remove()

    (args.output_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
