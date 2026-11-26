"""Export the Xiaomi-OCR-0 language graph with a pinned MNN exporter."""

import argparse
import os
from pathlib import Path
import site
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mnn_source", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra-site", type=Path)
    args = parser.parse_args()
    source = args.mnn_source.resolve()
    model = args.model.resolve()
    output = args.output.resolve()
    if args.extra_site:
        site.addsitedir(str(args.extra_site.resolve()))
    sys.path.insert(0, str(source / "transformers" / "llm" / "export"))

    from llmexport import LlmExporter, build_args

    export_parser = argparse.ArgumentParser()
    build_args(export_parser)
    export_args = export_parser.parse_args([
        "--path", str(model), "--export", "mnn", "--dst_path", str(output),
        "--quant_bit", "4", "--quant_block", "64",
    ])
    output.mkdir()
    os.chdir(output)
    exporter = LlmExporter(export_args)
    if exporter.model_type != "qwen3_5" or not exporter.llm_config["is_mrope"]:
        raise RuntimeError("expected the Qwen3.5 language model with M-RoPE")
    # The separate RKNN graph supplies image features. Keep M-RoPE in the language export.
    exporter.visual = None
    exporter.llm_config["is_visual"] = False
    # The Qwen3.5 exporter emits four language inputs. DeepStack inputs are only emitted for Qwen3-VL.
    exporter.llm_config["has_deepstack"] = False
    exporter.export("mnn")


if __name__ == "__main__":
    main()
