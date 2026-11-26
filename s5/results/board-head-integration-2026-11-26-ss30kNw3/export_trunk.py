"""Expose the last normalized token vector while preserving the language graph's stateful trunk."""

import argparse
import hashlib
import json
from pathlib import Path
import site
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "config", "mnn_site", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    operators = graph["oplists"]
    head, convert, reshape = operators[-3:]
    hidden = graph["tensorName"].index("hidden_states")
    logits = graph["tensorName"].index("logits")
    if (head["name"] != "/lm/lm_head/Linear" or head["inputIndexes"] != [hidden]
            or head["type"] != "Convolution" or convert["type"] != "ConvertTensor"
            or reshape["type"] != "Reshape" or reshape["outputIndexes"] != [logits]
            or head["outputIndexes"] != convert["inputIndexes"]
            or convert["outputIndexes"] != reshape["inputIndexes"]
            or reshape["main"]["dims"] != [1, -1, 248320]):
        raise ValueError("unexpected language model tail")
    # Keep selection by logits_index, normalization, and every stateful block unchanged.
    convert["inputIndexes"] = [hidden]
    convert["name"] = "/custom_head/hidden_convert"
    reshape["name"] = "/custom_head/hidden_reshape"
    reshape["main"]["dims"] = [1, -1, 1024]
    graph["oplists"] = operators[:-3] + [convert, reshape]
    graph["outputName"] = ["logits"]
    source = args.output / "trunk.json"
    target = args.output / "trunk.mnn"
    source.write_text(json.dumps(graph, separators=(",", ":")), encoding="utf-8")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["llm_model"] = "trunk.mnn"
    (args.output / "opencl-trunk.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    site.addsitedir(str(args.mnn_site))
    from MNN.tools import mnnconvert

    sys.argv = ["mnnconvert", "-f", "JSON", "--modelFile", str(source), "--MNNModel", str(target)]
    mnnconvert.main()
    if not target.is_file() or not target.stat().st_size:
        raise RuntimeError("MNN trunk conversion failed")
    report = {"source_graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
              "trunk_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "removed_operator": head["name"], "head_external": head["main"]["external"],
              "trunk_output": "logits", "trunk_output_shape": [1, -1, 1024],
              "weights": "reuse original external weight file, existing offsets unchanged"}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
