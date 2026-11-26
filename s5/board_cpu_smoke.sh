#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    printf 'Usage: sh %s <model directory> <new output directory>\n' "$0" >&2
    exit 2
fi

bundle=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
models=$(CDPATH='' cd -- "$1" && pwd)
output=$2

(cd "$models" && sha256sum -c "$bundle/models.sha256")
"$bundle/edgedoc_cli" \
    --model "$models/Xiaomi-OCR-0.Q4_K_M.gguf" \
    --mmproj "$models/Xiaomi-OCR-0.mmproj-bf16.gguf" \
    --prompt-file "$bundle/ocr_prompt.txt" \
    --image "$bundle/zh-notice.png" \
    --output-dir "$output"

cmp "$bundle/expected.md" "$output/page-1.md"
printf '%s\n' 'The recognized page matches the PC reference byte for byte.'
