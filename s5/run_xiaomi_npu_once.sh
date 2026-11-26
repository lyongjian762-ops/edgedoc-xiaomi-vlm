#!/bin/sh
set -eu

bundle=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
prompt=$(cat "$bundle/ocr_prompt.txt")

printf '%s\n' "<image>$prompt" exit |
    LD_LIBRARY_PATH="$bundle/lib" "$bundle/demo" \
        "$bundle/vision-input.png" "$bundle/xiaomi_vision_rk3588.rknn" \
        '' '' "$bundle/xiaomi_ocr_0_w8a8_rk3588.rkllm" \
        512 4096 3 rk3588 '<|vision_start|>' '<|vision_end|>' '<|image_pad|>'
