#!/bin/sh
set -eu

if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
    echo "Usage: integration_pipeline.sh REPLAY_BUNDLE VISION_BUNDLE IMAGE BINARY CONFIG CACHE [GEMV_DIR]" >&2
    exit 2
fi

replay=$1
vision=$2
image=$3
binary=$4
config=$5
cache=$6
cp "$cache" mnn_cachefile.bin

LD_LIBRARY_PATH="$vision/lib" "$vision/imgenc" "$vision/xiaomi_vision_rk3588.rknn" "$image" 3 \
    >vision.stdout.log 2>vision.stderr.log
test -s img_vec.bin
export LD_LIBRARY_PATH="$replay"

if [ "$#" -eq 7 ]; then
    exec stdbuf -oL -eL taskset -c 4-7 "$binary" "$config" "$replay/request.json" "$PWD/img_vec.bin" . --custom-head "$7"
else
    exec stdbuf -oL -eL taskset -c 4-7 "$binary" "$config" "$replay/request.json" "$PWD/img_vec.bin" .
fi
