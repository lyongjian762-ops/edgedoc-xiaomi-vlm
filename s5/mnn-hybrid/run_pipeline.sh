#!/bin/sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: sh run_pipeline.sh REPLAY_DIR VISION_BUNDLE cpu|opencl [OPENCL_CACHE]" >&2
    exit 2
fi

replay=$1
vision=$2
backend=$3
case "$backend" in
    cpu|opencl) ;;
    *) echo "Expected cpu or opencl" >&2; exit 2 ;;
esac

if [ "$#" -eq 4 ]; then
    test "$backend" = opencl
    cp "$4" mnn_cachefile.bin
fi

LD_LIBRARY_PATH="$vision/lib" "$vision/imgenc" "$vision/xiaomi_vision_rk3588.rknn" \
    "$vision/vision-input.png" 3 > vision.stdout.log 2> vision.stderr.log
cmp img_vec.bin "$replay/archive_img_vec.bin"

export LD_LIBRARY_PATH="$replay"
exec stdbuf -oL -eL taskset -c 4-7 "$replay/mnn_image_replay" \
    "$replay/model-replay/$backend.json" "$replay/request.json" "$PWD/img_vec.bin" .
