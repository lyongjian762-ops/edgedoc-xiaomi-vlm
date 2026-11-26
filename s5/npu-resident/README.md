# RK3588S 单进程多页 OCR

`main.cpp` 把模型初始化放在页面循环之前。循环每次读一张新图，重新运行视觉编码，并用当前的视觉特征调用 RKLLM。`keep_history = 0` 让每页使用独立的推理请求。固定视觉模型的输入为 `448×448` RGB，输出为 `196×1024`。

## 在 PC 上构建

从项目目录在 WSL Ubuntu 中执行，使用 Debian 11 ARM64 交叉工具链和下列固定版本的瑞芯微源码。

```bash
mkdir -p third_party
git clone https://github.com/airockchip/rknn-llm.git third_party/rknn-llm
git -C third_party/rknn-llm checkout --detach f7390530443bf84f0394255a449d7cbe81e69d1c

cache="$HOME/.cache/edgedoc-xiaomi-vlm"
toolchain="$cache/debian11-cross/sysroot"
build="$cache/s5-resident-debian11"
cmake -S s5/npu-resident -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DROCKCHIP_SOURCE="$PWD/third_party/rknn-llm" \
  -DCMAKE_SYSROOT="$toolchain" \
  -DCMAKE_CXX_COMPILER="$toolchain/usr/bin/aarch64-linux-gnu-g++-10" \
  -DCMAKE_CXX_FLAGS=-B/usr/aarch64-linux-gnu/bin/
cmake --build "$build" --target edgedoc_npu_resident -j 4
```

## 在板上复现

把构建产物、`benchmark.py` 和测试批次的 `inputs/` 目录放到板上。模型目录沿用 eMMC 的固定尺寸部署包。

```bash
models=/path/to/emmc-debian11-448
(cd inputs && sha256sum -c SHA256SUMS)
LD_LIBRARY_PATH="$models/lib" python3 benchmark.py ./edgedoc_npu_resident "$models" inputs trials
```

脚本先每图启动一个进程，之后用一个进程依次识别这批图。最后独立运行 `A → B → 坏图 → A`。坏图页面写入 `invalid_image` 状态，整个故障对照以状态码 3 退出，后续有效图仍可写出正文。

`init.json` 记录两份模型的初始化时间。每页 JSON 的 `first_text_ms` 从开始处理该图片到 RKLLM 回调第一次交付可见文本。`request_ms` 从读图前计到 RKLLM 调用结束。进程墙钟时间单独记在各组的 `time.json`，包含写文件和退出。
