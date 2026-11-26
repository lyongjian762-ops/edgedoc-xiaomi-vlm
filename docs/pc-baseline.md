# PC 基线与 C++ 验收记录

## S1 · C++ 程序

`edgedoc_cli` 在一个进程内直接调用 llama.cpp 与 libmtmd。模型权重和 Context 可供多页请求复用，图文输入、历史记忆、位置和采样器按页建立或重置。输出包括原始 Markdown 与逐页 JSON，记录像素和视觉特征形状、图文块的 Token 数和位置、停止原因。S1 支持 PNG、JPEG 静态图片，构建关闭视频解码。

使用 Ubuntu（WSL）构建并识别一页。

```bash
build="$HOME/.cache/edgedoc-xiaomi-vlm/s1-build"
assets="$PWD/models"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DEDGEDOC_LLAMA_SOURCE="$PWD/third_party/llama.cpp" -DGGML_NATIVE=ON -DGGML_CUDA=OFF
cmake --build "$build" --target edgedoc_cli -j 6

"$build/edgedoc_cli" \
  --model "$assets/Xiaomi-OCR-0.Q4_K_M.gguf" \
  --mmproj "$assets/Xiaomi-OCR-0.mmproj-bf16.gguf" \
  --prompt-file s1/ocr_prompt.txt \
  --image s0/samples/zh-notice.png \
  --output-dir "s1/results/local-$(date +%Y%m%d-%H%M%S)"
```

多次传入 `--image` 可以在同一个进程中按顺序识别多页。`--max-tokens` 默认 512。退出码 `0` 表示所有页自然结束，`2` 表示有页达到输出上限并保存了部分内容，`1` 表示发生错误并停止。`EDGEDOC_LLAMA_SOURCE` 可以指向另一处相同提交的源码。构建产物及其共享库在 WSL 的构建目录中。

三种页面的原始 C++ 输出均与 S0 逐字节一致。独立运行的 A 与 A→B→A→C 中的两个 A 一致，四次清空后序列的位置上界都是 `-1`。中文页产生 850 个宽度为 1024 的 `float32` 视觉特征，图像块推进 34 个多模态位置。两 Token 上限、损坏图片与上下文超额的分支均有实测。用下面的命令可一次复现这些对照。[S1 验收摘要和完整命令](../s1/results/pc-cpu-acceptance/acceptance.json)

```bash
python3 s1/verify_s1.py --cli "$build/edgedoc_cli" --assets "$assets" \
  --run-dir "s1/results/check-$(date +%Y%m%d-%H%M%S)"
```

S2 从这条链路建立分段计时。S1 的 `request_ms` 只覆盖开始读图到生成结束。S0 记录的是本地 HTTP 请求，两套计时的起点不同。

## S0 实测

同一个已就绪的模型进程依次处理三张新制样本。每张使用全页提示词和相同配置，跨请求提示缓存关闭。三页均自然停止，17 个关键字段全部命中。全文经全半角、空白和 Markdown 标题或加粗标记归一化后，三页均与底稿一致。

| 样本和底稿 | 客户端请求耗时 | 输出 Token | 关键字段 |
| --- | ---: | ---: | ---: |
| [中文交接通知](../s0/samples/zh-notice.png) · [底稿](../s0/samples/zh-notice.txt) | 14.48 s | 99 | 5/5 |
| [英文设备检修单](../s0/samples/en-service.png) · [底稿](../s0/samples/en-service.txt) | 13.88 s | 91 | 5/5 |
| [中英混排库存单](../s0/samples/mixed-inventory.png) · [底稿](../s0/samples/mixed-inventory.txt) | 13.97 s | 108 | 7/7 |

计时从客户端发起本地请求到读完响应。进程启动至可接收请求用时 7.04 秒，单独记录。首次仅运行中文页的结果为 14.10 秒，全文也与底稿一致。本轮样本由底稿排版生成，耗时用于功能复现，正式性能分析和真实照片质量留到后续阶段。

- [三页运行摘要、完整启动参数和资产 SHA-256](../s0/results/pc-cpu-s0/summary.json)
- [三页原始响应、输出文本、请求配置和日志](../s0/results/pc-cpu-s0/)
- [首次单图结果](../s0/results/pc-cpu-first/summary.json)

日志出现运行库对视觉定位任务的图像 Token 数提示，以及 `non-consecutive token position` 警告。后一条来自 `llama-memory-recurrent.cpp` 的 `find_slot()` 位置检查，每页各出现 16 次。S1 观察到同一图像的 M-RoPE 时间位置重复，并在独立运行中复现了警告。更大样本上的影响待验证。[三页运行日志](../s0/results/pc-cpu-s0/server.log)

## 环境与模型

| 项目 | 实测环境 |
| --- | --- |
| 主机 | Windows 11 专业版 22631，AMD Ryzen 7 5800H，8 核 16 线程，物理 RAM 16 GiB |
| 推理环境 | WSL2 Ubuntu 24.04.4 LTS x86_64，内核 `6.18.40.1-microsoft-standard-WSL2`，WSL 可见 RAM 7.5 GiB |
| GPU | Windows 和 WSL 均识别到 RTX 3050 Ti Laptop GPU，4 GiB 显存，驱动 566.36。本次为 CPU 构建与 CPU 推理，CUDA 构建待后续验证。 |
| 运行库 | `llama-server 0.8.0-dev`，llama.cpp `8642541b255bfb5dc39cd351a178be09013a55d6`，GNU 13.3.0，Release，`GGML_NATIVE=ON`、`GGML_CUDA=OFF` |
| 语言和视觉 | `Xiaomi-OCR-0.Q4_K_M.gguf`（529,297,664 字节）和 `Xiaomi-OCR-0.mmproj-bf16.gguf`（207,346,624 字节）。两份 SHA-256 均与本地保存的社区转换 LFS 清单一致。 |
| 请求配置 | 4 线程，4,096 上下文，生成上限 512 Token，温度 0，单并发，关闭跨请求提示缓存。使用小米 Demo 的整页提示词，直接提交给本地 `llama-server`。 |

## 复现

在 **Ubuntu（WSL）** 中运行下面的命令。脚本只使用 Python 标准库，先校验运行版本和两份权重，再在回环地址临时启动模型进程。结束后关闭该进程，运行产物保存在新的结果目录。

```bash
python3 s0/run_smoke.py \
  --server "$HOME/.cache/xiaomi-ocr-a0/build/bin/llama-server" \
  --assets "$PWD/models" \
  --run-dir "s0/results/repro-$(date +%Y%m%d-%H%M%S)"
```

加 `--sample zh-notice` 可只跑一页。权重和 llama.cpp 源码来自项目目录之外，`--assets` 和 `--server` 可以指向其他位置的同版本文件。若需要重建 CPU 二进制，在同一项目目录用下面的命令。构建目录单独放在 WSL 里，随后把 `--server` 改为 `$build/bin/llama-server`。

```bash
src="$PWD/third_party/llama.cpp"
build="$HOME/.cache/edgedoc-xiaomi-vlm/s0-build"
test "$(git -C "$src" rev-parse HEAD)" = 8642541b255bfb5dc39cd351a178be09013a55d6 &&
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF &&
cmake --build "$build" --target llama-server -j 4
```
