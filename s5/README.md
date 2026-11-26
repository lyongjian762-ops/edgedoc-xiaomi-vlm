# S5 静态单图的 CPU 与 NPU 适配

## 首次上板实测

2026-11-23 实测为鲁班猫 4 V1、RK3588S、Debian 12、内核 `6.1.99-rk3588`，可见内存约 3.81 GiB，系统从 SD 卡启动，RKNPU 驱动为 `0.9.8`。试验使用静态图片。板卡的日期仍为 2026-04-28，运行耗时用单调时钟测量。

| 检查 | 真机实测 |
| --- | --- |
| S1 CPU 原图 | 输入 `1100×790`，输出与 PC 基线逐字节一致，模型自然结束。请求 196.64 秒，包含校验和启动的整个测试约 208.10 秒。采样到的大核最高温度 90.3 °C。[CPU 结果](results/board-2026-11-23-Rx8Pse/cpu-run/result/page-1.md) |
| RKNN 小卷积 | 板端运行库 `2.3.2` 和驱动 `0.9.8` 完成推理。起初传入 NCHW 被板端归一化拒绝，测试程序改传 NHWC 后，128 个输出的最大绝对误差为 `0.0003554`，低于 `0.003` 阈值。[数值结果](results/board-2026-11-23-Rx8Pse/npu-l08-nhwc/summary.json) |
| RKNN 视觉独立测试 | `448×448` 固定 PNG 得到 `196×1024`。两次编码分别为 851.46 和 765.58 毫秒，输出逐字节相同。与 PyTorch 的余弦相似度 `0.9989592`，低于原定 `0.999`，差异来源待查。[数值对照](results/board-2026-11-23-Rx8Pse/vision-run/comparison.json) |
| RKNN → RKLLM 整页识别 | 同一张 `448×448` 固定图经视觉编码和 W8A8 语言模型，完整输出与 PC BF16 参考逐字节相同，五个字段全部命中。整个进程耗时 12.61 秒，包含加载。[NPU 结果](results/board-2026-11-23-Rx8Pse/ocr-run/page-1.md) |

CPU 和 NPU 使用的输入分辨率及计时范围见上表。完整模型版本、时间、内存和温度采样见[真机验证摘要](results/board-2026-11-23-Rx8Pse/summary.json)。

## eMMC 上的 Debian 11 复测

拔出 TF 卡后，实测根文件系统位于 eMMC 的 `/dev/mmcblk0p3`。系统是 Debian 11，内核 `5.10.160`，glibc `2.31`，RKNPU 驱动 `0.9.8`，传输模型前约有 24 GiB 空间。`pwm-fan` 已绑定，初次读取时温度为 37 °C、PWM 为 0。

| 检查 | eMMC 实测 |
| --- | --- |
| S1 CPU 短标签 | `640×256`，22 个输出 Token，自然结束，结果与 PC 逐字节一致。单次请求 37.19 秒。采样最高温度 76.7 °C，风扇 PWM 依次采样到 `100、150、200、255`。[CPU 输出](results/board-emmc-2026-11-23-ZcXVOr/cpu-short/result/page-1.md) |
| RKNN 小卷积 | 运行库 `2.3.2`，128 个输出的最大绝对误差 `0.0003554`，低于 `0.003`。[小卷积结果](results/board-emmc-2026-11-23-ZcXVOr/rknn-small/summary.json) |
| RKNN 视觉独立测试 | `448×448` 固定图，编码 774.94 毫秒。板端特征与 TF 卡 Debian 12 的同一输入逐字节一致。对 PyTorch 的余弦相似度仍是 `0.9989592`，低于 `0.999`。[数值对照](results/board-emmc-2026-11-23-ZcXVOr/vision-run/comparison.json) |
| RKNN → RKLLM 静态单图 | 沿用同一张 `448×448` 固定图和 W8A8 语言权重，完整输出与 PC BF16 参考逐字节一致，五个关键字段均命中。从程序启动到退出用时 13.62 秒，采样最高温度 55.5 °C。[NPU 输出](results/board-emmc-2026-11-23-ZcXVOr/ocr-run/page-1.md) |

WSL 中对应的部署目录为 `~/.cache/edgedoc-xiaomi-vlm/board-xiaomi-npu-448-debian11`。用 Debian 11 GCC 10 和目标用户态库重新编译了 `demo`、`imgenc` 和 `rknn_board_smoke`，沿用原 RKNN、RKLLM 模型文件和运行库。温度数据对应各自的图像和测试流程。[程序兼容性记录](results/board-emmc-2026-11-23-ZcXVOr/debian11-runtime-abi.log) · [eMMC 验证摘要](results/board-emmc-2026-11-23-ZcXVOr/summary.json)

## eMMC 同图 CPU 与 NPU 对照

两条路径使用同一张 `448×448` PNG、同一提示词、4,096 Token 上下文和 512 Token 输出上限。按 NPU、CPU、CPU、NPU 的顺序启动四个独立进程，启动时 SoC 温度均低于 45 °C，沿用系统文件缓存。统一用单调时钟测量从启动前到进程退出的墙钟时间。四份 OCR 正文均与 PC BF16 基准逐字节一致。

| 指标 | CPU，4 线程 | NPU，3 核 |
| --- | ---: | ---: |
| 两次完整进程耗时 | 51.77 / 51.51 秒 | 12.90 / 13.36 秒 |
| 完整进程耗时中位数 | 51.64 秒 | 13.13 秒 |
| 视觉阶段中位数 | 约 37.15 秒 | 0.755 秒 |
| 采样到的最高温度 | 85.9 °C | 61.0 °C |

完整链路的实测耗时比为 **3.93 倍**。CPU 使用 S1 的静态 ARMv8-A 构建，语言模型是 Q4_K_M GGUF，视觉使用 BF16 投影权重。NPU 使用 FP16 视觉 RKNN 与 W8A8 RKLLM。这是两套完整部署方案的速度对照。

CPU 识别请求中位数为 50.09 秒，视觉阶段约占 74%。CPU 的视觉时间来自 `clip_encode` 的复制输入和输出形状日志，NPU 的视觉时间来自示例程序的 `run_imgenc` 计时。CPU 总时长与识别时长之差约 1.55 秒，包含初始化、文件输出与退出。NPU 日志报告加载约 2.42 秒，语言 Prefill 处理 268 个输入 Token 约 2.58 秒，生成 98 个 Token 用时约 6.21 秒，约 15.8 Token/s。视觉时间下降后，生成阶段成为 NPU 路径里最大的已测阶段。[数据与测量边界](results/board-comparison-2026-11-23-ow5SEL/summary.json) · [NPU 运行日志](results/board-comparison-2026-11-23-ow5SEL/npu-a/stdout.log)

## 2026-11-24 模型常驻首轮对照

同一 eMMC 板和同一组 RKNN、RKLLM W8A8 权重。将 OCR 开发组的 5 张原图用 RGB 双三次缩放生成固定的 `448×448` 输入。两种模式用同一个新 C++ 程序，分别每图重启进程和一个进程连续处理。保留组用于后续复测。

| 指标 | 每图启动 | 常驻进程 |
| --- | ---: | ---: |
| 五张累计进程墙钟耗时 | 94.01 秒 | 81.30 秒 |
| 两份模型累计初始化 | 12.44 秒 | 2.49 秒 |
| 五张请求累计（读图到 RKLLM 返回） | 76.18 秒 | 77.89 秒 |

首轮累计少 **12.71 秒（13.5%）**，完整进程墙钟包含初始化、文件输出和退出。两种模式的五份正文逐字节一致，生成数均低于 512。额外的 `A → B → 坏图 → A` 中，三次有效输出和各自独立运行一致，坏图写入错误状态后继续处理。票据开发页的总额、实付和找零三项与标注一致，文档正文 CER 留到后续评测。[实验摘要](results/board-resident-2026-11-24-1YZuKC/summary.json) · [模型常驻源码与复现](npu-resident/README.md)

## eMMC GPU 初步验证

实机识别为四核 Mali-G610，`mali` 驱动和 `/dev/mali0` 可用。OpenCL 成功在 GPU 上执行 1,024 个浮点数的向量加法，最大绝对误差为 0。当前镜像的 Vulkan 加载器创建实例时返回 `-9`，系统未配置 Vulkan ICD。GPU 已具备用于小型 OpenCL 算子实验的条件，整模型适配和性能仍待验证。[GPU 检测结果](results/board-comparison-2026-11-23-ow5SEL/gpu.json) · [检测代码](results/board-comparison-2026-11-23-ow5SEL/gpu_probe.py)

### 2026-11-24 原图缩放与颜色转换

5 张开发组原图在实机上解码后，CPU OpenCV 和 GPU OpenCL 都做双线性缩放和 BGR → RGB，输出 `448×448`。每种 OpenCV 线程配置两轮共 300 对计时。下表为中位数，GPU 墙钟包含上传、提交命令、等待和读回。

| CPU 配置 | CPU 预处理 | GPU 含拷贝 | GPU 核函数 | 预处理加速比 |
| --- | ---: | ---: | ---: | ---: |
| OpenCV 默认 8 线程 | 5.106 ms | 4.434 ms | 0.515 ms | 1.15× |
| OpenCV 1 线程 | 8.108 ms | 4.642 ms | 0.540 ms | 1.75× |

默认配置下，GPU 若计入每种尺寸的坐标表和缓冲设置，中位数为 4.753 ms。CPU 解码中位数为 35.484 ms，两路共用此环节。20 组 CPU 和 GPU 的 PNG 都逐字节一致。将一张票据的两份 PNG 交给板端 NPU，OCR 正文逐字节一致。默认 CPU 配置下 GPU 的收益约 0.67 ms/张，可先作为图像预处理的候选后端。[实测数据](results/board-gpu-preprocess-2026-11-24-6FnYRe/summary.json) · [源码与计时边界](gpu-preprocess/README.md)

### 2026-11-24 第一层视觉 QKV 矩阵乘法

用同一张 `448×448` 固定图捕获第一层视觉 QKV 的输入，用原始模型权重完成 `[784,768] × [768,2304]` 的板端 CPU/GPU 对照。GPU 和 PyTorch 参考之间最大绝对误差 `1.1444e-5`。15 对计时中，四大核 OpenBLAS 中位数 23.17 ms，Mali-G610 核函数 70.78 ms，GPU 含传输与等待 72.70 ms。计时和内存采样覆盖这个独立算子。[源码与复跑](gpu-matmul/README.md) · [实测数据](results/board-mali-matmul-2026-11-24-4vs7mI/summary.json)

同数据补测四条路径各 30 次。含输入和读回的中位数，Mali 手写 16 项为 72.66 ms，32 项为 91.70 ms，CLBlast FP32 为 46.57 ms，RKNPU 独立 MatMul 单核为 12.89 ms。与 16 项核配对的四大核 OpenBLAS 为 23.30 ms。两轮输出均通过参考校验。[对照记录](results/board-qkv-backends-2026-11-24-l0E4mu/summary.json) · [计时边界和复跑](gpu-matmul/README.md)

## NPU 视觉与 MNN 语言接力

固定单图的 NPU 视觉特征先在 PC 接入 llama.cpp，正文与原 CPU 视觉路线一致。随后在 eMMC 板用新生成的特征接入 MNN 3.6.1。两轮整页耗时中位数为 RKNN → RKLLM 13.10 秒，RKNN → MNN 四大核 CPU 7.51 秒，RKNN → MNN Mali OpenCL 有缓存 9.43 秒。六次板端正文均与 PC 基线逐字节一致。[代码与复跑](mnn-hybrid/README.md) · [真机数据](results/board-npu-mnn-2026-11-24-eAgvzp/summary.json)

用第一层语言四比特投影比较 1、8、32、128、268 行。Mali 每行的合并中位数从 976.4 µs 降到 65.2 µs，四大核 CPU 在这些形状的总耗时都更短。单行使用更重的 OpenCL 调优没有观察到稳定提速。[算子对照与复跑](mnn-linear/README.md) · [原始数据](results/board-mnn-linear-2026-11-24-CQVuiW/summary.json)

## PC 归档与离板复盘

同图对照目录保留了固定图片、原提示词、参考输出、四次运行的原始日志及温度采样。另存了 S1 和瑞芯微示例的关键入口源码快照。完整依赖源码仍在 `output/xiaomi_ocr_setup`，llama.cpp 固定到 `8642541b255bfb5dc39cd351a178be09013a55d6`，rknn-llm 固定到 `f7390530443bf84f0394255a449d7cbe81e69d1c`。[固定图片](results/board-comparison-2026-11-23-ow5SEL/input/vision-input.png) · [C++ CPU 入口](results/board-comparison-2026-11-23-ow5SEL/code/edgedoc/edgedoc_cli.cpp) · [瑞芯微 NPU 示例](https://github.com/airockchip/rknn-llm/tree/f7390530443bf84f0394255a449d7cbe81e69d1c/examples/multimodal_model_demo) · [同图对照脚本](results/board-comparison-2026-11-23-ow5SEL/compare.py)

在 WSL x86_64 用归档中的同图和提示词重新执行 S1，输入像素为 `448×448×3`，得到 `196×1024` 图像特征。生成 98 个内容 Token 后自然结束，正文与 PC 参考逐字节一致。用扩展名为 PNG 的文本文件测试读图分支，进程退出码为 1，日志报 `cannot decode a still image`，没有写出该页 Markdown。[离板复跑摘要](results/board-comparison-2026-11-23-ow5SEL/pc-replay/summary.json) · [正常页 JSON](results/board-comparison-2026-11-23-ow5SEL/pc-replay/ok/page-1.json) · [坏图日志](results/board-comparison-2026-11-23-ow5SEL/pc-replay/bad-image.stderr.log)

eMMC 对照使用的模型、工具链和运行库版本见各节记录。复跑时从上游准备对应依赖。

## ARM64 单图 CPU 基线

板端 CPU 首次验证使用 S1 的 `zh-notice.png`、原提示词及同一对 Q4_K_M/BF16 权重。电脑已交叉构建 `edgedoc_cli`，确认它是 AArch64 静态 ELF，使用 `armv8-a` CPU 目标。构建产物在 WSL 的 `~/.cache/edgedoc-xiaomi-vlm/s5-arm64-static/edgedoc_cli`，SHA-256 为 `b7297a65e45fb23caa6fda34afd591fb9ab6f401204fa763256a3dd43dfcdfff`。真机运行时对照 [S1 中文页输出](../s1/results/pc-cpu-acceptance/sequence/page-1.md)，并记录模型能否自然结束、请求耗时和温度。

原板端 ARM64 单图测试包包含可执行文件、图片、提示词、预期输出和两份权重的校验清单。另带用于验证 NPU 通路的卷积模型、主机 CPU 参考输出和 `rknn_board_smoke`，该程序会报告实际库版本和驱动版本。实验使用的 `librknnrt.so` 为瑞芯微 `2.3.2`，可由单个测试进程加载。备齐这些文件后，在测试目录运行以下命令。

```bash
sha256sum -c SHA256SUMS
sh board_cpu_smoke.sh /path/to/models ./cpu-result
```

`/path/to/models` 指向板上两份 GGUF 所在目录。脚本先校验权重，再识别单页并与随包的 PC 原始输出逐字节比较。本轮使用板上已有的 `~/ai/xiaomi-ocr-0/models`。

在 PC 上用 QEMU 模拟 Cortex-A76，ARM64 程序识别了 `640×256` 的设备铭牌，输出与 x86 版逐字节相同。原始 `1100×790` 中文页在 QEMU 进入视觉编码后达到 720 秒运行时限，整页结果和耗时待板上实测。[ARM64 短标签输出](results/pc-arm64-short-label/arm/page-1.md)

## RKNN 最小通路

板端运行库核对完成后，从已解包的 `edgedoc-s5` 目录执行小卷积，输入由程序按固定规则生成。将 128 个输出值与 `expected_l08.json` 比较，并保留运行库和驱动版本日志。

```bash
LD_LIBRARY_PATH=./lib ./rknn_board_smoke ./conv_bn_rk3588.rknn > l08-output.json 2> l08-runtime.log
```

在 WSL Ubuntu 的独立 Python 环境中，用 RKNN-Toolkit2 `2.3.2` 把已有的 L08 卷积测试模型编译为目标平台 `rk3588`。构建时关闭量化。使用相同的 `[1, 1, 8, 8]` float32 输入，RKNN PC 模拟器与 ONNX Runtime 的输出形状均为 `[1, 2, 8, 8]`，最大绝对误差约 `0.000179`，低于这次测试选用的 `0.003` 阈值。[结果与文件校验值](results/pc-rknn-smoke/summary.json)

```bash
rknn_python="$HOME/.cache/edgedoc-xiaomi-vlm/s5-rknn-2.3.2/bin/python"
"$rknn_python" s5/rknn_smoke.py \
  --onnx ../edgedoc-cam/experiments/l08_onnx_audit/results/recorded/conv_bn_original.onnx \
  --output-dir "s5/results/pc-rknn-$(date +%Y%m%d-%H%M%S)"
```

已验证环境使用 Python 3.12、CPU 版 PyTorch `2.4.0+cpu`、NumPy `1.26.4`、ONNX `1.17.0`、ONNX Runtime `1.20.1` 和 setuptools `80.9.0`。这个版本的 RKNN 工具仍依赖 `pkg_resources`，用已验证的 setuptools 兼容版本可以初始化 API。

## Xiaomi-OCR-0 的 PC 实验结果

官方 checkpoint 固定到 `bbd8cf8d21f256b3644b44cb860f3a7cecb93c30`。已下载 `1,746,937,856` 字节的 `model.safetensors`，SHA-256 与仓库记录一致。PC 使用瑞芯微的 Qwen3.5 视觉适配代码和 RKLLM `1.3.1`。[小米模型仓库](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/tree/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30) · [瑞芯微多模态示例](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/examples/multimodal_model_demo/README.md)

| 环节 | PC 实测 |
| --- | --- |
| PyTorch → ONNX | 固定 `448×448` RGB 图像，网格 `[1, 28, 28]`，输出 `196×1024`。适配层之后与 PyTorch 的最大绝对误差约 `0.0000947`。[导出结果](results/pc-xiaomi-vision-onnx.json) |
| ONNX → 视觉 RKNN | `rk3588` 目标，关闭权重量化。真机示例的输入格式为 `uint8` NHWC，已在编译配置中设定 RGB 归一化。PC 模拟器输出 `196×1024`，与 PyTorch 的余弦相似度 `0.9999897895`，均方根误差约 `0.000574`。[模拟器结果](results/pc-xiaomi-vision-rknn-u8.json) |
| 小米原始权重 → 语言 RKLLM | `rk3588`、W8A8、4,096 Token 上下文。用三种 S0 页面实际捕获的语言输入校准，每个样本的输入嵌入为 `[1, 268, 1024]`，位置为 `[3, 1, 268]`。成功导出 `1,300,331,444` 字节的语言模型。[校准输入](results/pc-xiaomi-calibration.json) · [转换结果](results/pc-xiaomi-rkllm.json) |
| 固定 PNG 的 CPU 文本对照 | 官方 BF16 和 S1 的 GGUF 跑同一张 `448×448` PNG、同一份 OCR 提示词，输出逐字节相同，中文通知的五个关键字段全部命中。[固定图输入的参考文本](results/pc-xiaomi-reference-448/page-1.md) · [字段验证](results/pc-xiaomi-reference-448/summary.json) |

图像先由 RKNN 转成 196 个向量，每个向量有 1024 个数。瑞芯微 C++ 示例把这些向量、图像尺寸和 OCR 提示词一起交给 RKLLM，由 RKLLM 生成字符。PC 数值模拟只覆盖视觉侧。`448×448` 是这次选用的固定试验尺寸，CPU 首轮基线使用整页原始分辨率。

PC 复现使用这次已经准备好的 WSL 环境。从项目目录运行，结果写入新的 WSL 缓存子目录。

```bash
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
model="$cache/xiaomi-ocr-0-bbd8cf8"
vision_python="$cache/s5-xiaomi-export/bin/python"
run=$(mktemp -d "$cache/xiaomi-repro.XXXXXXXX")
vendor=../output/xiaomi_ocr_setup/rknn-llm/examples/multimodal_model_demo

"$vision_python" s5/xiaomi_vision_pc.py --model-dir "$model" \
  --vendor-export "$vendor/export/export_vision.py" --image s0/samples/zh-notice.png \
  --output-dir "$run/vision"
"$cache/s5-rknn-2.3.2/bin/python" s5/rknn_xiaomi_vision.py \
  --vision-dir "$run/vision" --runtime-input uint8 --output-dir "$run/rknn"
"$vision_python" s5/xiaomi_calibration_pc.py --model-dir "$model" \
  --prompt-file s1/ocr_prompt.txt --output-dir "$run/calibration" \
  --image s0/samples/zh-notice.png --image s0/samples/en-service.png \
  --image s0/samples/mixed-inventory.png
PYTHONPATH="$cache/s5-rkllm-site:$cache/s5-rkllm-deps" \
  "$vision_python" s5/rkllm_xiaomi_pc.py --model-dir "$model" \
  --dataset "$run/calibration/dataset.json" --output-dir "$run/rkllm"
```

## 固定单图的 NPU 测试目录

原实验使用的 WSL 目录为 `~/.cache/edgedoc-xiaomi-vlm/board-xiaomi-npu-448`，包含两份模型、示例程序 `demo` 和 `imgenc`、随进程加载的运行库、固定图、文本及视觉参考值。[文件清单与校验值](results/pc-xiaomi-npu-ready.json)

两份瑞芯微程序按 Debian 12 ARM64 工具链交叉构建，并在实际板卡上执行。复现时先执行随 CPU 包提供的小卷积测试，确认运行库和驱动可以执行 RKNN，再用固定图分别检查视觉数值和 OCR 文本。

将目录传到板上后，从目录内运行以下命令。`imgenc` 会生成 `img_vec.bin`，运行前使用新目录以区分实验产物。如果视觉日志报告失败，保留日志并结束该次检查。

```bash
sha256sum -c SHA256SUMS
chmod u+x demo imgenc
mkdir vision-result
(cd vision-result && LD_LIBRARY_PATH=../lib ../imgenc ../xiaomi_vision_rk3588.rknn ../vision-input.png 3) > vision-result/vision.log 2>&1
sh run_xiaomi_npu_once.sh > npu-ocr.log 2>&1
```

把 `vision-result/img_vec.bin` 传回 PC 后，在 WSL 项目目录比较输出形状和数值。OCR 字段从 `npu-ocr.log` 提取，与 `expected.md` 对照。日志里同时包含程序提示、模型耗时和生成文本。

```bash
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
"$cache/s5-rknn-2.3.2/bin/python" s5/compare_vision_output.py \
  /path/to/img_vec.bin "$cache/board-xiaomi-npu-448/reference.npy"
```
