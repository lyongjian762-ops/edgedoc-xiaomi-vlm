# EdgeDoc Cam · Xiaomi-OCR-0

在鲁班猫 4（RK3588S）上部署单页文档 OCR，使用 C++ 串接推理库，实测 CPU、NPU 和 Mali-G610 GPU。覆盖 PC 功能基线、板端推理、模型常驻、GPU 预处理和算子对照。

[示例输入](s0/samples/zh-notice.png) · [OCR 输出](s1/results/pc-cpu-acceptance/single/page-1.md) · [C++ 入门讲解](docs/reading-the-cpp-pipeline.md) · [板端实验](s5/README.md)

## 真机结果

下表为鲁班猫 4 V1（4 GB，eMMC Debian 11）代表性结果，计时范围各自注明。

| 实验 | 结果 | 测量范围 |
| --- | --- | --- |
| CPU 与 RKNN → RKLLM | **51.64 → 13.13 秒，3.93×** | 同一张 448×448 合成页的完整进程耗时。两路使用各自的权重精度，OCR 正文一致。[数据](s5/results/board-comparison-2026-11-23-ow5SEL/summary.json) |
| NPU 模型常驻 | **94.01 → 81.30 秒，减少 13.5%** | 五页累计进程耗时。模型加载从五次降到一次，输出逐字节一致。[数据](s5/results/board-resident-2026-11-24-1YZuKC/summary.json) |
| OpenCV 与 Mali OpenCL 预处理 | **5.106 → 4.434 ms，1.15×** | 缩放和颜色转换，GPU 含数据交接。两路共用 CPU 解码。[数据](s5/results/board-gpu-preprocess-2026-11-24-6FnYRe/summary.json) |
| RKNN 视觉接 MNN 语言 | CPU **37.27**，Mali **22.34 Token/s** | 生成阶段。两种 MNN 后端输入相同，输出均与参考正文一致。[数据](s5/results/board-npu-mnn-2026-11-24-eAgvzp/summary.json) |
| 四比特语言投影 | 单行 CPU **0.086 ms**，Mali **0.976 ms** | 同一层、同一权重，计到主机可读结果。Mali 一次处理 128 行时，均摊约 **0.066 ms/行**。[数据](s5/results/board-mnn-linear-2026-11-24-CQVuiW/summary.json) |

结果对应固定样本与上述配置。各实验说明记录了计时边界、逐轮数据和输出校验。多页结果来自小型开发集，保留组和文档正文 CER 留待后续评测。

## 程序怎么走

```text
图片 + OCR 提示词
  ├─ C++ / llama.cpp + libmtmd → CPU 视觉与语言推理
  └─ RKNN 视觉编码 → 图像特征 → RKLLM 或 MNN 语言生成
                                └─ MNN 对照 CPU 与 Mali OpenCL
  → Markdown 正文 + JSON 运行记录
```

| 内容 | 入口 |
| --- | --- |
| PC 单页基线与 C++ 请求状态 | [S0/S1 验收](docs/pc-baseline.md) · [C++ 入口](src/edgedoc_cli.cpp) |
| RKNN、RKLLM 转换与上板流程 | [S5 实验记录](s5/README.md) |
| 同一进程连续识别多页 | [NPU 常驻](s5/npu-resident/README.md) |
| GPU 缩放和视觉矩阵乘法 | [预处理](s5/gpu-preprocess/README.md) · [QKV 对照](s5/gpu-matmul/README.md) |
| NPU 视觉特征接 MNN 与生成性能 | [MNN 混合链路](s5/mnn-hybrid/README.md) · [四比特单层](s5/mnn-linear/README.md) |

## 在 PC 上跑第一张图

在 Ubuntu 或 WSL 从仓库根目录执行。需要 Git、Git LFS、CMake 3.20+、C/C++ 编译器和 Python 3.11+。

```sh
mkdir -p third_party
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout --detach 8642541b255bfb5dc39cd351a178be09013a55d6

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF models
git -C models lfs pull --include="Xiaomi-OCR-0.Q4_K_M.gguf,Xiaomi-OCR-0.mmproj-bf16.gguf"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DEDGEDOC_LLAMA_SOURCE="$PWD/third_party/llama.cpp" \
  -DGGML_NATIVE=ON -DGGML_CUDA=OFF
cmake --build build --target edgedoc_cli -j 4

./build/edgedoc_cli --model models/Xiaomi-OCR-0.Q4_K_M.gguf \
  --mmproj models/Xiaomi-OCR-0.mmproj-bf16.gguf \
  --prompt-file s1/ocr_prompt.txt --image s0/samples/zh-notice.png \
  --output-dir runs/first-page
```

结果保存在 `runs/first-page/page-1.md`，运行记录同目录见 `page-1.json`。板端实验需要 ARM64 工具链、对应驱动、运行库及转换后的模型。[板端版本和命令](s5/README.md)

## 文件与来源

随仓库提供实验源码、三张自制文档、选取的原始计时、汇总与复现说明。模型及运行库按上游说明获取。第三方评测样本可按索引在数据集原处定位。[测试集索引](data/ocr-eval-mini/README.md) · [小米模型](https://github.com/xiaomi-research/Xiaomi-OCR-0) · [GGUF 转换](https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF) · [RKNN](https://github.com/airockchip/rknn-toolkit2) · [RKLLM](https://github.com/airockchip/rknn-llm) · [MNN](https://github.com/alibaba/MNN)
