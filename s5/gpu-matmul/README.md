# 用 Mali-G610 跑 Xiaomi-OCR-0 的第一层视觉 QKV 矩阵乘法

## 这次算了什么

固定的 `448×448` 图片被分成 `28×28 = 784` 个图像块。图像块进入第一个视觉注意力块时，每个块用 768 个数表示。QKV 投影把每个块的这 768 个数映射为 2,304 个数，依次对应 Q、K、V。实验对照的是这一步的矩阵乘法。

```text
784 个块 × 768 个输入特征        768 个输入特征 × 2304 个输出特征
             X                 ×                 Wᵀ
                              ↓
                784 个块 × 2304 个 QKV 特征
```

[extract_qkv.py](extract_qkv.py) 用模型自己的处理器读取样图，在第一个 QKV 层收到输入时取出该矩阵，再从对应层读取并转置真实权重。视觉前段用 FP32 生成激活，实验的输入和权重存为 FP16，矩阵乘法用 FP32 累加并输出。校验范围为矩阵乘法本身，止于加 bias 和计算注意力之前。[固定模型版本](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/tree/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30) · [输入矩阵清单](../results/board-mali-matmul-2026-11-24-4vs7mI/operands/manifest.json)

## 从哪里看代码

1. 先看 [compare.cpp](compare.cpp) 里的 `main()`。它从文件读出输入和权重，把同一组数交给 CPU 和 GPU。CPU 计算调用 `cblas_sgemm`，实际矩阵乘法由 OpenBLAS 完成。
2. 再看 `MaliMatMul::configure()` 和 `run()`。前者把权重传给 GPU 一次，后者每轮把输入写进缓冲区、派发核函数、等结果读回。
3. 最后看 [qkv.cl](qkv.cl)。一个工作项负责输出的一行中连续 4 列。它沿 768 个输入特征，每次取 16 个，反复执行“输入值乘对应权重，再加到输出”。同组工作项复用局部内存中的输入和权重。代码中第一次 `barrier` 等整组读完这一批数据，第二次等整组用完，再覆写这块内存。[OpenCL C 的向量读写函数](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#vector-data-load-and-store-functions)

## 鲁班猫 4 真机结果

eMMC Debian 11.11，内核 `5.10.160`，GPU 为 Mali-G610 r0p0。CPU 用 OpenBLAS 0.3.13，4 个线程，进程约束在已确认的 4 个 Cortex-A76 大核。预热各 2 次，然后交替先跑 CPU 或 GPU，共 15 对。CPU 预先把同一份 FP16 操作数转成 FP32。GPU 初始化、编译以及一次性权重上传独立计时，逐轮计时包含输入上传、运算和输出读回。

| 指标 | 中位数 | 最小～最大 |
| --- | ---: | ---: |
| CPU 矩阵乘法 | 23.17 ms | 21.82～44.17 ms |
| GPU 核函数 | 70.78 ms | 70.63～71.04 ms |
| GPU 含传输和等待 | 72.70 ms | 72.09～73.89 ms |

GPU 含传输耗时是 CPU 的 **3.14 倍**。单独看 GPU 核函数，耗时仍高于 CPU 最慢的一次。优化这一实现时，可以继续把注意力放在核函数的取数、复用和同步上。当前计时覆盖整个核函数，三者的耗时还无法分别量化。

两路均对照 PC PyTorch 用同一份 FP16 操作数进行的 FP32 矩阵乘法。GPU 的 1,806,336 个输出中，最大绝对误差为 `1.1444e-5`，低于 `1e-3` 阈值，整体相对 L2 误差为 `4.61e-7`，低于 `1e-5` 阈值。额外测试缺失输入文件时，程序报告路径并以状态码 1 退出。

三份 GPU 矩阵缓冲合计 11.41 MiB。一个进程内完成 CPU 和 GPU 对照，峰值 RSS 为 92.59 MiB，含运行库与主机侧缓冲。采样到的全板最低可用内存为 3.21 GiB，测试目录含测试库和两轮输出约 49.8 MiB，eMMC 测后还有约 21.0 GiB。[性能、误差和内存摘要](../results/board-mali-matmul-2026-11-24-4vs7mI/summary.json) · [逐轮计时](../results/board-mali-matmul-2026-11-24-4vs7mI/board/timing.csv) · [存储记录](../results/board-mali-matmul-2026-11-24-4vs7mI/environment/storage-after.txt)

## 16 项、32 项、CLBlast 和 RKNPU 的对照

仍用同一组 `[784,768] × [768,2304]` 操作数，在同一块 eMMC Debian 11 鲁班猫 4 上复测。每个后端各两轮，每轮预热 2 次后测 15 次。两轮结束时的输出分别和同一份 PyTorch 参考逐项对照。每条路径的两份输出逐字节一致，GPU 的最大绝对误差为 `1.1444e-5`，NPU 为 `3.8147e-6`。

| 实现 | 计算阶段中位数 | 含输入和输出的中位数 |
| --- | ---: | ---: |
| CPU，四大核 OpenBLAS | 23.30 ms | 23.30 ms |
| Mali 手写核，每轮 16 项 | 70.78 ms | 72.66 ms |
| Mali 手写核，每轮 32 项 | 89.41 ms | 91.70 ms |
| Mali CLBlast 1.5.2，FP32 SGEMM | 44.47 ms | 46.57 ms |
| RKNPU 单核，FP16 输入，FP32 输出 | 10.41 ms | 12.89 ms |

CPU 行使用与 16 项 GPU 核配对的 OpenBLAS 计时，输入已转成 FP32。GPU 权重预先上传，CLBlast 使用同值的 FP32 输入和权重，设备计算阶段用队列中的两个标记计时，涵盖该库调用提交的所有计算和布局转换。NPU 使用预先打包的权重，计算列是同步 `rknn_matmul_run()` 调用的墙钟时间。NPU 总时长另含板上内存复制和缓存同步。这是一个 QKV 矩阵乘法的对照。[两轮计时与输出误差](../results/board-qkv-backends-2026-11-24-l0E4mu/summary.json)

- 32 项版本把每组同步次数从 96 降到 48，组内临时数组从 2.5 KiB 增至 5 KiB。两轮都慢了约 19 ms，说明改变分块大小的综合效果要用实测判断。当前采样无法把同步、取数和寄存器占用分别计时。[32 项核函数](qkv_tile32.cl)
- CLBlast 中位耗时比 16 项版本少 35.9%，仍高于同轮 CPU 的 26.58 ms。30 次测量中有约 134 ms 的慢样本，P95 为 96.51 ms。第一次调用约 1.76 秒，包含库内的编译和初始化开销。初次测试在结果落盘后退出崩溃，在 OpenCL 上下文释放前调用 `CLBlastClearCache()` 后，两轮均正常结束。[CLBlast 调用](compare.cpp) · [初次日志](../results/board-qkv-backends-2026-11-24-l0E4mu/initial-failures/clblast/process.json)
- 单核 NPU 两轮的含输入和读回中位数为 7.35 和 13.02 ms。30 次中最慢的样本为 13.86 ms。当前独立 MatMul 接口拒绝三核掩码，表中使用核心 0。[NPU 实现](compare_npu.cpp) · [三核请求日志](../results/board-qkv-backends-2026-11-24-l0E4mu/initial-failures/npu/benchmark.log) · [瑞芯微 API 示例](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknpu2/examples/rknn_matmul_api_demo/README.md)

### 复跑本次对照

板端目录是 `/path/to/board-run`，含操作数、头文件和运行库。CLBlast 用 Debian 11 ARM64 软件包的本地解包，RKNN 使用板上已验证的运行库。[CLBlast 软件包](https://packages.debian.org/bullseye/libclblast1) · [构建校验值](../results/board-qkv-backends-2026-11-24-l0E4mu/environment/final-build-sha256.txt)

```bash
export LD_LIBRARY_PATH="$PWD/lib" OPENBLAS_NUM_THREADS=4
sha256sum -c operands/SHA256SUMS
g++ -O3 -std=c++17 -Wall -Wextra -Werror -Iinclude $(pkg-config --cflags opencv4) \
    -DEDGEDOC_USE_CLBLAST compare.cpp -o compare_qkv \
    -lopencv_core -L"$PWD/lib" -l:libopenblas.so.0 -l:libclblast.so.1 -l:libOpenCL.so.1
g++ -O3 -std=c++17 -Wall -Wextra -Werror -Iinclude compare_npu.cpp -o compare_npu \
    -L"$PWD/lib" -l:librknnrt.so

replay=$(mktemp -d ./replay.XXXXXX)
for backend in tile16 tile32 clblast npu-core0; do
    python3 run_board.py "$replay/$backend" --cpus 4-7 --backend "$backend"
done
```

在 PC 的项目目录校验两轮归档输出并重算统计。

```bash
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
PYTHONDONTWRITEBYTECODE=1 "$cache/s5-xiaomi-export/bin/python" s5/gpu-matmul/summarize_backends.py \
    s5/results/board-qkv-backends-2026-11-24-l0E4mu \
    s5/results/board-mali-matmul-2026-11-24-4vs7mI/operands
```

## 复跑首次 GPU 实验

板端工作目录是 `/path/to/board-run`。其中有首次测量用的 C++、OpenCL 和采样脚本，以及已校验的 `operands`、`include` 和 `lib`。OpenCL 头文件与 OpenBLAS 来自 Debian 软件包的本地解包，动态库路径只设给测试进程。[首次源码快照](../results/board-mali-matmul-2026-11-24-4vs7mI/source/) · [Debian Bullseye OpenBLAS](https://packages.debian.org/bullseye/libopenblas0-pthread)

```bash
sha256sum -c operands/SHA256SUMS
g++ -O3 -std=c++17 -Wall -Wextra -Werror -Iinclude $(pkg-config --cflags opencv4) \
    compare.cpp -o compare_qkv -lopencv_core -L"$PWD/lib" -l:libopenblas.so.0 -l:libOpenCL.so.1
LD_LIBRARY_PATH="$PWD/lib" OPENBLAS_NUM_THREADS=4 python3 run_board.py board-retry --cpus 4-7
```

PC 归档中保留了原始输入、PyTorch 参考输出和两路板端输出。从项目目录可重新检查已归档的结果。

```bash
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
"$cache/s5-xiaomi-export/bin/python" s5/gpu-matmul/summarize.py \
    s5/results/board-mali-matmul-2026-11-24-4vs7mI
```
