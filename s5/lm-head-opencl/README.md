# Mali-G610 的自定义四比特 GEMV

目标为独立优化 Xiaomi-OCR-0 的 `lm_head` 单行 `[1,1024] × [1024,248320]` 投影。沿用已经校验的原 MNN 四比特字节、FP16 量化参数和 BF16 隐藏向量。与 MNN 同图算子在同一板上比较。本篇记录独立算子测试；后续整链路接入及真实 OCR 验证见 [integration.md](integration.md)。

## 实现

原权重每个输出通道占 512 个 Q4 字节，量化组包含 64 个权重。每组的实际权重为 `offset + scale * q`。因此一组的点积可以写成：

$$
\sum_i x_i w_i = scale\sum_i x_i q_i + offset\sum_i x_i
$$

输入组和只算一次，传给所有输出工作项使用。GPU 从每个字节拆出高、低四比特，通过向量 FMA 累积，组结束后再乘量化参数。全矩阵始终保持四比特存储。[核函数 gemv.cl](gemv.cl)

基础布局中，相邻工作项负责相邻输出通道，读取相同量化组的地址会相隔 512 字节。`tile64` 将 64 个输出通道内的同组同块权重排在一起，相邻工作项读取相邻的八字节块。重排后的量化参数遵循相同输出分组。从重排文件按逆排列恢复原始 packed 权重和量化参数，逐字节校验通过。[prepare.py](prepare.py)

FP32 版本采用 `float8` 组内累积。FP16 版本采用 `half8` 组内累积，组间仍用 FP32。本张图的输入转成 FP16 后可精确恢复原浮点值，实际 FMA 舍入会产生小量误差。

## 真机数据

18 个进程，每个预热 5 次、测量 30 次，每种配置两轮。自定义 GPU 的墙钟从输入和组和上传开始，直到完整 248320 个浮点 logits 读回 CPU。权重加载和输入重排位于计时外，MNN 基线的权重加载与输入打包也位于计时外。OpenCL 事件单独记录核函数。

| 版本 | 工作组大小 | 核函数中位数 | 完整耗时中位数 |
| --- | ---: | ---: | ---: |
| MNN CPU | — | — | 6.539 ms |
| MNN OpenCL | — | — | 10.402 ms |
| 原排列 FP32 | 64 | 9.598 ms | 10.551 ms |
| 重排 FP32 | 32 | 18.893 ms | 19.969 ms |
| 重排 FP32 | 64 | **7.670 ms** | **8.648 ms** |
| 重排 FP32 | 128 | 7.681 ms | 8.696 ms |
| 重排 FP16 | 32 | 7.834 ms | 8.808 ms |
| 重排 FP16 | 64 | 7.785 ms | 8.708 ms |
| 重排 FP16 | 128 | 7.818 ms | 8.708 ms |

FP32、64 工作项版本，两轮完整耗时中位数各为 8.607、8.748 ms。对自定义原排列版降低约 **18.0%**，对同轮 MNN OpenCL 降低约 **16.9%**，仍比 MNN CPU 耗时多约 32.2%。FP32/32 工作组第二轮有明显慢样本，具体原因待查。FP16 的组内计算未观察到进一步稳定提速。

18 次运行的 248320 个输出均通过独立 Q4 参考校验，最高分对应 Token ID 550。自定义 FP32 最大全量相对 L2 误差为约 1.27e-7，FP16 为约 5.34e-5。MNN CPU 与 GPU 也通过参考校验。所有 GPU 路径均打开 Mali 设备。

按照该层约 143.03 MB 序列化的权重和量化参数除以 7.670 ms 核函数时间，得到约 18.65 GB/s 的等效 payload 处理速度。上一次独立顺序读取程序的 Mali 实测为 21.05 GB/s。两者的量级接近，加上 FP16 未稳定提速，为访存成本占较大比重提供了证据。这个换算没有使用硬件实际 DDR 事务计数器。

[完整统计与误差](../results/board-q4-gemv-2026-11-26-fXwmLVXJ/summary.json) · [完整板端结果目录](../results/board-q4-gemv-2026-11-26-fXwmLVXJ)

## 复跑

在 PC 的项目根目录使用前次 NumPy 环境提取权重重排和输入数据。

```sh
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
"$cache/s5-xiaomi-export/bin/python" s5/lm-head-opencl/prepare.py \
  s5/results/board-lm-head-2026-11-26-Ue2nidO0/operands /tmp/gemv-prepared
```

板上使用已有的 OpenCL 头文件，编译 [bench.cpp](bench.cpp)。将 `gemv.cl` 和 prepared 目录放入测试目录。

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -isystem /path/to/OpenCL-headers bench.cpp -l:libOpenCL.so.1 -o q4_gemv_bench
taskset -c 4-7 ./q4_gemv_bench tile-fp32 gemv.cl /path/to/lm_head.mnn.weight prepared 64
```

PC 上重新统计需使用原来的 `lm_head` 操作数和独立参考 logits。

```sh
OPENBLAS_NUM_THREADS=2 "$cache/s5-xiaomi-export/bin/python" s5/lm-head-opencl/summarize.py \
  s5/results/board-q4-gemv-2026-11-26-fXwmLVXJ \
  s5/results/board-lm-head-2026-11-26-Ue2nidO0/operands
```
