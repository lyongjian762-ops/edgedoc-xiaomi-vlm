# lm_head 批量输入与顺序内存读取（2026-11-26）

针对单行 GPU 性能和内存带宽问题，复用上一轮 [lm_head 操作数](../results/board-lm-head-2026-11-26-Ue2nidO0/operands/manifest.json)。同一份 `[1024,248320]` 四比特权重和量化参数约 143.03 MB（136.4 MiB）。从已捕获的隐藏矩阵取最后 1、2、4、8 行供两个后端计算，每组预热五次、测量三十次，两轮反转后端及输入行数顺序。计时至主机 CPU 可读取结果。

| 输入行数 | 四大核 CPU 总耗时 | Mali 总耗时 | CPU 均摊/行 | Mali 均摊/行 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.533 ms | 22.049 ms | 6.533 ms | 22.049 ms |
| 2 | 12.891 ms | 16.352 ms | 6.445 ms | 8.176 ms |
| 4 | 9.452 ms | 19.845 ms | 2.363 ms | 4.961 ms |
| 8 | 18.530 ms | 38.696 ms | 2.316 ms | 4.837 ms |

总耗时取合并两轮 60 个样本的中位数，均摊耗时为总耗时除以行数。GPU 单行两轮各为 11.113、22.159 ms，具有明显波动。8 行 GPU 两轮为 38.758、38.612 ms。CPU 的 4 行总耗时低于 2 行，两轮复现，需进一步核对具体执行分支。独立解包 Q4 权重逐块重算每行，CPU、Mali 的最大逐行相对 L2 误差分别约 0.465%、0.173%。所有 GPU 运行均打开 Mali 设备。

已知多行可以共享权重和并行计算，提高吞吐。单请求普通自回归生成中，下一个输入 Token 在前一个输出后才确定。多请求 Decode batching 和 speculative decoding 可以创造多行输入机会。该实验只衡量一个线性投影，尚未完成多请求完整 OCR 并发和状态隔离。

## 独立顺序读取速度

[read_bandwidth.cpp](read_bandwidth.cpp) 初始化 128 MiB 无符号整数数组，每轮读取全部数据并计算校验和。CPU 使用 4 个 Cortex-A76 上的 OpenMP SIMD 并行规约，Mali 使用 OpenCL `uint4` 向量加载与分区累加。GPU 初始化、最后读取校验和在核函数计时之外。事件计时覆盖 GPU 上的读取、累加及小型部分和输出。两个后端各两轮，每轮预热五次、计时三十次，所有校验和一致。

| 后端 | 有效读取带宽中位数 | 两轮中位数 |
| --- | ---: | ---: |
| CPU 四大核 | 26.709 GB/s | 26.695 / 26.751 GB/s |
| Mali 设备端事件 | 21.050 GB/s | 20.916 / 21.101 GB/s |
| Mali 主机提交至设备完成 | 19.458 GB/s | 见原始记录 |

这是指定顺序读取和累加程序的有效数据率。若以此速度顺序读取与 `lm_head` 序列化权重相同的 143.03 MB，CPU、Mali 对应约 5.36 和 6.79 ms。真实 MNN 运行时权重可能重排、解包、缓存，不能据此直接认定实际 DDR 流量或硬件带宽已经饱和。单行 head 的访存模式、解包成本、核函数事件耗时值得继续剖析。[数据汇总](../results/board-batch-bandwidth-2026-11-26-HejYdFnl/summary.json)

相关研究和实现：NVIDIA 的 [CUDA 性能指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 解释内存访问布局与有效带宽。[Marlin](https://github.com/IST-DASLab/marlin) 在 NVIDIA GPU 上通过 INT4 权重布局、访存及数据流水优化利用 Tensor Core 执行线性层。[投机解码论文](https://arxiv.org/abs/2211.17192) 讨论先草拟再并行验证多个 Token。

## 复跑

同前次 [lm_head 构建说明](../lm-head/README.md) 编译 `mnn-linear/bench.cpp`，输出宽度设为 248320。将最后一个参数替换为 `batch`，运行 1、2、4、8 行模式。带宽程序在板端采用原 OpenCL 头文件及 GCC 10 构建。

```sh
g++ -std=c++17 -O3 -march=armv8-a+simd -fopenmp -Wall -Wextra -Werror \
  -isystem /path/to/opencl-headers read_bandwidth.cpp -l:libOpenCL.so.1 -o read_bandwidth
taskset -c 4-7 ./read_bandwidth cpu
taskset -c 4-7 ./read_bandwidth opencl
```

PC 归档包含各轮的计时和输出、编译源码、二进制和校验值。[独立校验与汇总代码](summarize.py)。
