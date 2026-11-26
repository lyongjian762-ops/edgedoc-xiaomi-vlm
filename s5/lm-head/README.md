# lm_head 单行真实权重对照（2026-11-26）

前面的完整 Decode 剖析发现 `/lm/lm_head/Linear` 为最大单个热点。这次从原 MNN 图和权重文件精确提取这一层，保留四比特权重及 FP16 量化参数（136.4 MiB）。输入取原始 BF16 模型同一张 448×448 OCR 图的最后一行 Prefill 隐藏向量，形状为 `[1,1024]`。它与单行 Decode 的矩阵形状一致，两个后端均使用此向量。输出宽度 248320，单次 254,279,680 次乘加。

## 实测

复用 [MNN 单层测试](../mnn-linear/bench.cpp)，编译时将输出宽度设为 248320，使用 `single` 模式。四大核 CPU、Mali OpenCL 缓冲区低内存低精度模式均在板上加载同一个独立模型，每个进程预热 5 次，记录 30 次。普通模式按 CPU A、GPU A、GPU B、CPU B 的顺序运行。

| 普通模式 | 合并两轮耗时中位数 | `onForward` 返回前 | 主机可读结果的剩余等待 |
| --- | ---: | ---: | ---: |
| CPU | 6.993 ms | 6.846 ms | 0.143 ms |
| Mali | 10.527 ms | 0.193 ms | 10.300 ms |

GPU 两轮中位数为 10.451、21.910 ms，CPU 为 6.538、7.373 ms，运行存在波动。为分开队列完成和结果访问，另做两轮 `single wait`，在 `onForward()` 返回后、`readMap()` 前调用 `Tensor::wait(MAP_TENSOR_READ, true)`。固定 MNN 3.6.1 的 OpenCL `onSync` 实现调用 `commandQueue().finish()`。结果如下。

| 补测 Mali | 中位数 |
| --- | ---: |
| `onForward` 返回前 | 0.185 ms |
| 等 OpenCL 队列完成 | 11.664 ms |
| 剩余等待与主机结果访问 | 2.327 ms |
| 总耗时 | 14.173 ms |

补测的同步位置与普通模式不同，各阶段中位数分别计算。队列完成时间包含已提交命令的调度、执行和等待，CPU 可读结果时间还可能包含布局转换与复制。等待队列本身占了主要延迟，GPU 核心实际访存字节数、算术单元利用率及单个核函数事件耗时仍待测。[Tensor::wait](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/source/core/Tensor.cpp#L489) · [OpenCL 同步实现](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/source/backend/opencl/core/OpenCLBackend.cpp#L884)

所有 GPU 进程均打开 `/dev/mali0`，采样峰值负载为 100%。独立解包四比特权重并在 PC 逐块计算参考 logits。普通 CPU 最大相对 L2 误差约 0.104%，普通 GPU 约 0.053%，两次显式等待补测约 0.039%。六次预测的最大分数位置均为 Token ID 550，与独立参考一致。 [完整数据](../results/board-lm-head-2026-11-26-Ue2nidO0/summary.json)

## 性能原理及下一步

一行输入对完整权重做一次矩阵向量乘法（GEMV）。该权重的序列化负载约 143 MB，每个权重对当前向量只参与一次乘加。如果读取权重一次，单行约为 1.78 MAC/字节。缓存、量化解包与实际运行时权重布局会改变物理访存量，因此此数只描述当前序列化数据规模。GPU 的并行单元需要及时收到权重，四大核 CPU 的权重读取、解包和累加实现也会影响最终速度。下一步针对 GEMV 比较实际内存访问和核函数执行。

## 复跑和核验

使用前次固定环境，在 WSL 项目根目录执行。

```sh
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
source="$cache/s5-mnn-3.6.1"
build="$cache/s5-mnn-arm64-debian11"
toolchain="$cache/debian11-cross/sysroot"
"$toolchain/usr/bin/aarch64-linux-gnu-g++-10" --sysroot="$toolchain" \
  -B/usr/aarch64-linux-gnu/bin/ -O2 -std=c++17 -fno-rtti -Wall -Wextra -Werror \
  -DEDGEDOC_LINEAR_OUTPUT_WIDTH=248320 -isystem "$source/include" \
  s5/mnn-linear/bench.cpp -L"$build" -lMNN -lpthread -o /tmp/mnn_lm_head_bench
```

真实图和权重的提取见 [prepare.py](prepare.py)，所有操作数已保存在 PC 归档。运行测试时分别选择 `cpu`、`opencl` 和 `forward single`，追加 `wait` 可打开显式同步。用下面命令重新统计保存的六轮记录。

```sh
OPENBLAS_NUM_THREADS=2 "$cache/s5-xiaomi-export/bin/python" \
  s5/lm-head/summarize.py s5/results/board-lm-head-2026-11-26-Ue2nidO0
```
