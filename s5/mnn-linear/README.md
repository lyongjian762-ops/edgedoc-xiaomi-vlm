# Mali 四比特语言投影的输入行数对照

使用 Xiaomi-OCR-0 第一个语言块的 `mlp.up_proj`。从部署用的 MNN 模型中提取原样的四比特权重和 FP16 量化参数。用项目里的同一张通知单和提示词，从原始模型捕获进入该层的 268 行激活。每组取激活矩阵的最后 `M` 行，计算 `[M,1024] × [1024,3584]`。[权重与输入清单](../results/board-mnn-linear-2026-11-24-CQVuiW/operands/manifest.json)

## 测量与校验

鲁班猫 4 eMMC Debian 11.11，MNN 3.6.1。CPU 使用 4 个 Cortex-A76，OpenCL 使用此前部署的低内存、低精度、缓冲区配置。两轮交替调换 CPU/GPU 顺序和输入行数顺序，每组预热 5 次、测 30 次。计时从 MNN 调用开始，到主机可以读取结果为止。固定的输入打包在计时前，输出回读在计时内。两轮 OpenCL 都观察到 Mali 设备和 GPU 负载。

首轮将逐行数据直接写入了交错的 NC4HW4 布局，导致多行输出无法和原始输入的参考结果对应。修正了每四通道一组的打包和结果恢复后，重新跑完两轮。独立解包同一份四比特权重，在 PC 计算参考值。五种形状的相对 L2 误差最大为 CPU 4.04%、Mali 0.49%。两路精度和计算路径由各自的 MNN 后端决定。首次布局错误的原始记录保存在归档，下面使用修正后的测量。[校验代码](summarize.py) · [逐轮数据](../results/board-mnn-linear-2026-11-24-CQVuiW/summary.json)

| 输入行数 | 四大核 CPU 总耗时 | Mali 总耗时 | Mali 平均到每行 |
| --- | ---: | ---: | ---: |
| 1 | 0.086 ms | 0.976 ms | 976.4 µs |
| 8 | 0.507 ms | 3.073 ms | 384.1 µs |
| 32 | 1.151 ms | 6.215 ms | 194.2 µs |
| 128 | 5.683 ms | 8.391 ms | 65.6 µs |
| 268 | 10.616 ms | 17.482 ms | 65.2 µs |

表中总耗时为合并两轮 60 次的中位数，平均到每行用总耗时除以行数。其中 CPU 单行两轮中位数为 0.247 和 0.055 ms，GPU 32 行分别为 6.426 和 3.151 ms，可看到较明显的运行波动。GPU 从 1 行增加到 268 行后，每行耗时约降至原来的 1/15。CPU 在这五种配置下的总耗时都更短。

另做两轮单行 OpenCL 调优对照，保持权重、输入和缓冲区不变，只把 MNN 的 `WIDE` 调优改成 `HEAVY`。合并中位数为 0.963 和 0.981 ms，两轮的快慢方向相反，输出保持一致。最初尝试用 `HEAVY` 扫过所有行数，在 80 秒时限内未完成。[MNN 调优选项](https://github.com/alibaba/MNN/blob/3.6.1/include/MNN/MNNForwardType.h) · [多形状调优日志](../results/board-mnn-linear-2026-11-24-CQVuiW/runs/round-a-opencl-heavy/process.json)

这个试验衡量单层投影和边界开销。完整模型的逐 Token 生成还包含其它投影、注意力、状态更新和取样。要决定优先优化哪段，可以在真实生成路径上测 GPU 命令间隙、回读点和算子耗时，定位后再评估连续算子融合或单行四比特核的收益。

## 读代码和复跑

- [prepare.py](prepare.py) 从 MNN 权重里精确取该投影使用的字节，并保存真实激活与原始权重参考。
- [bench.cpp](bench.cpp) 把同一输入交给两个后端。先看 `benchmark_rows()` 里的输入行切片、打包和计时起止位置。
- [summarize.py](summarize.py) 用独立解包后的四比特权重重算参考，并汇总两轮时长。

从板端目录 `/path/to/board-run` 运行。

```sh
export LD_LIBRARY_PATH="$PWD"
python3 run_board.py --timeout 80 example-cpu taskset -c 4-7 \
    "$PWD/mnn_linear_bench_packed" cpu "$PWD/operands/up_proj.mnn" "$PWD/operands/input.f32" forward
python3 run_board.py --timeout 80 example-opencl taskset -c 4-7 \
    "$PWD/mnn_linear_bench_packed" opencl "$PWD/operands/up_proj.mnn" "$PWD/operands/input.f32" forward
```

在 PC 的项目目录重算归档统计。

```sh
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
OPENBLAS_NUM_THREADS=2 "$cache/s5-xiaomi-export/bin/python" s5/mnn-linear/summarize.py \
    s5/results/board-mnn-linear-2026-11-24-CQVuiW
```
