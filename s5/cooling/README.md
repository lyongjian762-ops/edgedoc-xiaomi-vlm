# 鲁班猫 4 散热改善后的重复测量（2026-11-26）

用户完成散热安装后，沿用 eMMC Debian 11、Linux 5.10.160、`ondemand` 大核调频、同一张 `448×448` 通知单、同一 OCR 提示词、4096 上下文和 512 输出上限。CPU GGUF、RKNN/RKLLM、MNN 模型和运行库先通过旧校验清单，输入图片 SHA-256 与 11 月 23 日同图测试一致。每个后端两轮独立进程，启动前 SoC 均不高于 45°C。CPU/NPU 顺序与旧同图对照一致，MNN CPU 与 OpenCL 各两轮且相反顺序。Mali 使用之前的调优缓存。四条路径均从进程启动前计到退出，12 次运行均核对了 OCR 正文。

## 前后对照

| 部署路径 | 原进程中位耗时 | 本次进程中位耗时 | 耗时变化 | SoC 最高采样温度（原 → 本次） |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp 四线程 CPU | 51.64 s | 52.26 s | +1.2% | 81.3 → 53.6°C |
| RKNN → RKLLM | 13.13 s | 13.78 s | +4.9% | 57.3 → 46.2°C |
| RKNN → MNN CPU | 7.51 s | 7.98 s | +6.3% | 68.4 → 53.6°C |
| RKNN → MNN Mali OpenCL | 9.43 s | 8.95 s | −5.1% | 61.9 → 42.5°C |

CPU 同图两轮为 53.70 和 50.82 秒，视觉日志间隔中位数从 37.152 降到 36.723 秒，整页请求从 50.091 降到 49.473 秒。第一轮的进程启动开销较高。MNN 的生成速率中位数为 CPU 38.61（原 37.27）和 Mali 24.24（原 22.34）Token/s，RKLLM 为 15.76（原 15.80）Token/s。Mali 另用旧采样脚本独立核验设备，打开了 `/dev/mali0`，GPU 负载达到 100%，完整耗时 9.32 秒，生成速率 22.71 Token/s。三个 Mali 耗时为 8.89、9.00、9.32 秒。两次主测和补测存在波动，因此此次只能观察到小幅性能变化。两次不同日期的测试环境存在启动温度、文件缓存和后台任务差异，缺少旧 CPU 频率时间序列，无法单独量化散热对延迟的因果贡献。

## 持续满载

使用原来的纯 CPU 路径连续识别三次，第一轮等待 SoC ≤45°C，后两轮结束前一进程后立即启动下一进程。

| 页 | 启动前 SoC | 进程耗时 | SoC 峰值 | 全部热区的最大采样值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 37.0°C | 50.806 s | 53.6°C | 57.3°C |
| 2 | 49.9°C | 50.716 s | 53.6°C | 58.2°C |
| 3 | 49.0°C | 50.806 s | 53.6°C | 58.2°C |

每页视觉阶段两个大核频率组分别有 73 个采样，全部为 2304 MHz。记录到的四个热冷却设备的 `cur_state` 最大值均为 0。风扇 PWM 最高 100。对照原来的整页 CPU 实验，所有热区的最大采样温度由 85.9°C 降至 58.2°C。约 152 秒连续工作期间没有观察到延迟持续增长，三个输出与参考正文逐字节一致。

## 数据及复跑

原始日志、热区与频率记录、输出、资产校验和汇总保存于 [本次 PC 归档](../results/board-cooling-2026-11-26-Vn8Rr81e/summary.json)。归档中 `benchmark.py` 为实际执行版本。主测的 GPU 负载字段因设备以 `百分比@频率Hz` 格式报告而未被转成整数，另有 [GPU 核验](../results/board-cooling-2026-11-26-Vn8Rr81e/gpu-confirm/process.json) 和 `samples.csv`，工作区脚本已补充对应解析。系统调频、热保护及风扇配置均沿用板上现有值。

板端复跑时复制本目录 `benchmark.py` 到单独测试目录，指定已有部署路径。输出目录须为新目录。

```sh
base=/home/cat/ai/edgedoc-xiaomi-vlm
python3 benchmark.py baseline-new "$base/s5-emmc-2026-11-23.ZcXVOr" \
    "$base/s5-npu-mnn-2026-11-24.eAgvzp"
python3 benchmark.py sustained-new "$base/s5-emmc-2026-11-23.ZcXVOr" \
    "$base/s5-npu-mnn-2026-11-24.eAgvzp" --suite sustained
```

在 PC 的项目根目录重新汇总原数据：

```sh
python3 s5/cooling/summarize.py s5/results/board-cooling-2026-11-26-Vn8Rr81e \
    --previous-comparison s5/results/board-comparison-2026-11-23-ow5SEL \
    --previous-hybrid s5/results/board-npu-mnn-2026-11-24-eAgvzp
```
