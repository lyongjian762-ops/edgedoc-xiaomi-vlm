# S5：自定义 Q4 lm_head 接入整页 OCR

2026-11-26，鲁班猫 4 / RK3588S / eMMC Debian 11 / Linux 5.10.160 / Mali-G610。沿用 MNN 3.6.1（`d407447ed56c4121a11ccbd266dc184ca1ead0c2`）和 RKNN 448×448 视觉编码器。在此前的独立算子优化基础上，本轮实际生成完整 OCR 正文，而不只计算一个固定输入的矩阵乘法。

## 1. 从一个算子到整条生成链路

```text
448×448 PNG
    │
    ▼ NPU：RKNN 视觉编码
[196,1024] 图像特征 + 文字 embedding（总共 268 个输入位置）
    │
    ▼ GPU：MNN OpenCL 语言主干
注意力、MLP、KV Cache 更新 → 选取最后一个位置 → RMSNorm
    │
    ▼ [1,1,1024] hidden_states（经主机内存交接）
自定义 Mali OpenCL Q4 lm_head（权重与 OpenCL 缓冲区跨 Token 复用）
    │
    ├── 完整 logits：[1,1,248320] → CPU 调 MNN 原采样器
    └── 分组最大值：3880 对 (score, token id) → CPU 选全局最大值
    │
    ▼ 下一个 Token → embedding → MNN 语言主干 → 重复，直到 EOG
```

`lm_head` 的工作是将最后一个 Token 的 1024 个隐藏特征映射成 248320 个词表分数。Q4 矩阵本身就有约 143 MB 权重和量化参数；这一步并不是注意力，也不是 KV Cache。这里仍然逐 Token 生成，不是将未知的下一个 Token 提前并行计算。

## 2. 代码与原理

| 文件 | 负责的工作 |
| --- | --- |
| [export_trunk.py](export_trunk.py) | 检查原图最后三个算子及连线，移除 `/lm/lm_head/Linear`，让 MNN 输出最后 RMSNorm 后的隐藏向量。保留主干、原外部权重文件及偏移。接口仍命名 `logits`，但形状为 `[1,-1,1024]`，以匹配 MNN 3.6.1 的加载逻辑。 |
| [q4_gemv.hpp](q4_gemv.hpp) | `Q4Gemv` 创建和持有 OpenCL context、queue、program、权重及缓冲区。`compute_hidden()` 处理完整 logits；`select_hidden()` 处理分组最大值。 |
| [gemv.cl](gemv.cl) | 64 个工作项处理相邻输出通道。向量化拆 Q4、组内 FMA、乘量化参数；启用 `RETURN_ARGMAX` 时另外进行工作组内最大值规约。 |
| [replay.cpp](../mnn-hybrid/replay.cpp) | 可选编译宏 `EDGEDOC_CUSTOM_HEAD`。覆盖 `Llm::forwardRaw()` 记录主干时间；覆盖 `Llm::sample()` 完成自定义投影与选词，仍由原 MNN 生成循环维护历史状态。 |
| [integration_pipeline.sh](integration_pipeline.sh) | 每页生成真实 RKNN 视觉特征、复制已有 MNN 调优缓存、启动语言程序并选用相应配置。 |

简化后，`sample()` 的关键分支如下（实际代码还检查输入尺寸及浮点数有限性，并记录每次调用的耗时）：

```cpp
if (custom_head_ == nullptr) return Llm::sample(logits, offset, size);
const auto* input = logits->readMap<float>(); // 这里实际是 1024 个隐藏特征
if (custom_head_->argmax_only()) {
    const auto selection = custom_head_->select_hidden(input);
    return selection.token;
}
auto* output = logits_->writeMap<float>();
custom_head_->compute_hidden(input, output);
return Llm::sample(logits_, 0, 0); // 恢复真正的 248320 个词表分数
```

MNN 3.6.1 的生成参数是私有成员，但 `sample()` 是虚函数，因此不用重编 MNN 库即可接入实验算子。注意当前自定义算子使用**独立 OpenCL context 和队列**，隐藏向量通过主机内存交接，并非 MNN 后端原生融合，也不是 GPU 零拷贝。

### 为什么试分组最大值

当前配置 `sampler_type=greedy`，只需要最大分数及其 Token ID。原先每 Token 回读 `248320 × 4 = 993280` 字节；每 64 项求一个局部最大值，只回读 `3880 × 2 × 4 = 31040` 字节，减少 **96.875%（1/32）**。同分数时取较小 Token ID，与当前 MNN greedy 扫描规则一致。非 greedy 配置不允许进入此路径，实际 Top-K 错误测试以状态码 1 退出且未生成正文。分组规约额外消耗 GPU 计算和同步，并非回读字节减少多少就能加速多少。

## 3. 真机计时

三条路径都采用 RKNN NPU 视觉和 Mali OpenCL 语言主干，同样的图片、提示词、512 Token 上限、Q4/64 量化权重及缓存。每次独立进程。通知单和混排页各两轮，票据一轮；另有三次独立诊断及一次独立算子回归。主表仅计预先配对的两轮或单轮，不把额外诊断选入速度统计。

| 页面 | 内容 Token 数 | 原 MNN GPU 整页 | 自定义完整 logits | 自定义分组最大值 |
| --- | ---: | ---: | ---: | ---: |
| 通知单 | 98 | 9.393 s | 9.553 s | 9.267 s |
| 票据 | 38 | 6.407 s | 6.184 s | 6.223 s |
| 中英混排页 | 325–326 | 20.237 s | 18.863 s | 18.828 s |

这里的整页墙钟包括视觉、文件交接、模型加载、语言生成及退出。最长样本分别降低约 6.8%、7.0%；短样本没有稳定的大幅收益。混排页第二轮原 MNN 输出与其他路径存在差异，故不能把该速度差当作严格等输出加速证明。

MNN 内部的 `decode_us` 不含采样；实验版把投影移到 `sample()` 后，如果直接看原字段，会将投影排除，造成虚假提速。因此将首步自定义 head 耗时加回 Prefill，将后续每步 head 加回 Decode，原始字段单独保留；分组最大值在 head 内包含了选词的一部分，`request_ms` 和进程墙钟是更直接可比的指标。

| 页面 | 原 MNN GPU Decode | 自定义完整 logits（校正） | 分组最大值（校正） |
| --- | ---: | ---: | ---: |
| 通知单 | 4.116 s | 4.212 s | 4.261 s |
| 票据 | 1.653 s | 1.677 s | 1.632 s |
| 中英混排页 | 13.828 s | 13.805 s | 13.757 s |

混排页语言请求耗时中位数分别为 17.296、16.182、16.098 s。该页 Decode 吞吐分别为 23.54、23.62、23.70 Token/s；大部分整页下降不是 Decode 提速，而是 Prefill、采样与其他开销差异。两轮自定义完整 logits 平均每 Token 的 head 总时间为 9.04、9.08 ms，MNN 主干为 33.05、33.35 ms；分组最大值 head 总时间 8.57、8.94 ms。设备侧读取事件由约 0.185–0.193 ms 降至 0.009–0.010 ms，但核函数由约 7.72–7.80 ms 增至 7.99–8.37 ms，两轮整页结果也不能证明分组最大值持续快于完整 logits。

## 4. 数值和输出验证

- 全部 18 次整页运行自然结束，所有 GPU 进程确认打开 Mali 设备。同一张图的 RKNN FP32 特征哈希完全一致，18 次未发生超时。
- 通知单所有输出逐字节匹配既有 PC 参考（98 Token）。票据三份正文逐字节一致，标注的总额、实付、找零三项均命中。
- 中英混排页：自定义两路各两次的 326 Token 正文相同，与旧版测试程序的原 MNN 两次一致。新版测试程序关闭自定义 head 后，两次均生成另一份 325 Token 正文。差异与测试程序版本相关，但根因未查明，不能断言是 GPU 随机误差或编译问题。人工标注选项 A、D 分别为 `dinner`、`actors`；旧版原 MNN 与两种自定义路径误识别为 `cleaner`、`motors`，新版原 MNN 为 `answer`、`doctors`，两者均有错。这是局部比较，不是完整 CER 评估。
- 独立算子重构后再次和全部 248320 个参考 logits 比较，相对 L2 误差 `1.27e-7`，最大值 Token ID 550，与此前实验相符。

**结论：** 已验证自定义 GPU 算子能够替换真实 OCR 的 `lm_head`，而不是只在微基准中工作。尚未证明整模型 Decode 明显加速或完全等价的 OCR 准确率。下一阶段优先在 MNN 自身的 OpenCL 内存/command queue 上接入同一计算，减少跨 context 的同步及隐藏向量中转，并复核同一构建的逐步 logits 与 Token ID。先测主干中占时较大的 QKV、MLP 投影，再决定是否推广权重重排；不必盲目把所有工作改到 GPU。

## 5. 复现与归档

> GitHub 公开版保留源码、配置、计时与校验结果，不包含模型、权重、二进制、视觉/浮点操作数或第三方样本正文。下述 217 文件的校验及数值复核针对完整 PC 归档；公开子集可用各目录的 PUBLIC_SHA256SUMS 核验。

已存在的板端实验目录 `$HOME/ai/edgedoc-xiaomi-vlm/s5-head-integration-2026-11-26.ss30kNw3` 中保存了两个 ARM64 可执行文件和模型配置。`model/` 中使用原语言权重、tokenizer 及语言配置的链接，`select-gemv/prepared` 指向先前验证的 tile64 权重。其余部署依赖说明参见 [MNN 语言实验](../mnn-hybrid/README.md) 与 [Q4 权重重排](README.md)。重新编译 replay 时在原 MNN ARM64 构建参数基础上增加 `-DEDGEDOC_CUSTOM_HEAD`、OpenCL 头文件路径及 `-l:libOpenCL.so.1`；不加该宏仍可构建为原来的程序。应保持源码相对位置（`s5/mnn-hybrid` 和 `s5/lm-head-opencl`）。

```sh
cd "$HOME/ai/edgedoc-xiaomi-vlm/s5-head-integration-2026-11-26.ss30kNw3"
replay="$HOME/ai/edgedoc-xiaomi-vlm/s5-npu-mnn-2026-11-24.eAgvzp"
vision="$HOME/ai/edgedoc-xiaomi-vlm/s5-emmc-2026-11-23.ZcXVOr/npu"
image="$HOME/ai/edgedoc-xiaomi-vlm/s5-resident-2026-11-24.1YZuKC/inputs/omni-mixed-single-dev.png"
cache="$replay/fixed-opencl/mnn_cachefile.bin"
python3 "$replay/run_board.py" --timeout 150 example-mnn sh "$PWD/select-source/integration_pipeline.sh" \
  "$replay" "$vision" "$image" "$PWD/mnn_image_replay_argmax" "$replay/model-replay/opencl.json" "$cache"
python3 "$replay/run_board.py" --timeout 150 example-custom sh "$PWD/select-source/integration_pipeline.sh" \
  "$replay" "$vision" "$image" "$PWD/mnn_image_replay_argmax" "$PWD/model/opencl-trunk.json" "$cache" "$PWD/select-gemv"
python3 "$replay/run_board.py" --timeout 150 example-max sh "$PWD/select-source/integration_pipeline.sh" \
  "$replay" "$vision" "$image" "$PWD/mnn_image_replay_argmax" "$PWD/model/opencl-trunk.json" "$cache" "$PWD/select-gemv" argmax
```

PC 归档 [board-head-integration-2026-11-26-ss30kNw3](../results/board-head-integration-2026-11-26-ss30kNw3) 包含两版程序、源码快照、trunk 模型及导出图、每页输出、视觉特征、逐 Token 计时、进程监控、错误检查和 SHA256 清单。217 个板端普通文件均核验通过；原有大权重不重复存储，链接及外部依赖按上文复用。[汇总 JSON](../results/board-head-integration-2026-11-26-ss30kNw3/summary.json)。

在 PC 的项目根目录可重建统计：

```sh
python3 s5/lm-head-opencl/summarize_integration.py \
  s5/results/board-head-integration-2026-11-26-ss30kNw3 \
  data/ocr-eval-mini/dev/cord-068/key-fields.json \
  s5/results/board-lm-head-2026-11-26-Ue2nidO0/operands/reference-q4.f32 \
  data/ocr-eval-mini/dev/omni-mixed-single/reference.txt
```
