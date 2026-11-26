# Decode 算子剖析（2026-11-26）

本阶段从一次独立的语言推理开始。沿用已经核验的 NPU 图像特征、268 行图文输入、四个 Cortex-A76、MNN 3.6.1（`d407447ed56c4121a11ccbd266dc184ca1ead0c2`）与同一四比特语言模型。先解决一个问题：生成一个 Token 时，各类算子花费多少时间？

## 源码调用关系

```text
replay.cpp: main()
  Llm::generate_init()
  Llm::generate(input_embeds, max_tokens)
    forwardVec(input_embeds)       ── Prefill，得到首轮 logits
    ArGeneration::generate()
      sample(logits)                ── 选择当前 Token
      tokenizer_decode()            ── 得到对应文字
      forwardVec({current_token})   ── 查表、生成 mask 与位置
        forwardRaw(...)
          Module::onForward(...)
            Pipeline: before → onExecute → after
```

`ArGeneration::generate()` 中的 `updateContext(0, 1)` 先推进生成数量，随后遇到结束 Token 就退出。正常内容 Token 才调用 `forwardVec({current_token})`。`decode_us` 的计时器在判断结束之后、`tokenizer_decode()` 前启动，记录文字解码、下一轮前向和状态更新。采样器内部时间另由 `sample_us` 记录。因此 98 个内容 Token 对应本次 98 次 Decode 前向，最后还有结束符的采样。 [生成循环（固定版本）](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/transformers/llm/engine/src/speculative_decoding/generate.cpp#L44) · [Llm 前向实现](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/transformers/llm/engine/src/llm.cpp#L534) · [Pipeline 执行](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/source/core/Pipeline.cpp#L1249)

## 怎么测量

在 [replay.cpp](../mnn-hybrid/replay.cpp) 增加可选的 `--profile-decode`。加载前开启 MNN `enable_debug` 并注册回调，加载后启用记录。只在 `forwardRaw()` 且 `gen_seq_len > 0` 时记录，以排除 Prefill 和模型加载。

CPU 的命名执行命令调用 `before()`，再调用 `onExecute()`，最后调用 `after()`。在两次回调之间计时，保存 Token 序号、算子名称、类型和微秒数，同时独立计整个 `forwardRaw()`。回调中不打印日志，生成完成后统一写 JSONL。无元数据的命令可能绕过回调，因此用全层前向计时检验覆盖范围。

## 真机结果

顺序为普通 A、剖析 A、剖析 B、普通 B。全部结束符、268 输入 Token、98 输出内容 Token、OCR 正文 SHA-256 都一致。两轮剖析各 98 次前向、54,586 条算子记录，每次前向 557 条命名算子记录。普通模式平均每 Token 的 Decode 中位数 27.43 ms，剖析模式合并平均 31.03 ms。相差约 13.1%，包括剖析成本及运行波动。以下算子数字仅描述剖析模式。

| MNN 算子类型 | 每 Token 调用次数 | 每 Token 累计耗时 | 占剖析 Decode |
| --- | ---: | ---: | ---: |
| Convolution（图中的 1×1 线性投影） | 193 | 25.783 ms | 83.1% |
| LinearAttention | 18 | 2.843 ms | 9.2% |
| Attention | 6 | 0.690 ms | 2.2% |
| LayerNorm | 67 | 0.274 ms | 0.9% |
| 其余命名算子 | 273 | 0.630 ms | 2.0% |

全 `forwardRaw()` 平均为 30.987 ms，命名算子累计 30.220 ms，约覆盖剖析 Decode 的 97.4%。普通运行中内置采样器计时为 31.71 和 30.59 ms，按生成与结束符的 99 次采样均摊约 0.31 ms。两轮逐算子数据见 [汇总](../results/board-decode-profile-2026-11-26-xXFBJ3io/summary.json)。

最大单个算子为 `/lm/lm_head/Linear`，两轮各平均 7.26、8.53 ms/Token，合并 7.894 ms，约占剖析 Decode 的 25.4%，属于上表的 Convolution。核对原始 MNN 导出图，193 个 Convolution 全部为 1×1。`lm_head` 输入 1024 维，输出 248320 维，权重四比特，FP16 量化参数，序列化权重与量化参数约 136.4 MiB。对单行输入执行 `[1,1024] × [1024,248320]` 的数学计算，需要 254,279,680 次乘加。这是把隐藏向量投影到词表分数的线性层。实际 DDR 读取字节数、内核指令和访存瓶颈还需要继续测量。 [导出图投影清单](../results/board-decode-profile-2026-11-26-xXFBJ3io/model-projections.json)

## 复跑

使用前次 Debian 11 ARM64 工具链和 MNN 构建目录，在 WSL 项目根目录执行。

```sh
cache="$HOME/.cache/edgedoc-xiaomi-vlm"
source="$cache/s5-mnn-3.6.1"
build="$cache/s5-mnn-arm64-debian11"
toolchain="$cache/debian11-cross/sysroot"
"$toolchain/usr/bin/aarch64-linux-gnu-g++-10" --sysroot="$toolchain" \
  -B/usr/aarch64-linux-gnu/bin/ -O2 -std=c++17 -fno-rtti -Wall -Wextra -Werror \
  -isystem "$source/include" -isystem "$source/transformers/llm/engine/include" \
  -isystem "$source/transformers/llm/engine/src" -isystem "$source/3rd_party" \
  -isystem "$source/apps/frameworks/3rd_party/include" \
  s5/mnn-hybrid/replay.cpp -L"$build" -lMNN -lpthread -o /tmp/mnn_image_replay_profile
```

把二进制复制到板端独立目录后，使用之前混合推理包内的模型、请求与特征。下例沿用原采样脚本和大核亲和性。

```sh
hybrid=/home/cat/ai/edgedoc-xiaomi-vlm/s5-npu-mnn-2026-11-24.eAgvzp
export LD_LIBRARY_PATH="$hybrid"
python3 "$hybrid/run_board.py" --timeout 90 profile-new taskset -c 4-7 \
  "$PWD/mnn_image_replay_profile" "$hybrid/model-replay/cpu.json" \
  "$hybrid/request.json" "$hybrid/archive_img_vec.bin" . --profile-decode
```

剖析开关暂限同步 CPU 路径。使用 OpenCL 配置加此开关的边界测试在加载前返回 1，未产生 OCR 正文。不开剖析开关时，原有 CPU 与 OpenCL 命令格式均保持兼容，本轮实测复验了 CPU。

原始日志、二进制、实际执行源码和校验值已归档。可在 PC 项目根目录独立重新统计。

```sh
python3 s5/decode-profile/summarize.py s5/results/board-decode-profile-2026-11-26-xXFBJ3io
```
