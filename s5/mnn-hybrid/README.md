# NPU 视觉和 MNN 语言接力实验

实验沿用同一张 `448×448` 通知单、同一份 OCR 提示词和 512 个输出 Token 上限。NPU 用已有的 RKNN 视觉模型，MNN 语言部分从同一份 Xiaomi-OCR-0 官方权重导出，使用 4 比特权重和 64 元素量化块。CPU 和 OpenCL 两路读取同一份 MNN 模型。

## 嵌入和位置一起交接

```text
图像 → RKNN 视觉编码 → [196,1024] FP32 特征
                                    ↓
     4 个文字 Token + 196 行图像特征 + 68 个文字 Token
                                    ↓
                         MNN 语言模型 → OCR
```

PC 上的 `edgedoc_cli --image-embedding` 用板端特征替换原来的视觉计算，并保留原有提示词模板和多维位置编号。一次调用接收一张图及其特征。图片仍用于确定 Token 布局。输入的 `img_vec.bin` 为 802,816 字节，截短到 4 字节会收到长度错误，没有生成页面。正常视觉和外部特征的两份 OCR 正文逐字节一致。[PC 实验](../results/pc-npu-embedding-2026-11-24-uYvhQ5) · [C++ 视觉入口](../../src/edgedoc_cli.cpp)

板端 [replay.cpp](replay.cpp) 读取 PC 记录的 Token ID 和图像位置。它用 MNN 查表得到文字向量，把新鲜的 RKNN 特征插在中间，然后对整段提示做预填充。物理输入有 268 行。图像块位置是 `14×14` 的二维网格，语言生成的起始位置是 86。两者都从已验证的 PC 路线记录下来。

## 鲁班猫 4 真机结果

系统为 eMMC Debian 11.11，内核 `5.10.160`。两轮使用相反的执行顺序。每次从新进程运行，MNN 两条路径先生成新鲜的 RKNN 特征，再由独立的 MNN 进程读取。OpenCL 行沿用首轮保存的 MNN 缓存。计时覆盖从每页启动到退出，含模型加载、视觉编码、文件交接和生成。所有六份 OCR 正文与 PC 基线逐字节一致，MNN 每次生成 98 个内容 Token 后自然结束。

| 整页路径 | 两次耗时 | 中位数 |
| --- | ---: | ---: |
| RKNN → RKLLM | 13.54 / 12.65 秒 | 13.10 秒 |
| RKNN → MNN，4 个 Cortex-A76 大核 | 7.61 / 7.40 秒 | 7.51 秒 |
| RKNN → MNN，Mali OpenCL 有缓存 | 9.57 / 9.30 秒 | 9.43 秒 |

MNN 语言生成的两轮中位数是 CPU 37.27 token/s、Mali 22.34 token/s。Mali 进程实际打开了 `/dev/mali0`，运行期间采样到 GPU 负载。首次在空的运行目录启动 OpenCL，语言请求耗时 45.37 秒，其中预填充 38.50 秒。随后仅复放语言阶段，带缓存的两个语言请求为 6.72 和 6.36 秒。整页 OpenCL 路径采样到的最低板端可用内存约 2.64 GiB。[原始计时、GPU 负载和校验值](../results/board-npu-mnn-2026-11-24-eAgvzp/summary.json)

初次导出的 MNN 配置声明了 `deepstack_embeds`，实际语言图只有四个输入。CPU 和 OpenCL 都在生成前崩溃。将语言配置设为 `has_deepstack=false` 后，保留同一份权重，两路均正常完成。MNN 的 Linux OpenCL 加载器查找 `libOpenCL.so`，板端实验目录提供了指向现有 `libOpenCL.so.1` 的本地链接。[导出封装](export_language.py) · [首轮故障日志](../results/board-npu-mnn-2026-11-24-eAgvzp/initial-failures) · [MNN 3.6.1](https://github.com/alibaba/MNN/releases/tag/3.6.1)

## 复跑

复跑需要 MNN 3.6.1 的 ARM64 运行库、从官方模型导出的语言权重、PC 端记录的请求配置，以及用同版本 MNN 构建的回放程序。准备好这些文件后，进入板端实验目录并指定视觉模型目录。

```sh
cd /path/to/mnn-board-run
vision=/path/to/rknn-vision
python3 run_board.py --timeout 90 example-cpu sh "$PWD/run_pipeline.sh" "$PWD" "$vision" cpu
python3 run_board.py --timeout 120 example-opencl sh "$PWD/run_pipeline.sh" "$PWD" "$vision" opencl \
    "$PWD/fixed-opencl/mnn_cachefile.bin"
cmp example-cpu/page.md example-opencl/page.md
```

从部署包新建测试目录时，可用 `sha256sum -c SHA256SUMS` 核验文件。在本次 Debian 11 环境，可从该测试目录运行 `ln -s /lib/aarch64-linux-gnu/libOpenCL.so.1 libOpenCL.so`，让 MNN 找到已有的 OpenCL 加载器。首次 OpenCL 运行省略最后的缓存路径，它会在当次输出目录写入缓存。随后把该文件的绝对路径交给下一次运行。现有视觉包里的 `imgenc`、RKNN 权重和图片可直接复用。
