# EdgeDoc Cam S1 · 从图片到第一个 Token

## 一、先看全程

面对一张拍下来的检修单，我们希望一台普通电脑或小型设备把单据编号、日期和正文转成可以复制、搜索的文字。这个项目选用 Xiaomi-OCR-0 来做识别。它属于视觉语言模型（Vision-Language Model，VLM），既能把图片转换为计算机可处理的特征，也能按顺序生成文字。

**部署**是把训练好的模型、运行库、输入和输出接在一起，让它能在指定设备上完成工作。**推理**是拿着训练好的参数计算一次结果。**推理优化**是确认结果质量后，找出时间或内存花在哪里，再修改实现并用对照实验检验。有些优化只让一段代码变快，最终用户等待的时间取决于整条链路。

S0 通过本地服务验证了三张测试页。S1 把同样的工作收进一个自己能读懂、能编译的 C++ 程序。程序直接调用 llama.cpp 的语言库和 libmtmd 多模态库。这给了我们几个真实的观察和修改点，例如图片怎么变成特征、第一枚输出 Token 在哪个调用产生，以及处理第二张图前需要清掉什么。

```text
文件 page.png ──图像解码──→ RGB 像素 ──视觉预处理、编码和适配──→ 图像特征向量
                                                                         ↓
提示词 ──套用对话模板──→ 前段文字 ID、图像占位、后段文字 ID ──按顺序送入语言模型
                                                                         ↓
                                              Prefill 处理整份输入，得到下一 Token 的分数
                                                                         ↓
                                              采样第一个 Token → Decode → 采样下一个
                                                                         ↓
                                              终止 Token 或达到上限 → Markdown 与记录
```

把“模型”理解为一套训练得到的数字参数，把“运行库”理解为负责执行计算的一组 C/C++ 函数，把 `edgedoc_cli` 理解为选择输入、安排这些函数、判断结果是否完整的应用程序。学习顺序是先跑一次，接着读图文输入、生成、状态隔离，最后用实验把它们串起来。

**第一次读源码只追一张中文页。** 在 `edgedoc_cli.cpp` 找到 `main()`，沿 `recognize()` → `prefill()` → `generate()` 画出四个函数的调用关系。每次跳转先找传进来的数据、返回或写出的数据，以及哪行可能失败。下面的源码片段省略了部分日志和检查，并标明了省略处。第一次读到陌生的 C++ 语法时，先用片段旁边的解释认出数据和控制流。

本章使用 WSL Ubuntu 的 **CPU 路径**。源码固定在 llama.cpp `8642541b255bfb5dc39cd351a178be09013a55d6`，资产沿用 S0 的 `Q4_K_M` 语言 GGUF 和 `BF16` 视觉 GGUF。GGUF 是装模型张量及配套元数据的文件格式。`Q4_K_M` 是一种混合量化格式，用较少的位数保存许多语言权重。BF16 是一种 16 位浮点格式。实际架构、维度和位置行为以这一版源码与实测为依据。[环境与资产记录](../README.md)

## 二、单元 1 · 先用命令拿到文字

在 Windows 打开 Ubuntu（WSL）终端，执行下面的命令。`cd` 是进入项目目录。`cmake -S . -B "$build"` 根据当前目录的 `CMakeLists.txt` 生成构建配置，`cmake --build` 调用编译器，把 C++ 源码和所需的库做成 `edgedoc_cli`。`build` 和 `assets` 是这次终端会话里的路径变量。库源码和权重来自工作区外部依赖目录，构建产物在 Ubuntu 的用户目录。

```bash
build="$PWD/build"
assets="$PWD/models"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DEDGEDOC_LLAMA_SOURCE="$PWD/third_party/llama.cpp" -DGGML_NATIVE=ON -DGGML_CUDA=OFF
cmake --build "$build" --target edgedoc_cli -j 6

out="s1/results/my-first-$(date +%Y%m%d-%H%M%S)"
"$build/edgedoc_cli" \
  --model "$assets/Xiaomi-OCR-0.Q4_K_M.gguf" \
  --mmproj "$assets/Xiaomi-OCR-0.mmproj-bf16.gguf" \
  --prompt-file s1/ocr_prompt.txt \
  --image s0/samples/zh-notice.png \
  --output-dir "$out"
cat "$out/page-1.md"
cat "$out/page-1.json"
```

`--model` 提供语言参数与词表、对话模板等元数据。`--mmproj` 提供视觉编码和特征适配所需的参数。`--prompt-file` 提供任务说明，告诉模型按阅读顺序输出正文。每次 `--image` 提交一张 PNG 或 JPEG 静态图片，`--output-dir` 指向本次新建的目录。`.md` 是生成的文字，`.json` 记录数据形状、分块、结束原因等可核对的信息。`run.json` 保存运行配置。

打开[中文样本图](../s0/samples/zh-notice.png)，再读 `.md`。先对照标题、`WX-1049-28`、`36 套` 和最后一行。看到正确的识别结果后，再读 `finish_reason`。`eog` 是模型选择了结束 Token，是否认全仍要对照原文。下一单元用 `page-1.json` 回答图片到底送了什么数据进去。

在 [参数解析处](../src/edgedoc_cli.cpp#L49) 找 `options.images.emplace_back(value)`。`emplace_back` 在这里就是把路径放到列表末尾。命令里每多写一次 `--image 路径`，列表就多一项。此时 `s0/samples/zh-notice.png` 还是路径字符串。

回到 [`main()`](../src/edgedoc_cli.cpp#L312)，读出命令行怎样变成一次识别。

```cpp
const Options options = parse_options(argc, argv);
const std::string prompt = read_prompt(options.prompt_file);
// …创建本次输出目录
Backend backend;
OcrEngine engine(options);
// …写 run.json，准备退出状态
for (size_t i = 0; i < options.images.size(); ++i) {
    const auto result = engine.recognize(options.images[i], prompt, options.max_tokens);
    const std::string name = "page-" + std::to_string(i + 1);
    write_file(options.output_dir / (name + ".md"), result.text);
    write_file(options.output_dir / (name + ".json"), result.trace.dump(2) + '\n');
    // …显示停止原因并更新退出状态
}
```

`OcrEngine engine(options)` 创建一个 `OcrEngine` 对象，构造函数按参数加载语言和视觉模型。它在循环外，所以一条命令加载一次模型。循环里的 `engine.recognize(...)` 才按页调用。`options.images[i]` 是第 `i` 项路径，`i` 从 `0` 开始，保存时用 `i + 1` 生成 `page-1`。`auto` 让编译器从函数返回值推导类型，这里推导出 `Result`，其中 `text` 保存生成内容，`trace` 保存本次识别的记录。`dump(2)` 把记录排成易读的 JSON 字符串。认识这几行后，顺着 `recognize()` 跳进去。[模型构造函数](../src/edgedoc_cli.cpp#L114)

## 三、单元 2 · 图像怎样进入语言模型

PNG 文件存的是编码后的图片。图像解码把它展开成 RGB 像素。每个像素有红、绿、蓝三个通道，`[790, 1100, 3]` 表示 790 行、每行 1100 个像素、每像素 3 个通道。这个维度列表叫 **Shape**。`uint8` 表示每个通道在这里用 1 字节的无符号整数保存。

文字和图片的输入形式不同。文字先按词表编码成整数 **Token ID**。例如一句话可以被拆成若干词、字或字的一部分，具体由词表决定。图片由 libmtmd 按模型规则预处理，再由视觉编码计算出一组特征向量。每个向量是描述某处图像内容的一排数字。适配层把这些向量对齐到语言模块能接收的宽度。

在 `recognize()` 里跟着三步走。

1. `mtmd_helper_bitmap_init_from_file()` 得到 `bitmap`。文件是否能解码，在这里第一次能确定。
2. `common_chat_templates_apply()` 用 GGUF 自带的对话模板拼好用户消息和生成起始内容，消息里有一个视觉占位。模板中相同的提示词会变成带位置和角色控制符的输入，接着交给运行库解析。
3. `mtmd_tokenize()` 读模板结果与图像，形成有顺序的 `chunks`。文本块持有 Token ID。图像块包含预处理后的图像数据及用于后续执行的位置信息。此时开始拥有图像块，还要经过视觉编码才能拿到浮点特征。[项目的 `recognize()`](../src/edgedoc_cli.cpp#L150) · [同版本 libmtmd 接口](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/mtmd/mtmd.h#L316)

```cpp
const auto media = mtmd_helper_bitmap_init_from_file(
    vision_.get(), image_path.c_str(), false, mtmd_helper_init_opt_default());
mtmd::bitmap_ptr bitmap(media.bitmap);
// …检查静态图片，组装用户消息，设置模板选项
const auto rendered = common_chat_templates_apply(templates_.get(), chat).prompt.str();

mtmd::input_chunks chunks(mtmd_input_chunks_init());
const mtmd_input_text input{rendered.data(), rendered.size(), true, true};
const mtmd_bitmap * images[] = {bitmap.get()};
const int code = mtmd_tokenize(vision_.get(), chunks.ptr.get(), &input, images, 1);
```

`.get()` 把托管对象里的裸指针借给 C 接口，原对象继续管理释放。`mtmd_tokenize(...)` 的第二个实参 `chunks.ptr.get()` 指向要填充的分块容器。后面的 `&input, images, 1` 表示给它模板产生的文本、图片指针数组和图片数量。返回值 `code` 表示成功或错误，分块数据写入 `chunks`。可以用 [中文页的实际 JSON](../s1/results/pc-cpu-acceptance/sequence/page-1.json) 找到这些块的数量和顺序。

[`prefill()`](../src/edgedoc_cli.cpp#L203) 按分块顺序执行。只看循环里的这一处分支，体会文字与图像走的是哪条路。

```cpp
if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    record["type"] = "text";
    code = mtmd_helper_eval_chunk_single(vision_.get(), context_.get(), chunk,
        next, kSequence, kBatchSize, i + 1 == count, &next);
} else {
    code = prefill_image(chunk, next, record);
}
```

`next` 记录该从哪个位置处理当前块。在文字分支中，传值 `next` 给运行库作为起点，传地址 `&next` 让运行库把新位置写回同一个变量。图片分支的函数签名采用 C++ 引用 `llama_pos & next`，所以它也能更新本次循环的 `next`。`i + 1 == count` 只在最后一块为真，告诉文字分支在收尾时留下用于生成的候选分数。

进入 [`prefill_image()`](../src/edgedoc_cli.cpp#L231)，找三处动作，分别对应算视觉特征、拿到特征地址、提交给语言模型。

```cpp
const int code = mtmd_encode_chunk(vision_.get(), chunk);
if (code != 0) {
    return code;
}
float * features = mtmd_get_output_embd(vision_.get());
// …检查指针，记录特征形状和位置
return mtmd_helper_decode_image_chunk(vision_.get(), context_.get(), chunk,
    features, next, kSequence, kBatchSize, &next, nullptr, nullptr);
```

沿调用再往库里看一层。`mtmd_encode_chunk()` 把本次视觉编码的输出存到 `ctx->out_embd`。接着 `mtmd_get_output_embd()` 返回这个缓冲的地址，所以 `features` 的所有者仍是 libmtmd，下一次视觉编码可能复用缓冲。`mtmd_helper_decode_image_chunk()` 用这个地址和图像位置构建批次，在循环里把批次传给 `llama_process()`。中文页的日志里四次图像批次分别处理 `256 + 256 + 256 + 82 = 850` 项。现在能在代码和日志两边指出图像特征真正进入语言计算的位置。[输出缓冲源码](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/mtmd/mtmd.cpp#L2207) · [语言计算入口](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/mtmd/mtmd-helper.cpp#L172) · [实测批次日志](../s1/results/pc-cpu-acceptance/single.stderr#L1948)

**中文页的实测数据。** `page-1.json` 中有 4 个前段文字 Token、850 个视觉特征、68 个后段文字 Token，共 922 项输入。图像特征是 `[850, 1024]` 的 `float32` 矩阵。逐项占 4 字节，因此这份特征矩阵的理论数据量是 `850 × 1024 × 4 = 3,481,600` 字节。程序的峰值内存还涉及权重、历史状态、工作区和其它临时数据。`Q4_K_M` 描述语言权重文件的量化方案，视觉输出接口实测仍是 `float32`。

| 块 | 输入项数 | 位置变化 |
| --- | ---: | --- |
| 前段文字 | 4 | 0 → 4 |
| 图像 | 850 | 4 → 38 |
| 后段文字 | 68 | 38 → 106 |

**这里用了多模态位置编码 M-RoPE。** 相邻文字的位置通常递增，图像特征还要保留二维位置。测试图的首个特征位置为 `(t,x,y,z)=(4,4,4,0)`，最后一个为 `(4,37,28,0)`。前者到后者，`t` 一直是 4，横纵坐标在网格内展开。图像块的位置跨度由运行库给出，是 34，因此下一块从 38 开始。应用同时记录特征数量与位置跨度，并把 `next_position` 交还给运行库推进。[位置映射源码](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/mtmd/mtmd.cpp#L3078)

**动手看一项变式。** 用另外一张尺寸差异明显、可在本地使用的清晰页面复跑，记录 `input_rgb_shape`、`embedding_shape` 与 `position_span`。先写下预测，再观察原始图尺寸变化是否一定带来视觉特征变化。只凭这些记录能解释到哪个边界，哪里还要看视觉预处理源码？

## 四、单元 3 · 首个 Token 和后续生成

**Prefill** 是处理整份输入的阶段，内容包括模板里的文字和一张图的特征。处理到最后一个输入位置时，语言模块输出一组候选分数，常称为 **logits**。词表里每个可以输出的 Token 有一个分数。项目用贪心采样（greedy sampling）取分数最高的 Token。

`prefill()` 给后续生成准备了首轮分数。接着打开 [`generate()` 的循环](../src/edgedoc_cli.cpp#L263)，只跟第一轮看代码。

```cpp
while (generated < max_tokens) {
    const llama_token token = llama_sampler_sample(sampler.get(), context_.get(), -1);
    if (llama_vocab_is_eog(vocab, token)) {
        complete = true;
        break;
    }
    text += common_token_to_piece(vocab, token);
    if (++generated == max_tokens) {
        break;
    }
    // …把本轮选出的 token 装进 batch，设为 next 位置并请求 logits
    const int code = llama_process(context_.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (code != 0) {
        throw std::runtime_error("decode failed, code=" + std::to_string(code));
    }
    ++next;
}
```

第一次执行 `llama_sampler_sample()` 时，参数 `-1` 取最近一份输出分数，即 Prefill 留下的 logits。`token` 是选中的整数 ID。`common_token_to_piece()` 把 ID 变成能拼入 `text` 的字节片段。`llama_process()` 用本轮 Token 和历史状态更新分数，下一次循环就能选出下一枚。这个重复更新后续 Token 的阶段叫 **Decode**。底层文本和图片输入也会调用 `llama_process()`，可以看传入的是已有图文输入的一批，还是刚生成的一个 Token 来判断它在执行哪个阶段。

```text
最后一个输入位置的分数 → 采样 A → A 放进输出
                                   ↓
                     用 A 和已建立的历史状态更新模型
                                   ↓
                        新的分数 → 采样 B → B 放进输出
```

程序根据两种信号结束。采到模型的结束 Token，记录 `eog`，并省去把结束符写入正文。达到 `--max-tokens` 则保存已有内容，记录 `max_tokens`，进程退出码为 `2`，让调用者知道输出可能只是前缀。`generated_content_tokens` 只计输出内容。中文页的 C++ 记录是 98，S0 接口的 `completion_tokens` 是 99，两者的计数口径包含的结束符不同，原始正文逐字节一致。[项目生成循环](../src/edgedoc_cli.cpp#L256) · [S0 响应](../s0/results/pc-cpu-s0/zh-notice.response.json)

**动手改一个参数。** 原命令只把生成上限改为 `--max-tokens 2`，并换一个新输出目录。运行后立刻执行 `echo $?`，再检查 `.md` 和 `.json`。先预测图文输入的 922 项会出现什么变化、第二个输出 Token 之后还有没有必要继续调用模型，再用输出和代码核对。

## 五、单元 4 · 模型复用与请求隔离

当同一个进程识别 A、B、A 三张图时，模型权重和对话模板可以共用。每张图的历史状态都有自己的起点。**KV cache** 保存注意力计算已经得到的部分历史数据，模型的混合记忆还包含递归状态。处理 B 前需要重置这些记录并从自己的图片建立历史。C++ 中 `context_` 是持有这些运行资源的对象。`llama_memory_clear(memory, true)` 让运行库清空序列元数据和数据缓冲，再复用已分配的 Context。新请求的采样器、下一位置和输出字符串也都重新建立。[项目重置位置](../src/edgedoc_cli.cpp#L180) · [上游混合记忆清理](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/src/llama-memory-hybrid.cpp#L139)

```cpp
llama_memory_t memory = llama_get_memory(context_.get());
const llama_pos previous = llama_memory_seq_pos_max(memory, kSequence);
llama_memory_clear(memory, true);
const llama_pos cleared = llama_memory_seq_pos_max(memory, kSequence);
// …检查清空是否成功，并把 previous、cleared 写入 trace
const llama_pos next_position = prefill(chunks.ptr.get(), trace["chunks"]);
trace["next_position_after_prefill"] = next_position;
Result result = generate(next_position, max_tokens, std::move(trace));
```

按执行顺序找到清理上一页历史、处理当前页输入和开始生成的三处调用。`previous` 和 `cleared` 读自同一个 `memory`，因此它们直接对应清理前后的状态。`prefill()` 每次把自己的 `next` 从 `0` 起算，结束时返回新位置。`generate()` 接过这个位置用来安排后续 Token。

如果你学过 C，可以把 `llama_model_ptr` 看成同时持有裸指针和对应释放函数的对象。C++ 在其作用域结束时自动释放资源，这个习惯称为 RAII。`Backend` 先创建，`OcrEngine` 后创建，退出时按相反顺序释放。图像、块、批次与采样器由一页请求持有。解码失败时，异常沿调用链退出，局部资源照样会销毁。

**复现 A→B→A 再加一张混排页。** 下列脚本调用刚编译的 C++ 程序，验证独立 A、同进程 A→B→A→C、两 Token 上限、损坏图片与超出上下文的情况。正常页与 S0 原始文本进行逐字节对照，日志留在新目录。

```bash
python3 s1/verify_s1.py --cli "$build/edgedoc_cli" --assets "$assets" \
  --run-dir "s1/results/my-check-$(date +%Y%m%d-%H%M%S)"
```

先看 `sequence/page-2.json` 里重置前后的位置字段。实测 B 开始时，`memory_pos_max_before_reset` 是上一页留下的 203，清空后变成 `-1`，即序列为空。两个 A 的 `.md` 都与独立运行的 A 一致。接着看一个失败日志。`corrupt.stderr` 对应图像解码失败。`over_budget.stderr` 对应图片与提示词可以展开，但输入和请求的输出额度合起来超过 4096，程序在 Prefill 前拒绝。两种失败都以 `1` 退出。[代理的验收记录](../s1/results/pc-cpu-acceptance/acceptance.json)

**S0 位置警告的源码对照。** 在第一个图像微批次，日志显示 `4 after 3`，批次有 128 项。上游递归记忆的 `find_slot()` 用 `last_pos == 前位置 + 批次项数` 检查连续性。图像批次的位置由 M-RoPE 生成，第一分量 `t` 在该图上保持 4，另外两个空间分量表示网格。这解释了实测日志里为什么反复比较到 4，也解释了空状态的单独 A 仍然出现警告。三张材料的逐字节输出对照覆盖的是当前页和配置。更长、形状更复杂的输入还需要单独验证质量。[位置填充代码](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/mtmd/mtmd-helper-common.h#L101) · [警告检查](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/src/llama-memory-recurrent.cpp#L668)

**机制题。** 如果 C++ 入口保持每一页的 `next` 从 0 开始，却跳过 `llama_memory_clear()`，第二张图的哪些对象可能仍持有 A 的信息？请说出你想看哪两个记录或源码位置，并说明为什么只比较两个输出文件可能漏掉问题。先作答，再做反馈和订正。

## 六、把这一章接到工程训练

复述从 **PNG → RGB → 图文分块 → 图像特征 → Prefill → 首 Token → Decode** 的过程。每到一个箭头，指出负责函数、数据的 Shape 或大小、资源由谁持有、失败时能从哪里获得证据。这四个问题也会用于板端适配和以后读其他推理引擎。

| 里程碑 | 你要能独立完成的工程动作 | 可以交付的证据 |
| --- | --- | --- |
| S1 应用与运行库 | 看懂一条真实调用链，亲手改变上限，解释两次请求为何能有各自的状态 | 命令、代码行、输出和状态记录 |
| S2 性能定位 | 划分图像解码、视觉编码、语言输入处理和逐 Token 生成，给每段确定可靠的完成边界 | 一条有测量口径的时间线，和一个可反驳的热点假设 |
| S3 配置与量化 | 固定其他因素，比较权重和视觉输入预算对错误率、时延、内存的影响 | 同一测试集上的配置对照与取舍 |
| S4 代码优化 | 用真实 Shape 复现热点，修改一处实现，接回完整模型后复测 | 可审查的改动、正确性边界、端到端收益或负结果 |
| S5–S6 端侧部署 | 在 RK3588S 重建基线，验证视觉和语言接口，再接入相机输入 | 实际设备上的部署记录、资源占用和失败复现 |

学完 S1，自己跑一次完整页和一个边界条件。用三到五句话解释 850 个视觉特征为何会推进 34 个位置、首 Token 所用的分数来自哪里，以及第二页前哪些状态需要重置。下一次讨论从你的预测与命令结果开始，再进入 S2 的计时边界。文档中的验证是当前工程结果，独立操作与机制理解由你的实作和讲解来确认。
