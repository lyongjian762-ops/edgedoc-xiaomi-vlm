# Mali 图像预处理对照

开发组 5 张原图在板端解码。CPU 用 OpenCV `INTER_LINEAR_EXACT` 缩放，再做 BGR 到 RGB 转换。Mali-G610 用 OpenCL 在同一个核里完成相同定点双线性插值和通道转换。结果为 `448×448` RGB，与现有 RKNN 的输入尺寸一致。本实验统一使用双线性规则。

`gpu_wall_ms` 从提交原图像素的复制开始，到输出复制回 CPU 后结束。OpenCL 事件单独记录设备上核函数的执行时间。`decode_ms` 记录共用的 CPU 解码，`gpu_configure_ms` 记录该输入尺寸的坐标表和缓冲设置，初始化另见标准错误日志。写 PNG 在计时结束后进行。[OpenCL 事件计时定义](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetEventProfilingInfo.html)

## 板端复现

Debian 11 环境使用已存在的 OpenCV 4.5.1 开发文件和 `libOpenCL.so.1`。将校验过的 Debian `opencl-c-headers` 中的 `CL/` 目录、源码和开发组原图复制到同一个实验目录。五张图按下面的示例放在 `inputs/`。

```bash
g++ -O3 -std=c++17 -Wall -Wextra -Werror -I. $(pkg-config --cflags opencv4) \
    compare.cpp -o compare_gpu \
    -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -l:libOpenCL.so.1
./compare_gpu resize_rgb.cl run-default inputs/*.jpg
EDGEDOC_OPENCV_THREADS=1 ./compare_gpu resize_rgb.cl run-single inputs/*.jpg
```

每张图预热 5 次，交替顺序做 30 对计时。实测每种线程配置启动两个进程。每轮保存 `timing.csv`、`pixels.csv` 和两路 PNG。 [逐次结果和汇总](../results/board-gpu-preprocess-2026-11-24-6FnYRe/summary.json)
