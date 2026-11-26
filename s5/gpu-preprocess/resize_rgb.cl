// Each work-item resizes one output pixel and changes BGR to RGB.
__kernel void resize_rgb(__global const uchar * image, int width,
                         __global const int * xs, __global const int * ys,
                         __global uchar * output) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int x0 = xs[3 * x];
    const int x1 = xs[3 * x + 1];
    const int y0 = ys[3 * y];
    const int y1 = ys[3 * y + 1];
    const uint wx = xs[3 * x + 2];
    const uint wy = ys[3 * y + 2];

    const uint3 a = convert_uint3(vload3(y0 * width + x0, image));
    const uint3 b = convert_uint3(vload3(y0 * width + x1, image));
    const uint3 c = convert_uint3(vload3(y1 * width + x0, image));
    const uint3 d = convert_uint3(vload3(y1 * width + x1, image));
    const uint3 top = a * (256 - wx) + b * wx;
    const uint3 bottom = c * (256 - wx) + d * wx;
    const uchar3 bgr = convert_uchar3((top * (256 - wy) + bottom * wy + 32768) >> 16);
    vstore3(bgr.zyx, y * get_global_size(0) + x, output);
}
