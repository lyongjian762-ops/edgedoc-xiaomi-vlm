#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GROUPS 16
#define TILE 64
#define HALF_WIDTH 512
#define OUTPUTS 248320

#if HALF_ACC
typedef half Value;
typedef half8 Value8;
#define TO_VALUE8 convert_half8
#else
typedef float Value;
typedef float8 Value8;
#define TO_VALUE8 convert_float8
#endif

inline float reduce8(float8 value) {
    float4 reduced = value.lo + value.hi;
    return reduced.x + reduced.y + reduced.z + reduced.w;
}

__kernel void q4_gemv(__global const uchar *weights,
                      __global const half *scales,
                      __constant const Value *input,
                      __constant const float *group_sum,
                      __global float *output) {
    const uint channel = get_global_id(0);
    if (channel >= OUTPUTS) return;
    const uint tile = channel / TILE;
    const uint lane = channel % TILE;
    float total = 0.0f;

    for (uint group = 0; group < GROUPS; ++group) {
        Value8 partial = (Value8)(0);
        const uint input_base = group * 32;
        // Each group contains 64 original weights in 32 packed bytes.
        #pragma unroll 4
        for (uint chunk = 0; chunk < 4; ++chunk) {
#if TILED
            const uint weight_index = (((tile * GROUPS + group) * 4 + chunk) * TILE + lane) * 8;
#else
            const uint weight_index = (channel * GROUPS + group) * 32 + chunk * 8;
#endif
            uchar8 packed = vload8(0, weights + weight_index);
            // The stored lookup table is [-8, -7, ..., 7], high nibble first.
            Value8 high = TO_VALUE8(packed >> (uchar8)(4)) - (Value8)(8);
            Value8 low = TO_VALUE8(packed & (uchar8)(15)) - (Value8)(8);
            Value8 even = vload8(0, input + input_base + chunk * 8);
            Value8 odd = vload8(0, input + HALF_WIDTH + input_base + chunk * 8);
            partial = fma(high, even, partial);
            partial = fma(low, odd, partial);
        }
#if TILED
        const uint scale_index = (tile * GROUPS + group) * TILE + lane;
#else
        const uint scale_index = channel * GROUPS + group;
#endif
        float2 alpha = vload_half2(0, scales + scale_index * 2);
        total = fma(alpha.y, reduce8(convert_float8(partial)), total);
        total = fma(alpha.x, group_sum[group], total);
    }
#if RETURN_ARGMAX
    // The launch size is an exact multiple of the local size (at most 128).
    __local float scores[128];
    __local uint indices[128];
    const uint lid = get_local_id(0);
    scores[lid] = total;
    indices[lid] = channel;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            const float other = scores[lid + stride];
            const uint other_index = indices[lid + stride];
            if (!isfinite(other) || (isfinite(scores[lid]) &&
                (other > scores[lid] || (other == scores[lid] && other_index < indices[lid])))) {
                scores[lid] = other;
                indices[lid] = other_index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        output[2 * get_group_id(0)] = scores[0];
        output[2 * get_group_id(0) + 1] = (float)indices[0];
    }
#else
    output[channel] = total;
#endif
}
