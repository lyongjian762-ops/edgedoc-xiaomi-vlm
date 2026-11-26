#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Each 8x8 work-group produces 8 rows by 32 columns of the output.
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void qkv_matmul(__global const half * input,
                         __global const half * weight,
                         __global float * output) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row = get_global_id(1);
    const int column = 4 * get_global_id(0);
    __local float input_tile[8][16];
    __local float4 weight_tile[16][8];
    float4 sum = (float4)(0.0f);

    for (int start = 0; start < INPUT_WIDTH; start += 16) {
        input_tile[ly][lx] = vload_half(row * INPUT_WIDTH + start + lx, input);
        input_tile[ly][lx + 8] = vload_half(row * INPUT_WIDTH + start + lx + 8, input);
        weight_tile[ly][lx] = vload_half4(0, weight + (start + ly) * OUTPUT_WIDTH + column);
        weight_tile[ly + 8][lx] = vload_half4(0, weight + (start + ly + 8) * OUTPUT_WIDTH + column);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; ++k) {
            sum = fma((float4)(input_tile[ly][k]), weight_tile[k][lx], sum);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    vstore4(sum, 0, output + row * OUTPUT_WIDTH + column);
}
