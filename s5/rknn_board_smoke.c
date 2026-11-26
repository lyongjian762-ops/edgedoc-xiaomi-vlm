/* Run the small L08 RKNN fixture once and report the driver and runtime versions. */
#include <stdio.h>
#include <string.h>

#include "rknn_api.h"

static int failed(const char *stage, int status)
{
    if (status != RKNN_SUCC) {
        fprintf(stderr, "%s returned %d\n", stage, status);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2 || strcmp(argv[1], "--help") == 0) {
        fprintf(stderr, "Usage: rknn_board_smoke <conv_bn_rk3588.rknn>\n");
        return argc == 2 ? 0 : 2;
    }

    rknn_context context = 0;
    if (failed("rknn_init", rknn_init(&context, argv[1], 0, 0, NULL))) {
        return 1;
    }

    int result = 1;
    rknn_sdk_version version = {0};
    rknn_input_output_num count = {0};
    rknn_tensor_attr input_attr = {0};
    if (failed("query version", rknn_query(context, RKNN_QUERY_SDK_VERSION, &version, sizeof(version))) ||
        failed("query I/O count", rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count))) ||
        failed("query input", rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)))) {
        goto cleanup;
    }
    fprintf(stderr, "runtime=%.256s driver=%.256s\n", version.api_version, version.drv_version);
    if (count.n_input != 1 || count.n_output != 1 || input_attr.n_elems != 64) {
        fprintf(stderr, "expected the L08 single-input, single-output convolution fixture\n");
        goto cleanup;
    }

    float sample[64];
    for (int i = 0; i < 64; ++i) {
        sample[i] = (float)(2.0 * i / 63.0 - 1.0);
    }
    rknn_input input = {
        // The board runtime accepts NHWC as the source layout of this input.
        .index = 0, .buf = sample, .size = sizeof(sample), .type = RKNN_TENSOR_FLOAT32, .fmt = RKNN_TENSOR_NHWC,
    };
    rknn_output output = {.index = 0, .want_float = 1};
    if (failed("set input", rknn_inputs_set(context, 1, &input)) ||
        failed("rknn_run", rknn_run(context, NULL)) ||
        failed("get output", rknn_outputs_get(context, 1, &output, NULL))) {
        goto cleanup;
    }

    if (output.buf && output.size == 128 * sizeof(float)) {
        const float *values = output.buf;
        printf("[");
        for (int i = 0; i < 128; ++i) {
            printf("%s%.9g", i ? "," : "", values[i]);
        }
        printf("]\n");
        result = 0;
    } else {
        fprintf(stderr, "unexpected output size: %u bytes\n", output.size);
    }
    if (failed("release output", rknn_outputs_release(context, 1, &output))) {
        result = 1;
    }

cleanup:
    if (failed("destroy context", rknn_destroy(context))) {
        result = 1;
    }
    return result;
}
