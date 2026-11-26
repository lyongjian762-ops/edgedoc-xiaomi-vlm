#include "bench_io.hpp"
#include <rknn_matmul_api.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using namespace qkv;
constexpr size_t kInputBytes = kRows * kInputWidth * sizeof(uint16_t);
constexpr size_t kWeightBytes = kInputWidth * kOutputWidth * sizeof(uint16_t);
constexpr size_t kOutputBytes = kRows * kOutputWidth * sizeof(float);

void check(int status, const char * operation) {
    if (status != RKNN_SUCC) {
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(status) + ")");
    }
}

struct NpuTiming {
    double wall_ms;
    double upload_ms;
    double run_ms;
    double readback_ms;
};

class RknnMatMul {
public:
    RknnMatMul() = default;
    RknnMatMul(const RknnMatMul &) = delete;
    RknnMatMul & operator=(const RknnMatMul &) = delete;

    ~RknnMatMul() {
        for (auto * memory : {output_, weight_, input_}) {
            if (memory) {
                rknn_destroy_mem(context_, memory);
            }
        }
        if (context_) {
            rknn_matmul_destroy(context_);
        }
    }

    void configure(std::vector<uint16_t> & weights, rknn_core_mask core_mask) {
        rknn_matmul_info info{};
        info.M = kRows;
        info.K = kInputWidth;
        info.N = kOutputWidth;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        rknn_matmul_io_attr io{};
        check(rknn_matmul_create(&context_, &info, &io), "create RKNN MatMul");
        check(rknn_matmul_set_core_mask(context_, core_mask), "set RKNN MatMul core mask");
        std::cerr << "NPU core mask: " << core_mask << ", B layout native, A and C row-major\n";
        std::cerr << "NPU buffer sizes bytes: A=" << io.A.size << ", B=" << io.B.size
                  << ", C=" << io.C.size << '\n';
        if (io.A.size != kInputBytes || io.B.size < kWeightBytes || io.C.size != kOutputBytes ||
            io.A.type != RKNN_TENSOR_FLOAT16 || io.B.type != RKNN_TENSOR_FLOAT16 ||
            io.C.type != RKNN_TENSOR_FLOAT32) {
            throw std::runtime_error("unexpected RKNN MatMul tensor layout or dtype");
        }
        input_ = allocate(io.A.size);
        weight_ = allocate(io.B.size);
        output_ = allocate(io.C.size);
        check(rknn_B_normal_layout_to_native_layout(weights.data(), weight_->virt_addr,
                                                    kInputWidth, kOutputWidth, &info), "pack NPU weights");
        check(rknn_mem_sync(context_, weight_, RKNN_MEMORY_SYNC_TO_DEVICE), "sync NPU weights");
        check(rknn_matmul_set_io_mem(context_, input_, &io.A), "bind NPU input");
        check(rknn_matmul_set_io_mem(context_, weight_, &io.B), "bind NPU weights");
        check(rknn_matmul_set_io_mem(context_, output_, &io.C), "bind NPU output");
    }

    NpuTiming run(const std::vector<uint16_t> & input, std::vector<float> & output) {
        const auto start = Clock::now();
        std::memcpy(input_->virt_addr, input.data(), kInputBytes);
        check(rknn_mem_sync(context_, input_, RKNN_MEMORY_SYNC_TO_DEVICE), "sync NPU input");
        const double upload_ms = elapsed_ms(start);

        const auto run_start = Clock::now();
        check(rknn_matmul_run(context_), "run RKNN MatMul");
        const double run_ms = elapsed_ms(run_start);

        const auto read_start = Clock::now();
        check(rknn_mem_sync(context_, output_, RKNN_MEMORY_SYNC_FROM_DEVICE), "sync NPU output");
        std::memcpy(output.data(), output_->virt_addr, kOutputBytes);
        return {elapsed_ms(start), upload_ms, run_ms, elapsed_ms(read_start)};
    }

private:
    rknn_tensor_mem * allocate(uint32_t bytes) {
        auto * memory = rknn_create_mem(context_, bytes);
        if (!memory) {
            throw std::runtime_error("allocate RKNN MatMul buffer failed");
        }
        if (!memory->virt_addr || memory->size < bytes) {
            rknn_destroy_mem(context_, memory);
            throw std::runtime_error("invalid RKNN MatMul buffer");
        }
        return memory;
    }

    rknn_matmul_ctx context_ = 0;
    rknn_tensor_mem * input_ = nullptr;
    rknn_tensor_mem * weight_ = nullptr;
    rknn_tensor_mem * output_ = nullptr;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4 || (std::string(argv[3]) != "7" && std::string(argv[3]) != "1")) {
        std::cerr << "Usage: compare_npu <operand_dir> <existing_output_dir> <core_mask 7|1>\n";
        return 2;
    }
    try {
        const fs::path operands(argv[1]);
        const fs::path output_dir(argv[2]);
        const rknn_core_mask core_mask = argv[3][0] == '7' ? RKNN_NPU_CORE_0_1_2 : RKNN_NPU_CORE_0;
        const auto input = read_half(operands / "x.f16", kRows * kInputWidth);
        auto weights = read_half(operands / "w_transposed.f16", kInputWidth * kOutputWidth);
        std::vector<float> output(kRows * kOutputWidth);
        RknnMatMul npu;
        const auto start = Clock::now();
        npu.configure(weights, core_mask);
        std::cerr << "NPU initialization and one-time weight packing ms: " << elapsed_ms(start) << '\n';
        for (int i = 0; i < kWarmups; ++i) {
            auto warmup = npu.run(input, output);
            if (i == 0) {
                std::cerr << "First NPU request ms: " << warmup.wall_ms << '\n';
            }
        }
        std::ofstream csv(output_dir / "timing.csv");
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv << "iteration,npu_wall_ms,npu_upload_ms,npu_run_ms,npu_readback_ms\n";
        csv << std::fixed << std::setprecision(6);
        for (int i = 0; i < kRepeats; ++i) {
            const auto result = npu.run(input, output);
            csv << i + 1 << ',' << result.wall_ms << ',' << result.upload_ms << ',' << result.run_ms << ','
                << result.readback_ms << '\n';
        }
        csv.close();
        save_float(output_dir / "npu.f32", output);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
