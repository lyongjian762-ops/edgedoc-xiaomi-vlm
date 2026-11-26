#include "q4_gemv.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace {
void run(const std::string & mode, const edgedoc::fs::path & source_file,
         const edgedoc::fs::path & original_weight, const edgedoc::fs::path & prepared,
         size_t local_size) {
    edgedoc::Q4Gemv gpu(mode, source_file, original_weight, prepared, local_size);
    const auto token = edgedoc::read_exact(prepared / (mode == "tile-fp16" ? "input-f16.bin" : "input-f32.bin"),
                                          gpu.token_bytes());
    const auto sums = edgedoc::read_exact(prepared / "group-sum.f32", edgedoc::Q4Gemv::groups * sizeof(float));
    std::vector<float> output(edgedoc::Q4Gemv::output_width);
    std::ofstream csv("timing.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "iteration,upload_ms,kernel_ms,readback_ms,wall_ms\n" << std::fixed << std::setprecision(6);
    for (int step = -5; step < 30; ++step) {
        const auto timing = gpu.compute(token.data(), sums.data(), output.data());
        if (step >= 0) {
            csv << step + 1 << ',' << timing.upload_ms << ',' << timing.kernel_ms << ','
                << timing.readback_ms << ',' << timing.wall_ms << '\n';
        }
    }
    if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("GEMV produced nonfinite logits");
    }
    std::ofstream logits("output.f32", std::ios::binary);
    logits.exceptions(std::ios::badbit | std::ios::failbit);
    logits.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::cerr << "Usage: q4_gemv_bench row-fp32|tile-fp32|tile-fp16 KERNEL ORIGINAL_WEIGHT PREPARED_DIR LOCAL_SIZE\n";
        return 2;
    }
    try {
        run(argv[1], argv[2], argv[3], argv[4], static_cast<size_t>(std::stoul(argv[5])));
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

