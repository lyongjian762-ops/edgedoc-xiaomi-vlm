#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace MNN;
using namespace MNN::Express;
using Clock = std::chrono::steady_clock;
constexpr int kInputs = 1024;
constexpr int kOutputs = 3584;
constexpr int kChannelPack = 4;
constexpr int kMaxRows = 268;
constexpr int kCpuThreads = 4;
constexpr int kWarmups = 5;
constexpr int kRepeats = 30;

double elapsed_us(Clock::time_point begin) {
    return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

std::vector<float> read_input(const char * path) {
    std::vector<float> input(kMaxRows * kInputs);
    const auto bytes = static_cast<std::streamsize>(input.size() * sizeof(float));
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.tellg() != bytes) {
        throw std::runtime_error("unexpected input size");
    }
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(input.data()), bytes)) {
        throw std::runtime_error("cannot read input");
    }
    return input;
}

void save_rows(int rows, const float * output) {
    if (!std::all_of(output, output + rows * kOutputs, [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("projection returned nonfinite values");
    }
    std::ofstream file("rows-" + std::to_string(rows) + ".f32", std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    std::vector<float> ordered(rows * kOutputs);
    for (int channel = 0; channel < kOutputs / kChannelPack; ++channel) {
        for (int row = 0; row < rows; ++row) {
            const auto * packed = output + (channel * rows + row) * kChannelPack;
            std::copy_n(packed, kChannelPack, ordered.data() + row * kOutputs + channel * kChannelPack);
        }
    }
    file.write(reinterpret_cast<const char *>(ordered.data()), ordered.size() * sizeof(float));
    file.close();
}

void benchmark_rows(Module & model, const std::vector<float> & input, int rows, std::ofstream & csv) {
    const auto * values = input.data() + (kMaxRows - rows) * kInputs;
    auto tensor = _Input({rows, kInputs, 1, 1}, NC4HW4);
    const float * result = nullptr;
    for (int iteration = -kWarmups; iteration < kRepeats; ++iteration) {
        auto * mapped = tensor->writeMap<float>();
        if (!mapped) {
            throw std::runtime_error("cannot map input");
        }
        // MNN's host NC4HW4 tensor stores four channels together, then advances to the next row.
        for (int channel = 0; channel < kInputs / kChannelPack; ++channel) {
            for (int row = 0; row < rows; ++row) {
                const auto * source = values + row * kInputs + channel * kChannelPack;
                std::copy_n(source, kChannelPack, mapped + (channel * rows + row) * kChannelPack);
            }
        }
        const auto start = Clock::now();
        auto outputs = model.onForward({tensor});
        if (outputs.size() != 1 || !outputs[0].get()) {
            throw std::runtime_error("projection failed");
        }
        result = outputs[0]->readMap<float>();
        const double elapsed = elapsed_us(start);
        if (!result || !outputs[0]->getInfo() ||
            outputs[0]->getInfo()->size != static_cast<size_t>(rows) * kOutputs) {
            throw std::runtime_error("unexpected output size");
        }
        if (iteration >= 0) {
            csv << rows << ',' << iteration + 1 << ',' << elapsed << '\n';
        }
        if (iteration == kRepeats - 1) {
            save_rows(rows, result);
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if ((argc != 5 && argc != 6) || (argc == 6 && std::string(argv[5]) != "single")) {
        std::cerr << "Usage: mnn_linear_bench cpu|opencl|opencl-heavy MODEL INPUT forward|reverse [single]\n";
        return 2;
    }
    try {
        const std::string mode(argv[1]);
        const std::string order(argv[4]);
        if ((mode != "cpu" && mode != "opencl" && mode != "opencl-heavy") ||
            (order != "forward" && order != "reverse")) {
            throw std::runtime_error("unrecognized backend or order");
        }
        const bool gpu = mode != "cpu";
        const auto input = read_input(argv[3]);
        BackendConfig backend;
        backend.precision = BackendConfig::Precision_Low;
        backend.memory = BackendConfig::Memory_Low;
        const auto executor = Executor::newExecutor(MNN_FORWARD_CPU, backend, kCpuThreads);
        ExecutorScope scope(executor);
        ScheduleConfig schedule;
        schedule.type = gpu ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
        schedule.numThread = kCpuThreads;
        if (gpu) {
            const auto tuning = mode == "opencl-heavy" ? MNN_GPU_TUNING_HEAVY : MNN_GPU_TUNING_WIDE;
            schedule.numThread = tuning | MNN_GPU_MEMORY_BUFFER | MNN_GPU_RECORD_BATCH;
        }
        schedule.backendConfig = &backend;
        using Runtime = Executor::RuntimeManager;
        auto runtime = std::shared_ptr<Runtime>(Runtime::createRuntimeManager(schedule), Runtime::destroy);
        if (!runtime) {
            throw std::runtime_error("cannot initialize MNN backend");
        }
        const std::string model_path(argv[2]);
        runtime->setExternalFile(model_path + ".weight");
        const std::shared_ptr<Module> model(Module::load({"input"}, {"output"}, argv[2], runtime),
                                            Module::destroy);
        if (!model) {
            throw std::runtime_error("cannot load isolated projection");
        }

        std::vector<int> row_counts{1, 8, 32, 128, kMaxRows};
        if (argc == 6) {
            row_counts = {1};
        }
        if (order == "reverse") {
            std::reverse(row_counts.begin(), row_counts.end());
        }
        std::ofstream csv("timing.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << "rows,iteration,wall_us\n" << std::fixed << std::setprecision(3);
        for (int rows : row_counts) {
            benchmark_rows(*model, input, rows, csv);
            csv.flush();
        }
        csv.close();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
