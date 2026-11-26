#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cblas.h>
#include <opencv2/core.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kRows = 784;
constexpr int kInputWidth = 768;
constexpr int kOutputWidth = 2304;
constexpr int kCpuThreads = 4;
constexpr int kWarmups = 2;
constexpr int kRepeats = 15;
static_assert(kRows % 8 == 0 && kInputWidth % 16 == 0 && kOutputWidth % 32 == 0);

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<uint16_t> read_half(const fs::path & path, size_t count) {
    std::vector<uint16_t> data(count);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(uint16_t));
    if (file.tellg() != bytes) {
        throw std::runtime_error("missing or wrong size: " + path.string());
    }
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(data.data()), bytes)) {
        throw std::runtime_error("could not read " + path.string());
    }
    return data;
}

void save_float(const fs::path & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    file.close();
}

void check(cl_int code, const char * operation) {
    if (code != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(code) + ")");
    }
}

template <class T, cl_int (*Release)(T)>
using ClPtr = std::unique_ptr<std::remove_pointer_t<T>, decltype(Release)>;

struct GpuTiming {
    double wall_ms;
    double upload_ms;
    double kernel_ms;
    double readback_ms;
};

double event_ms(cl_event event) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
          "read event start");
    check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
          "read event end");
    if (end < start) {
        throw std::runtime_error("OpenCL event ended before it began");
    }
    return (end - start) / 1e6;
}

class MaliMatMul {
public:
    explicit MaliMatMul(const std::string & source) {
        cl_uint count = 0;
        check(clGetPlatformIDs(0, nullptr, &count), "enumerate OpenCL platforms");
        if (count == 0) {
            throw std::runtime_error("no OpenCL platform");
        }
        std::vector<cl_platform_id> platforms(count);
        check(clGetPlatformIDs(count, platforms.data(), nullptr), "list OpenCL platforms");
        for (const auto platform : platforms) {
            const auto code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
            if (code != CL_DEVICE_NOT_FOUND) {
                check(code, "select OpenCL GPU");
                break;
            }
        }
        if (!device_) {
            throw std::runtime_error("no OpenCL GPU");
        }
        std::array<char, 128> name{};
        check(clGetDeviceInfo(device_, CL_DEVICE_NAME, name.size(), name.data(), nullptr), "read GPU name");
        std::cerr << "OpenCL device: " << name.data() << '\n';

        cl_int code = 0;
        context_.reset(clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &code));
        check(code, "create context");
        queue_.reset(clCreateCommandQueue(context_.get(), device_, CL_QUEUE_PROFILING_ENABLE, &code));
        check(code, "create profiling queue");
        const char * text = source.c_str();
        program_.reset(clCreateProgramWithSource(context_.get(), 1, &text, nullptr, &code));
        check(code, "create program");
        const std::string options = "-cl-std=CL1.2 -DINPUT_WIDTH=" + std::to_string(kInputWidth) +
                                    " -DOUTPUT_WIDTH=" + std::to_string(kOutputWidth);
        code = clBuildProgram(program_.get(), 1, &device_, options.c_str(), nullptr, nullptr);
        if (code != CL_SUCCESS) {
            size_t length = 0;
            if (clGetProgramBuildInfo(program_.get(), device_, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                      &length) == CL_SUCCESS && length > 0) {
                std::string log(length, '\0');
                if (clGetProgramBuildInfo(program_.get(), device_, CL_PROGRAM_BUILD_LOG,
                                          log.size(), log.data(), nullptr) == CL_SUCCESS) {
                    std::cerr << log << '\n';
                }
            }
            check(code, "build QKV kernel");
        }
        kernel_.reset(clCreateKernel(program_.get(), "qkv_matmul", &code));
        check(code, "create QKV kernel");
    }

    void configure(const std::vector<uint16_t> & weight) {
        input_.reset(buffer(CL_MEM_READ_ONLY, kRows * kInputWidth * sizeof(uint16_t)));
        weight_.reset(buffer(CL_MEM_READ_ONLY, weight.size() * sizeof(uint16_t)));
        output_.reset(buffer(CL_MEM_WRITE_ONLY, kRows * kOutputWidth * sizeof(float)));
        check(clEnqueueWriteBuffer(queue_.get(), weight_.get(), CL_TRUE, 0,
                                   weight.size() * sizeof(uint16_t), weight.data(), 0, nullptr, nullptr),
              "upload resident weights");
        set_buffer(0, input_.get());
        set_buffer(1, weight_.get());
        set_buffer(2, output_.get());
    }

    GpuTiming run(const std::vector<uint16_t> & input, std::vector<float> & output) {
        const auto start = Clock::now();
        const size_t global[] = {kOutputWidth / 4, kRows};
        const size_t local[] = {8, 8};
        ClPtr<cl_event, clReleaseEvent> upload_event(nullptr, clReleaseEvent);
        ClPtr<cl_event, clReleaseEvent> kernel_event(nullptr, clReleaseEvent);
        ClPtr<cl_event, clReleaseEvent> read_event(nullptr, clReleaseEvent);
        cl_event event = nullptr;
        check(clEnqueueWriteBuffer(queue_.get(), input_.get(), CL_FALSE, 0,
                                   input.size() * sizeof(uint16_t), input.data(), 0, nullptr, &event),
              "upload activations");
        upload_event.reset(event);
        try {
            check(clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), 2, nullptr, global, local,
                                         0, nullptr, &event), "dispatch QKV kernel");
            kernel_event.reset(event);
            check(clEnqueueReadBuffer(queue_.get(), output_.get(), CL_TRUE, 0,
                                      output.size() * sizeof(float), output.data(), 0, nullptr, &event),
                  "read QKV output");
            read_event.reset(event);
        } catch (...) {
            // Keep the host activations alive until the queued upload completes.
            clFinish(queue_.get());
            throw;
        }
        const double wall = elapsed_ms(start);
        return {wall, event_ms(upload_event.get()), event_ms(kernel_event.get()), event_ms(read_event.get())};
    }

private:
    cl_mem buffer(cl_mem_flags flags, size_t bytes) {
        cl_int code = 0;
        cl_mem memory = clCreateBuffer(context_.get(), flags, bytes, nullptr, &code);
        check(code, "allocate OpenCL buffer");
        return memory;
    }

    void set_buffer(cl_uint index, cl_mem memory) {
        check(clSetKernelArg(kernel_.get(), index, sizeof(memory), &memory), "set OpenCL argument");
    }

    cl_device_id device_ = nullptr;
    ClPtr<cl_context, clReleaseContext> context_{nullptr, clReleaseContext};
    ClPtr<cl_command_queue, clReleaseCommandQueue> queue_{nullptr, clReleaseCommandQueue};
    ClPtr<cl_program, clReleaseProgram> program_{nullptr, clReleaseProgram};
    ClPtr<cl_kernel, clReleaseKernel> kernel_{nullptr, clReleaseKernel};
    ClPtr<cl_mem, clReleaseMemObject> input_{nullptr, clReleaseMemObject};
    ClPtr<cl_mem, clReleaseMemObject> weight_{nullptr, clReleaseMemObject};
    ClPtr<cl_mem, clReleaseMemObject> output_{nullptr, clReleaseMemObject};
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "Usage: compare_qkv <qkv.cl> <operand_dir> <existing_output_dir>\n";
        return 2;
    }
    try {
        const fs::path operands(argv[2]);
        const fs::path output(argv[3]);
        auto input = read_half(operands / "x.f16", kRows * kInputWidth);
        auto weight = read_half(operands / "w_transposed.f16", kInputWidth * kOutputWidth);
        std::vector<float> cpu_output(kRows * kOutputWidth);
        std::vector<float> gpu_output(kRows * kOutputWidth);

        std::ifstream kernel(argv[1]);
        const std::string source((std::istreambuf_iterator<char>(kernel)), {});
        if (source.empty()) {
            throw std::runtime_error("empty or missing OpenCL source");
        }
        const auto init = Clock::now();
        MaliMatMul mali(source);
        std::cerr << "OpenCL initialization and build ms: " << elapsed_ms(init) << '\n';
        const auto configure = Clock::now();
        mali.configure(weight);
        std::cerr << "OpenCL buffer allocation and one-time weight upload ms: " << elapsed_ms(configure) << '\n';

        openblas_set_num_threads(kCpuThreads);
        std::cerr << "OpenBLAS " << openblas_get_config() << ", core " << openblas_get_corename()
                  << ", configured threads " << openblas_get_num_threads() << '\n';
        const auto convert = Clock::now();
        cv::Mat cpu_input;
        cv::Mat cpu_weight;
        cv::convertFp16(cv::Mat(kRows, kInputWidth, CV_16SC1, input.data()), cpu_input);
        cv::convertFp16(cv::Mat(kInputWidth, kOutputWidth, CV_16SC1, weight.data()), cpu_weight);
        if (!cpu_input.isContinuous() || cpu_input.type() != CV_32FC1 ||
            !cpu_weight.isContinuous() || cpu_weight.type() != CV_32FC1) {
            throw std::runtime_error("expected contiguous FP32 CPU operands");
        }
        std::cerr << "CPU one-time operand conversion ms: " << elapsed_ms(convert) << '\n';
        auto run_cpu = [&] {
            const auto start = Clock::now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kRows, kOutputWidth, kInputWidth,
                        1.f, cpu_input.ptr<float>(), kInputWidth, cpu_weight.ptr<float>(), kOutputWidth,
                        0.f, cpu_output.data(), kOutputWidth);
            return elapsed_ms(start);
        };

        for (int i = 0; i < kWarmups; ++i) {
            run_cpu();
            mali.run(input, gpu_output);
        }

        std::ofstream csv(output / "timing.csv");
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv << "iteration,cpu_ms,gpu_wall_ms,gpu_upload_ms,gpu_kernel_ms,gpu_readback_ms\n";
        csv << std::fixed << std::setprecision(6);
        for (int i = 0; i < kRepeats; ++i) {
            double cpu_time;
            GpuTiming gpu_time;
            if (i % 2 == 0) {
                cpu_time = run_cpu();
                gpu_time = mali.run(input, gpu_output);
            } else {
                gpu_time = mali.run(input, gpu_output);
                cpu_time = run_cpu();
            }
            csv << i + 1 << ',' << cpu_time << ',' << gpu_time.wall_ms << ',' << gpu_time.upload_ms << ','
                << gpu_time.kernel_ms << ',' << gpu_time.readback_ms << '\n';
        }
        csv.close();
        save_float(output / "cpu.f32", cpu_output);
        save_float(output / "gpu.f32", gpu_output);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
