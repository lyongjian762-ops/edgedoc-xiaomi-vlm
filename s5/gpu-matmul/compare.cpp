#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cblas.h>
#include <opencv2/core.hpp>
#ifdef EDGEDOC_USE_CLBLAST
#include <clblast_c.h>
#endif

#include "bench_io.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace qkv;
constexpr int kCpuThreads = 4;
static_assert(kRows % 8 == 0 && kInputWidth % 32 == 0 && kOutputWidth % 32 == 0);

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

double interval_ms(cl_event first, cl_profiling_info first_point, cl_event last) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    check(clGetEventProfilingInfo(first, first_point, sizeof(start), &start, nullptr),
          "read event start");
    check(clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
          "read event end");
    if (end < start) {
        throw std::runtime_error("OpenCL event ended before it began");
    }
    return (end - start) / 1e6;
}

double event_ms(cl_event event) {
    return interval_ms(event, CL_PROFILING_COMMAND_START, event);
}

class MaliMatMul {
public:
    MaliMatMul(const std::string & source, bool use_clblast) : use_clblast_(use_clblast) {
#ifndef EDGEDOC_USE_CLBLAST
        if (use_clblast_) {
            throw std::runtime_error("CLBlast support was omitted from this build");
        }
#endif
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
        if (use_clblast_) {
            std::cerr << "Mali backend: CLBlast FP32 SGEMM\n";
            return;
        }
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

    ~MaliMatMul() {
#ifdef EDGEDOC_USE_CLBLAST
        if (use_clblast_) {
            // Release CLBlast's cached OpenCL objects while the context is still alive.
            const auto status = CLBlastClearCache();
            if (status != CLBlastSuccess) {
                std::cerr << "CLBlastClearCache failed (" << status << ")\n";
            }
        }
#endif
    }

    void configure(const void * weight) {
        const size_t scalar_bytes = use_clblast_ ? sizeof(float) : sizeof(uint16_t);
        input_bytes_ = kRows * kInputWidth * scalar_bytes;
        const size_t weight_bytes = kInputWidth * kOutputWidth * scalar_bytes;
        input_.reset(buffer(CL_MEM_READ_ONLY, input_bytes_));
        weight_.reset(buffer(CL_MEM_READ_ONLY, weight_bytes));
        output_.reset(buffer(CL_MEM_READ_WRITE, kRows * kOutputWidth * sizeof(float)));
        check(clEnqueueWriteBuffer(queue_.get(), weight_.get(), CL_TRUE, 0, weight_bytes, weight,
                                   0, nullptr, nullptr),
              "upload resident weights");
        if (!use_clblast_) {
            set_buffer(0, input_.get());
            set_buffer(1, weight_.get());
            set_buffer(2, output_.get());
        }
    }

    GpuTiming run(const void * input, std::vector<float> & output) {
        const auto start = Clock::now();
        const size_t global[] = {kOutputWidth / 4, kRows};
        const size_t local[] = {8, 8};
        ClPtr<cl_event, clReleaseEvent> upload_event(nullptr, clReleaseEvent);
        ClPtr<cl_event, clReleaseEvent> compute_begin(nullptr, clReleaseEvent);
        ClPtr<cl_event, clReleaseEvent> kernel_event(nullptr, clReleaseEvent);
        ClPtr<cl_event, clReleaseEvent> read_event(nullptr, clReleaseEvent);
        cl_event event = nullptr;
        check(clEnqueueWriteBuffer(queue_.get(), input_.get(), CL_FALSE, 0, input_bytes_,
                                   input, 0, nullptr, &event),
              "upload activations");
        upload_event.reset(event);
        try {
#ifdef EDGEDOC_USE_CLBLAST
            if (use_clblast_) {
                compute_begin.reset(marker());
                enqueue_sgemm();
                kernel_event.reset(marker());
            } else
#endif
            {
                check(clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), 2, nullptr, global, local,
                                             0, nullptr, &event), "dispatch QKV kernel");
                kernel_event.reset(event);
            }
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
        // A CLBlast call may submit packing and compute kernels. Measure the whole queue interval.
        const double compute_ms = compute_begin
            ? interval_ms(compute_begin.get(), CL_PROFILING_COMMAND_END, kernel_event.get())
            : event_ms(kernel_event.get());
        return {wall, event_ms(upload_event.get()), compute_ms, event_ms(read_event.get())};
    }

private:
#ifdef EDGEDOC_USE_CLBLAST
    cl_event marker() {
        cl_event event = nullptr;
        check(clEnqueueMarkerWithWaitList(queue_.get(), 0, nullptr, &event), "enqueue profiling marker");
        return event;
    }

    void enqueue_sgemm() {
        auto queue = queue_.get();
        const auto status = CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                                        kRows, kOutputWidth, kInputWidth, 1.f,
                                        input_.get(), 0, kInputWidth, weight_.get(), 0, kOutputWidth,
                                        0.f, output_.get(), 0, kOutputWidth, &queue, nullptr);
        if (status != CLBlastSuccess) {
            throw std::runtime_error("CLBlastSgemm failed (" + std::to_string(status) + ")");
        }
    }
#endif

    cl_mem buffer(cl_mem_flags flags, size_t bytes) {
        cl_int code = 0;
        cl_mem memory = clCreateBuffer(context_.get(), flags, bytes, nullptr, &code);
        check(code, "allocate OpenCL buffer");
        return memory;
    }

    void set_buffer(cl_uint index, cl_mem memory) {
        check(clSetKernelArg(kernel_.get(), index, sizeof(memory), &memory), "set OpenCL argument");
    }

    bool use_clblast_;
    size_t input_bytes_ = 0;
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
        std::cerr << "Usage: compare_qkv <qkv.cl|--clblast> <operand_dir> <existing_output_dir>\n";
        return 2;
    }
    try {
        const fs::path operands(argv[2]);
        const fs::path output(argv[3]);
        auto input = read_half(operands / "x.f16", kRows * kInputWidth);
        auto weight = read_half(operands / "w_transposed.f16", kInputWidth * kOutputWidth);
        std::vector<float> cpu_output(kRows * kOutputWidth);
        std::vector<float> gpu_output(kRows * kOutputWidth);

        const bool use_clblast = std::string(argv[1]) == "--clblast";
        std::string source;
        if (!use_clblast) {
            std::ifstream kernel(argv[1]);
            source.assign(std::istreambuf_iterator<char>(kernel), {});
            if (source.empty()) {
                throw std::runtime_error("empty or missing OpenCL source");
            }
        }
        const auto init = Clock::now();
        MaliMatMul mali(source, use_clblast);
        std::cerr << "OpenCL initialization and build ms: " << elapsed_ms(init) << '\n';

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
        const void * gpu_input = use_clblast ? static_cast<const void *>(cpu_input.ptr<float>()) : input.data();
        const void * gpu_weight = use_clblast ? static_cast<const void *>(cpu_weight.ptr<float>()) : weight.data();
        const auto configure = Clock::now();
        mali.configure(gpu_weight);
        std::cerr << "OpenCL buffer allocation and one-time weight upload ms: " << elapsed_ms(configure) << '\n';
        auto run_cpu = [&] {
            const auto start = Clock::now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kRows, kOutputWidth, kInputWidth,
                        1.f, cpu_input.ptr<float>(), kInputWidth, cpu_weight.ptr<float>(), kOutputWidth,
                        0.f, cpu_output.data(), kOutputWidth);
            return elapsed_ms(start);
        };

        for (int i = 0; i < kWarmups; ++i) {
            run_cpu();
            const auto warmup = mali.run(gpu_input, gpu_output);
            if (i == 0) {
                std::cerr << "First GPU request ms: " << warmup.wall_ms << '\n';
            }
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
                gpu_time = mali.run(gpu_input, gpu_output);
            } else {
                gpu_time = mali.run(gpu_input, gpu_output);
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
