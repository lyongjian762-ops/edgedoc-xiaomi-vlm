#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace edgedoc {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

inline void check_cl(cl_int code, const char * action) {
    if (code != CL_SUCCESS) throw std::runtime_error(std::string(action) + ": " + std::to_string(code));
}

template <class T, cl_int (*Release)(T)>
using ClPtr = std::unique_ptr<std::remove_pointer_t<T>, decltype(Release)>;
using Mem = ClPtr<cl_mem, clReleaseMemObject>;
using Event = ClPtr<cl_event, clReleaseEvent>;

inline std::vector<uint8_t> read_exact(const fs::path & path, size_t size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.tellg() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error("unexpected operand size: " + path.string());
    }
    file.seekg(0);
    std::vector<uint8_t> result(size);
    if (!file.read(reinterpret_cast<char *>(result.data()), size)) throw std::runtime_error("could not read " + path.string());
    return result;
}

inline double event_interval_ms(cl_event first, cl_event last) {
    cl_ulong start = 0, end = 0;
    check_cl(clGetEventProfilingInfo(first, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr), "read event start");
    check_cl(clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "read event end");
    if (end < start) throw std::runtime_error("invalid OpenCL timestamps");
    return (end - start) / 1e6;
}

inline cl_device_id select_gpu() {
    cl_uint count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &count), "enumerate platforms");
    std::vector<cl_platform_id> platforms(count);
    check_cl(clGetPlatformIDs(count, platforms.data(), nullptr), "list platforms");
    for (const auto platform : platforms) {
        cl_device_id device = nullptr;
        const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (status != CL_DEVICE_NOT_FOUND) {
            check_cl(status, "select GPU");
            return device;
        }
    }
    throw std::runtime_error("no OpenCL GPU");
}

struct GemvTiming {
    double upload_ms;
    double kernel_ms;
    double readback_ms;
    double wall_ms;
};

// The weights, program, queue and GPU buffers stay alive for the complete request.
class Q4Gemv {
public:
    static constexpr size_t input_width = 1024;
    static constexpr size_t output_width = 248320;
    static constexpr size_t groups = 16;

    Q4Gemv(const std::string & mode, const fs::path & source_file, const fs::path & original_weight,
          const fs::path & prepared, size_t local_size) : local_size_(local_size), half_(mode == "tile-fp16") {
        const bool tiled = mode != "row-fp32";
        if ((mode != "row-fp32" && mode != "tile-fp32" && mode != "tile-fp16") ||
            (local_size != 32 && local_size != 64 && local_size != 128)) {
            throw std::runtime_error("invalid kernel mode or workgroup size");
        }
        const auto device = select_gpu();
        char name[256]{};
        check_cl(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "device name");
        std::cerr << "OpenCL device: " << name << ", mode=" << mode << ", local_size=" << local_size << '\n';
        cl_int status = 0;
        context_.reset(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status));
        check_cl(status, "create context");
        queue_.reset(clCreateCommandQueue(context_.get(), device, CL_QUEUE_PROFILING_ENABLE, &status));
        check_cl(status, "create profiling queue");

        std::ifstream source_stream(source_file);
        const std::string source((std::istreambuf_iterator<char>(source_stream)), std::istreambuf_iterator<char>());
        if (source.empty()) throw std::runtime_error("could not read GEMV kernel source");
        const char * source_ptr = source.c_str();
        program_.reset(clCreateProgramWithSource(context_.get(), 1, &source_ptr, nullptr, &status));
        check_cl(status, "create program");
        const std::string options = std::string("-cl-std=CL1.2 -DTILED=") + (tiled ? "1" : "0")
                                    + " -DHALF_ACC=" + (half_ ? "1" : "0");
        status = clBuildProgram(program_.get(), 1, &device, options.c_str(), nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t length = 0;
            clGetProgramBuildInfo(program_.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
            if (length) {
                std::string log(length, '\0');
                clGetProgramBuildInfo(program_.get(), device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
                std::cerr << log << '\n';
            }
            check_cl(status, "build GEMV kernel");
        }
        kernel_.reset(clCreateKernel(program_.get(), "q4_gemv", &status));
        check_cl(status, "create GEMV kernel");
        size_t maximum_group = 0;
        check_cl(clGetKernelWorkGroupInfo(kernel_.get(), device, CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(maximum_group), &maximum_group, nullptr), "check workgroup size");
        if (local_size_ > maximum_group || output_width % local_size_ != 0) {
            throw std::runtime_error("unsupported local workgroup size");
        }

        constexpr size_t weight_bytes = output_width * input_width / 2;
        constexpr size_t scale_bytes = output_width * groups * 2 * sizeof(uint16_t);
        const size_t header = tiled ? 0 : 26;
        auto weight_file = read_exact(tiled ? prepared / "tile64.bin" : original_weight,
                                      header + weight_bytes + scale_bytes);
        weight_ = buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weight_bytes, weight_file.data() + header);
        scale_ = buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scale_bytes, weight_file.data() + header + weight_bytes);
        input_ = buffer(CL_MEM_READ_ONLY, token_bytes());
        sum_ = buffer(CL_MEM_READ_ONLY, groups * sizeof(float));
        output_ = buffer(CL_MEM_WRITE_ONLY, output_width * sizeof(float));
        const cl_mem arguments[] = {weight_.get(), scale_.get(), input_.get(), sum_.get(), output_.get()};
        for (cl_uint i = 0; i < sizeof(arguments) / sizeof(arguments[0]); ++i) {
            check_cl(clSetKernelArg(kernel_.get(), i, sizeof(cl_mem), &arguments[i]), "set kernel argument");
        }
    }

    size_t token_bytes() const { return input_width * (half_ ? sizeof(uint16_t) : sizeof(float)); }

    GemvTiming compute(const void * token, const void * group_sum, float * logits) {
        const auto begin = Clock::now();
        auto input_event = upload(input_.get(), token_bytes(), token);
        auto sum_event = upload(sum_.get(), groups * sizeof(float), group_sum);
        cl_event raw_kernel = nullptr;
        check_cl(clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), 1, nullptr, &output_width,
                                        &local_size_, 0, nullptr, &raw_kernel), "submit GEMV");
        Event kernel_event(raw_kernel, clReleaseEvent);
        cl_event raw_read = nullptr;
        check_cl(clEnqueueReadBuffer(queue_.get(), output_.get(), CL_TRUE, 0, output_width * sizeof(float),
                                     logits, 0, nullptr, &raw_read), "read logits");
        Event read_event(raw_read, clReleaseEvent);
        const double wall = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        return {event_interval_ms(input_event.get(), sum_event.get()),
                event_interval_ms(kernel_event.get(), kernel_event.get()),
                event_interval_ms(read_event.get(), read_event.get()), wall};
    }

    GemvTiming compute_hidden(const float * hidden, float * logits) {
        if (half_) throw std::runtime_error("live hidden-state path requires FP32 input mode");
        std::array<float, input_width> packed{};
        std::array<float, groups> sums{};
        for (size_t group = 0; group < groups; ++group) {
            for (size_t i = 0; i < 32; ++i) {
                const float even = hidden[group * 64 + 2 * i];
                const float odd = hidden[group * 64 + 2 * i + 1];
                packed[group * 32 + i] = even;
                packed[input_width / 2 + group * 32 + i] = odd;
                sums[group] += even + odd;
            }
        }
        return compute(packed.data(), sums.data(), logits);
    }

private:
    Mem buffer(cl_mem_flags flags, size_t size, void * data = nullptr) {
        cl_int status = 0;
        Mem result(clCreateBuffer(context_.get(), flags, size, data, &status), clReleaseMemObject);
        check_cl(status, "allocate buffer");
        return result;
    }

    Event upload(cl_mem memory, size_t bytes, const void * data) {
        cl_event event = nullptr;
        check_cl(clEnqueueWriteBuffer(queue_.get(), memory, CL_FALSE, 0, bytes, data, 0, nullptr, &event), "upload token vector");
        return Event(event, clReleaseEvent);
    }

    size_t local_size_;
    bool half_;
    ClPtr<cl_context, clReleaseContext> context_{nullptr, clReleaseContext};
    ClPtr<cl_command_queue, clReleaseCommandQueue> queue_{nullptr, clReleaseCommandQueue};
    ClPtr<cl_program, clReleaseProgram> program_{nullptr, clReleaseProgram};
    ClPtr<cl_kernel, clReleaseKernel> kernel_{nullptr, clReleaseKernel};
    Mem weight_{nullptr, clReleaseMemObject}, scale_{nullptr, clReleaseMemObject};
    Mem input_{nullptr, clReleaseMemObject}, sum_{nullptr, clReleaseMemObject}, output_{nullptr, clReleaseMemObject};
};
}  // namespace edgedoc
