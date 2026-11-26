#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kBytes = 128 * 1024 * 1024;
constexpr int kWarmups = 5;
constexpr int kRepeats = 30;

double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

void check(cl_int code, const char * action) {
    if (code != CL_SUCCESS) {
        throw std::runtime_error(std::string(action) + ": " + std::to_string(code));
    }
}

template <class T, cl_int (*Release)(T)>
using ClPtr = std::unique_ptr<std::remove_pointer_t<T>, decltype(Release)>;

void record(std::ofstream & csv, const char * backend, int iteration, double compute_ms,
            double wall_ms, uint32_t checksum, uint32_t expected) {
    if (checksum != expected || compute_ms <= 0) {
        throw std::runtime_error("read checksum or timing mismatch");
    }
    if (iteration >= 0) {
        csv << backend << ',' << kBytes << ',' << iteration + 1 << ',' << compute_ms << ','
            << wall_ms << ',' << kBytes / (compute_ms * 1e6) << ',' << checksum << '\n';
    }
}

void benchmark_cpu(const std::vector<uint32_t> & data, uint32_t expected, std::ofstream & csv) {
    for (int iteration = -kWarmups; iteration < kRepeats; ++iteration) {
        uint32_t sum = 0;
        const auto begin = Clock::now();
#pragma omp parallel for simd num_threads(4) schedule(static) reduction(+:sum)
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        const double ms = elapsed_ms(begin);
        record(csv, "cpu", iteration, ms, ms, sum, expected);
    }
}

void benchmark_gpu(std::vector<uint32_t> & data, uint32_t expected, std::ofstream & csv) {
    cl_uint platform_count = 0;
    check(clGetPlatformIDs(0, nullptr, &platform_count), "enumerate platforms");
    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "list platforms");
    cl_device_id device = nullptr;
    for (const auto platform : platforms) {
        const auto code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (code != CL_DEVICE_NOT_FOUND) {
            check(code, "select GPU");
            break;
        }
    }
    if (!device) {
        throw std::runtime_error("no GPU device");
    }
    char name[256]{};
    check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "read device name");
    std::cerr << "OpenCL device: " << name << '\n';

    cl_int status = CL_SUCCESS;
    ClPtr<cl_context, clReleaseContext> context(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status), clReleaseContext);
    check(status, "create context");
    ClPtr<cl_command_queue, clReleaseCommandQueue> queue(
        clCreateCommandQueue(context.get(), device, CL_QUEUE_PROFILING_ENABLE, &status), clReleaseCommandQueue);
    check(status, "create profiling queue");

    const char * source = R"CLC(
__kernel void read_reduce(__global const uint4 *input, __global uint *partial, uint count) {
    uint id = get_global_id(0);
    uint4 sum = (uint4)(0);
    for (uint i = id; i < count; i += get_global_size(0)) {
        sum += input[i];
    }
    partial[id] = sum.x + sum.y + sum.z + sum.w;
}
)CLC";
    ClPtr<cl_program, clReleaseProgram> program(
        clCreateProgramWithSource(context.get(), 1, &source, nullptr, &status), clReleaseProgram);
    check(status, "create program");
    status = clBuildProgram(program.get(), 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t length = 0;
        clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
        if (length) {
            std::string log(length, '\0');
            clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
            std::cerr << log << '\n';
        }
        check(status, "build program");
    }
    ClPtr<cl_kernel, clReleaseKernel> kernel(clCreateKernel(program.get(), "read_reduce", &status), clReleaseKernel);
    check(status, "create kernel");
    constexpr size_t local = 128;
    constexpr size_t global = 65536;
    std::vector<uint32_t> partial(global);
    ClPtr<cl_mem, clReleaseMemObject> buffer(clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                          kBytes, data.data(), &status), clReleaseMemObject);
    check(status, "create initialized source buffer");
    ClPtr<cl_mem, clReleaseMemObject> output(clCreateBuffer(context.get(), CL_MEM_WRITE_ONLY,
                                                          global * sizeof(uint32_t), nullptr, &status), clReleaseMemObject);
    check(status, "create partial sum buffer");
    auto in_buffer = buffer.get();
    auto out_buffer = output.get();
    const auto vector_count = static_cast<cl_uint>(data.size() / 4);
    check(clSetKernelArg(kernel.get(), 0, sizeof(in_buffer), &in_buffer), "set source");
    check(clSetKernelArg(kernel.get(), 1, sizeof(out_buffer), &out_buffer), "set output");
    check(clSetKernelArg(kernel.get(), 2, sizeof(vector_count), &vector_count), "set vector count");
    for (int iteration = -kWarmups; iteration < kRepeats; ++iteration) {
        cl_event event = nullptr;
        const auto begin = Clock::now();
        check(clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, nullptr, &global, &local, 0, nullptr, &event), "enqueue read kernel");
        ClPtr<cl_event, clReleaseEvent> owned_event(event, clReleaseEvent);
        check(clWaitForEvents(1, &event), "wait for read kernel");
        const double wall_ms = elapsed_ms(begin);
        cl_ulong start = 0;
        cl_ulong end = 0;
        check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr), "read device start");
        check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "read device end");
        check(clEnqueueReadBuffer(queue.get(), output.get(), CL_TRUE, 0, global * sizeof(uint32_t),
                                  partial.data(), 0, nullptr, nullptr), "read check sums");
        const uint32_t sum = std::accumulate(partial.begin(), partial.end(), uint32_t{0});
        record(csv, "opencl", iteration, (end - start) / 1e6, wall_ms, sum, expected);
    }
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2 || (std::string(argv[1]) != "cpu" && std::string(argv[1]) != "opencl")) {
        std::cerr << "Usage: read_bandwidth cpu|opencl\n";
        return 2;
    }
    try {
        std::vector<uint32_t> data(kBytes / sizeof(uint32_t));
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint32_t>(i) * 1664525U + 1013904223U;
        }
        const uint32_t expected = std::accumulate(data.begin(), data.end(), uint32_t{0});
        std::ofstream csv("bandwidth.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << "backend,bytes,iteration,compute_ms,wall_ms,read_gb_per_s,checksum\n"
            << std::fixed << std::setprecision(6);
        if (std::string(argv[1]) == "cpu") {
            benchmark_cpu(data, expected, csv);
        } else {
            benchmark_gpu(data, expected, csv);
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
