#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
constexpr size_t kOutputs = 248320;
constexpr size_t kWeightBytes = kOutputs * 512;
constexpr size_t kScaleBytes = kOutputs * 16 * 2 * sizeof(uint16_t);
constexpr int kWarmups = 5;
constexpr int kRepeats = 30;

void check(cl_int code, const char * action) {
    if (code != CL_SUCCESS) {
        throw std::runtime_error(std::string(action) + ": " + std::to_string(code));
    }
}

template <class T, cl_int (*Release)(T)>
using ClPtr = std::unique_ptr<std::remove_pointer_t<T>, decltype(Release)>;

using Mem = ClPtr<cl_mem, clReleaseMemObject>;
using Event = ClPtr<cl_event, clReleaseEvent>;

std::vector<uint8_t> read_exact(const fs::path & path, size_t size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.tellg() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error("unexpected operand size: " + path.string());
    }
    file.seekg(0);
    std::vector<uint8_t> result(size);
    if (!file.read(reinterpret_cast<char *>(result.data()), size)) {
        throw std::runtime_error("could not read: " + path.string());
    }
    return result;
}

Mem buffer(cl_context context, cl_mem_flags flags, size_t size, void * data = nullptr) {
    cl_int status = 0;
    Mem result(clCreateBuffer(context, flags, size, data, &status), clReleaseMemObject);
    check(status, "allocate OpenCL buffer");
    return result;
}

double interval_ms(cl_event first, cl_event last) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    check(clGetEventProfilingInfo(first, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr), "read start timestamp");
    check(clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "read end timestamp");
    if (end < start) {
        throw std::runtime_error("invalid OpenCL event timestamps");
    }
    return (end - start) / 1e6;
}

cl_device_id select_gpu() {
    cl_uint count = 0;
    check(clGetPlatformIDs(0, nullptr, &count), "enumerate OpenCL platforms");
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "list OpenCL platforms");
    for (const auto platform : platforms) {
        cl_device_id device = nullptr;
        const cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (status != CL_DEVICE_NOT_FOUND) {
            check(status, "select GPU");
            return device;
        }
    }
    throw std::runtime_error("no OpenCL GPU");
}

Event upload(cl_command_queue queue, cl_mem memory, const std::vector<uint8_t> & data) {
    cl_event event = nullptr;
    check(clEnqueueWriteBuffer(queue, memory, CL_FALSE, 0, data.size(), data.data(), 0, nullptr, &event), "upload token vector");
    return Event(event, clReleaseEvent);
}

void run(const std::string & mode, const fs::path & source_file, const fs::path & original_weight,
         const fs::path & prepared, size_t local_size) {
    const bool tiled = mode != "row-fp32";
    const bool half = mode == "tile-fp16";
    const auto device = select_gpu();
    char name[256]{};
    check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "device name");
    std::cerr << "OpenCL device: " << name << ", mode=" << mode << ", local_size=" << local_size << '\n';

    cl_int status = 0;
    ClPtr<cl_context, clReleaseContext> context(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status), clReleaseContext);
    check(status, "create context");
    ClPtr<cl_command_queue, clReleaseCommandQueue> queue(
        clCreateCommandQueue(context.get(), device, CL_QUEUE_PROFILING_ENABLE, &status), clReleaseCommandQueue);
    check(status, "create profiling queue");

    std::ifstream source_stream(source_file);
    const std::string source((std::istreambuf_iterator<char>(source_stream)), std::istreambuf_iterator<char>());
    if (source.empty()) throw std::runtime_error("could not load OpenCL kernel source");
    const char * source_ptr = source.c_str();
    ClPtr<cl_program, clReleaseProgram> program(
        clCreateProgramWithSource(context.get(), 1, &source_ptr, nullptr, &status), clReleaseProgram);
    check(status, "create source program");
    const std::string options = std::string("-cl-std=CL1.2 -DTILED=") + (tiled ? "1" : "0")
                                + " -DHALF_ACC=" + (half ? "1" : "0");
    status = clBuildProgram(program.get(), 1, &device, options.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t length = 0;
        clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
        if (length) {
            std::string log(length, '\0');
            clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
            std::cerr << log << '\n';
        }
        check(status, "build GEMV kernel");
    }
    ClPtr<cl_kernel, clReleaseKernel> kernel(clCreateKernel(program.get(), "q4_gemv", &status), clReleaseKernel);
    check(status, "create GEMV kernel");
    size_t maximum_group = 0;
    check(clGetKernelWorkGroupInfo(kernel.get(), device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maximum_group), &maximum_group, nullptr), "check workgroup size");
    if (local_size > maximum_group || kOutputs % local_size != 0) {
        throw std::runtime_error("unsupported local workgroup size");
    }

    const size_t header = tiled ? 0 : 26;
    auto weight_file = read_exact(tiled ? prepared / "tile64.bin" : original_weight, header + kWeightBytes + kScaleBytes);
    auto weight = buffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kWeightBytes, weight_file.data() + header);
    auto scale = buffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kScaleBytes, weight_file.data() + header + kWeightBytes);
    weight_file.clear();
    weight_file.shrink_to_fit();
    const auto token = read_exact(prepared / (half ? "input-f16.bin" : "input-f32.bin"), 1024 * (half ? 2 : 4));
    const auto group_sum = read_exact(prepared / "group-sum.f32", 16 * sizeof(float));
    auto input_device = buffer(context.get(), CL_MEM_READ_ONLY, token.size());
    auto sum_device = buffer(context.get(), CL_MEM_READ_ONLY, group_sum.size());
    auto output_device = buffer(context.get(), CL_MEM_WRITE_ONLY, kOutputs * sizeof(float));
    const cl_mem arguments[] = {weight.get(), scale.get(), input_device.get(), sum_device.get(), output_device.get()};
    for (cl_uint i = 0; i < sizeof(arguments) / sizeof(arguments[0]); ++i) {
        check(clSetKernelArg(kernel.get(), i, sizeof(cl_mem), &arguments[i]), "set kernel argument");
    }
    std::vector<float> output(kOutputs);
    std::ofstream csv("timing.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "iteration,upload_ms,kernel_ms,readback_ms,wall_ms\n" << std::fixed << std::setprecision(6);
    for (int step = -kWarmups; step < kRepeats; ++step) {
        const auto begin = Clock::now();
        auto upload_input = upload(queue.get(), input_device.get(), token);
        auto upload_sum = upload(queue.get(), sum_device.get(), group_sum);
        cl_event kernel_event = nullptr;
        check(clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, nullptr, &kOutputs, &local_size, 0, nullptr, &kernel_event), "submit GEMV");
        Event kernel_done(kernel_event, clReleaseEvent);
        cl_event read_event = nullptr;
        check(clEnqueueReadBuffer(queue.get(), output_device.get(), CL_TRUE, 0, output.size() * sizeof(float),
                                  output.data(), 0, nullptr, &read_event), "read vocabulary logits");
        Event read_done(read_event, clReleaseEvent);
        const double wall = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (step >= 0) {
            csv << step + 1 << ',' << interval_ms(upload_input.get(), upload_sum.get()) << ','
                << interval_ms(kernel_done.get(), kernel_done.get()) << ','
                << interval_ms(read_done.get(), read_done.get()) << ',' << wall << '\n';
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
        const std::string mode(argv[1]);
        const size_t local = static_cast<size_t>(std::stoul(argv[5]));
        if ((mode != "row-fp32" && mode != "tile-fp32" && mode != "tile-fp16") ||
            (local != 32 && local != 64 && local != 128)) {
            throw std::runtime_error("invalid kernel mode or local size");
        }
        run(mode, argv[2], argv[3], argv[4], local);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
