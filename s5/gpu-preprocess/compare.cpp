#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
constexpr int kSize = 448;
constexpr int kRepeats = 30;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(cl_int code, const char * operation) {
    if (code != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(code) + ")");
    }
}

template <class T, cl_int (*Release)(T)>
using ClPtr = std::unique_ptr<std::remove_pointer_t<T>, decltype(Release)>;

std::vector<cl_int> sampling_map(int length) {
    std::vector<cl_int> result;
    result.reserve(kSize * 3);
    const int denominator = 2 * kSize;
    for (int i = 0; i < kSize; ++i) {
        const int center = (2 * i + 1) * length - kSize;
        const int clamped = std::clamp(center, 0, denominator * (length - 1));
        const int first = clamped / denominator;
        const int fraction = clamped % denominator;
        result.push_back(first);
        result.push_back(std::min(first + 1, length - 1));
        result.push_back((fraction * 256 + denominator / 2) / denominator);
    }
    return result;
}

class MaliResize {
public:
    explicit MaliResize(const std::string & source) {
        cl_uint count = 0;
        check(clGetPlatformIDs(0, nullptr, &count), "enumerate OpenCL platforms");
        std::vector<cl_platform_id> platforms(count);
        check(clGetPlatformIDs(count, platforms.data(), nullptr), "list OpenCL platforms");
        for (const auto platform : platforms) {
            const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
            if (status != CL_DEVICE_NOT_FOUND) {
                check(status, "select OpenCL GPU");
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
        code = clBuildProgram(program_.get(), 1, &device_, "-cl-std=CL1.2", nullptr, nullptr);
        if (code != CL_SUCCESS) {
            size_t length = 0;
            clGetProgramBuildInfo(program_.get(), device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
            std::string log(length, '\0');
            clGetProgramBuildInfo(program_.get(), device_, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
            std::cerr << log << '\n';
            check(code, "build resize kernel");
        }
        kernel_.reset(clCreateKernel(program_.get(), "resize_rgb", &code));
        check(code, "create resize kernel");
    }

    void configure(const cv::Mat & image) {
        source_bytes_ = image.total() * image.elemSize();
        output_.reset(buffer(CL_MEM_WRITE_ONLY, kSize * kSize * 3, nullptr));
        source_.reset(buffer(CL_MEM_READ_ONLY, source_bytes_, nullptr));
        auto xs = sampling_map(image.cols);
        auto ys = sampling_map(image.rows);
        xs_.reset(buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xs.size() * sizeof(cl_int), xs.data()));
        ys_.reset(buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ys.size() * sizeof(cl_int), ys.data()));

        set_buffer(0, source_.get());
        const cl_int width = image.cols;
        check(clSetKernelArg(kernel_.get(), 1, sizeof(width), &width), "set source width");
        set_buffer(2, xs_.get());
        set_buffer(3, ys_.get());
        set_buffer(4, output_.get());
    }

    std::array<double, 2> run(const cv::Mat & image, cv::Mat & rgb) {
        const auto start = Clock::now();
        check(clEnqueueWriteBuffer(queue_.get(), source_.get(), CL_FALSE, 0, source_bytes_,
                                   image.data, 0, nullptr, nullptr), "upload decoded pixels");
        const size_t global[] = {kSize, kSize};
        cl_event completion = nullptr;
        ClPtr<cl_event, clReleaseEvent> event(nullptr, clReleaseEvent);
        try {
            check(clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), 2, nullptr, global, nullptr,
                                        0, nullptr, &completion), "enqueue resize kernel");
            event.reset(completion);
            check(clEnqueueReadBuffer(queue_.get(), output_.get(), CL_TRUE, 0, rgb.total() * rgb.elemSize(),
                                      rgb.data, 0, nullptr, nullptr), "read resized pixels");
        } catch (...) {
            // The host pixels must stay alive until the queued upload has finished.
            clFinish(queue_.get());
            throw;
        }
        const double wall = elapsed(start);

        cl_ulong begin = 0;
        cl_ulong end = 0;
        check(clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_START, sizeof(begin), &begin, nullptr),
              "read kernel start");
        check(clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
              "read kernel end");
        return {wall, (end - begin) / 1e6};
    }

private:
    cl_mem buffer(cl_mem_flags flags, size_t size, void * data) {
        cl_int code = 0;
        auto result = clCreateBuffer(context_.get(), flags, size, data, &code);
        check(code, "allocate OpenCL buffer");
        return result;
    }

    void set_buffer(cl_uint index, cl_mem memory) {
        check(clSetKernelArg(kernel_.get(), index, sizeof(memory), &memory), "set kernel buffer");
    }

    cl_device_id device_ = nullptr;
    ClPtr<cl_context, clReleaseContext> context_{nullptr, clReleaseContext};
    ClPtr<cl_command_queue, clReleaseCommandQueue> queue_{nullptr, clReleaseCommandQueue};
    ClPtr<cl_program, clReleaseProgram> program_{nullptr, clReleaseProgram};
    ClPtr<cl_kernel, clReleaseKernel> kernel_{nullptr, clReleaseKernel};
    ClPtr<cl_mem, clReleaseMemObject> source_{nullptr, clReleaseMemObject};
    ClPtr<cl_mem, clReleaseMemObject> output_{nullptr, clReleaseMemObject};
    ClPtr<cl_mem, clReleaseMemObject> xs_{nullptr, clReleaseMemObject};
    ClPtr<cl_mem, clReleaseMemObject> ys_{nullptr, clReleaseMemObject};
    size_t source_bytes_ = 0;
};

void save_rgb(const fs::path & path, const cv::Mat & rgb) {
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path.string(), bgr)) {
        throw std::runtime_error("could not save " + path.string());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "Usage: compare_gpu <resize_rgb.cl> <new_output_dir> <original_image>...\n";
        return 2;
    }
    try {
        if (const char * setting = std::getenv("EDGEDOC_OPENCV_THREADS")) {
            size_t parsed = 0;
            const int threads = std::stoi(setting, &parsed);
            if (threads < 1 || setting[parsed] != '\0') {
                throw std::runtime_error("EDGEDOC_OPENCV_THREADS must specify a positive integer");
            }
            cv::setNumThreads(threads);
        }
        const fs::path output(argv[2]);
        if (!fs::create_directory(output)) {
            throw std::runtime_error("expected a new output directory");
        }
        const auto init_start = Clock::now();
        std::ifstream kernel(argv[1]);
        const std::string source((std::istreambuf_iterator<char>(kernel)), {});
        if (source.empty()) {
            throw std::runtime_error("empty or missing kernel source");
        }
        MaliResize mali(source);
        std::cerr << "OpenCV " << CV_VERSION << ", threads " << cv::getNumThreads()
                  << ", OpenCL initialization ms " << elapsed(init_start) << '\n';
        std::ofstream csv(output / "timing.csv");
        csv << "image,width,height,iteration,decode_ms,gpu_configure_ms,cpu_ms,gpu_wall_ms,gpu_kernel_ms\n";
        std::ofstream pixels(output / "pixels.csv");
        pixels << "image,differing_channels,max_abs_difference\n";
        for (int i = 3; i < argc; ++i) {
            const auto decode_start = Clock::now();
            const auto bgr = cv::imread(argv[i], cv::IMREAD_COLOR);
            const double decode_ms = elapsed(decode_start);
            if (bgr.empty() || !bgr.isContinuous()) {
                throw std::runtime_error(std::string("cannot read packed BGR pixels from ") + argv[i]);
            }
            const auto name = fs::path(argv[i]).stem().string();
            const auto configure_start = Clock::now();
            mali.configure(bgr);
            const double configure_ms = elapsed(configure_start);
            cv::Mat tmp, cpu, gpu(kSize, kSize, CV_8UC3);
            const auto run_cpu = [&]() {
                const auto start = Clock::now();
                cv::resize(bgr, tmp, {kSize, kSize}, 0, 0, cv::INTER_LINEAR_EXACT);
                cv::cvtColor(tmp, cpu, cv::COLOR_BGR2RGB);
                return elapsed(start);
            };
            for (int warm = 0; warm < 5; ++warm) {
                run_cpu();
                mali.run(bgr, gpu);
            }
            for (int repeat = 0; repeat < kRepeats; ++repeat) {
                double cpu_ms;
                std::array<double, 2> gpu_ms;
                if (repeat % 2 == 0) {
                    cpu_ms = run_cpu();
                    gpu_ms = mali.run(bgr, gpu);
                } else {
                    gpu_ms = mali.run(bgr, gpu);
                    cpu_ms = run_cpu();
                }
                csv << name << ',' << bgr.cols << ',' << bgr.rows << ',' << repeat << ','
                    << decode_ms << ',' << configure_ms << ',' << cpu_ms << ',' << gpu_ms[0] << ',' << gpu_ms[1] << '\n';
            }
            cv::Mat difference;
            cv::absdiff(cpu, gpu, difference);
            const auto scalars = difference.reshape(1);
            double max_difference = 0;
            cv::minMaxLoc(scalars, nullptr, &max_difference);
            pixels << name << ',' << cv::countNonZero(scalars) << ',' << max_difference << '\n';
            save_rgb(output / (name + ".cpu.png"), cpu);
            save_rgb(output / (name + ".gpu.png"), gpu);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
