#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace qkv {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kRows = 784;
constexpr int kInputWidth = 768;
constexpr int kOutputWidth = 2304;
constexpr int kWarmups = 2;
constexpr int kRepeats = 15;

inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline std::vector<uint16_t> read_half(const fs::path & path, size_t count) {
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

inline void save_float(const fs::path & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    file.close();
}

}  // namespace qkv
