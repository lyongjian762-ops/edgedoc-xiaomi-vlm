#include <llm/llm.hpp>
#include "llmconfig.hpp"
#include "nlohmann/json.hpp"
#ifdef EDGEDOC_CUSTOM_HEAD
#include "../lm-head-opencl/q4_gemv.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace {

using namespace MNN::Express;
using namespace MNN::Transformer;
using nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class DecodeProfile {
public:
    void begin_forward(int step) {
        step_ = step;
        forward_start_ = Clock::now();
    }

    bool before(const MNN::OperatorInfo * info) {
        if (step_ > 0) {
            if (current_ != nullptr) {
                throw std::runtime_error("overlapping operator callbacks");
            }
            current_ = info;
            operator_start_ = Clock::now();
        }
        return true;
    }

    bool after(const MNN::OperatorInfo * info) {
        if (step_ > 0) {
            const double duration_us = elapsed_ms(operator_start_) * 1000;
            if (info != current_) {
                throw std::runtime_error("unmatched operator callbacks");
            }
            operators_.push_back({step_, info->name(), info->type(), duration_us});
            current_ = nullptr;
        }
        return true;
    }

    void end_forward() {
        const double duration_us = elapsed_ms(forward_start_) * 1000;
        if (current_ != nullptr) {
            throw std::runtime_error("unfinished operator callback");
        }
        forwards_.push_back({step_, duration_us});
        step_ = 0;
    }

    void save(const fs::path & directory) const {
        std::ofstream operators(directory / "decode-ops.jsonl");
        operators.exceptions(std::ios::badbit | std::ios::failbit);
        for (const auto & op : operators_) {
            operators << json{{"step", op.step}, {"name", op.name}, {"type", op.type},
                              {"duration_us", op.duration_us}}.dump() << '\n';
        }
        std::ofstream forwards(directory / "decode-forwards.jsonl");
        forwards.exceptions(std::ios::badbit | std::ios::failbit);
        for (const auto & forward : forwards_) {
            forwards << json{{"step", forward.step}, {"duration_us", forward.duration_us}}.dump() << '\n';
        }
    }

private:
    struct OperatorTiming {
        int step;
        std::string name;
        std::string type;
        double duration_us;
    };
    struct ForwardTiming {
        int step;
        double duration_us;
    };

    int step_ = 0;
    const MNN::OperatorInfo * current_ = nullptr;
    Clock::time_point operator_start_;
    Clock::time_point forward_start_;
    std::vector<OperatorTiming> operators_;
    std::vector<ForwardTiming> forwards_;
};

struct Request {
    std::vector<int> prefix;
    std::vector<int> suffix;
    std::vector<std::vector<int>> positions;
    int visual_rows;
    int hidden;
    int next_position;
    int max_tokens;
};

Request read_request(const fs::path & path) {
    std::ifstream file(path);
    const auto trace = json::parse(file);
    const auto & chunks = trace.at("chunks");
    if (chunks.size() != 3 || chunks[0].at("type") != "text" ||
            chunks[1].at("type") != "image" || chunks[2].at("type") != "text" ||
            !trace.at("mrope").get<bool>()) {
        throw std::runtime_error("expected an M-RoPE request with text, one image, then text");
    }
    Request request{
        chunks[0].at("token_ids").get<std::vector<int>>(),
        chunks[2].at("token_ids").get<std::vector<int>>(),
        {}, chunks[1].at("embedding_shape")[0].get<int>(),
        chunks[1].at("embedding_shape")[1].get<int>(),
        trace.at("next_position_after_prefill").get<int>(),
        trace.at("max_tokens").get<int>(),
    };
    const auto & visual_positions = chunks[1].at("thw_positions");
    const auto total = request.prefix.size() + visual_positions.size() + request.suffix.size();
    if (request.prefix.empty() || request.suffix.empty() || request.visual_rows <= 0 ||
            request.hidden != 1024 || visual_positions.size() != static_cast<size_t>(request.visual_rows) ||
            total != trace.at("prompt_tokens").get<size_t>() || request.max_tokens < 1 ||
            total + request.max_tokens > 4096) {
        throw std::runtime_error("invalid dimensions or token counts in request trace");
    }
    for (const size_t i : {0, 2}) {
        const int start = chunks[i].at("start_position").get<int>();
        const auto count = chunks[i].at("token_ids").size();
        for (size_t token = 0; token < count; ++token) {
            const int position = start + static_cast<int>(token);
            request.positions.push_back({position, position, position});
        }
        if (i == 0) {
            for (const auto & axes : visual_positions) {
                if (axes.size() != 3) {
                    throw std::runtime_error("each image row needs three M-RoPE coordinates");
                }
                request.positions.push_back(axes.get<std::vector<int>>());
            }
        }
    }
    return request;
}

std::vector<float> read_features(const fs::path & path, size_t size) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(size * sizeof(float));
    if (stream.tellg() != bytes) {
        throw std::runtime_error("unexpected feature file size: " + path.string());
    }
    stream.seekg(0);
    std::vector<float> features(size);
    if (!stream.read(reinterpret_cast<char *>(features.data()), bytes) ||
            !std::all_of(features.begin(), features.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("invalid FP32 features: " + path.string());
    }
    return features;
}

class FirstText : public std::streambuf {
public:
    explicit FirstText(Clock::time_point start) : start_(start) {}
    std::string text;
    double first_text_ms = -1;

protected:
    std::streamsize xsputn(const char * data, std::streamsize size) override {
        if (size > 0) {
            if (first_text_ms < 0) {
                first_text_ms = elapsed_ms(start_);
            }
            text.append(data, static_cast<size_t>(size));
        }
        return size;
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        const char value = traits_type::to_char_type(ch);
        xsputn(&value, 1);
        return ch;
    }

private:
    Clock::time_point start_;
};

class ImageFeatureReplay : public Llm {
public:
    ImageFeatureReplay(std::shared_ptr<LlmConfig> config, const Request & request)
        : Llm(config), request_(request) {
        if (!config->is_mrope() || config->is_visual() || config->hidden_size() != request.hidden) {
            throw std::runtime_error("expected the corresponding language-only M-RoPE model");
        }
    }

    void enable_decode_profile(DecodeProfile & profile) {
        profile_ = &profile;
    }

#ifdef EDGEDOC_CUSTOM_HEAD
    void enable_custom_head(edgedoc::Q4Gemv & head) {
        custom_head_ = &head;
        ExecutorScope scope(mExecutor);
        logits_ = _Input({1, 1, static_cast<int>(edgedoc::Q4Gemv::output_width)}, NCHW, halide_type_of<float>());
    }

    void save_head_timings(const fs::path & directory) const {
        std::ofstream file(directory / "custom-head.jsonl");
        file.exceptions(std::ios::badbit | std::ios::failbit);
        for (const auto & row : head_timings_) {
            file << json{{"step", row.step}, {"trunk_ms", row.trunk_ms}, {"head_total_ms", row.head_ms},
                         {"upload_ms", row.device.upload_ms}, {"kernel_ms", row.device.kernel_ms},
                         {"readback_ms", row.device.readback_ms}, {"head_wall_ms", row.device.wall_ms}}.dump() << '\n';
        }
    }

    double head_time_ms(bool prefill) const {
        double total = 0;
        for (const auto & row : head_timings_) {
            if ((row.step == 0) == prefill) total += row.head_ms;
        }
        return total;
    }

    size_t head_calls() const { return head_timings_.size(); }

    int sample(VARP logits, int offset = 0, int size = 0) override {
        if (custom_head_ == nullptr) return Llm::sample(logits, offset, size);
        const auto head_start = Clock::now();
        if (offset != 0 || size != 0 || !logits.get() || !logits->getInfo() ||
            logits->getInfo()->size != edgedoc::Q4Gemv::input_width) {
            throw std::runtime_error("custom head requires a single 1024-element trunk output");
        }
        ExecutorScope scope(mExecutor);
        const auto * input = logits->readMap<float>();
        auto * output = logits_->writeMap<float>();
        if (input == nullptr || output == nullptr ||
            !std::all_of(input, input + edgedoc::Q4Gemv::input_width, [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("invalid hidden vector for custom GEMV");
        }
        const auto timing = custom_head_->compute_hidden(input, output);
        if (!std::all_of(output, output + edgedoc::Q4Gemv::output_width, [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("custom head returned nonfinite logits");
        }
        head_timings_.push_back({mContext->gen_seq_len, last_trunk_ms_, elapsed_ms(head_start), timing});
        return Llm::sample(logits_, 0, 0);
    }
#endif

    std::vector<VARP> forwardRaw(VARP hidden_state, VARP mask, VARP input_pos,
                                 VARPS extra_args = {}) override {
#ifdef EDGEDOC_CUSTOM_HEAD
        if (custom_head_ != nullptr) {
            const auto trunk_start = Clock::now();
            auto hidden = Llm::forwardRaw(hidden_state, mask, input_pos, extra_args);
            last_trunk_ms_ = elapsed_ms(trunk_start);
            return hidden;
        }
#endif
        // The first graph execution (Prefill) has gen_seq_len == 0.
        if (profile_ == nullptr || mContext->gen_seq_len == 0) {
            return Llm::forwardRaw(hidden_state, mask, input_pos, extra_args);
        }
        profile_->begin_forward(mContext->gen_seq_len);
        auto output = Llm::forwardRaw(hidden_state, mask, input_pos, extra_args);
        profile_->end_forward();
        return output;
    }

    VARP gen_position_ids(int length) override {
        ExecutorScope scope(mExecutor);
        if (!positionIds.get() || positionIds->getInfo()->dim[1] != length) {
            positionIds = _Input({3, length}, NCHW, halide_type_of<int>());
        }
        auto * positions = positionIds->writeMap<int>();
        if (!positions) {
            throw std::runtime_error("could not allocate position tensor");
        }
        const int first_row = mContext->all_seq_len;
        const int prompt_rows = static_cast<int>(request_.positions.size());
        for (int i = 0; i < length; ++i) {
            const int row = first_row + i;
            for (int axis = 0; axis < 3; ++axis) {
                positions[axis * length + i] = row < prompt_rows
                    ? request_.positions[row][axis]
                    : request_.next_position + row - prompt_rows;
            }
        }
        return positionIds;
    }

    VARP prompt(const std::vector<float> & visual) {
        ExecutorScope scope(mExecutor);
        auto combined = _Input({static_cast<int>(request_.positions.size()), 1, request_.hidden}, NCHW);
        const auto before = embedding(request_.prefix);
        const auto after = embedding(request_.suffix);
        auto * destination = combined->writeMap<float>();
        const auto * prefix = before->readMap<float>();
        const auto * suffix = after->readMap<float>();
        if (!destination || !prefix || !suffix) {
            throw std::runtime_error("could not map prompt embeddings");
        }
        destination = std::copy_n(prefix, request_.prefix.size() * request_.hidden, destination);
        destination = std::copy(visual.begin(), visual.end(), destination);
        std::copy_n(suffix, request_.suffix.size() * request_.hidden, destination);
        return combined;
    }

private:
    const Request & request_;
    DecodeProfile * profile_ = nullptr;
#ifdef EDGEDOC_CUSTOM_HEAD
    struct HeadTiming {
        int step;
        double trunk_ms;
        double head_ms;
        edgedoc::GemvTiming device;
    };
    edgedoc::Q4Gemv * custom_head_ = nullptr;
    VARP logits_;
    double last_trunk_ms_ = 0;
    std::vector<HeadTiming> head_timings_;
#endif
};

}  // namespace

int main(int argc, char ** argv) {
    const bool profile_enabled = argc == 6 && std::string(argv[5]) == "--profile-decode";
#ifdef EDGEDOC_CUSTOM_HEAD
    const bool custom_enabled = argc == 7 && std::string(argv[5]) == "--custom-head";
#else
    const bool custom_enabled = false;
#endif
    if (argc != 5 && !profile_enabled && !custom_enabled) {
        std::cerr << "Usage: mnn_image_replay CONFIG REQUEST_JSON IMAGE_EMBEDDING OUTPUT_DIR [--profile-decode | --custom-head GEMV_DIR]\n";
        return 2;
    }
    try {
        const auto request = read_request(argv[2]);
        const auto visual = read_features(argv[3], request.visual_rows * request.hidden);
        const auto config = std::make_shared<LlmConfig>(argv[1]);
        if (profile_enabled && config->backend_type() != "cpu") {
            throw std::runtime_error("decode profiling requires the synchronous CPU backend");
        }
#ifdef EDGEDOC_CUSTOM_HEAD
        if (custom_enabled && config->backend_type() != "opencl") {
            throw std::runtime_error("custom head experiment requires an OpenCL trunk");
        }
        std::unique_ptr<edgedoc::Q4Gemv> custom_head;
        double head_load_ms = 0;
#endif
        DecodeProfile profile;
        const auto load_start = Clock::now();
        ImageFeatureReplay llm(config, request);
        if (profile_enabled) {
            if (!llm.set_config(R"({"enable_debug":true})")) {
                throw std::runtime_error("could not enable MNN operator callbacks");
            }
            llm.setDebugCallback(
                [&profile](const std::vector<MNN::Tensor *> &, const MNN::OperatorInfo * info) {
                    return profile.before(info);
                },
                [&profile](const std::vector<MNN::Tensor *> &, const MNN::OperatorInfo * info) {
                    return profile.after(info);
                });
        }
        if (!llm.load()) {
            throw std::runtime_error("MNN language model failed to load");
        }
#ifdef EDGEDOC_CUSTOM_HEAD
        if (custom_enabled) {
            const auto head_start = Clock::now();
            const fs::path directory(argv[6]);
            custom_head = std::make_unique<edgedoc::Q4Gemv>("tile-fp32", directory / "gemv.cl", fs::path{}, directory / "prepared", 64);
            llm.enable_custom_head(*custom_head);
            head_load_ms = elapsed_ms(head_start);
        }
#endif
        const double load_ms = elapsed_ms(load_start);
        if (profile_enabled) {
            llm.enable_decode_profile(profile);
        }
        const auto request_start = Clock::now();
        FirstText buffer(request_start);
        std::ostream output(&buffer);
        llm.generate_init(&output, "");
        llm.generate(llm.prompt(visual), request.max_tokens);
        const double request_ms = elapsed_ms(request_start);
        const auto * context = llm.getContext();
        if (buffer.text.empty() || (context->status != LlmStatus::NORMAL_FINISHED &&
                                   context->status != LlmStatus::MAX_TOKENS_FINISHED)) {
            throw std::runtime_error("MNN OCR did not produce a complete response, status="
                + std::to_string(static_cast<int>(context->status)));
        }
        const bool ended = context->status == LlmStatus::NORMAL_FINISHED;
        const int content_tokens = context->gen_seq_len - (ended ? 1 : 0);
        double prefill_ms = context->prefill_us / 1e3;
        double decode_ms = context->decode_us / 1e3;
#ifdef EDGEDOC_CUSTOM_HEAD
        if (custom_enabled) {
            prefill_ms += llm.head_time_ms(true);
            decode_ms += llm.head_time_ms(false);
        }
#endif
        const fs::path target(argv[4]);
        std::ofstream text(target / "page.md", std::ios::binary);
        text.exceptions(std::ios::badbit | std::ios::failbit);
        text << buffer.text;
        text.close();
        std::ofstream metrics(target / "metrics.json");
        metrics.exceptions(std::ios::badbit | std::ios::failbit);
        auto report = json{
            {"backend", config->backend_type()}, {"model_load_ms", load_ms},
            {"request_ms", request_ms}, {"first_text_ms", buffer.first_text_ms},
            {"prompt_tokens", context->prompt_len}, {"generated_tokens", content_tokens},
            {"prefill_ms", prefill_ms}, {"decode_ms", decode_ms},
            {"sample_ms", context->sample_us / 1e3},
            {"decode_tokens_per_sec", decode_ms > 0 ? content_tokens * 1e3 / decode_ms : 0},
            {"finish_reason", ended ? "eog" : "limit"},
            {"status", static_cast<int>(context->status)},
        };
#ifdef EDGEDOC_CUSTOM_HEAD
        if (custom_enabled) {
            report["custom_head_load_ms"] = head_load_ms;
            report["custom_head_calls"] = llm.head_calls();
            report["raw_mnn_prefill_ms"] = context->prefill_us / 1e3;
            report["raw_mnn_decode_ms"] = context->decode_us / 1e3;
            report["custom_head_prefill_ms"] = llm.head_time_ms(true);
            report["custom_head_decode_ms"] = llm.head_time_ms(false);
            llm.save_head_timings(target);
        }
#endif
        metrics << report.dump(2) << '\n';
        metrics.close();
        if (profile_enabled) {
            profile.save(target);
        }
        return ended ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
