#include "chat.h"
#include "common.h"
#include "llama-cpp.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "nlohmann/json.hpp"
#include "unicode.h"

#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr int kContextSize = 4096;
constexpr int kBatchSize = 256;
constexpr int kMicroBatchSize = 128;
constexpr int kThreads = 4;
constexpr llama_seq_id kSequence = 0;
constexpr const char * kUsage =
    "Usage: edgedoc_cli --model TEXT.gguf --mmproj VISION.gguf --prompt-file PROMPT.txt\n"
    "                   --output-dir NEW_DIR [--max-tokens N] --image PAGE [--image PAGE ...]\n"
    "                   [--image-embedding FP32_FILE (single image)]\n";

struct Options {
    std::string model, mmproj, prompt_file;
    std::string image_embedding;
    fs::path output_dir;
    std::vector<std::string> images;
    int max_tokens = 512;
};

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (i + 1 >= argc || std::string_view(argv[i + 1]).substr(0, 2) == "--") {
            throw std::runtime_error("missing value for " + std::string(flag));
        }
        const std::string_view value = argv[++i];
        if (flag == "--model") {
            options.model = value;
        } else if (flag == "--mmproj") {
            options.mmproj = value;
        } else if (flag == "--prompt-file") {
            options.prompt_file = value;
        } else if (flag == "--image") {
            options.images.emplace_back(value);
        } else if (flag == "--image-embedding") {
            if (!options.image_embedding.empty() || value.empty()) {
                throw std::runtime_error("--image-embedding expects one nonempty FP32 file path");
            }
            options.image_embedding = value;
        } else if (flag == "--output-dir") {
            options.output_dir = value;
        } else if (flag == "--max-tokens") {
            const char * end = value.data() + value.size();
            const auto result = std::from_chars(value.data(), end, options.max_tokens);
            if (result.ec != std::errc{} || result.ptr != end ||
                    options.max_tokens < 1 || options.max_tokens >= kContextSize) {
                throw std::runtime_error("--max-tokens expects an integer in [1, 4095]");
            }
        } else {
            throw std::runtime_error("unknown option " + std::string(flag));
        }
    }
    if (options.model.empty() || options.mmproj.empty() || options.prompt_file.empty() ||
            options.images.empty() || options.output_dir.empty()) {
        throw std::runtime_error(kUsage);
    }
    if (!options.image_embedding.empty() && options.images.size() != 1) {
        throw std::runtime_error("--image-embedding requires exactly one --image");
    }
    return options;
}

std::string read_prompt(const std::string & filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open prompt: " + filename);
    }
    std::string prompt{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const auto last = prompt.find_last_not_of(" \t\r\n");
    if (last == std::string::npos) {
        throw std::runtime_error("prompt is empty: " + filename);
    }
    prompt.resize(last + 1);
    return prompt;
}

std::vector<float> read_image_embedding(const std::string & path, size_t values) {
    const auto bytes = static_cast<std::streamsize>(values * sizeof(float));
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.tellg() != bytes) {
        throw std::runtime_error("missing image embedding or unexpected size, expected "
            + std::to_string(bytes) + " bytes: " + path);
    }
    file.seekg(0);
    std::vector<float> embedding(values);
    if (!file.read(reinterpret_cast<char *>(embedding.data()), bytes)) {
        throw std::runtime_error("cannot read image embedding: " + path);
    }
    for (float value : embedding) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("image embedding contains a nonfinite value: " + path);
        }
    }
    return embedding;
}

void write_file(const fs::path & path, const std::string & content) {
    std::ofstream file(path, std::ios::binary);
    file << content;
    file.close();
    if (!file) {
        throw std::runtime_error("cannot write: " + path.string());
    }
}

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

struct Result {
    std::string text;
    json trace;
    bool complete;
};

class OcrEngine {
public:
    explicit OcrEngine(const Options & options) {
        auto model_params = llama_model_default_params();
        ggml_backend_dev_t cpu_only[] = {nullptr};
        model_params.devices = cpu_only;
        model_params.n_gpu_layers = 0;
        model_.reset(llama_model_load_from_file(options.model.c_str(), model_params));
        if (!model_) {
            throw std::runtime_error("cannot load language model: " + options.model);
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = kContextSize;
        context_params.n_batch = kBatchSize;
        context_params.n_ubatch = kMicroBatchSize;
        context_params.n_threads = kThreads;
        context_params.n_threads_batch = kThreads;
        context_params.n_seq_max = 1;
        context_params.kv_unified = false;
        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_ || !llama_get_memory(context_.get())) {
            throw std::runtime_error("cannot initialize language context with request memory");
        }

        auto vision_params = mtmd_context_params_default();
        vision_params.use_gpu = false;
        vision_params.n_threads = kThreads;
        vision_.reset(mtmd_init_from_file(options.mmproj.c_str(), model_.get(), vision_params));
        if (!vision_ || !mtmd_support_vision(vision_.get())) {
            throw std::runtime_error("cannot load a compatible vision model: " + options.mmproj);
        }
        if (!llama_model_chat_template(model_.get(), nullptr)) {
            throw std::runtime_error("language model has no embedded chat template");
        }
        templates_ = common_chat_templates_init(model_.get(), "");
    }

    Result recognize(const std::string & image_path, const std::string & prompt,
                     int max_tokens, const std::string & image_embedding) {
        const auto started = std::chrono::steady_clock::now();
        const auto media = mtmd_helper_bitmap_init_from_file(
            vision_.get(), image_path.c_str(), false, mtmd_helper_init_opt_default());
        mtmd::bitmap_ptr bitmap(media.bitmap);
        mtmd_helper::video_ptr video(media.video_ctx);
        if (!bitmap || video || mtmd_bitmap_is_audio(bitmap.get())) {
            throw std::runtime_error("cannot decode a still image: " + image_path);
        }

        common_chat_msg message;
        message.role = "user";
        message.content_parts = {{"media_marker", mtmd_get_marker(vision_.get()), {}}, {"text", prompt, {}}};
        common_chat_templates_inputs chat;
        chat.messages.push_back(std::move(message));
        chat.enable_thinking = false;
        const auto rendered = common_chat_templates_apply(templates_.get(), chat).prompt.str();

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        const mtmd_input_text input{rendered.data(), rendered.size(), true, true};
        const mtmd_bitmap * images[] = {bitmap.get()};
        const int code = mtmd_tokenize(vision_.get(), chunks.ptr.get(), &input, images, 1);
        if (code != 0) {
            throw std::runtime_error("multimodal tokenization failed, code=" + std::to_string(code));
        }
        const size_t prompt_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
        if (prompt_tokens + static_cast<size_t>(max_tokens) > llama_n_ctx_seq(context_.get())) {
            throw std::runtime_error("input and requested output exceed context capacity");
        }

        llama_memory_t memory = llama_get_memory(context_.get());
        const llama_pos previous = llama_memory_seq_pos_max(memory, kSequence);
        llama_memory_clear(memory, true);
        const llama_pos cleared = llama_memory_seq_pos_max(memory, kSequence);
        if (cleared != -1) {
            throw std::runtime_error("request memory remained populated after reset");
        }
        json trace = {
            {"image", image_path}, {"prompt_tokens", prompt_tokens}, {"max_tokens", max_tokens},
            {"input_rgb_shape", {mtmd_bitmap_get_ny(bitmap.get()), mtmd_bitmap_get_nx(bitmap.get()), 3}},
            {"memory_pos_max_before_reset", previous}, {"memory_pos_max_after_reset", cleared},
            {"mrope", mtmd_decode_use_mrope(vision_.get())}, {"chunks", json::array()},
        };

        const llama_pos next_position = prefill(chunks.ptr.get(), trace["chunks"], image_embedding);
        trace["next_position_after_prefill"] = next_position;
        Result result = generate(next_position, max_tokens, std::move(trace));
        result.trace["request_ms"] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }

private:
    llama_pos prefill(const mtmd_input_chunks * chunks, json & trace, const std::string & image_embedding) {
        const size_t count = mtmd_input_chunks_size(chunks);
        if (count == 0 || mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, count - 1)) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            throw std::runtime_error("expected a final text chunk with generation prompt");
        }
        if (!image_embedding.empty()) {
            size_t image_chunks = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto * chunk = mtmd_input_chunks_get(chunks, i);
                image_chunks += mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
            }
            if (image_chunks != 1) {
                throw std::runtime_error("--image-embedding requires exactly one image chunk");
            }
        }

        llama_pos next = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks, i);
            const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
            json record = {{"tokens", n}, {"start_position", next}, {"position_span", mtmd_input_chunk_get_n_pos(chunk)}};
            int code;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                record["type"] = "text";
                if (!image_embedding.empty()) {
                    size_t token_count = 0;
                    const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &token_count);
                    if (!ids || token_count != n) {
                        throw std::runtime_error("unexpected text chunk tokens");
                    }
                    record["token_ids"] = std::vector<llama_token>(ids, ids + token_count);
                }
                code = mtmd_helper_eval_chunk_single(vision_.get(), context_.get(), chunk,
                    next, kSequence, kBatchSize, i + 1 == count, &next);
            } else {
                code = prefill_image(chunk, next, record, image_embedding);
            }
            if (code != 0) {
                throw std::runtime_error("prefill failed at chunk " + std::to_string(i) + ", code=" + std::to_string(code));
            }
            record["next_position"] = next;
            trace.push_back(std::move(record));
        }
        return next;
    }

    int prefill_image(const mtmd_input_chunk * chunk, llama_pos & next, json & trace,
                      const std::string & image_embedding) {
        const auto * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
        if (!image_tokens || count == 0) {
            throw std::runtime_error("expected a nonempty image chunk");
        }
        const size_t width = llama_model_n_embd_inp(model_.get());
        std::vector<float> external_features;
        float * features = nullptr;
        if (image_embedding.empty()) {
            const int code = mtmd_encode_chunk(vision_.get(), chunk);
            if (code != 0) {
                return code;
            }
            features = mtmd_get_output_embd(vision_.get());
        } else {
            external_features = read_image_embedding(image_embedding, count * width);
            features = external_features.data();
            trace["embedding_file"] = image_embedding;
        }
        if (!features) {
            throw std::runtime_error("vision encoder produced no features");
        }
        const auto first = mtmd_image_tokens_get_decoder_pos(image_tokens, next, 0);
        const auto last = mtmd_image_tokens_get_decoder_pos(image_tokens, next, count - 1);
        trace["type"] = "image";
        trace["embedding_shape"] = {count, width};
        trace["embedding_dtype"] = "float32";
        trace["first_txyz"] = {first.t, first.x, first.y, first.z};
        trace["last_txyz"] = {last.t, last.x, last.y, last.z};
        if (!image_embedding.empty() && mtmd_decode_use_mrope(vision_.get())) {
            trace["thw_positions"] = json::array();
            for (size_t i = 0; i < count; ++i) {
                const auto pos = mtmd_image_tokens_get_decoder_pos(image_tokens, next, i);
                trace["thw_positions"].push_back({pos.t, pos.y, pos.x});
            }
        }
        return mtmd_helper_decode_image_chunk(vision_.get(), context_.get(), chunk,
            features, next, kSequence, kBatchSize, &next, nullptr, nullptr);
    }

    Result generate(llama_pos next, int max_tokens, json trace) {
        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        llama_sampler_ptr sampler(llama_sampler_init_greedy());
        llama_batch_ext_ptr batch(llama_batch_ext_init(context_.get()));
        std::string text;
        int generated = 0;
        bool complete = false;
        while (generated < max_tokens) {
            const llama_token token = llama_sampler_sample(sampler.get(), context_.get(), -1);
            if (llama_vocab_is_eog(vocab, token)) {
                complete = true;
                break;
            }
            text += common_token_to_piece(vocab, token);
            if (++generated == max_tokens) {
                break;
            }
            llama_batch_ext_clear(batch.get());
            const int32_t index = llama_batch_ext_add_token(batch.get(), kSequence, token);
            if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &next) ||
                    !llama_batch_ext_set_output_logits(batch.get(), index, true)) {
                throw std::runtime_error("cannot prepare next token batch");
            }
            const int code = llama_process(context_.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (code != 0) {
                throw std::runtime_error("decode failed, code=" + std::to_string(code));
            }
            ++next;
        }
        trace["generated_content_tokens"] = generated;
        trace["finish_reason"] = complete ? "eog" : "max_tokens";
        if (complete && !common_utf8_is_complete(text)) {
            throw std::runtime_error("model ended with an incomplete UTF-8 character");
        }
        const size_t bytes_before_trim = text.size();
        while (!common_utf8_is_complete(text)) {
            text.pop_back();
        }
        trace["incomplete_utf8_tail_bytes"] = bytes_before_trim - text.size();
        return {std::move(text), std::move(trace), complete};
    }

    llama_model_ptr model_;
    llama_context_ptr context_;
    mtmd::context_ptr vision_;
    common_chat_templates_ptr templates_;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << kUsage;
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        const std::string prompt = read_prompt(options.prompt_file);
        if (!fs::create_directories(options.output_dir)) {
            throw std::runtime_error("output directory already exists: " + options.output_dir.string());
        }
        Backend backend;
        OcrEngine engine(options);
        write_file(options.output_dir / "run.json", json{
            {"runtime_commit", EDGEDOC_LLAMA_COMMIT}, {"backend", "CPU"},
            {"model", options.model}, {"mmproj", options.mmproj}, {"prompt", prompt},
            {"context", kContextSize}, {"batch", kBatchSize}, {"micro_batch", kMicroBatchSize},
            {"threads", kThreads}, {"sampler", "greedy"}, {"images", options.images},
        }.dump(2) + '\n');
        int status = 0;
        for (size_t i = 0; i < options.images.size(); ++i) {
            const auto result = engine.recognize(options.images[i], prompt, options.max_tokens, options.image_embedding);
            const std::string name = "page-" + std::to_string(i + 1);
            write_file(options.output_dir / (name + ".md"), result.text);
            write_file(options.output_dir / (name + ".json"), result.trace.dump(2) + '\n');
            std::cout << name << ' ' << result.trace["finish_reason"].get<std::string>()
                      << ' ' << options.images[i] << '\n';
            if (!result.complete) {
                status = 2;
            }
        }
        return status;
    } catch (const std::exception & error) {
        std::cerr << "edgedoc_cli: " << error.what() << '\n';
        return 1;
    }
}
