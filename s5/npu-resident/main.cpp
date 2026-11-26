#include <stddef.h>

#include "image_enc.h"
#include "rkllm.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int kImageSize = 448;
constexpr int kMaxTokens = 512;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void write_file(const fs::path & path, const std::string & contents) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(contents.data(), contents.size());
    stream.close();
    if (!stream) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

struct Page {
    Clock::time_point started = Clock::now();
    std::string text;
    std::optional<double> first_text_ms;
    bool finished = false;
    bool failed = false;
};

int collect_text(RKLLMResult * result, void * userdata, LLMCallState state) {
    auto & page = *static_cast<Page *>(userdata);
    if (state == RKLLM_RUN_FINISH) {
        page.finished = true;
    } else if (state == RKLLM_RUN_ERROR) {
        page.failed = true;
    } else if (state == RKLLM_RUN_NORMAL && result && result->text && *result->text) {
        if (!page.first_text_ms) {
            page.first_text_ms = elapsed_ms(page.started);
        }
        try {
            page.text += result->text;
        } catch (...) {
            page.failed = true;
        }
    }
    return 0;
}

class Models {
public:
    Models() = default;
    Models(const Models &) = delete;
    Models & operator=(const Models &) = delete;

    ~Models() {
        if (vision_.rknn_ctx) {
            release_imgenc(&vision_);
        }
        if (language_) {
            rkllm_destroy(language_);
        }
    }

    void load(const char * vision_path, const char * language_path) {
        params_ = rkllm_createDefaultParam();
        params_.model_path = language_path;
        params_.top_k = 1;
        params_.max_new_tokens = kMaxTokens;
        params_.max_context_len = 4096;
        params_.skip_special_token = true;
        params_.is_async = false;
        params_.extend_param.base_domain_id = 1;
        callback_.result_callback = collect_text;
        if (rkllm_init(&language_, &params_, &callback_) != 0) {
            throw std::runtime_error("cannot initialize RKLLM");
        }
        if (init_imgenc(vision_path, &vision_, 3) != 0) {
            throw std::runtime_error("cannot initialize RKNN");
        }
        if (vision_.model_width != kImageSize || vision_.model_height != kImageSize ||
            vision_.model_channel != 3 || vision_.io_num.n_output != 1 ||
            vision_.model_image_token != 196 || vision_.model_embed_size != 1024) {
            throw std::runtime_error("expected the 448px Xiaomi vision model with 196x1024 output");
        }
        features_.resize(vision_.model_image_token * vision_.model_embed_size);
    }

    bool recognize(const char * image_path, const std::string & prompt, const fs::path & output) {
        Page page;
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty() || image.cols != kImageSize || image.rows != kImageSize) {
            write_file(output.string() + ".json", "{\"status\":\"invalid_image\"}\n");
            return false;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        const double read_rgb_ms = elapsed_ms(page.started);

        const auto vision_start = Clock::now();
        if (run_imgenc(&vision_, image.data, features_.data()) != 0) {
            throw std::runtime_error("RKNN image inference failed");
        }
        const double vision_ms = elapsed_ms(vision_start);

        RKLLMInput input{};
        std::string request_prompt = prompt;
        input.role = "user";
        input.input_type = RKLLM_INPUT_MULTIMODAL;
        input.multimodal_input.prompt = request_prompt.data();
        auto & visual = input.multimodal_input.image;
        visual.image_embed = features_.data();
        visual.n_image_tokens = vision_.model_image_token;
        visual.n_image = 1;
        visual.image_height = vision_.model_height;
        visual.image_width = vision_.model_width;
        visual.image_start = "<|vision_start|>";
        visual.image_end = "<|vision_end|>";
        visual.image_content = "<|image_pad|>";
        RKLLMInferParam inference{};
        inference.mode = RKLLM_INFER_GENERATE;
        inference.keep_history = 0;

        const auto language_start = Clock::now();
        const int rc = rkllm_run(language_, &input, &inference, &page);
        const double language_ms = elapsed_ms(language_start);
        const double request_ms = elapsed_ms(page.started);
        if (rc != 0 || page.failed || !page.finished || !page.first_text_ms) {
            throw std::runtime_error("RKLLM did not return a completed text response (code " +
                                     std::to_string(rc) + ")");
        }
        write_file(output.string() + ".md", page.text);
        write_file(output.string() + ".json",
            "{\"status\":\"ok\",\"read_rgb_ms\":" + std::to_string(read_rgb_ms) +
            ",\"vision_ms\":" + std::to_string(vision_ms) +
            ",\"first_text_ms\":" + std::to_string(*page.first_text_ms) +
            ",\"language_ms\":" + std::to_string(language_ms) +
            ",\"request_ms\":" + std::to_string(request_ms) + "}\n");
        return true;
    }

private:
    LLMHandle language_ = nullptr;
    RKLLMParam params_{};
    RKLLMCallback callback_{};
    rknn_app_context_t vision_{};
    std::vector<float> features_;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::cerr << "Usage: edgedoc_npu_resident <vision.rknn> <language.rkllm> <prompt.txt> <new_output_dir> <448px_image>...\n";
        return 2;
    }
    try {
        std::ifstream source(argv[3]);
        std::string prompt;
        std::string extra;
        if (!std::getline(source, prompt) || prompt.empty() || std::getline(source, extra)) {
            throw std::runtime_error("expected a nonempty single-line prompt");
        }
        prompt.insert(0, "<image>");
        const fs::path output(argv[4]);
        if (!fs::create_directory(output)) {
            throw std::runtime_error("output directory already exists");
        }
        Models models;
        const auto start = Clock::now();
        models.load(argv[1], argv[2]);
        write_file(output / "init.json", "{\"model_load_ms\":" + std::to_string(elapsed_ms(start)) + "}\n");

        int status = 0;
        for (int i = 5; i < argc; ++i) {
            const int page = i - 4;
            std::cout << "PAGE_BEGIN " << page << std::endl;
            if (!models.recognize(argv[i], prompt, output / ("page-" + std::to_string(page)))) {
                std::cerr << "invalid image " << argv[i] << '\n';
                status = 3;
            }
            std::cout << "PAGE_END " << page << std::endl;
        }
        return status;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
