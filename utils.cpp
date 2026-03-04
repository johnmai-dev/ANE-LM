#include "utils.h"
#include "core/ane_runtime.h"
#include <ane_lm/common.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace ane_lm {

using json = nlohmann::json;

std::pair<std::unique_ptr<LLMModel>, Tokenizer> load(
    const std::string& model_dir, bool ane_cache)
{
    Timer timer;

    // Read model_type from config.json
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open " + config_path);
    }
    json j = json::parse(f);
    std::string model_type = j.value("model_type", "");
    LOG("Config loaded: model_type=%s (%.1f ms)\n", model_type.c_str(), timer.elapsed_ms());

    // Dispatch by model_type
    std::unique_ptr<LLMModel> model;
    if (model_type == "qwen3_5") {
        model = std::make_unique<Qwen35Model>();
    } else if (model_type == "qwen3") {
        model = std::make_unique<Qwen3Model>();
    } else {
        throw std::runtime_error("Unsupported model_type: " + model_type);
    }

    // Set ANE cache preference before model loads
    ane_set_persist_cache(ane_cache);

    // Model self-loads (config, safetensors, ANE)
    timer.reset();
    if (!model->load(model_dir)) {
        throw std::runtime_error("Failed to load model from " + model_dir);
    }
    LOG("Model init: %.1f ms\n", timer.elapsed_ms());

    // Initialize tokenizer
    timer.reset();
    Tokenizer tokenizer;
    if (!tokenizer.init(model_dir)) {
        throw std::runtime_error("Failed to initialize tokenizer from " + model_dir);
    }
    LOG("Tokenizer init: %.1f ms\n", timer.elapsed_ms());

    return {std::move(model), std::move(tokenizer)};
}

} // namespace ane_lm
