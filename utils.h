#pragma once

#include "models/llm/qwen3_5.h"
#include "models/llm/qwen3.h"
#include "core/tokenizer.h"
#include <memory>
#include <string>
#include <utility>

namespace ane_lm {

std::pair<std::unique_ptr<LLMModel>, Tokenizer> load(
    const std::string& model_dir, bool ane_cache = true);

} // namespace ane_lm
