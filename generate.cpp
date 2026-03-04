#include "generate.h"
#include "core/sampling.h"
#include <ane_lm/common.h>
#include <climits>
#include <cstdint>
#include <algorithm>

namespace ane_lm {

static bool is_utf8_continuation(uint8_t b) {
    return (b & 0xC0u) == 0x80u;
}

static size_t longest_common_prefix_len(const std::string& a, const std::string& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

// Move cut position to a UTF-8 codepoint boundary at or before cut.
static size_t utf8_boundary_at_or_before(const std::string& s, size_t cut) {
    if (cut >= s.size()) return s.size();
    while (cut > 0 && is_utf8_continuation(static_cast<uint8_t>(s[cut]))) {
        cut--;
    }
    return cut;
}

void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& messages,
    int max_tokens,
    bool enable_thinking,
    const SamplingParams& sampling,
    std::function<void(const GenerationResponse&)> callback)
{
    // Tokenize with chat template
    std::vector<int> prompt_tokens;
    if (tokenizer.has_chat_template()) {
        std::string formatted = tokenizer.apply_chat_template(messages, true, enable_thinking);
        prompt_tokens = tokenizer.encode(formatted);
    } else {
        // Fallback: concatenate all message contents
        std::string combined;
        for (auto& [role, content] : messages) {
            combined += content + "\n";
        }
        prompt_tokens = tokenizer.encode(combined);
    }

    // Prefill
    Timer prefill_timer;
    float* logits = nullptr;
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        logits = model.forward(prompt_tokens[i], i);
        if (!logits) {
            fprintf(stderr, "Forward failed during prefill at token index %d\n", i);
            return;
        }
    }
    double prefill_ms = prefill_timer.elapsed_ms();
    double prompt_tps = prompt_tokens.size() / (prefill_ms / 1000.0);

    // Sample only over token ids supported by both model logits and tokenizer decode.
    int sampler_vocab = std::min(model.vocab_size(), tokenizer.vocab_size());
    if (sampler_vocab <= 0) {
        fprintf(stderr, "Invalid sampler vocab size: %d\n", sampler_vocab);
        return;
    }

    // Decode
    Timer gen_timer;
    int n_generated = 0;
    std::vector<int> generated_tokens;
    std::string emitted_text;
    std::string prev_decoded;
    bool has_prev_decoded = false;
    int next_token = sample_token(logits, sampler_vocab, sampling, generated_tokens);

    int limit = (max_tokens > 0) ? max_tokens : INT_MAX;
    for (int i = 0; i < limit; i++) {
        if (next_token == tokenizer.eos_id() || next_token == tokenizer.im_end_id()) {
            break;
        }

        n_generated++;
        generated_tokens.push_back(next_token);
        std::string current_decoded = tokenizer.decode(generated_tokens);

        std::string piece;
        if (has_prev_decoded) {
            size_t lcp = longest_common_prefix_len(prev_decoded, current_decoded);
            size_t stable_len = utf8_boundary_at_or_before(prev_decoded, lcp);
            std::string stable_decoded = prev_decoded.substr(0, stable_len);
            if (stable_decoded.size() >= emitted_text.size() &&
                stable_decoded.compare(0, emitted_text.size(), emitted_text) == 0) {
                piece = stable_decoded.substr(emitted_text.size());
                emitted_text = std::move(stable_decoded);
            } else {
                // Fallback: find current common prefix with emitted text first.
                size_t p = longest_common_prefix_len(stable_decoded, emitted_text);
                p = utf8_boundary_at_or_before(stable_decoded, p);
                piece = stable_decoded.substr(p);
                emitted_text = std::move(stable_decoded);
            }
        }
        prev_decoded = std::move(current_decoded);
        has_prev_decoded = true;

        if (callback) {
            GenerationResponse r;
            r.text = piece;
            r.token = next_token;
            r.prompt_tokens = (int)prompt_tokens.size();
            r.prompt_tps = prompt_tps;
            r.generation_tokens = n_generated;
            r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
            callback(r);
        }

        int pos = (int)prompt_tokens.size() + i;
        logits = model.forward(next_token, pos);
        if (!logits) {
            fprintf(stderr, "Forward failed during generation at step %d\n", i);
            return;
        }
        next_token = sample_token(logits, sampler_vocab, sampling, generated_tokens);
    }

    // Flush any remaining tail at end.
    if (callback && has_prev_decoded) {
        std::string final_decoded = prev_decoded;
        std::string tail;
        if (final_decoded.size() >= emitted_text.size() &&
            final_decoded.compare(0, emitted_text.size(), emitted_text) == 0) {
            tail = final_decoded.substr(emitted_text.size());
        } else {
            size_t p = 0;
            while (p < final_decoded.size() && p < emitted_text.size() &&
                   final_decoded[p] == emitted_text[p]) p++;
            tail = final_decoded.substr(p);
        }

        if (!tail.empty()) {
            GenerationResponse r;
            r.text = tail;
            r.token = generated_tokens.back();
            r.prompt_tokens = (int)prompt_tokens.size();
            r.prompt_tps = prompt_tps;
            r.generation_tokens = n_generated;
            r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
            callback(r);
        }
    }

    // Final stats callback (token = -1 signals end)
    if (callback) {
        GenerationResponse r;
        r.token = -1;
        r.prompt_tokens = (int)prompt_tokens.size();
        r.prompt_tps = prompt_tps;
        r.generation_tokens = n_generated;
        r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
        callback(r);
    }
}

// Single-prompt overload wraps into messages vector
void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::string& prompt,
    int max_tokens,
    bool enable_thinking,
    const SamplingParams& sampling,
    std::function<void(const GenerationResponse&)> callback)
{
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    stream_generate(model, tokenizer, messages, max_tokens, enable_thinking, sampling, std::move(callback));
}

} // namespace ane_lm
