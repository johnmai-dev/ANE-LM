#pragma once

#include "qwen3_5.h"
#include "../../core/ane_runtime.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ane_lm {

struct Qwen3Args {
    int hidden_size = 1024;
    int num_hidden_layers = 28;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    int head_dim = 128;
    int intermediate_size = 3072;
    int vocab_size = 151936;
    int max_position_embeddings = 40960;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = true;

    static Qwen3Args from_json(const nlohmann::json& config);
};

class Qwen3Model : public LLMModel {
public:
    ~Qwen3Model() override;
    bool load(const std::string& model_dir) override;
    float* forward(int token_id, int pos) override;
    void reset() override;
    int vocab_size() const override { return vocab_size_; }

private:
    // Config
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int vocab_size_ = 0;
    int num_layers_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int rot_dim_ = 0;
    int max_pos_ = 0;
    float rope_theta_ = 0;
    float rms_eps_ = 0;
    bool tie_word_embeddings_ = true;

    int q_proj_dim_ = 0;
    int kv_proj_dim_ = 0;
    int full_out_dim_ = 0;

    int rope_cache_len_ = 0;

    static constexpr int KV_CACHE_CAPACITY = 2048;
    static constexpr int LM_HEAD_ANE_CHUNK_MAX = 16384;

    struct LayerWeights {
        float* input_layernorm = nullptr;
        float* post_attention_layernorm = nullptr;
        float* q_norm = nullptr;
        float* k_norm = nullptr;
    };

    struct KVCache {
        float* k_cache = nullptr;
        float* v_cache = nullptr;
        int len = 0;
        int start = 0;
        int capacity = 0;
    };

    std::vector<LayerWeights> layers_;
    std::vector<KVCache> kv_caches_;
    std::vector<LayerANEKernels> ane_layers_;

    float* embed_tokens_ = nullptr;
    float* lm_head_ = nullptr;
    float* final_norm_ = nullptr;

    std::vector<ANEKernel*> lm_head_kernels_;
    int lm_head_chunk_ = LM_HEAD_ANE_CHUNK_MAX;
    bool ane_lm_head_enabled_ = false;

    float* x_ = nullptr;
    float* x_norm_ = nullptr;
    float* logits_ = nullptr;
    float* scratch_qkv_ = nullptr;
    float* scratch_attn_ = nullptr;
    float* rope_cos_ = nullptr;
    float* rope_sin_ = nullptr;

    void apply_args(const Qwen3Args& args);
    bool load_weights(ModelWeights* sf);
    bool compile_ane(ModelWeights* sf, const std::string& blob_dir);
    bool compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir);
    void free_lm_head_ane();

    bool forward_full_attn_core(int L, float* x, float* pre_oproj, int pos);
};

} // namespace ane_lm
