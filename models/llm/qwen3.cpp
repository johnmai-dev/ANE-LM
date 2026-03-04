#include "qwen3.h"
#include "../../core/cpu_ops.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sys/stat.h>

namespace ane_lm {

using json = nlohmann::json;

static void apply_rope_qwen3(
    float* q, float* k,
    int n_q_heads, int n_kv_heads,
    int head_dim, int pos, float theta,
    const float* cos_row, const float* sin_row) {
    int half = head_dim / 2;
    for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
        float* v = (h < n_q_heads) ? q + (size_t)h * head_dim : k + (size_t)(h - n_q_heads) * head_dim;
        for (int i = 0; i < half; i++) {
            float cos_a, sin_a;
            if (cos_row && sin_row) {
                cos_a = cos_row[i];
                sin_a = sin_row[i];
            } else {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float angle = pos * freq;
                cos_a = cosf(angle);
                sin_a = sinf(angle);
            }
            float v0 = v[i];
            float v1 = v[i + half];
            v[i] = v0 * cos_a - v1 * sin_a;
            v[i + half] = v1 * cos_a + v0 * sin_a;
        }
    }
}

Qwen3Args Qwen3Args::from_json(const json& j) {
    Qwen3Args args;

    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    args.hidden_size = tc.value("hidden_size", args.hidden_size);
    args.num_hidden_layers = tc.value("num_hidden_layers", args.num_hidden_layers);
    args.num_attention_heads = tc.value("num_attention_heads", args.num_attention_heads);
    args.num_key_value_heads = tc.value("num_key_value_heads", args.num_key_value_heads);
    args.intermediate_size = tc.value("intermediate_size", args.intermediate_size);
    args.vocab_size = tc.value("vocab_size", args.vocab_size);
    args.max_position_embeddings = tc.value("max_position_embeddings", args.max_position_embeddings);
    args.rms_norm_eps = tc.value("rms_norm_eps", args.rms_norm_eps);

    args.head_dim = tc.value("head_dim", args.head_dim);
    if (!tc.contains("head_dim") && args.num_attention_heads > 0) {
        args.head_dim = args.hidden_size / args.num_attention_heads;
    }

    args.tie_word_embeddings = tc.value(
        "tie_word_embeddings",
        j.value("tie_word_embeddings", args.tie_word_embeddings));

    if (tc.contains("rope_parameters") && tc["rope_parameters"].is_object()) {
        const auto& rp = tc["rope_parameters"];
        args.rope_theta = rp.value("rope_theta", args.rope_theta);
    } else {
        args.rope_theta = tc.value("rope_theta", args.rope_theta);
    }

    return args;
}

Qwen3Model::~Qwen3Model() {
    free(embed_tokens_);
    free(final_norm_);
    if (!tie_word_embeddings_) {
        free(lm_head_);
    }

    free(x_);
    free(x_norm_);
    free(logits_);
    free(scratch_qkv_);
    free(scratch_attn_);
    free(rope_cos_);
    free(rope_sin_);

    for (auto& lw : layers_) {
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);
        free(lw.q_norm);
        free(lw.k_norm);
    }

    for (auto& kv : kv_caches_) {
        free(kv.k_cache);
        free(kv.v_cache);
    }

    for (auto& lk : ane_layers_) {
        ane_free_layer(&lk);
    }

    free_lm_head_ane();
}

void Qwen3Model::reset() {
    for (auto& kv : kv_caches_) {
        kv.len = 0;
        kv.start = 0;
        memset(kv.k_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
        memset(kv.v_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
    }
}

void Qwen3Model::apply_args(const Qwen3Args& args) {
    hidden_size_ = args.hidden_size;
    intermediate_size_ = args.intermediate_size;
    vocab_size_ = args.vocab_size;
    num_layers_ = args.num_hidden_layers;
    num_q_heads_ = args.num_attention_heads;
    num_kv_heads_ = args.num_key_value_heads;
    head_dim_ = args.head_dim;
    rot_dim_ = head_dim_;
    max_pos_ = args.max_position_embeddings;
    rope_theta_ = args.rope_theta;
    rms_eps_ = args.rms_norm_eps;
    tie_word_embeddings_ = args.tie_word_embeddings;

    q_proj_dim_ = num_q_heads_ * head_dim_;
    kv_proj_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = q_proj_dim_;
}

bool Qwen3Model::load(const std::string& model_dir) {
    // 1. Read config.json and parse args
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Qwen3Args args = Qwen3Args::from_json(j);
    apply_args(args);

    // 2. Open model weights (single-file or sharded)
    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    // Infer dims from safetensors
    const SFTensor* embed = sf->find("model.embed_tokens.weight");
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid model.embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find("model.layers.0.mlp.gate_proj.weight");
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_ = (int)embed->shape[1];
    vocab_size_ = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];
    if (head_dim_ <= 0 && num_q_heads_ > 0) {
        head_dim_ = hidden_size_ / num_q_heads_;
    }

    rot_dim_ = head_dim_;
    q_proj_dim_ = num_q_heads_ * head_dim_;
    kv_proj_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = q_proj_dim_;

    LOG("Model dims: hidden=%d intermediate=%d vocab=%d layers=%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_);

    // 3. Init ANE
    ane_init();

    x_ = (float*)calloc(hidden_size_, sizeof(float));
    x_norm_ = (float*)calloc(hidden_size_, sizeof(float));
    logits_ = (float*)calloc(vocab_size_, sizeof(float));
    scratch_qkv_ = (float*)calloc((size_t)q_proj_dim_ + 2 * kv_proj_dim_, sizeof(float));
    scratch_attn_ = (float*)calloc(std::max(full_out_dim_, hidden_size_), sizeof(float));

    int half_rot = rot_dim_ / 2;
    rope_cache_len_ = std::min(std::max(max_pos_, 1), 16384);
    rope_cos_ = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));
    rope_sin_ = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));

    // Precompute RoPE trig table for common context lengths.
    if (rope_cos_ && rope_sin_ && half_rot > 0) {
        std::vector<float> inv_freq((size_t)half_rot);
        for (int j = 0, i = 0; i < rot_dim_; i += 2, j++) {
            inv_freq[(size_t)j] = 1.0f / powf(rope_theta_, (float)i / (float)rot_dim_);
        }
        for (int pos = 0; pos < rope_cache_len_; pos++) {
            float* cos_row = rope_cos_ + (size_t)pos * half_rot;
            float* sin_row = rope_sin_ + (size_t)pos * half_rot;
            for (int j = 0; j < half_rot; j++) {
                float angle = pos * inv_freq[(size_t)j];
                cos_row[j] = cosf(angle);
                sin_row[j] = sinf(angle);
            }
        }
    }

    layers_.resize(num_layers_);
    kv_caches_.resize(num_layers_);
    ane_layers_.resize(num_layers_);

    for (int L = 0; L < num_layers_; L++) {
        auto& kv = kv_caches_[L];
        kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.len = 0;
        kv.start = 0;
        kv.capacity = KV_CACHE_CAPACITY;
    }

    // 4. Load weights + compile ANE kernels
    if (!load_weights(sf.get())) {
        return false;
    }

    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st_blob;
    bool has_blobs = (stat(blob_dir.c_str(), &st_blob) == 0 && S_ISDIR(st_blob.st_mode));
    if (has_blobs) {
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());
    }

    if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) {
        return false;
    }

    return true;
}

bool Qwen3Model::load_weights(ModelWeights* sf) {
    char name[256];

    embed_tokens_ = sf->load_bf16_to_f32(
        "model.embed_tokens.weight", (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    if (tie_word_embeddings_) {
        lm_head_ = embed_tokens_;
    } else {
        lm_head_ = sf->load_bf16_to_f32("lm_head.weight", (int64_t)vocab_size_ * hidden_size_);
        if (!lm_head_) return false;
    }

    final_norm_ = sf->load_bf16_to_f32("model.norm.weight", hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", L);
        lw.input_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", L);
        lw.post_attention_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", L);
        lw.q_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.q_norm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", L);
        lw.k_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.k_norm) return false;
    }

    LOG("All Qwen3 weights loaded successfully\n");
    return true;
}

// Convert tensor name to blob path: "a.b.c" -> "<dir>/a/b/c.bin"
static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++) {
        p += (*c == '.') ? '/' : *c;
    }
    p += ".bin";
    return p;
}

bool Qwen3Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available, cannot run\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling Qwen3 ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");

    char name[256], name2[256], name3[256];

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d...\r", L + 1, num_layers_);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.layers.%d.self_attn.k_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.layers.%d.self_attn.v_proj.weight", L);

        if (use_blobs) {
            ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                blob_path(blob_dir, name), q_proj_dim_,
                blob_path(blob_dir, name2), kv_proj_dim_,
                blob_path(blob_dir, name3), kv_proj_dim_, hidden_size_);
        } else {
            ane_layers_[L].first_proj = ane_compile_fused_3(
                sf->get_bf16_ptr(name), q_proj_dim_,
                sf->get_bf16_ptr(name2), kv_proj_dim_,
                sf->get_bf16_ptr(name3), kv_proj_dim_, hidden_size_);
        }

        if (!ane_layers_[L].first_proj) {
            fprintf(stderr, "ANE first_proj compile failed for layer %d\n", L);
            return false;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", L);
        if (use_blobs) {
            ane_layers_[L].o_proj = ane_compile_matmul_blob(blob_path(blob_dir, name), hidden_size_, full_out_dim_);
        } else {
            ane_layers_[L].o_proj = ane_compile_matmul(sf->get_bf16_ptr(name), hidden_size_, full_out_dim_);
        }
        if (!ane_layers_[L].o_proj) {
            fprintf(stderr, "ANE o_proj compile failed for layer %d\n", L);
            return false;
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.layers.%d.mlp.up_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.layers.%d.mlp.down_proj.weight", L);

        if (use_blobs) {
            ane_layers_[L].fused_ffn = ane_compile_fused_ffn_blob(
                blob_path(blob_dir, name), blob_path(blob_dir, name2),
                blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
        } else {
            ane_layers_[L].fused_ffn = ane_compile_fused_ffn(
                sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
        }
        if (!ane_layers_[L].fused_ffn) {
            fprintf(stderr, "ANE fused_ffn compile failed for layer %d\n", L);
            return false;
        }
    }

    int compiled = ane_compile_count();
    int cached = ane_cache_loads();
    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d)\n",
        compiled + cached, compiled, cached);

    if (!compile_lm_head_ane(sf, blob_dir)) {
        LOG("ANE LM head disabled, falling back to CPU\n");
    } else {
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());
    }

    return true;
}

bool Qwen3Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();
    const char* lm_name = tie_word_embeddings_ ? "model.embed_tokens.weight" : "lm_head.weight";

    const uint16_t* lm_bf16 = sf->get_bf16_ptr(lm_name);
    if (!lm_bf16) {
        fprintf(stderr, "ANE LM head: missing BF16 weights for %s\n", lm_name);
        return false;
    }

    int chunk = lm_head_chunk_;
    if (chunk > vocab_size_) chunk = vocab_size_;

    int chunks = (vocab_size_ + chunk - 1) / chunk;
    lm_head_kernels_.resize(chunks, nullptr);

    LOG("  LM head ANE: compiling %d chunks (chunk=%d)\n", chunks, chunk);
    for (int c = 0; c < chunks; c++) {
        int offset = c * chunk;
        int rows = vocab_size_ - offset;
        if (rows > chunk) rows = chunk;

        LOG("    LM head chunk %d/%d...\r", c + 1, chunks);

        // For blob mode we still use BF16 pointer here because lm_head is chunked dynamically.
        (void)use_blobs;
        const uint16_t* chunk_w = lm_bf16 + (int64_t)offset * hidden_size_;
        lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        if (!lm_head_kernels_[c]) {
            fprintf(stderr, "\nANE LM head: compile failed at chunk %d/%d\n", c + 1, chunks);
            free_lm_head_ane();
            return false;
        }
    }

    LOG("    LM head chunk %d/%d done          \n", chunks, chunks);
    ane_lm_head_enabled_ = true;
    lm_head_chunk_ = chunk;
    return true;
}

void Qwen3Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) {
        ane_free(k);
    }
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

bool Qwen3Model::forward_full_attn_core(int L, float* x, float* pre_oproj, int pos) {
    auto& lw = layers_[L];
    auto& cache = kv_caches_[L];

    float* qkv_buf = scratch_qkv_;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x,
                    hidden_size_, q_proj_dim_ + 2 * kv_proj_dim_)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d\n", L);
        return false;
    }

    float* q_raw = qkv_buf;
    float* k_raw = qkv_buf + q_proj_dim_;
    float* v_raw = qkv_buf + q_proj_dim_ + kv_proj_dim_;

    for (int h = 0; h < num_q_heads_; h++) {
        float* qh = q_raw + (size_t)h * head_dim_;
        rmsnorm(qh, qh, lw.q_norm, head_dim_, rms_eps_);
    }
    for (int h = 0; h < num_kv_heads_; h++) {
        float* kh = k_raw + (size_t)h * head_dim_;
        rmsnorm(kh, kh, lw.k_norm, head_dim_, rms_eps_);
    }

    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < rope_cache_len_ && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }

    apply_rope_qwen3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                     head_dim_, pos, rope_theta_, rope_cos_row, rope_sin_row);

    int slot;
    if (cache.len < cache.capacity) {
        slot = cache.start + cache.len;
        if (slot >= cache.capacity) slot -= cache.capacity;
        cache.len++;
    } else {
        slot = cache.start;
        cache.start++;
        if (cache.start >= cache.capacity) cache.start = 0;
    }

    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
    memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

    gqa_attention(pre_oproj, q_raw, cache.k_cache, cache.v_cache,
                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                  cache.start, cache.len, cache.capacity);

    return true;
}

float* Qwen3Model::forward(int token_id, int pos) {
    memcpy(x_, embed_tokens_ + (int64_t)token_id * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = scratch_attn_;

    for (int L = 0; L < num_layers_; L++) {
        rmsnorm(x_norm_, x_, layers_[L].input_layernorm, hidden_size_, rms_eps_);

        if (!forward_full_attn_core(L, x_norm_, pre_oproj, pos)) {
            return nullptr;
        }

        float* attn_out = x_norm_;
        if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, full_out_dim_, hidden_size_)) {
            fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
            return nullptr;
        }

        for (int i = 0; i < hidden_size_; i++) {
            x_[i] += attn_out[i];
        }

        rmsnorm(x_norm_, x_, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);

        float* mlp_out = scratch_attn_;
        if (!ane_matvec(ane_layers_[L].fused_ffn, mlp_out, x_norm_, hidden_size_, hidden_size_)) {
            fprintf(stderr, "ANE fused_ffn eval failed at layer %d\n", L);
            return nullptr;
        }

        for (int i = 0; i < hidden_size_; i++) {
            x_[i] += mlp_out[i];
        }
    }

    rmsnorm(x_, x_, final_norm_, hidden_size_, rms_eps_);

    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!ane_matvec(lm_head_kernels_[c], logits_ + offset, x_, hidden_size_, rows)) {
                fprintf(stderr, "ANE LM head eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
        }
        if (!ok) {
            free_lm_head_ane();
            matvec(logits_, lm_head_, x_, vocab_size_, hidden_size_);
        }
    } else {
        matvec(logits_, lm_head_, x_, vocab_size_, hidden_size_);
    }

    return logits_;
}

} // namespace ane_lm
