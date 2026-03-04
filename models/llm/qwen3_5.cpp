#include "qwen3_5.h"
#include "../../core/cpu_ops.h"
#include <cmath>
#include <fstream>
#include <sys/stat.h>

namespace ane_lm {

using json = nlohmann::json;

// --- Qwen35Args::from_json ---

Qwen35Args Qwen35Args::from_json(const json& j) {
    Qwen35Args args;

    // Parse text_config if present, otherwise read from top level
    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    args.hidden_size = tc.value("hidden_size", args.hidden_size);
    args.num_hidden_layers = tc.value("num_hidden_layers", args.num_hidden_layers);
    args.num_attention_heads = tc.value("num_attention_heads", args.num_attention_heads);
    args.num_key_value_heads = tc.value("num_key_value_heads", args.num_key_value_heads);
    args.head_dim = tc.value("head_dim", args.head_dim);
    args.intermediate_size = tc.value("intermediate_size", args.intermediate_size);
    args.vocab_size = tc.value("vocab_size", args.vocab_size);
    args.full_attention_interval = tc.value("full_attention_interval", args.full_attention_interval);
    args.rms_norm_eps = tc.value("rms_norm_eps", args.rms_norm_eps);
    args.tie_word_embeddings = tc.value("tie_word_embeddings", j.value("tie_word_embeddings", args.tie_word_embeddings));
    args.attn_output_gate = tc.value("attn_output_gate", args.attn_output_gate);
    args.linear_num_key_heads = tc.value("linear_num_key_heads", args.linear_num_key_heads);
    args.linear_key_head_dim = tc.value("linear_key_head_dim", args.linear_key_head_dim);
    args.linear_value_head_dim = tc.value("linear_value_head_dim", args.linear_value_head_dim);
    args.linear_num_value_heads = tc.value("linear_num_value_heads", args.linear_num_value_heads);
    args.linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", args.linear_conv_kernel_dim);

    // RoPE parameters
    if (tc.contains("rope_parameters")) {
        auto& rp = tc["rope_parameters"];
        args.rope_theta = rp.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    } else {
        args.rope_theta = tc.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    }

    // Layer types
    if (tc.contains("layer_types")) {
        for (auto& lt : tc["layer_types"]) {
            std::string s = lt.get<std::string>();
            if (s == "linear_attention") {
                args.layer_types.push_back(LayerType::LinearAttention);
            } else {
                args.layer_types.push_back(LayerType::FullAttention);
            }
        }
    } else {
        for (int i = 0; i < args.num_hidden_layers; i++) {
            if ((i + 1) % args.full_attention_interval == 0) {
                args.layer_types.push_back(LayerType::FullAttention);
            } else {
                args.layer_types.push_back(LayerType::LinearAttention);
            }
        }
    }

    return args;
}

// --- Qwen35Model ---

Qwen35Model::~Qwen35Model() {
    free(embed_tokens_);
    free(final_norm_);
    free(x_);
    free(x_norm_);
    free(logits_);
    free(scratch_qkv_);
    free(scratch_conv_);
    free(scratch_y_);
    free(scratch_attn_);
    free(scratch_tmp_);
    free(rope_cos_);
    free(rope_sin_);

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);

        if (lw.type == LayerType::LinearAttention) {
            free(lw.deltanet.in_proj_a);
            free(lw.deltanet.in_proj_b);
            free(lw.deltanet.conv1d_w);
            free(lw.deltanet.A);
            free(lw.deltanet.dt_bias);
            free(lw.deltanet.norm_w);
        } else {
            free(lw.full_attn.q_norm);
            free(lw.full_attn.k_norm);
        }
    }

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            free(kv_caches_[L].k_cache);
            free(kv_caches_[L].v_cache);
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            free(delta_states_[L].ssm_state);
            free(delta_states_[L].conv_state);
        }
        ane_free_layer(&ane_layers_[L]);
    }

    free_lm_head_ane();
}

void Qwen35Model::reset() {
    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            kv_caches_[L].len = 0;
            kv_caches_[L].start = 0;
            memset(kv_caches_[L].k_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
            memset(kv_caches_[L].v_cache, 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            memset(delta_states_[L].ssm_state, 0, (size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_ * sizeof(float));
            memset(delta_states_[L].conv_state, 0, (size_t)lin_qkv_dim_ * (conv_kernel_ - 1) * sizeof(float));
            delta_states_[L].conv_pos = 0;
        }
    }
}

void Qwen35Model::apply_args(const Qwen35Args& args) {
    hidden_size_ = args.hidden_size;
    intermediate_size_ = args.intermediate_size;
    vocab_size_ = args.vocab_size;
    num_layers_ = args.num_hidden_layers;
    num_q_heads_ = args.num_attention_heads;
    num_kv_heads_ = args.num_key_value_heads;
    head_dim_ = args.head_dim;
    rot_dim_ = args.rotation_dim();
    rope_theta_ = args.rope_theta;
    rms_eps_ = args.rms_norm_eps;
    lin_num_heads_ = args.linear_num_key_heads;
    lin_num_val_heads_ = args.linear_num_value_heads;
    lin_key_dim_ = args.linear_key_head_dim;
    lin_val_dim_ = args.linear_value_head_dim;
    lin_total_key_ = lin_num_heads_ * lin_key_dim_;
    lin_total_val_ = lin_num_val_heads_ * lin_val_dim_;
    lin_qkv_dim_ = lin_total_key_ * 2 + lin_total_val_;
    conv_kernel_ = args.linear_conv_kernel_dim;
    full_q_dim_ = num_q_heads_ * head_dim_ * 2;
    full_kv_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = num_q_heads_ * head_dim_;
    attn_output_gate_ = args.attn_output_gate;
    layer_types_ = args.layer_types;
}

bool Qwen35Model::load(const std::string& model_dir) {
    // 1. Read config.json and parse args
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Qwen35Args args = Qwen35Args::from_json(j);
    apply_args(args);

    // 2. Open model weights (single-file or sharded)
    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    // Infer dims from safetensors
    const SFTensor* embed = sf->find("model.language_model.embed_tokens.weight");
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find("model.language_model.layers.0.mlp.gate_proj.weight");
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_ = (int)embed->shape[1];
    vocab_size_ = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];

    LOG("Model dims: hidden=%d intermediate=%d vocab=%d layers=%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_);

    // 3. Init ANE
    ane_init();

    // Allocate scratch buffers
    x_ = (float*)calloc(hidden_size_, sizeof(float));
    x_norm_ = (float*)calloc(hidden_size_, sizeof(float));
    logits_ = (float*)calloc(vocab_size_, sizeof(float));
    scratch_qkv_ = (float*)calloc(lin_qkv_dim_ + lin_total_val_, sizeof(float));
    scratch_conv_ = (float*)calloc(lin_qkv_dim_, sizeof(float));
    scratch_y_ = (float*)calloc(lin_total_val_, sizeof(float));
    scratch_attn_ = (float*)calloc(full_out_dim_, sizeof(float));
    // scratch_tmp_: a_vec (lin_num_val_heads_) + b_vec (lin_num_val_heads_) + silu_tmp (lin_qkv_dim_)
    scratch_tmp_ = (float*)calloc((size_t)lin_num_val_heads_ * 2 + lin_qkv_dim_, sizeof(float));
    rope_cos_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));
    rope_sin_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));

    // Precompute RoPE trig table
    if (rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        float inv_freq[half_rot];
        for (int j2 = 0, i = 0; i < rot_dim_; i += 2, j2++) {
            inv_freq[j2] = 1.0f / powf(rope_theta_, (float)i / (float)rot_dim_);
        }
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
            float* cos_row = rope_cos_ + (size_t)pos * half_rot;
            float* sin_row = rope_sin_ + (size_t)pos * half_rot;
            for (int j2 = 0; j2 < half_rot; j2++) {
                float angle = pos * inv_freq[j2];
                cos_row[j2] = cosf(angle);
                sin_row[j2] = sinf(angle);
            }
        }
    }

    // Initialize layers
    layers_.resize(num_layers_);
    delta_states_.resize(num_layers_);
    kv_caches_.resize(num_layers_);
    ane_layers_.resize(num_layers_);

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            auto& kv = kv_caches_[L];
            kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
            kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
            kv.len = 0;
            kv.start = 0;
            kv.capacity = KV_CACHE_CAPACITY;
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            auto& ds = delta_states_[L];
            ds.ssm_state = (float*)calloc((size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_, sizeof(float));
            ds.conv_state = (float*)calloc((size_t)lin_qkv_dim_ * (conv_kernel_ - 1), sizeof(float));
            ds.conv_pos = 0;
        }
    }

    // 4. Load weights + compile ANE kernels
    if (!load_weights(sf.get())) { return false; }
    // Detect pre-converted ANE blob directory
    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st_blob;
    bool has_blobs = (stat(blob_dir.c_str(), &st_blob) == 0 && S_ISDIR(st_blob.st_mode));
    if (has_blobs) {
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());
    }

    if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) { return false; }

    return true;
}

bool Qwen35Model::load_weights(ModelWeights* sf) {
    char name[256];

    embed_tokens_ = sf->load_bf16_to_f32("model.language_model.embed_tokens.weight",
                                           (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    final_norm_ = sf->load_norm_weight("model.language_model.norm.weight", hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        lw.type = layer_types_[L];

        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", L);
        lw.input_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", L);
        lw.post_attention_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        if (lw.type == LayerType::LinearAttention) {
            auto& dw = lw.deltanet;

            // Note: in_proj_a/b, A_log, dt_bias use linear_num_value_heads (not key_heads)
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L);
            dw.in_proj_a = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L);
            dw.in_proj_b = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", L);
            dw.conv1d_w = sf->load_bf16_to_f32(name, (int64_t)lin_qkv_dim_ * conv_kernel_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", L);
            dw.A = sf->load_f32_direct(name, lin_num_val_heads_);
            if (dw.A) {
                for (int i = 0; i < lin_num_val_heads_; i++) dw.A[i] = expf(dw.A[i]);
            }

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", L);
            dw.dt_bias = sf->load_bf16_to_f32(name, lin_num_val_heads_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", L);
            dw.norm_w = sf->load_f32_direct(name, lin_val_dim_);

            if (!dw.in_proj_a || !dw.in_proj_b || !dw.conv1d_w ||
                !dw.A || !dw.dt_bias || !dw.norm_w) {
                fprintf(stderr, "Failed to load DeltaNet weights for layer %d\n", L);
                return false;
            }
        } else {
            auto& fw = lw.full_attn;

            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", L);
            fw.q_norm = sf->load_norm_weight(name, head_dim_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", L);
            fw.k_norm = sf->load_norm_weight(name, head_dim_);

            if (!fw.q_norm || !fw.k_norm) {
                fprintf(stderr, "Failed to load FullAttn weights for layer %d\n", L);
                return false;
            }
        }
    }

    LOG("All weights loaded successfully\n");
    return true;
}

// Convert tensor name to blob path: "a.b.c" → "<dir>/a/b/c.bin"
static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++) {
        p += (*c == '.') ? '/' : *c;
    }
    p += ".bin";
    return p;
}

bool Qwen35Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available, cannot run\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");
    char name[256], name2[256], name3[256];

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d (%s)...\r", L+1, num_layers_,
            layer_types_[L] == LayerType::LinearAttention ? "deltanet" : "full_attn");

        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", L);
            snprintf(name2, sizeof(name2), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_2_blob(
                    blob_path(blob_dir, name), lin_qkv_dim_,
                    blob_path(blob_dir, name2), lin_total_val_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_2(
                    sf->get_bf16_ptr(name), lin_qkv_dim_,
                    sf->get_bf16_ptr(name2), lin_total_val_, hidden_size_);
            }
        } else {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", L);
            snprintf(name2, sizeof(name2), "model.language_model.layers.%d.self_attn.k_proj.weight", L);
            snprintf(name3, sizeof(name3), "model.language_model.layers.%d.self_attn.v_proj.weight", L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                    blob_path(blob_dir, name), full_q_dim_,
                    blob_path(blob_dir, name2), full_kv_dim_,
                    blob_path(blob_dir, name3), full_kv_dim_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_3(
                    sf->get_bf16_ptr(name), full_q_dim_,
                    sf->get_bf16_ptr(name2), full_kv_dim_,
                    sf->get_bf16_ptr(name3), full_kv_dim_, hidden_size_);
            }
        }

        if (!ane_layers_[L].first_proj) {
            fprintf(stderr, "ANE first_proj compile failed for layer %d\n", L);
            return false;
        }

        // O projection
        int attn_dim;
        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
            attn_dim = lin_total_val_;
        } else {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", L);
            attn_dim = full_out_dim_;
        }
        if (use_blobs) {
            ane_layers_[L].o_proj = ane_compile_matmul_blob(blob_path(blob_dir, name), hidden_size_, attn_dim);
        } else {
            ane_layers_[L].o_proj = ane_compile_matmul(sf->get_bf16_ptr(name), hidden_size_, attn_dim);
        }
        if (!ane_layers_[L].o_proj) {
            fprintf(stderr, "ANE o_proj compile failed for layer %d\n", L);
            return false;
        }

        // Fused FFN
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.language_model.layers.%d.mlp.up_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.language_model.layers.%d.mlp.down_proj.weight", L);

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

    // Compile LM head
    if (!compile_lm_head_ane(sf, blob_dir)) {
        LOG("ANE LM head disabled, falling back to CPU\n");
    } else {
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());
    }

    return true;
}

bool Qwen35Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();

    // For blob mode, we need the embed blob; for bf16 mode, the bf16 pointer
    const uint16_t* embed_bf16 = nullptr;
    if (!use_blobs) {
        embed_bf16 = sf->get_bf16_ptr("model.language_model.embed_tokens.weight");
        if (!embed_bf16) {
            fprintf(stderr, "ANE LM head: missing embed_tokens BF16 weights\n");
            return false;
        }
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

        if (use_blobs) {
            // LM head reuses embed_tokens weight, chunked by row offset
            // Blob was written as one file; we need per-chunk blobs or fall back to bf16
            // For now: fall back to bf16 for LM head since embed_tokens is one big blob
            embed_bf16 = sf->get_bf16_ptr("model.language_model.embed_tokens.weight");
            if (!embed_bf16) return false;
            const uint16_t* chunk_w = embed_bf16 + (int64_t)offset * hidden_size_;
            lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        } else {
            const uint16_t* chunk_w = embed_bf16 + (int64_t)offset * hidden_size_;
            lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        }
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

void Qwen35Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) ane_free(k);
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

bool Qwen35Model::forward_deltanet_core(int L, float* x, float* pre_oproj) {
    auto& dw = layers_[L].deltanet;
    auto& st = delta_states_[L];

    float* qkv_z = scratch_qkv_;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_z, x,
                    hidden_size_, lin_qkv_dim_ + lin_total_val_)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d (DeltaNet)\n", L);
        return false;
    }

    float* mixed_qkv = qkv_z;
    float* z = qkv_z + lin_qkv_dim_;

    // Small projections on CPU
    // Note: in_proj_a/b output dim is lin_num_val_heads_
    float* a_vec = scratch_tmp_;
    float* b_vec = scratch_tmp_ + lin_num_val_heads_;
    matvec(a_vec, dw.in_proj_a, x, lin_num_val_heads_, hidden_size_);
    matvec(b_vec, dw.in_proj_b, x, lin_num_val_heads_, hidden_size_);

    // Causal conv1d + SiLU
    float* conv_out = scratch_conv_;
    conv1d_update(conv_out, st.conv_state, &st.conv_pos, mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
    silu_vec_inplace(conv_out, lin_qkv_dim_, scratch_tmp_ + lin_num_val_heads_ * 2);

    // Split into Q, K, V
    // Q and K have lin_num_heads_ heads, V has lin_num_val_heads_ heads
    // Each key head corresponds to (lin_num_val_heads_ / lin_num_heads_) value heads
    float* Q = conv_out;
    float* K = conv_out + lin_total_key_;
    float* V = conv_out + lin_total_key_ * 2;

    // Per-head SSM
    // Architecture: 16 key heads, 32 value heads
    // Each key head pairs with 2 value heads (val_heads_per_key = 2)
    float* y = scratch_y_;
    float q_scale = 1.0f / sqrtf((float)lin_key_dim_);
    int val_heads_per_key = lin_num_val_heads_ / lin_num_heads_;

    for (int kh = 0; kh < lin_num_heads_; kh++) {
        float* qh = Q + kh * lin_key_dim_;
        float* kh_ptr = K + kh * lin_key_dim_;

        l2_normalize(qh, lin_key_dim_);
        l2_normalize(kh_ptr, lin_key_dim_);
        float qs = q_scale;
        vDSP_vsmul(qh, 1, &qs, qh, 1, (vDSP_Length)lin_key_dim_);

        for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
            int vh = kh * val_heads_per_key + vsub;
            float* vh_ptr = V + vh * lin_val_dim_;
            float* yh = y + vh * lin_val_dim_;
            float* state = st.ssm_state + vh * lin_key_dim_ * lin_val_dim_;

            float beta = sigmoid_f(b_vec[vh]);
            float decay = expf(-dw.A[vh] * softplus_f(a_vec[vh] + dw.dt_bias[vh]));
            ssm_step(yh, state, qh, kh_ptr, vh_ptr, decay, beta, lin_key_dim_, lin_val_dim_);
        }
    }

    // RMSNorm gated
    for (int h = 0; h < lin_num_val_heads_; h++) {
        rmsnorm_gated(pre_oproj + h * lin_val_dim_,
                      y + h * lin_val_dim_,
                      z + h * lin_val_dim_,
                      dw.norm_w, lin_val_dim_);
    }
    return true;
}

bool Qwen35Model::forward_full_attn_core(int L, float* x, float* pre_oproj, int pos) {
    auto& fw = layers_[L].full_attn;
    auto& cache = kv_caches_[L];

    float* qkv_buf = scratch_qkv_;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x,
                    hidden_size_, full_q_dim_ + full_kv_dim_ * 2)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d (FullAttn)\n", L);
        return false;
    }

    float* q_gate_raw = qkv_buf;
    float* k_raw = qkv_buf + full_q_dim_;
    float* v_raw = qkv_buf + full_q_dim_ + full_kv_dim_;

    // RMSNorm on Q and K per-head
    for (int h = 0; h < num_q_heads_; h++) {
        float* qh = q_gate_raw + (size_t)h * head_dim_ * 2;
        rmsnorm(qh, qh, fw.q_norm, head_dim_, rms_eps_);
    }
    for (int h = 0; h < num_kv_heads_; h++) {
        rmsnorm(k_raw + h * head_dim_, k_raw + h * head_dim_, fw.k_norm, head_dim_, rms_eps_);
    }

    // RoPE
    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }
    apply_rope_cached(q_gate_raw, k_raw, num_q_heads_, num_kv_heads_,
                      head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                      rope_cos_row, rope_sin_row);

    // KV cache update
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

    // GQA attention
    gqa_attention(pre_oproj, q_gate_raw, cache.k_cache, cache.v_cache,
                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                  cache.start, cache.len, cache.capacity);

    // Output gate
    if (attn_output_gate_) {
        for (int h = 0; h < num_q_heads_; h++) {
            float* oh = pre_oproj + h * head_dim_;
            const float* gh = q_gate_raw + (size_t)h * head_dim_ * 2 + head_dim_;
            mul_sigmoid_inplace(oh, gh, head_dim_, scratch_tmp_);
        }
    }
    return true;
}

float* Qwen35Model::forward(int token, int pos) {
    // Embedding lookup
    memcpy(x_, embed_tokens_ + (int64_t)token * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = scratch_attn_;

    for (int L = 0; L < num_layers_; L++) {
        // Pre-attention norm
        rmsnorm(x_norm_, x_, layers_[L].input_layernorm, hidden_size_, rms_eps_);

        // Attention core
        if (layer_types_[L] == LayerType::LinearAttention) {
            if (!forward_deltanet_core(L, x_norm_, pre_oproj)) return nullptr;
        } else {
            if (!forward_full_attn_core(L, x_norm_, pre_oproj, pos)) return nullptr;
        }

        // O projection (ANE)
        int attn_dim = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;
        float* attn_out = x_norm_;
        if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, attn_dim, hidden_size_)) {
            fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
            return nullptr;
        }

        // Residual 1
        for (int i = 0; i < hidden_size_; i++) x_[i] += attn_out[i];

        // Post-attention norm
        rmsnorm(x_norm_, x_, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);

        // Fused FFN (ANE)
        float* mlp_out = scratch_attn_;
        if (!ane_matvec(ane_layers_[L].fused_ffn, mlp_out, x_norm_, hidden_size_, hidden_size_)) {
            fprintf(stderr, "ANE fused_ffn eval failed at layer %d\n", L);
            return nullptr;
        }

        // Residual 2
        for (int i = 0; i < hidden_size_; i++) x_[i] += mlp_out[i];
    }

    // Final norm
    rmsnorm(x_, x_, final_norm_, hidden_size_, rms_eps_);

    // LM head
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
            matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
        }
    } else {
        matvec(logits_, embed_tokens_, x_, vocab_size_, hidden_size_);
    }

    return logits_;
}

} // namespace ane_lm
