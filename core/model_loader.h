#pragma once

#include "safetensors.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ane_lm {

// Discover and load model safetensors files (single-file or sharded).
class ModelWeights {
public:
    static std::unique_ptr<ModelWeights> open(const std::string& model_dir);

    const SFTensor* find(const char* name) const;
    const uint16_t* get_bf16_ptr(const char* name) const;

    float* load_bf16_to_f32(const char* name, int64_t expected_numel = -1) const;
    float* load_f32_direct(const char* name, int64_t expected_numel = -1) const;
    float* load_norm_weight(const char* name, int64_t expected_numel) const;

    int write_ane_blobs(const std::string& output_dir) const;

    int shard_count() const { return (int)shards_.size(); }
    const std::vector<std::string>& files() const { return files_; }

private:
    std::vector<std::string> files_;
    std::vector<std::unique_ptr<SafeTensors>> shards_;
    std::unordered_map<std::string, const SafeTensors*> tensor_owner_;

    const SafeTensors* owner_for(const char* name) const;
};

} // namespace ane_lm
