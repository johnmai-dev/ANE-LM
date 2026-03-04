#include "model_loader.h"
#include <ane_lm/common.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace ane_lm {

using json = nlohmann::json;
namespace fs = std::filesystem;

static bool discover_safetensors_files(
    const std::string& model_dir,
    std::vector<std::string>* out_files,
    std::string* err) {
    out_files->clear();

    fs::path dir(model_dir);
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        if (err) *err = "Model directory not found: " + model_dir;
        return false;
    }

    // Prefer HF shard index when present.
    fs::path index = dir / "model.safetensors.index.json";
    if (fs::exists(index, ec) && fs::is_regular_file(index, ec)) {
        std::ifstream f(index);
        if (!f.is_open()) {
            if (err) *err = "Cannot open " + index.string();
            return false;
        }

        json j = json::parse(f, nullptr, false);
        if (j.is_discarded() || !j.contains("weight_map") || !j["weight_map"].is_object()) {
            if (err) *err = "Invalid shard index JSON: " + index.string();
            return false;
        }

        std::unordered_set<std::string> uniq;
        std::vector<std::string> rel_files;
        for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) {
            if (!it.value().is_string()) continue;
            std::string rel = it.value().get<std::string>();
            if (uniq.insert(rel).second) rel_files.push_back(rel);
        }
        std::sort(rel_files.begin(), rel_files.end());

        if (rel_files.empty()) {
            if (err) *err = "No shard files listed in " + index.string();
            return false;
        }

        for (const auto& rel : rel_files) {
            fs::path p = dir / rel;
            if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) {
                if (err) *err = "Missing shard file: " + p.string();
                return false;
            }
            out_files->push_back(p.string());
        }
        return true;
    }

    // Fallback: recursively enumerate all *.safetensors in model directory.
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const fs::path& p = entry.path();
        if (p.extension() == ".safetensors") files.push_back(p);
    }

    if (ec) {
        if (err) *err = "Failed to enumerate files in " + model_dir;
        return false;
    }

    std::sort(files.begin(), files.end());
    for (const auto& p : files) out_files->push_back(p.string());

    if (out_files->empty()) {
        if (err) *err = "No .safetensors files found in " + model_dir;
        return false;
    }

    return true;
}

std::unique_ptr<ModelWeights> ModelWeights::open(const std::string& model_dir) {
    std::string err;
    std::vector<std::string> files;
    if (!discover_safetensors_files(model_dir, &files, &err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return nullptr;
    }

    auto mw = std::make_unique<ModelWeights>();
    mw->files_ = files;

    for (const auto& path : files) {
        std::unique_ptr<SafeTensors> sf(SafeTensors::open(path));
        if (!sf) return nullptr;

        for (int i = 0; i < sf->n_tensors(); i++) {
            const SFTensor& t = sf->tensor(i);
            if (mw->tensor_owner_.find(t.name) == mw->tensor_owner_.end()) {
                mw->tensor_owner_[t.name] = sf.get();
            }
        }
        mw->shards_.push_back(std::move(sf));
    }

    LOG("Model weights: %d safetensors shard(s) discovered\n", mw->shard_count());
    return mw;
}

const SafeTensors* ModelWeights::owner_for(const char* name) const {
    auto it = tensor_owner_.find(name);
    if (it == tensor_owner_.end()) return nullptr;
    return it->second;
}

const SFTensor* ModelWeights::find(const char* name) const {
    const SafeTensors* sf = owner_for(name);
    return sf ? sf->find(name) : nullptr;
}

const uint16_t* ModelWeights::get_bf16_ptr(const char* name) const {
    const SafeTensors* sf = owner_for(name);
    return sf ? sf->get_bf16_ptr(name) : nullptr;
}

float* ModelWeights::load_bf16_to_f32(const char* name, int64_t expected_numel) const {
    const SafeTensors* sf = owner_for(name);
    if (!sf) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    return sf->load_bf16_to_f32(name, expected_numel);
}

float* ModelWeights::load_f32_direct(const char* name, int64_t expected_numel) const {
    const SafeTensors* sf = owner_for(name);
    if (!sf) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    return sf->load_f32_direct(name, expected_numel);
}

float* ModelWeights::load_norm_weight(const char* name, int64_t expected_numel) const {
    const SafeTensors* sf = owner_for(name);
    if (!sf) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    return sf->load_norm_weight(name, expected_numel);
}

int ModelWeights::write_ane_blobs(const std::string& output_dir) const {
    int total = 0;
    for (const auto& sf : shards_) {
        int written = SafeTensors::write_ane_blobs(*sf, output_dir);
        if (written < 0) return -1;
        total += written;
    }
    return total;
}

} // namespace ane_lm
