// main.cpp — ane-lm: Apple Neural Engine LLM inference tool
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <utility>
#include <ane_lm/common.h>
#include "utils.h"
#include "generate.h"
#include "core/model_loader.h"

// ObjC autorelease pool via C runtime API
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s generate --model <path> [--prompt <text>] [options]\n", prog);
    fprintf(stderr, "  %s chat --model <path> [options]\n", prog);
    fprintf(stderr, "  %s convert --model <path>\n", prog);
    fprintf(stderr, "\nSubcommands:\n");
    fprintf(stderr, "  generate    Single-shot text generation\n");
    fprintf(stderr, "  chat        Interactive multi-turn chat\n");
    fprintf(stderr, "  convert     Convert model weights from BF16 to FP16\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model <path>    Path to model directory (required)\n");
    fprintf(stderr, "  --prompt <text>   Input prompt (generate only, default: \"Hello\")\n");
    fprintf(stderr, "  --max-tokens N    Max tokens per response (default: unlimited)\n");
    fprintf(stderr, "  --temp T          Temperature (default: 0.6)\n");
    fprintf(stderr, "  --repeat-penalty P Repetition penalty (default: 1.2, 1.0=off)\n");
    fprintf(stderr, "  --enable-thinking Enable thinking/reasoning mode\n");
    fprintf(stderr, "  --no-ane-cache    Disable persistent ANE compile cache\n");
    fprintf(stderr, "  -v, --verbose     Show detailed initialization info\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s generate --model /path/to/Qwen3.5-0.8B --prompt \"Hello\" --max-tokens 50\n", prog);
    fprintf(stderr, "  %s chat --model /path/to/Qwen3.5-0.8B\n", prog);
}

struct Args {
    const char* model_dir = nullptr;
    const char* prompt = "Hello";
    float temperature = 0.6f;
    int max_tokens = 0;
    float repetition_penalty = 1.2f;
    bool ane_cache = true;
    bool enable_thinking = false;
};

static Args parse_args(int argc, char* argv[], int start) {
    Args args;
    for (int i = start; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repetition_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--enable-thinking") == 0) {
            args.enable_thinking = true;
        } else if (strcmp(argv[i], "--no-ane-cache") == 0) {
            args.ane_cache = false;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        }
    }
    return args;
}

static int cmd_generate(LLMModel& model, Tokenizer& tokenizer, const Args& args) {
    LOG("Prompt: \"%s\"\n", args.prompt);

    SamplingParams sampling;
    sampling.temperature = args.temperature;
    sampling.repetition_penalty = args.repetition_penalty;

    GenerationResponse last{};
    bool first = true;
    stream_generate(model, tokenizer, std::string(args.prompt),
        args.max_tokens, args.enable_thinking, sampling,
        [&](const GenerationResponse& r) {
            if (r.token == -1) {
                last = r;
                return;
            }
            if (!r.text.empty()) {
                if (first) { fprintf(stderr, "==========\n"); first = false; }
                fprintf(stderr, "%s", r.text.c_str());
            }
            last = r;
        });

    fprintf(stderr, "\n==========\n");
    fprintf(stderr, "Prompt: %d tokens, %.3f tokens-per-sec\n",
            last.prompt_tokens, last.prompt_tps);
    fprintf(stderr, "Generation: %d tokens, %.3f tokens-per-sec\n",
            last.generation_tokens, last.generation_tps);
    return 0;
}

static int cmd_chat(LLMModel& model, Tokenizer& tokenizer, const Args& args) {
    std::vector<std::pair<std::string, std::string>> messages;
    char buf[4096];

    while (true) {
        fprintf(stderr, ">>> ");
        if (!fgets(buf, sizeof(buf), stdin)) {
            // EOF (Ctrl-D)
            fprintf(stderr, "\n");
            break;
        }

        // Strip trailing newline
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

        // Skip empty input
        if (buf[0] == '\0') continue;

        // Exit commands
        if (strcmp(buf, "/bye") == 0 || strcmp(buf, "/exit") == 0) break;

        // Add user message
        messages.push_back({"user", std::string(buf)});

        // Reset model state and generate with full history
        model.reset();

        SamplingParams sampling;
        sampling.temperature = args.temperature;
        sampling.repetition_penalty = args.repetition_penalty;

        std::string assistant_text;
        GenerationResponse last{};
        stream_generate(model, tokenizer, messages,
            args.max_tokens, args.enable_thinking, sampling,
            [&](const GenerationResponse& r) {
                if (r.token == -1) {
                    last = r;
                    return;
                }
                if (!r.text.empty()) {
                    fprintf(stderr, "%s", r.text.c_str());
                    assistant_text += r.text;
                }
                last = r;
            });

        fprintf(stderr, "\n");

        // Add assistant response to history
        messages.push_back({"assistant", assistant_text});

        fprintf(stderr, "[%d prompt tokens, %.1f t/s | %d gen tokens, %.1f t/s]\n\n",
                last.prompt_tokens, last.prompt_tps,
                last.generation_tokens, last.generation_tps);
    }

    return 0;
}

static int cmd_convert(const Args& args) {
    std::string model_dir = args.model_dir;

    // Discover all safetensors files (single-file or sharded) and convert them.
    auto weights = ModelWeights::open(model_dir);
    if (!weights) {
        fprintf(stderr, "Error: failed to load model weights in %s\n", model_dir.c_str());
        return 1;
    }

    std::string output_dir = model_dir + "/ane_weights";

    Timer timer;
    int written = weights->write_ane_blobs(output_dir);
    double elapsed = timer.elapsed_ms();

    if (written < 0) {
        fprintf(stderr, "Error: conversion failed\n");
        return 1;
    }

    fprintf(stderr, "Done in %.1f ms\n", elapsed);
    return 0;
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    srand48(time(nullptr));
    setbuf(stdout, nullptr);

    // Need at least a subcommand
    if (argc < 2) {
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // Check for --help before subcommand
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 0;
    }

    // Determine subcommand
    const char* subcmd = argv[1];
    bool is_generate = (strcmp(subcmd, "generate") == 0);
    bool is_chat = (strcmp(subcmd, "chat") == 0);
    bool is_convert = (strcmp(subcmd, "convert") == 0);

    if (!is_generate && !is_chat && !is_convert) {
        fprintf(stderr, "Unknown subcommand: %s\n\n", subcmd);
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // Parse args after subcommand
    Args args = parse_args(argc, argv, 2);

    if (!args.model_dir) {
        fprintf(stderr, "Error: --model is required\n\n");
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // convert doesn't need model/tokenizer loading
    if (is_convert) {
        int ret = cmd_convert(args);
        objc_autoreleasePoolPop(pool);
        return ret;
    }

    LOG("=== ane-lm: Apple Neural Engine LLM Inference ===\n");
    LOG("Model: %s\n", args.model_dir);
    LOG("Mode: %s\n", is_chat ? "chat" : "generate");
    LOG("Temperature: %.2f, Max tokens: %d\n", args.temperature, args.max_tokens);
    LOG("ANE compile cache: %s\n", args.ane_cache ? "enabled" : "disabled");

    // Load model + tokenizer
    std::unique_ptr<LLMModel> model;
    Tokenizer tokenizer;
    try {
        auto result = load(args.model_dir, args.ane_cache);
        model = std::move(result.first);
        tokenizer = std::move(result.second);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    int ret;
    if (is_chat) {
        ret = cmd_chat(*model, tokenizer, args);
    } else {
        ret = cmd_generate(*model, tokenizer, args);
    }

    objc_autoreleasePoolPop(pool);
    return ret;
}
