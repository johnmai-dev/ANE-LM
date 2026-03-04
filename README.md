# ANE-LM

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs. 
## Supported Models

- Qwen3 (dense)
- Qwen3.5 (dense, text-only)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

![image](assets/image.png)

Download a supported model (e.g. `Qwen3-0.6B` or `Qwen3.5-0.8B` in safetensors format), then:

```bash
# Single-shot generation
./build/ane-lm generate --model /path/to/Qwen3.5-0.8B --prompt "Hello"

# Interactive chat
./build/ane-lm chat --model /path/to/Qwen3.5-0.8B

# Pre-convert weights (BF16 -> FP16, speeds up subsequent loads)
./build/ane-lm convert --model /path/to/Qwen3.5-0.8B
```

### Options

```
--model <path>       Path to model directory (required)
--prompt <text>      Input prompt (generate mode, default: "Hello")
--max-tokens N       Max tokens to generate (default: unlimited)
--temp T             Temperature (default: 0.6)
--repeat-penalty P   Repetition penalty (default: 1.2, 1.0=off)
--enable-thinking    Enable thinking/reasoning mode
--no-ane-cache       Disable persistent ANE compile cache
-v, --verbose        Show detailed initialization info
```

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [maderix/ANE](https://github.com/maderix/ANE) - Training neural networks on Apple Neural Engine via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++
