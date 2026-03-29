# SID-GPT support for llama.cpp

This fork adds support for [SID-GPT](https://github.com/...) -- a LLaMA-architecture
transformer that generates Commodore 64 chiptune music as SID register streams.

## What changed

SID-GPT is standard LLaMA except for one custom embedding: `frame_pos_emb`.
It tells the model which of the 26 SID register slots the current token
represents (cyclic: `position % 26`). Without it the model produces garbage.

The patch adds optional `frame_pos_embd` support to the LLaMA architecture.
When the tensor is absent (normal LLaMA models), nothing changes.

### Patched files

| File | Change |
|------|--------|
| `src/llama-arch.h` | `LLM_KV_FRAME_SIZE`, `LLM_TENSOR_FRAME_POS_EMBD` enums |
| `src/llama-arch.cpp` | KV/tensor name mappings, LLAMA tensor list, tensor info |
| `src/llama-hparams.h` | `uint32_t frame_size` field |
| `src/llama-model.h` | `frame_pos_embd` tensor pointer |
| `src/llama-model.cpp` | Load frame_size hparam + frame_pos_embd tensor |
| `src/llama-graph.h` | `llm_graph_input_frame_pos` class |
| `src/llama-graph.cpp` | `set_input()` (pos % frame_size) + `build_inp_frame_pos()` |
| `src/models/llama.cpp` | Conditional frame_pos_embd add in forward pass |

### New: `sidgpt-generate`

Custom inference program in `examples/sidgpt/`. Bypasses the text tokenizer
and works directly with raw token IDs, outputting uint16 LE binary.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sidgpt-generate -j$(nproc)
# binary: build/bin/sidgpt-generate
```

### AVX-512 (AMD Ryzen AI / Zen 5)

On CPUs with AVX-512 support (e.g. Tuxedo laptops with AMD Ryzen AI 9 HX 370),
enable the hand-tuned ggml AVX-512 kernels for better matmul performance:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON
cmake --build build -j$(nproc)
```

### Apple Silicon (M1/M2/M3/M4)

Metal GPU is auto-detected on macOS -- no extra flags needed.
All layers offloaded to GPU by default via `-ngl 99`.
GGUF model files are portable across platforms.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sidgpt-generate -j$(sysctl -n hw.ncpu)
```

## Usage

```bash
# Generate 500 frames (~10s of music)
build/bin/sidgpt-generate \
    -m model.gguf -n 500 --temp 0.85 -o output.sidgpt

# With sampling options
build/bin/sidgpt-generate \
    -m model.gguf -n 500 --temp 0.95 \
    --top-k 40 --seed 42 -o output.sidgpt

# Play with sidgpt-play (from SID-GPT repo)
sidgpt-play output.sidgpt
```

### Options

```
-m PATH      Model GGUF file (required)
-n N         Number of frames to generate (default: 500)
--temp F     Temperature (default: 0.85)
--top-k N    Top-K sampling (default: 0 = off)
--top-p F    Top-P sampling (default: 1.0)
--seed N     RNG seed (default: random)
-o PATH      Output file (default: stdout)
-ngl N       GPU layers to offload (default: 99)
```

## GGUF format details

The GGUF file uses `general.architecture = "llama"` with these additions:

- Metadata key `llama.frame_size = 26` (read as optional hparam)
- Tensor `frame_pos_embd.weight` shape `[26, n_embd]`
- No `output.weight` tensor (weight-tied with `token_embd`)
- Vocabulary: 258 tokens (bytes 0-255, SEP=256, FRAME=257)

## Status

- [x] Model loading with frame_pos_embd
- [x] Unconditional generation (SEP frame prompt)
- [x] F32 inference working
- [x] --seed-file support (prompt from existing .sidgpt)
- [x] Quantization testing (Q4_K_M, Q8_0 via llama-quantize)
- [x] Logit verification against PyTorch reference
