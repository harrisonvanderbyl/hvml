# Qwen3-ASR

[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech recognition, implemented with hvml tensors,
operations and modules (`tensor/models/qwen3asr/`).

```
audio (16 kHz mono)
  → Whisper log-mel [frames, 128]                        models/qwen3asr/mel.hpp
  → audio encoder                                        AudioEncoder
      100-frame chunks → 3× Conv2d(3×3, stride 2) + GELU → 13 tokens / chunk
      + sinusoidal positions → 18 transformer layers (attention in windows
      of 104 tokens) → LayerNorm → proj1 (GELU) → proj2 → 1024-d
  → prompt: <|im_start|>system\n…<|im_end|>\n<|im_start|>user\n
            <|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n<|im_start|>assistant\n
            (the audio embeddings replace the <|audio_pad|> embeddings)
  → Qwen3 decoder: 28 layers, GQA 16/8 heads × 128, q/k RMSNorm, RoPE θ=1e6,
    SwiGLU MLP, KV cache, greedy decoding              TextModel
  → "language English<asr_text>the transcript"          Qwen2Tokenizer + parse_output
```

## Use

```bash
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B

# CPU
g++ -std=c++20 -O3 -march=native -fopenmp -I./tensor examples/qwen3asr.cpp -o qwen3asr -ldl -rdynamic
# CUDA / HIP (through the shader tool, like the other examples)
make BASEFILE=examples/qwen3asr.cpp OUTPUT=qwen3asr

DEVICE_PLUGIN_DIR=tensor/device/plugins ./qwen3asr Qwen3-ASR-0.6B speech.wav --device hip
#   --language English   force the language (output is then text only)
#   --context "…"        system-prompt context (names, terms)
#   --max-tokens 512
#   --full-attention     encoder attends over the whole clip (see below)
#   --profile            time and operation count per stage (log-mel, encoder, prefill, decode)
```

From code:

```cpp
#include "models/qwen3asr/qwen3asr.hpp"
#include "file_loaders/wav.hpp"

auto model = qwen3asr::Qwen3ASR::from_pretrained("Qwen3-ASR-0.6B", MemoryType::kCUDA_VRAM);
auto result = model->transcribe(load_audio("speech.wav"));   // any WAV; resampled to 16 kHz
std::cout << result.language << ": " << result.text << "\n";
```

`from_pretrained` reads `config.json`, `model.safetensors` (or a sharded
`model.safetensors.index.json`), and `vocab.json` + `merges.txt` +
`tokenizer_config.json` (or `tokenizer.json`). The weights stay in the
checkpoint's bf16 and activations are fp32. If the checkpoint has no
`lm_head.weight`, the embedding table is used as the output projection
(tied embeddings).

## Modules and checkpoint keys

The module tree matches the checkpoint, so `load_from_safetensors` needs no
key mapping:

```
thinker.audio_tower.{conv2d1,conv2d2,conv2d3,conv_out,ln_post,proj1,proj2}
thinker.audio_tower.layers.N.{self_attn.{q,k,v,out}_proj, self_attn_layer_norm, fc1, fc2, final_layer_norm}
thinker.model.embed_tokens, thinker.model.norm
thinker.model.layers.N.{self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm, mlp.{gate,up,down}_proj,
                        input_layernorm, post_attention_layernorm}
thinker.lm_head
```

The building blocks are general: `Linear<W, bias>`, `LayerNorm`, `RMSNorm`,
`Embedding` and `Conv2d` (under `tensor/module/`), plus `ModuleList<T>` for
numbered layers and `move_to(MemoryLocation)` for putting a whole module on a
device. They are written with tensor views and a handful of element-wise ops
(see `docs/kernel.md`):

- `Embedding` returns `weight.tensor_index(ids)`.
- `Linear` is a `DotProduct` of `x.unsqueeze(1)` and `weight.unsqueeze(0)`.
- `Conv2d` stacks strided slices into an im2col tensor, then does one
  `DotProduct`.
- Attention is two `DotProduct`s over transposed views. The KV cache is kept
  as `[kv_heads, positions, head_dim]`, and new keys are written into a slice
  of it.
- The encoder's windowed attention runs attention on row slices, one window
  at a time.
- The log-mel frames are slices of the signal viewed as rows of 160 samples.

## Encoder attention windows

The encoder attends within blocks of `13 × n_window_infer / 100 = 104` tokens
(8 s of audio). vLLM and transformers' flash-attention path do this;
transformers' default eager/sdpa path ignores the blocks and attends over the
whole clip. Both reproduce the reference; windowed is the default, and
`set_windowed_encoder_attention(false)` / `--full-attention` switches.

## Verification

The port was checked against the official PyTorch implementation
(`qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`, transformers
4.57.6) with random weights saved as bf16 safetensors under the checkpoint's
key names:

| case                                             | log-mel | encoder | logits | 12 greedy tokens |
|--------------------------------------------------|---------|---------|--------|------------------|
| small dims, 3.7 s                                | 1.3e-5  | 3.6e-6  | 1.2e-5 | identical        |
| small dims, 12.3 s (two attention windows)       | 2.7e-5  | 3.3e-6  | 1.2e-5 | identical        |
| same, full attention vs. transformers sdpa       | 2.7e-5  | 3.9e-6  | 1.2e-5 | identical        |
| small dims, 5.0 s (whole chunks only)            | 2.7e-5  | 3.3e-6  | 1.0e-5 | identical        |
| real widths (896/14 heads, 1024/16q-8kv × 128), 3+3 layers, 4.2 s | 1.3e-5 | 6.7e-6 | 3.1e-5 | identical |

(Values are max absolute differences; features are ~1.3, encoder outputs ~5,
logits ~20.)  A checkpoint without `lm_head` and with float32 norms decodes
the same tokens as the PyTorch model with tied embeddings.

The tokenizer matches Hugging Face `tokenizers` on llama.cpp's 46 Qwen2 test
strings and on 400 random multilingual strings. Every case decodes back to its
input.

A full-size random checkpoint (0.9B parameters, bf16) takes 21 s for 3.7 s of
audio and 16 tokens on a 2-core CPU, with 3.6 GB peak memory. The real weights
and the GPU backends were not available in the environment used to build this.

## Not included

- Streaming and chunking of audio longer than one pass (the reference splits
  at 20 minutes).
- The forced aligner (timestamps).
- The repetition clean-up the Python package applies to the output.
