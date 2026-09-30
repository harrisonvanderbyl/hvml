#ifndef MODELS_QWEN3ASR_HPP
#define MODELS_QWEN3ASR_HPP

//
//  Qwen3-ASR (https://huggingface.co/Qwen/Qwen3-ASR-0.6B) for hvml.
//
//  Audio (16 kHz mono) → Whisper log-mel [frames, 128]
//    → audio encoder: 3× stride-2 conv (per 100-frame chunk) → 13 tokens per
//      chunk → windowed transformer (18 layers) → proj to text width
//    → spliced into the prompt in place of <|audio_pad|> tokens
//    → Qwen3 text decoder (28 layers, GQA, q/k RMSNorm, RoPE) → greedy decode
//    → "language <Lang><asr_text><transcript>"
//
//  Every module mirrors the checkpoint's key names, so
//      Qwen3ASR model(config);
//      model.load_from_safetensors(safetensors("model.safetensors"));
//  loads `thinker.audio_tower.*`, `thinker.model.*`, `thinker.lm_head.*`.
//  All compute is element-wise ops and reductions over tensor views, so the
//  model runs on whatever device it is moved to (`model.move_to(MemoryType::kHIP_VRAM)` etc.).
//

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <algorithm>
#include <chrono>
#include <optional>
#include <cstdio>

#include "tensor.hpp"
#include "ops/nn.hpp"
#include "file_loaders/json.hpp"
#include "file_loaders/safetensors.hpp"
#include "module/base/module.hpp"
#include "module/linear/linear.hpp"
#include "module/layernorm/layernorm.hpp"
#include "module/rmsnorm/rmsnorm.hpp"
#include "module/embedding/embedding.hpp"
#include "module/conv/conv2d.hpp"
#include "models/qwen3asr/mel.hpp"
#include "tokenizers/qwen2.hpp"

namespace qwen3asr {

using WeightType = bfloat16;   // checkpoint dtype

// ---------------------------------------------------------------------------
//  Configuration (thinker_config in config.json)
// ---------------------------------------------------------------------------

struct Config {
    // audio encoder
    long audio_d_model = 896;
    long audio_heads = 14;
    long audio_ffn = 3584;
    long audio_layers = 18;
    long audio_downsample = 480;
    long audio_output_dim = 1024;
    long num_mel_bins = 128;
    long n_window = 50;           // chunk = 2 * n_window mel frames
    long n_window_infer = 800;    // attention window, in mel frames
    // Block-diagonal encoder attention over n_window_infer-frame windows, as
    // vLLM and flash-attention run it.  false = attend over the whole clip
    // (what transformers' eager/sdpa path does).
    bool windowed_encoder_attention = true;

    // text decoder
    long hidden = 1024;
    long heads = 16;
    long kv_heads = 8;
    long head_dim = 128;
    long intermediate = 3072;
    long layers = 28;
    long vocab = 151936;
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;

    // special tokens
    int audio_start_id = 151669;
    int audio_end_id = 151670;
    int audio_pad_id = 151676;
    int im_start_id = 151644;
    int im_end_id = 151645;
    int endoftext_id = 151643;
    int asr_text_id = 151704;

    static Config from_json(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("qwen3asr: cannot open " + path);
        nlohmann::json j = nlohmann::json::parse(f);
        nlohmann::json t = j.contains("thinker_config") ? j["thinker_config"] : j;
        Config c;
        auto a = t["audio_config"];
        auto x = t["text_config"];
        auto get = [](const nlohmann::json& o, const char* k, auto dflt) {
            return o.contains(k) && !o[k].is_null() ? o[k].get<decltype(dflt)>() : dflt;
        };
        c.audio_d_model = get(a, "d_model", c.audio_d_model);
        c.audio_heads = get(a, "encoder_attention_heads", c.audio_heads);
        c.audio_ffn = get(a, "encoder_ffn_dim", c.audio_ffn);
        c.audio_layers = get(a, "encoder_layers", c.audio_layers);
        c.audio_downsample = get(a, "downsample_hidden_size", c.audio_downsample);
        c.audio_output_dim = get(a, "output_dim", c.audio_output_dim);
        c.num_mel_bins = get(a, "num_mel_bins", c.num_mel_bins);
        c.n_window = get(a, "n_window", c.n_window);
        c.n_window_infer = get(a, "n_window_infer", c.n_window_infer);
        c.hidden = get(x, "hidden_size", c.hidden);
        c.heads = get(x, "num_attention_heads", c.heads);
        c.kv_heads = get(x, "num_key_value_heads", c.kv_heads);
        c.head_dim = get(x, "head_dim", c.head_dim);
        c.intermediate = get(x, "intermediate_size", c.intermediate);
        c.layers = get(x, "num_hidden_layers", c.layers);
        c.vocab = get(x, "vocab_size", c.vocab);
        c.rms_eps = (float)get(x, "rms_norm_eps", (double)c.rms_eps);
        c.rope_theta = (float)get(x, "rope_theta", (double)c.rope_theta);
        c.audio_start_id = get(t, "audio_start_token_id", c.audio_start_id);
        c.audio_end_id = get(t, "audio_end_token_id", c.audio_end_id);
        c.audio_pad_id = get(t, "audio_token_id", c.audio_pad_id);
        return c;
    }

    long chunk_frames() const { return 2 * n_window; }

    // Number of audio tokens for `frames` mel frames (matches the reference's
    // _get_feat_extract_output_lengths).
    static long audio_tokens(long frames) {
        auto ceil_half = [](long L) { return L <= 0 ? 0 : (L - 1) / 2 + 1; };
        long leave = frames % 100;
        long tail = leave == 0 ? 0 : ceil_half(ceil_half(ceil_half(leave)));
        return tail + (frames / 100) * 13;
    }
};

// ---------------------------------------------------------------------------
//  Audio encoder
// ---------------------------------------------------------------------------

struct AudioAttention : public Module<Linear<WeightType, true>, Linear<WeightType, true>,
                                      Linear<WeightType, true>, Linear<WeightType, true>> {
    Linear<WeightType, true> q_proj, k_proj, v_proj, out_proj;
    long heads;

    AudioAttention(long heads)
        : Module({q_proj, "q_proj"}, {k_proj, "k_proj"}, {v_proj, "v_proj"}, {out_proj, "out_proj"}),
          heads(heads) {}

    // x [N, D]; tokens attend within consecutive blocks of `window` rows
    // (0 = one block over everything)
    Tensor<float, 2> forward(const Tensor<float, 2>& x, long window) const {
        long N = x.shape[0], D = x.shape[1], hd = D / heads;
        auto q = q_proj(x), k = k_proj(x), v = v_proj(x);
        Tensor<float, 2> out(Shape<2>{N, D}, MemoryLocation(*x.device));
        long step = window > 0 ? window : N;
        for (long w0 = 0; w0 < N; w0 += step) {
            long w1 = std::min(N, w0 + step), n = w1 - w0;
            Tensor<float, 3> qw = q[{{w0, w1}}].view(Shape<3>{n, heads, hd});
            Tensor<float, 3> kw = k[{{w0, w1}}].view(Shape<3>{n, heads, hd}).transpose(0, 1);   // [heads, n, hd]
            Tensor<float, 3> vw = v[{{w0, w1}}].view(Shape<3>{n, heads, hd}).transpose(0, 1);
            out[{{w0, w1}}] = attention(qw, kw, vw, 1.0f / std::sqrt((float)hd), 0, false);
        }
        return out_proj(out);
    }
};

struct AudioEncoderLayer : public Module<AudioAttention, LayerNorm<WeightType>, Linear<WeightType, true>,
                                         Linear<WeightType, true>, LayerNorm<WeightType>> {
    AudioAttention self_attn;
    LayerNorm<WeightType> self_attn_layer_norm;
    Linear<WeightType, true> fc1, fc2;
    LayerNorm<WeightType> final_layer_norm;

    AudioEncoderLayer(const Config& c)
        : Module({self_attn, "self_attn"}, {self_attn_layer_norm, "self_attn_layer_norm"}, {fc1, "fc1"},
                 {fc2, "fc2"}, {final_layer_norm, "final_layer_norm"}),
          self_attn(c.audio_heads), self_attn_layer_norm(1e-5f), final_layer_norm(1e-5f) {}

    Tensor<float, 2> forward(const Tensor<float, 2>& x, long window) const {
        Tensor<float, 2> h = x + self_attn.forward(self_attn_layer_norm(x), window);
        return h + fc2(fc1.forward(final_layer_norm(h), /*gelu=*/true));
    }
};

struct AudioEncoder : public Module<Conv2d<WeightType>, Conv2d<WeightType>, Conv2d<WeightType>,
                                    Linear<WeightType, false>, ModuleList<AudioEncoderLayer>,
                                    LayerNorm<WeightType>, Linear<WeightType, true>, Linear<WeightType, true>> {
    Conv2d<WeightType> conv2d1, conv2d2, conv2d3;
    Linear<WeightType, false> conv_out;
    ModuleList<AudioEncoderLayer> layers;
    LayerNorm<WeightType> ln_post;
    Linear<WeightType, true> proj1, proj2;
    Config cfg;

    AudioEncoder(const Config& c)
        : Module({conv2d1, "conv2d1"}, {conv2d2, "conv2d2"}, {conv2d3, "conv2d3"}, {conv_out, "conv_out"},
                 {layers, "layers"}, {ln_post, "ln_post"}, {proj1, "proj1"}, {proj2, "proj2"}),
          conv2d1(2, 1), conv2d2(2, 1), conv2d3(2, 1),
          layers(c.audio_layers, [&](size_t) { return new AudioEncoderLayer(c); }),
          ln_post(1e-5f), cfg(c) {}

    // Sinusoidal position table rows [0, rows) (SinusoidsPositionEmbedding).
    std::vector<float> positions(long rows) const {
        long C = cfg.audio_d_model, half = C / 2;
        double inc = std::log(10000.0) / (double)(half - 1);
        std::vector<float> p(rows * C);
        for (long t = 0; t < rows; t++) {
            for (long i = 0; i < half; i++) {
                float inv = (float)std::exp(-inc * (double)i);
                float a = (float)t * inv;
                p[t * C + i] = std::sin(a);
                p[t * C + half + i] = std::cos(a);
            }
        }
        return p;
    }

    // mel: [frames, n_mels] on the model's device → [tokens, output_dim]
    Tensor<float, 2> forward(const Tensor<float, 2>& mel) const {
        MemoryLocation loc = working_location(conv_out.weight.device);
        long F = mel.shape[0], nmel = mel.shape[1];
        long chunk = cfg.chunk_frames();
        long nC = (F + chunk - 1) / chunk;
        long C = cfg.audio_downsample, d = cfg.audio_d_model;

        // Zero-padded chunks of `chunk` frames, as images [nC, 1, n_mels, chunk]
        Tensor<float, 2> padded(Shape<2>{nC * chunk, nmel}, loc);
        padded = 0.0f;
        padded[{{0, F}}] = mel;
        Tensor<float, 4> images = padded.view(Shape<3>{nC, chunk, nmel}).transpose(1, 2).unsqueeze(1);

        Tensor<float, 4> c1 = conv2d1.forward(images, /*gelu=*/true);
        Tensor<float, 4> c2 = conv2d2.forward(c1, true);
        Tensor<float, 4> c = conv2d3.forward(c2, true);                                // [nC, C, H3, W3]
        long H3 = c.shape[2], W3 = c.shape[3];

        // One token per output column: [nC, W3, C, H3] (the reference's
        // permute(0, 3, 1, 2)) → rows of C·H3 features
        Tensor<float, 4> columns = c.transpose(1, 3).transpose(2, 3).contiguous();
        Tensor<float, 2> x = conv_out(columns.view(Shape<2>{nC * W3, C * H3}));

        std::vector<float> pos_host = positions(W3);
        auto pos = tensor_from_host(Shape<2>{W3, d}, pos_host.data(), loc);
        x.view(Shape<3>{nC, W3, d}) += pos.unsqueeze(0);

        // Only the last chunk can be partial, so the valid tokens are a prefix.
        long N = Config::audio_tokens(F);
        // (optional::emplace rebinds h to each layer's output; Tensor's
        // operator= would copy into the previous buffer instead)
        std::optional<Tensor<float, 2>> h(x[{{0, N}}]);

        // Attention in blocks of W3 · (n_window_infer / chunk) tokens
        long window = cfg.windowed_encoder_attention ? W3 * (cfg.n_window_infer / chunk) : 0;
        for (size_t i = 0; i < layers.size(); i++) h.emplace(layers[i].forward(*h, window));

        return proj2(proj1.forward(ln_post(*h), /*gelu=*/true));
    }
};

// ---------------------------------------------------------------------------
//  Text decoder (Qwen3)
// ---------------------------------------------------------------------------

struct KVCache {
    std::vector<Tensor<float, 3>> k, v;   // per layer [kv_heads, capacity, head_dim]
    long capacity = 0;
    long length = 0;
};

struct TextAttention : public Module<Linear<WeightType, false>, Linear<WeightType, false>, Linear<WeightType, false>,
                                     Linear<WeightType, false>, RMSNorm<WeightType>, RMSNorm<WeightType>> {
    Linear<WeightType, false> q_proj, k_proj, v_proj, o_proj;
    RMSNorm<WeightType> q_norm, k_norm;
    Config cfg;

    TextAttention(const Config& c)
        : Module({q_proj, "q_proj"}, {k_proj, "k_proj"}, {v_proj, "v_proj"}, {o_proj, "o_proj"},
                 {q_norm, "q_norm"}, {k_norm, "k_norm"}),
          q_norm(c.rms_eps), k_norm(c.rms_eps), cfg(c) {}

    // x [T, hidden] at positions pos0 .. pos0+T
    Tensor<float, 2> forward(const Tensor<float, 2>& x, const RopeTables& rotary, long pos0,
                             Tensor<float, 3>& k_cache, Tensor<float, 3>& v_cache) const {
        long T = x.shape[0], H = cfg.heads, KV = cfg.kv_heads, D = cfg.head_dim;
        Tensor<float, 3> q = q_norm(q_proj(x).view(Shape<2>{T * H, D})).view(Shape<3>{T, H, D});
        Tensor<float, 3> k = k_norm(k_proj(x).view(Shape<2>{T * KV, D})).view(Shape<3>{T, KV, D});
        Tensor<float, 3> v = v_proj(x).view(Shape<3>{T, KV, D});
        rope(q, rotary);
        rope(k, rotary);

        k_cache[{{}, {pos0, pos0 + T}}] = k.transpose(0, 1);
        v_cache[{{}, {pos0, pos0 + T}}] = v.transpose(0, 1);
        Tensor<float, 3> keys = k_cache[{{}, {0, pos0 + T}}];
        Tensor<float, 3> values = v_cache[{{}, {0, pos0 + T}}];

        return o_proj(attention(q, keys, values, 1.0f / std::sqrt((float)D), pos0, T > 1));
    }
};

struct TextMLP : public Module<Linear<WeightType, false>, Linear<WeightType, false>, Linear<WeightType, false>> {
    Linear<WeightType, false> gate_proj, up_proj, down_proj;

    TextMLP() : Module({gate_proj, "gate_proj"}, {up_proj, "up_proj"}, {down_proj, "down_proj"}) {}

    Tensor<float, 2> forward(const Tensor<float, 2>& x) const {
        return down_proj(OpSiluMul::run(gate_proj(x), up_proj(x)));
    }
};

struct TextDecoderLayer : public Module<TextAttention, TextMLP, RMSNorm<WeightType>, RMSNorm<WeightType>> {
    TextAttention self_attn;
    TextMLP mlp;
    RMSNorm<WeightType> input_layernorm, post_attention_layernorm;

    TextDecoderLayer(const Config& c)
        : Module({self_attn, "self_attn"}, {mlp, "mlp"}, {input_layernorm, "input_layernorm"},
                 {post_attention_layernorm, "post_attention_layernorm"}),
          self_attn(c), input_layernorm(c.rms_eps), post_attention_layernorm(c.rms_eps) {}

    Tensor<float, 2> forward(const Tensor<float, 2>& x, const RopeTables& rotary, long pos0,
                             Tensor<float, 3>& k_cache, Tensor<float, 3>& v_cache) const {
        Tensor<float, 2> h = x + self_attn.forward(input_layernorm(x), rotary, pos0, k_cache, v_cache);
        return h + mlp.forward(post_attention_layernorm(h));
    }
};

struct TextModel : public Module<Embedding<WeightType>, ModuleList<TextDecoderLayer>, RMSNorm<WeightType>> {
    Embedding<WeightType> embed_tokens;
    ModuleList<TextDecoderLayer> layers;
    RMSNorm<WeightType> norm;
    Config cfg;

    TextModel(const Config& c)
        : Module({embed_tokens, "embed_tokens"}, {layers, "layers"}, {norm, "norm"}),
          layers(c.layers, [&](size_t) { return new TextDecoderLayer(c); }), norm(c.rms_eps), cfg(c) {}

    MemoryLocation location() const { return working_location(embed_tokens.weight.device); }

    // Token ids → float embeddings [T, hidden]
    Tensor<float, 2> embed(const std::vector<int>& ids) const {
        std::vector<unsigned long> index(ids.begin(), ids.end());
        auto id_tensor = tensor_from_host(Shape<1>{(long)index.size()}, index.data(), location());
        Tensor<float, 2> out(Shape<2>{(long)index.size(), cfg.hidden}, location());
        out = embed_tokens(id_tensor);
        return out;
    }

    KVCache make_cache(long capacity) const {
        KVCache cache;
        cache.capacity = capacity;
        for (long i = 0; i < cfg.layers; i++) {
            cache.k.emplace_back(Shape<3>{cfg.kv_heads, capacity, cfg.head_dim}, location());
            cache.v.emplace_back(Shape<3>{cfg.kv_heads, capacity, cfg.head_dim}, location());
        }
        return cache;
    }

    // embeds: [T, hidden] for positions [cache.length, cache.length + T)
    // → final hidden state of the last position, normalised: [1, hidden]
    Tensor<float, 2> forward(const Tensor<float, 2>& embeds, KVCache& cache) const {
        long T = embeds.shape[0], pos0 = cache.length;
        if (pos0 + T > cache.capacity) throw std::runtime_error("qwen3asr: KV cache full");
        std::vector<long> positions(T);
        for (long t = 0; t < T; t++) positions[t] = pos0 + t;
        RopeTables rotary = rope_tables(positions, cfg.head_dim, cfg.rope_theta, location());

        std::optional<Tensor<float, 2>> h(embeds);
        for (size_t i = 0; i < layers.size(); i++) h.emplace(layers[i].forward(*h, rotary, pos0, cache.k[i], cache.v[i]));
        cache.length += T;
        return norm((*h)[{{T - 1, T}}]);
    }
};

// ---------------------------------------------------------------------------
//  Whole model
// ---------------------------------------------------------------------------

struct Thinker : public Module<AudioEncoder, TextModel, Linear<WeightType, false>> {
    AudioEncoder audio_tower;
    TextModel model;
    Linear<WeightType, false> lm_head;

    Thinker(const Config& c)
        : Module({audio_tower, "audio_tower"}, {model, "model"}, {lm_head, "lm_head"}),
          audio_tower(c), model(c) {}

    // logits for the final hidden state [1, hidden] → [1, vocab]
    Tensor<float, 2> logits(const Tensor<float, 2>& h) const {
        if (lm_head.weight.storage_pointer) return lm_head(h);
        return linear(h, model.embed_tokens.weight);   // tied embeddings
    }
};

struct TranscribeOptions {
    std::string context;           // system prompt text (optional)
    std::string language;          // force a language, e.g. "English" (optional)
    long max_new_tokens = 512;
    bool verbose = false;
    bool profile = false;          // print per-stage times and operation counts (synchronises the device)
    std::function<void(const std::string&)> on_text;   // streaming callback (decoded so far)
};

struct Transcription {
    std::string language;
    std::string text;
    std::string raw;
    std::vector<int> tokens;
};

struct Qwen3ASR : public Module<Thinker> {
    Thinker thinker;
    Config cfg;
    LogMel mel;
    Qwen2Tokenizer tokenizer;

    Qwen3ASR(const Config& c) : Module({thinker, "thinker"}), thinker(c), cfg(c), mel(c.num_mel_bins) {}

    // Load config.json, model*.safetensors and the tokenizer from a model
    // directory, then move the weights to `loc`.
    static std::unique_ptr<Qwen3ASR> from_pretrained(const std::string& dir, MemoryLocation loc = MemoryType::kDDR) {
        auto model = std::make_unique<Qwen3ASR>(Config::from_json(dir + "/config.json"));
        std::vector<std::string> files;
        std::ifstream index(dir + "/model.safetensors.index.json");
        if (index) {
            nlohmann::json j = nlohmann::json::parse(index);
            std::set<std::string> uniq;
            for (auto& [k, v] : j["weight_map"].items()) uniq.insert(v.get<std::string>());
            files.assign(uniq.begin(), uniq.end());
        } else {
            files.push_back("model.safetensors");
        }
        for (auto& f : files) model->load_from_safetensors(safetensors(dir + "/" + f));
        if (!model->thinker.lm_head.weight.storage_pointer) {
            std::cout << "qwen3asr: no lm_head in checkpoint — using tied embed_tokens" << std::endl;
        }
        model->move_to(loc);
        model->mel.move_to(loc);
        model->tokenizer.load(dir);
        return model;
    }

    // false: the audio encoder attends over the whole clip (transformers'
    // eager/sdpa behaviour) instead of n_window_infer windows.
    void set_windowed_encoder_attention(bool windowed) {
        cfg.windowed_encoder_attention = windowed;
        thinker.audio_tower.cfg.windowed_encoder_attention = windowed;
    }

    // Token ids of the chat prompt (see chat_template.json) with `n_audio`
    // audio placeholders.
    std::vector<int> prompt_ids(long n_audio, const TranscribeOptions& opt) const {
        const int NL = 198, SYSTEM = 8948, USER = 872, ASSISTANT = 77091;   // "\n", "system", "user", "assistant"
        std::vector<int> ids = {cfg.im_start_id, SYSTEM, NL};
        if (!opt.context.empty()) {
            auto ctx = tokenizer.encode(opt.context);
            ids.insert(ids.end(), ctx.begin(), ctx.end());
        }
        ids.insert(ids.end(), {cfg.im_end_id, NL, cfg.im_start_id, USER, NL, cfg.audio_start_id});
        ids.insert(ids.end(), n_audio, cfg.audio_pad_id);
        ids.insert(ids.end(), {cfg.audio_end_id, cfg.im_end_id, NL, cfg.im_start_id, ASSISTANT, NL});
        if (!opt.language.empty()) {
            auto lang = tokenizer.encode("language " + opt.language);
            ids.insert(ids.end(), lang.begin(), lang.end());
            ids.push_back(cfg.asr_text_id);
        }
        return ids;
    }

    // Audio features → audio embeddings: [tokens, hidden]
    Tensor<float, 2> encode_audio(const Tensor<float, 2>& features) const { return thinker.audio_tower.forward(features); }

    // Greedy decoding from 16 kHz mono samples.
    Transcription transcribe(const std::vector<float>& samples, const TranscribeOptions& opt = {}) {
        // Profiling: wait for the device at stage boundaries and report the
        // wall time and number of operations of each stage.
        auto clock = std::chrono::steady_clock::now();
        unsigned long ops = OperationStats::launches;
        auto stage = [&](const char* name, long steps = 1) {
            if (!opt.profile) return;
            thinker.model.embed_tokens.weight.device->synchronize_function();
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - clock).count();
            unsigned long n = OperationStats::launches - ops;
            if (steps > 1)
                fprintf(stderr, "[profile] %-8s %8.1f ms  %6lu ops  (%ld steps: %.2f ms, %lu ops each)\n", name, ms, n,
                        steps, ms / steps, n / steps);
            else
                fprintf(stderr, "[profile] %-8s %8.1f ms  %6lu ops\n", name, ms, n);
            clock = now;
            ops = OperationStats::launches;
        };

        auto features = mel.compute(samples);
        stage("log-mel");
        auto audio = encode_audio(features);
        long n_audio = audio.shape[0];
        stage("encoder");

        auto ids = prompt_ids(n_audio, opt);
        long audio_at = std::find(ids.begin(), ids.end(), cfg.audio_pad_id) - ids.begin();
        Tensor<float, 2> embeds = thinker.model.embed(ids);
        embeds[{{audio_at, audio_at + n_audio}}] = audio;

        KVCache cache = thinker.model.make_cache((long)ids.size() + opt.max_new_tokens + 1);
        Transcription result;
        std::optional<Tensor<float, 2>> h(thinker.model.forward(embeds, cache));
        stage("prefill");
        long steps = 0;
        for (long step = 0; step < opt.max_new_tokens; step++) {
            std::vector<float> logits = tensor_to_host(thinker.logits(*h));
            int tok = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            steps++;
            if (tok == cfg.im_end_id || tok == cfg.endoftext_id) break;
            result.tokens.push_back(tok);
            if (opt.on_text) opt.on_text(tokenizer.decode(result.tokens));
            if (step + 1 == opt.max_new_tokens) break;
            h.emplace(thinker.model.forward(thinker.model.embed({tok}), cache));
        }
        stage("decode", steps);

        result.raw = tokenizer.decode(result.tokens);
        parse_output(result, opt.language);
        return result;
    }

    // "language English<asr_text>hello" → ("English", "hello")
    static void parse_output(Transcription& r, const std::string& forced_language) {
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        };
        std::string s = trim(r.raw);
        if (!forced_language.empty()) {
            r.language = forced_language;
            r.text = s;
            return;
        }
        size_t tag = s.find("<asr_text>");
        if (tag == std::string::npos) {
            r.text = s;
            return;
        }
        std::string meta = trim(s.substr(0, tag));
        r.text = trim(s.substr(tag + 10));
        if (meta.rfind("language ", 0) == 0) meta = trim(meta.substr(9));
        r.language = (meta == "None") ? "" : meta;
        if (meta == "None") r.text.clear();
    }
};

} // namespace qwen3asr

#endif // MODELS_QWEN3ASR_HPP
