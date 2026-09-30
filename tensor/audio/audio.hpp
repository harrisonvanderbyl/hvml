#ifndef AUDIO_AUDIO_HPP
#define AUDIO_AUDIO_HPP

//
//  Audio as tensors.
//
//  AudioSample<Format, Rate>   one sample, stored as Format:
//                                uint8_t  8-bit PCM (unsigned, centred on 128)
//                                int16_t  16-bit PCM        pcm24  24-bit PCM
//                                int32_t  32-bit PCM        float16, float, double
//                              at Rate Hz (0: unspecified).  Assigning one
//                              format to another converts through [-1, 1]
//                              floats; between two different known rates it
//                              does not compile — that is resample<To>().
//
//  Tensor<AudioSample<F, Rate>, 2>   audio: {samples, channels}, interleaved
//                              as in a WAV file.  Every such tensor has
//                              mono(), resample<To>(), convert<S>(),
//                              channel(c), num_samples(), num_channels(),
//                              seconds() (TensorExtensions below).
//                              Aliases: AudioTensor<Sample>, Audio<F, Rate>.
//
//  Files: AudioSample's file format is WAV, so on the disk map
//      Audio<int16_t, 16000> out({1000, 2}, "out.wav");   // a new WAV file, mapped read/write
//      audio.to(MemoryLocation("out.wav"));                // written as a WAV file
//      Audio<int16_t, 16000> in({0, 0}, "in.wav");         // an existing one of that format
//  and AudioFile (file_loaders/wav.hpp) opens a WAV file of any format.
//  The header is in the allocation's metadata:
//      tensor.storage_pointer->metadata.header_as<WavHeader>()
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "tensor.hpp"
#include "ops/nn.hpp"
#include "bfloat16/bf16.hpp"

// 24-bit PCM: three little-endian bytes.
struct pcm24 {
    uint8_t b[3];
};

// Sample format ↔ [-1, 1].
__host__ __device__ inline float audio_decode(uint8_t v) { return ((float)v - 128.0f) * (1.0f / 128.0f); }
__host__ __device__ inline float audio_decode(int16_t v) { return (float)v * (1.0f / 32768.0f); }
__host__ __device__ inline float audio_decode(int32_t v) { return (float)v * (1.0f / 2147483648.0f); }
__host__ __device__ inline float audio_decode(float v) { return v; }
__host__ __device__ inline float audio_decode(double v) { return (float)v; }
__host__ __device__ inline float audio_decode(float16 v) { return (float)v; }
__host__ __device__ inline float audio_decode(pcm24 v) {
    int32_t x = (int32_t)v.b[0] | ((int32_t)v.b[1] << 8) | ((int32_t)(int8_t)v.b[2] << 16);
    return (float)x * (1.0f / 8388608.0f);
}

// round to nearest, clamped to [lo, hi]
__host__ __device__ inline float audio_quantise(float x, float scale, float lo, float hi) {
    float q = floorf(x * scale + 0.5f);
    return q < lo ? lo : (q > hi ? hi : q);
}

template <typename F> struct audio_encoder;
template <> struct audio_encoder<uint8_t> {
    __host__ __device__ static uint8_t encode(float x) { return (uint8_t)(audio_quantise(x, 128.0f, -128.0f, 127.0f) + 128.0f); }
};
template <> struct audio_encoder<int16_t> {
    __host__ __device__ static int16_t encode(float x) { return (int16_t)audio_quantise(x, 32768.0f, -32768.0f, 32767.0f); }
};
template <> struct audio_encoder<int32_t> {
    __host__ __device__ static int32_t encode(float x) {
        if (x >= 1.0f) return 2147483647;                       // 2^31 - 1 is not a float
        return (int32_t)audio_quantise(x, 2147483648.0f, -2147483648.0f, 2147483520.0f);
    }
};
template <> struct audio_encoder<pcm24> {
    __host__ __device__ static pcm24 encode(float x) {
        int32_t v = (int32_t)audio_quantise(x, 8388608.0f, -8388608.0f, 8388607.0f);
        pcm24 p;
        p.b[0] = (uint8_t)(v & 0xFF);
        p.b[1] = (uint8_t)((v >> 8) & 0xFF);
        p.b[2] = (uint8_t)((v >> 16) & 0xFF);
        return p;
    }
};
template <> struct audio_encoder<float> {
    __host__ __device__ static float encode(float x) { return x; }
};
template <> struct audio_encoder<double> {
    __host__ __device__ static double encode(float x) { return (double)x; }
};
template <> struct audio_encoder<float16> {
    __host__ __device__ static float16 encode(float x) { return float16(x); }
};

template <typename Format, int Rate = 0>
struct AudioSample {
    using format = Format;
    static constexpr int rate = Rate;
    Format value;

    __host__ __device__ AudioSample() {}
    // from a [-1, 1] float
    __host__ __device__ AudioSample(float x) : value(audio_encoder<Format>::encode(x)) {}
    // another format (same rate, or an unspecified one): through [-1, 1]
    template <typename Other, int OtherRate>
    __host__ __device__ AudioSample(const AudioSample<Other, OtherRate>& o) : value(audio_encoder<Format>::encode(o.to_float())) {
        static_assert(Rate == 0 || OtherRate == 0 || Rate == OtherRate,
                      "different sample rates: resample<Rate>() the audio instead of converting samples");
    }

    __host__ __device__ float to_float() const { return audio_decode(value); }

    // files of AudioSample tensors are WAV files (defined below)
    static const FileFormat* file_format();
};

// ---------------------------------------------------------------------------
//  Audio functions on plain tensors (defined below)
// ---------------------------------------------------------------------------
template <typename S2, typename S1> Tensor<S2, 2> audio_relabel(const Tensor<S1, 2>& x);
template <typename S2, typename S1> Tensor<S2, 2> audio_convert(const Tensor<S1, 2>& x);
template <int R> Tensor<float, 1> audio_channel(const Tensor<AudioSample<float, R>, 2>& x, long c);
template <typename SOut, typename SIn> Tensor<SOut, 2> audio_resample(const Tensor<SIn, 2>& x, int from, int to);
template <typename S> Tensor<AudioSample<float, S::rate>, 2> audio_mono(const Tensor<S, 2>& x);

// ---------------------------------------------------------------------------
//  Member functions of every Tensor<AudioSample<F, Rate>, 2>
// ---------------------------------------------------------------------------

template <typename F, int Rate>
struct TensorExtensions<Tensor<AudioSample<F, Rate>, 2>, AudioSample<F, Rate>, 2> {
    using Sample = AudioSample<F, Rate>;
    using Self = Tensor<Sample, 2>;

    static constexpr int sample_rate() { return Rate; }
    long num_samples() const { return self().shape[0]; }
    long num_channels() const { return self().shape[1]; }
    double seconds() const {
        static_assert(Rate != 0, "seconds(): the rate is not part of this tensor's type");
        return (double)num_samples() / Rate;
    }

    // another sample format, same rate
    template <typename S2>
    Tensor<S2, 2> convert() const { return audio_convert<S2>(self()); }

    // another rate, same format (from or to an unspecified rate: relabelled)
    template <int To>
    Tensor<AudioSample<F, To>, 2> resample() const {
        if constexpr (To == Rate) return self();
        else if constexpr (Rate == 0 || To == 0) return audio_relabel<AudioSample<F, To>>(self());
        else return audio_resample<AudioSample<F, To>>(self(), Rate, To);
    }

    // channels averaged: {samples, 1} floats
    Tensor<AudioSample<float, Rate>, 2> mono() const { return audio_mono(self()); }

    // channel c as plain floats (float samples): a view
    Tensor<float, 1> channel(long c) const {
        static_assert(std::is_same_v<F, float>, "channel(): float samples only (convert<AudioSample<float, Rate>>() first)");
        return audio_channel(self(), c);
    }

private:
    const Self& self() const { return static_cast<const Self&>(*this); }
};

// ---------------------------------------------------------------------------
//  WAV files
// ---------------------------------------------------------------------------

struct WavHeader : public FileHeader {
    int tag = 0;               // 1: integer PCM, 3: IEEE float
    int channels = 0;
    int sample_rate = 0;
    int bits = 0;
    long samples = 0;          // per channel
};

// fmt and data chunks of a RIFF/WAVE file
inline void parse_wav(const char* buf, size_t size, WavHeader& h, size_t& data_at, size_t& data_len) {
    auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, buf + o, 2); return v; };
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, buf + o, 4); return v; };
    if (size < 12 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) throw std::runtime_error("not a RIFF/WAVE file");
    data_at = data_len = 0;
    for (size_t at = 12; at + 8 <= size;) {
        uint32_t len = u32(at + 4);
        if (memcmp(buf + at, "fmt ", 4) == 0) {
            h.tag = u16(at + 8);
            h.channels = u16(at + 10);
            h.sample_rate = (int)u32(at + 12);
            h.bits = u16(at + 22);
            if (h.tag == 0xFFFE && len >= 26) h.tag = u16(at + 32);   // WAVE_FORMAT_EXTENSIBLE
        } else if (memcmp(buf + at, "data", 4) == 0) {
            data_at = at + 8;
            data_len = std::min<size_t>(len, size - data_at);
            break;
        }
        at += 8 + len + (len & 1);
    }
    if (!data_at || !h.channels || !h.bits || h.bits % 8) throw std::runtime_error("missing or unsupported fmt/data chunk");
    h.samples = (long)(data_len / ((size_t)h.channels * (h.bits / 8)));
}

// WAV format tag and bits of a sample format
template <typename F> struct wav_format;
template <> struct wav_format<uint8_t> { static constexpr int tag = 1, bits = 8; };
template <> struct wav_format<int16_t> { static constexpr int tag = 1, bits = 16; };
template <> struct wav_format<pcm24> { static constexpr int tag = 1, bits = 24; };
template <> struct wav_format<int32_t> { static constexpr int tag = 1, bits = 32; };
template <> struct wav_format<float16> { static constexpr int tag = 3, bits = 16; };
template <> struct wav_format<float> { static constexpr int tag = 3, bits = 32; };
template <> struct wav_format<double> { static constexpr int tag = 3, bits = 64; };

// A WAV file of Sample, as a {samples, channels} tensor.
template <typename Sample>
struct WavFormat : public FileFormat {
    using F = typename Sample::format;
    static constexpr int Rate = Sample::rate;

    const char* name() const override { return "WAV"; }

    void read(const char* file, size_t size, AllocationMetadata& meta) const override {
        WavHeader h;
        size_t at, len;
        parse_wav(file, size, h, at, len);
        if (h.tag != wav_format<F>::tag || h.bits != wav_format<F>::bits)
            throw std::runtime_error("holds " + std::to_string(h.bits) + "-bit " + (h.tag == 3 ? "float" : "integer") +
                                     " samples, not this sample type (AudioFile opens any format)");
        if (Rate != 0 && h.sample_rate != Rate)
            throw std::runtime_error("is " + std::to_string(h.sample_rate) + " Hz, not " + std::to_string(Rate));
        meta.data_offset = at;
        meta.shape = Shape<-1>{h.samples, (long)h.channels};
        meta.byte_size = (size_t)h.samples * h.channels * sizeof(Sample);
        meta.header = std::make_shared<WavHeader>(h);
    }

    std::vector<char> write(AllocationMetadata& meta) const override {
        if (Rate == 0) throw std::runtime_error("a new WAV file needs the sample rate in the type: AudioSample<F, Rate>");
        if (meta.shape.ndim() != 2) throw std::runtime_error("a WAV file is a {samples, channels} tensor");
        WavHeader h;
        h.tag = wav_format<F>::tag;
        h.bits = wav_format<F>::bits;
        h.channels = (int)meta.shape[1];
        h.sample_rate = Rate;
        h.samples = meta.shape[0];
        unsigned long long data_bytes = (unsigned long long)h.samples * h.channels * sizeof(Sample);
        if (data_bytes + 36 > 0xFFFFFFFFull) throw std::runtime_error("WAV files are limited to 4 GiB");

        std::vector<char> b(44);
        auto put16 = [&](int at, uint16_t v) { memcpy(b.data() + at, &v, 2); };
        auto put32 = [&](int at, uint32_t v) { memcpy(b.data() + at, &v, 4); };
        uint32_t block = (uint32_t)(h.channels * sizeof(Sample));
        memcpy(b.data(), "RIFF", 4);
        put32(4, (uint32_t)(36 + data_bytes));
        memcpy(b.data() + 8, "WAVEfmt ", 8);
        put32(16, 16);
        put16(20, (uint16_t)h.tag);
        put16(22, (uint16_t)h.channels);
        put32(24, (uint32_t)Rate);
        put32(28, (uint32_t)Rate * block);
        put16(32, (uint16_t)block);
        put16(34, (uint16_t)h.bits);
        memcpy(b.data() + 36, "data", 4);
        put32(40, (uint32_t)data_bytes);
        meta.data_offset = b.size();
        meta.header = std::make_shared<WavHeader>(h);
        return b;
    }

    int fill_byte() const override { return std::is_same_v<F, uint8_t> ? 128 : 0; }   // 8-bit silence

    static const WavFormat* instance() {
        static WavFormat format;
        return &format;
    }
};

template <typename Format, int Rate>
const FileFormat* AudioSample<Format, Rate>::file_format() { return WavFormat<AudioSample<Format, Rate>>::instance(); }

// ---------------------------------------------------------------------------
//  Band-limited resampling of one float signal (Kaiser-windowed sinc, 64
//  taps).  For to / from = L / M, output i reads the 64 inputs around i·M/L
//  with weights that depend only on i mod L, so each phase p is one
//  DotProduct of a strided view of the signal (window k starts M samples
//  after window k - 1) with that phase's weights, written to every L-th
//  output.  x may be strided; the result is on x's device.
// ---------------------------------------------------------------------------
inline Tensor<float, 1> resample_signal(const Tensor<float, 1>& x, int from, int to) {
    const long half = 32, taps = 2 * half;
    const double beta = 8.6;
    long g = std::gcd(from, to);
    long L = to / g, M = from / g;
    long N = x.shape[0];
    long n_out = (long)std::floor((double)N * to / from);
    MemoryLocation loc = working_location(x);

    auto bessel_i0 = [](double v) {
        double sum = 1, term = 1;
        for (int k = 1; k < 30; k++) { term *= (v / (2 * k)) * (v / (2 * k)); sum += term; }
        return sum;
    };
    double cutoff = std::min(1.0, (double)to / from) * 0.95;
    double i0b = bessel_i0(beta);
    std::vector<float> w(L * taps);
    std::vector<double> row(taps);
    for (long p = 0; p < L; p++) {
        double t = (double)p * M / L;
        double frac = t - std::floor(t);
        double wsum = 0;
        for (long j = 0; j < taps; j++) {
            double d = frac + half - 1 - j;           // t - (floor(t) - half + 1 + j)
            double sinc = d == 0 ? 1.0 : std::sin(M_PI * d * cutoff) / (M_PI * d * cutoff);
            double r = d / half;
            double win = std::fabs(r) >= 1 ? 0 : bessel_i0(beta * std::sqrt(1 - r * r)) / i0b;
            row[j] = sinc * win;
            wsum += row[j];
        }
        for (long j = 0; j < taps; j++) w[p * taps + j] = (float)(wsum != 0 ? row[j] / wsum : 0);
    }
    Tensor<float, 2> weights = tensor_from_host(Shape<2>{L, taps}, w.data(), loc);

    // zero-padded signal: input sample n is padded[n + half]
    Tensor<float, 1> padded(Shape<1>{N + 2 * half}, loc);
    padded = 0.0f;
    padded[{{half, half + N}}] = x;

    Tensor<float, 1> out(Shape<1>{n_out}, loc);
    for (long p = 0; p < L && p < n_out; p++) {
        long K = (n_out - p + L - 1) / L;             // outputs p, p + L, ...
        long start = (p * M) / L + 1;                 // padded index of window 0
        Tensor<float, 2> windows(Shape<2>{K, taps}, padded.data + (size_t)start, padded.location(), padded.storage_pointer);
        windows.strides = Shape<2>{M, 1};
        Tensor<float, 1> r = DotProduct<-1>::run(windows, weights[{{p, p + 1}}]);
        out[{{p, n_out, L}}] = r;
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Audio on plain tensors: {samples, channels} of AudioSample<...>
// ---------------------------------------------------------------------------

// Same samples, another rate label (same format): a view.
template <typename S2, typename S1>
Tensor<S2, 2> audio_relabel(const Tensor<S1, 2>& x) {
    static_assert(std::is_same_v<typename S1::format, typename S2::format>, "audio_relabel: same sample format only");
    Tensor<S2, 2> t(x.shape, x.data.template reinterpret<S2>(), MemoryLocation(*x.device), x.storage_pointer);
    t.strides = x.strides;
    return t;
}

// Another sample format (element-wise, on x's device; disk → host memory).
// Same format with another rate label is a view.
template <typename S2, typename S1>
Tensor<S2, 2> audio_convert(const Tensor<S1, 2>& x) {
    if constexpr (std::is_same_v<S1, S2>) {
        return x;
    } else if constexpr (std::is_same_v<typename S1::format, typename S2::format>) {
        return audio_relabel<S2>(x);
    } else {
        Tensor<S2, 2> out(x.shape, x.scratch_location());
        out = x;
        return out;
    }
}

// Channel c of float samples as plain floats: a view.
template <int R>
Tensor<float, 1> audio_channel(const Tensor<AudioSample<float, R>, 2>& x, long c) {
    Tensor<float, 1> t(Shape<1>{x.shape[0]}, x.data.template reinterpret<float>() + (size_t)(c * x.strides[1]),
                       MemoryLocation(*x.device), x.storage_pointer);
    t.strides = Shape<1>{x.strides[0]};
    return t;
}

// from → to Hz, into another sample type if wanted (resampled as floats;
// from == to is a conversion only, a view when the format is the same).
template <typename SOut, typename SIn>
Tensor<SOut, 2> audio_resample(const Tensor<SIn, 2>& x, int from, int to) {
    using Unrated = AudioSample<typename SIn::format, 0>;
    using Float = AudioSample<float, 0>;
    Tensor<Unrated, 2> u = audio_relabel<Unrated>(x);
    if (from == to) return audio_convert<SOut>(u);
    Tensor<Float, 2> f = audio_convert<Float>(u);
    long n_out = (long)std::floor((double)f.shape[0] * to / from);
    Tensor<Float, 2> out(Shape<2>{n_out, f.shape[1]}, f.scratch_location());
    for (long c = 0; c < f.shape[1]; c++) {
        Tensor<float, 1> column = audio_channel(out, c);
        column = resample_signal(audio_channel(f, c), from, to);
    }
    return audio_convert<SOut>(out);
}

// Channels averaged: {samples, 1} floats.
template <typename S>
Tensor<AudioSample<float, S::rate>, 2> audio_mono(const Tensor<S, 2>& x) {
    using Float = AudioSample<float, S::rate>;
    Tensor<Float, 2> f = audio_convert<Float>(x);
    if (f.shape[1] == 1) return f;
    Tensor<float, 2> all(f.shape, f.data.template reinterpret<float>(), MemoryLocation(*f.device), f.storage_pointer);
    all.strides = f.strides;
    Tensor<float, 1> m = ReduceSum<1>::run(all);
    m *= 1.0f / f.shape[1];
    return Tensor<Float, 2>(Shape<2>{f.shape[0], 1}, m.data.template reinterpret<Float>(), m.location(), m.storage_pointer);
}

template <typename Sample>
using AudioTensor = Tensor<Sample, 2>;
template <typename F, int Rate = 0>
using Audio = Tensor<AudioSample<F, Rate>, 2>;

static_assert(sizeof(Audio<float, 16000>) == sizeof(Tensor<float, 2>), "audio functions add no state to Tensor");

// WAV files of any format: AudioFile
#include "file_loaders/wav.hpp"

#endif // AUDIO_AUDIO_HPP
