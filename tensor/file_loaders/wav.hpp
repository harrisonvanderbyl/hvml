#ifndef FILE_LOADERS_WAV_HPP
#define FILE_LOADERS_WAV_HPP

//
//  Minimal WAV reader: PCM 8/16/24/32-bit and IEEE float 32/64, any channel
//  count (mixed to mono), resampled to a target rate with a windowed-sinc
//  filter.  Returns float samples in [-1, 1].
//

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct WavAudio {
    std::vector<float> samples;   // mono
    int sample_rate = 0;
};

inline WavAudio read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_wav: cannot open " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, &buf[o], 2); return v; };
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, &buf[o], 4); return v; };
    if (buf.size() < 12 || memcmp(&buf[0], "RIFF", 4) != 0 || memcmp(&buf[8], "WAVE", 4) != 0) {
        throw std::runtime_error("read_wav: not a RIFF/WAVE file: " + path);
    }

    int format = 0, channels = 0, bits = 0, rate = 0;
    size_t data_at = 0, data_len = 0;
    for (size_t o = 12; o + 8 <= buf.size();) {
        uint32_t len = u32(o + 4);
        if (memcmp(&buf[o], "fmt ", 4) == 0) {
            format = u16(o + 8);
            channels = u16(o + 10);
            rate = (int)u32(o + 12);
            bits = u16(o + 22);
            if (format == 0xFFFE && len >= 26) format = u16(o + 32);   // WAVE_FORMAT_EXTENSIBLE sub-format
        } else if (memcmp(&buf[o], "data", 4) == 0) {
            data_at = o + 8;
            data_len = std::min<size_t>(len, buf.size() - data_at);
        }
        o += 8 + len + (len & 1);
    }
    if (!data_at || !channels || !bits) throw std::runtime_error("read_wav: missing fmt/data chunk");

    size_t bytes = bits / 8, frames = data_len / (bytes * channels);
    WavAudio out;
    out.sample_rate = rate;
    out.samples.resize(frames);
    for (size_t i = 0; i < frames; i++) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
            const char* p = &buf[data_at + (i * channels + c) * bytes];
            double v = 0;
            if (format == 3 && bits == 32) { float x; memcpy(&x, p, 4); v = x; }
            else if (format == 3 && bits == 64) { double x; memcpy(&x, p, 8); v = x; }
            else if (bits == 8) v = ((unsigned char)p[0] - 128) / 128.0;
            else if (bits == 16) { int16_t x; memcpy(&x, p, 2); v = x / 32768.0; }
            else if (bits == 24) {
                int32_t x = ((unsigned char)p[0]) | ((unsigned char)p[1] << 8) | ((signed char)p[2] << 16);
                v = x / 8388608.0;
            } else if (bits == 32) { int32_t x; memcpy(&x, p, 4); v = x / 2147483648.0; }
            else throw std::runtime_error("read_wav: unsupported sample format");
            acc += v;
        }
        out.samples[i] = (float)(acc / channels);
    }
    return out;
}

// Band-limited resampling (Kaiser-windowed sinc).
inline std::vector<float> resample(const std::vector<float>& in, int from, int to) {
    if (from == to || in.empty()) return in;
    double ratio = (double)to / from;
    double cutoff = std::min(1.0, ratio) * 0.95;
    const int half = 32;
    const double beta = 8.6;
    auto bessel_i0 = [](double x) {
        double sum = 1, term = 1;
        for (int k = 1; k < 30; k++) { term *= (x / (2 * k)) * (x / (2 * k)); sum += term; }
        return sum;
    };
    double i0b = bessel_i0(beta);
    size_t n_out = (size_t)std::floor(in.size() * ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; i++) {
        double t = i / ratio;
        long center = (long)std::floor(t);
        double acc = 0, wsum = 0;
        for (long k = center - half + 1; k <= center + half; k++) {
            double x = t - k;
            double sinc = x == 0 ? 1.0 : std::sin(M_PI * x * cutoff) / (M_PI * x * cutoff);
            double r = x / half;
            double win = std::fabs(r) >= 1 ? 0 : bessel_i0(beta * std::sqrt(1 - r * r)) / i0b;
            double w = sinc * win;
            wsum += w;
            if (k >= 0 && k < (long)in.size()) acc += in[k] * w;
        }
        out[i] = (float)(wsum != 0 ? acc / wsum : 0);
    }
    return out;
}

// Mono float samples at `rate` Hz.
inline std::vector<float> load_audio(const std::string& path, int rate = 16000) {
    WavAudio a = read_wav(path);
    return resample(a.samples, a.sample_rate, rate);
}

#endif // FILE_LOADERS_WAV_HPP
