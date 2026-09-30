#ifndef MODELS_QWEN3ASR_MEL_HPP
#define MODELS_QWEN3ASR_MEL_HPP

//
//  Whisper-style log-mel spectrogram (WhisperFeatureExtractor):
//  16 kHz, n_fft 400, hop 160, periodic Hann window, centred reflect padding,
//  power spectrum → Slaney mel filterbank (0–8 kHz) → log10(max(x, 1e-10))
//  → max(x, max - 8) → (x + 4) / 4.   Output: [frames, n_mels], frames =
//  samples / 160.  The spectrogram runs on the device the tables live on.
//

#include <vector>
#include <cmath>
#include "tensor.hpp"
#include "ops/nn.hpp"
#include "module/base/reflist.hpp"

struct LogMel {
    long n_mels;
    long n_fft = 400;
    long hop = 160;
    long sample_rate = 16000;

    Tensor<float, 1> window;                  // [n_fft]
    Tensor<float, 2> dft_cos, dft_sin;        // [n_fft / 2 + 1, n_fft]
    Tensor<float, 2> filters;                 // [n_mels, n_fft / 2 + 1]

    static double hz_to_mel(double f) {               // Slaney
        const double min_log_hz = 1000.0, min_log_mel = 15.0, logstep = 27.0 / std::log(6.4);
        return f < min_log_hz ? 3.0 * f / 200.0 : min_log_mel + std::log(f / min_log_hz) * logstep;
    }
    static double mel_to_hz(double m) {
        const double min_log_hz = 1000.0, min_log_mel = 15.0, logstep = std::log(6.4) / 27.0;
        return m < min_log_mel ? 200.0 * m / 3.0 : min_log_hz * std::exp(logstep * (m - min_log_mel));
    }

    explicit LogMel(long n_mels = 128) : n_mels(n_mels) {
        long bins = n_fft / 2 + 1;
        MemoryLocation host(MemoryType::kDDR);

        std::vector<float> w(n_fft), c(bins * n_fft), s(bins * n_fft);
        for (long n = 0; n < n_fft; n++) w[n] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * n / n_fft));
        for (long k = 0; k < bins; k++) {
            for (long n = 0; n < n_fft; n++) {
                double a = 2.0 * M_PI * (double)((k * n) % n_fft) / n_fft;
                c[k * n_fft + n] = (float)std::cos(a);
                s[k * n_fft + n] = (float)std::sin(a);
            }
        }
        window = tensor_from_host(Shape<1>{n_fft}, w.data(), host);
        dft_cos = tensor_from_host(Shape<2>{bins, n_fft}, c.data(), host);
        dft_sin = tensor_from_host(Shape<2>{bins, n_fft}, s.data(), host);

        // Triangular Slaney filters, normalised to constant energy per band
        std::vector<double> mel_f(n_mels + 2), fft_f(bins);
        double mmin = hz_to_mel(0.0), mmax = hz_to_mel(sample_rate / 2.0);
        for (long i = 0; i < n_mels + 2; i++) mel_f[i] = mel_to_hz(mmin + (mmax - mmin) * i / (n_mels + 1));
        for (long k = 0; k < bins; k++) fft_f[k] = (sample_rate / 2.0) * k / (bins - 1);
        std::vector<float> fb(n_mels * bins);
        for (long m = 0; m < n_mels; m++) {
            double enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
            for (long k = 0; k < bins; k++) {
                double down = (fft_f[k] - mel_f[m]) / (mel_f[m + 1] - mel_f[m]);
                double up = (mel_f[m + 2] - fft_f[k]) / (mel_f[m + 2] - mel_f[m + 1]);
                fb[m * bins + k] = (float)(std::max(0.0, std::min(down, up)) * enorm);
            }
        }
        filters = tensor_from_host(Shape<2>{n_mels, bins}, fb.data(), host);
    }

    void move_to(MemoryLocation loc) {
        move_tensor_to(window, loc);
        move_tensor_to(dft_cos, loc);
        move_tensor_to(dft_sin, loc);
        move_tensor_to(filters, loc);
    }

    long frames(size_t samples) const { return (long)samples / hop; }

    // samples: 16 kHz mono → [frames, n_mels] on the tables' device
    Tensor<float, 2> compute(const std::vector<float>& samples) const {
        long N = (long)samples.size();
        long pad = n_fft / 2;
        long F = frames(samples.size());
        if (F <= 0) throw std::runtime_error("LogMel: audio shorter than one hop");
        if (N <= pad) throw std::runtime_error("LogMel: audio too short for reflect padding");

        // numpy-style reflect padding (edge sample not repeated)
        std::vector<float> padded(N + 2 * pad);
        for (long i = 0; i < pad; i++) padded[i] = samples[pad - i];
        for (long i = 0; i < N; i++) padded[pad + i] = samples[i];
        for (long i = 0; i < pad; i++) padded[pad + N + i] = samples[N - 2 - i];

        MemoryLocation loc = working_location(window);
        auto signal = tensor_from_host(Shape<1>{(long)padded.size()}, padded.data(), loc);

        // Frame f is signal[f·hop, f·hop + n_fft).  With the signal viewed as
        // rows of `hop` samples, columns [j·hop, (j+1)·hop) of the frames are
        // rows j .. j+F of that view.
        long pieces = (n_fft + hop - 1) / hop;
        long rows = F + pieces - 1;
        Tensor<float, 2> hops = signal[{{0, rows * hop}}].view(Shape<2>{rows, hop});
        Tensor<float, 2> frames(Shape<2>{F, n_fft}, loc);
        for (long j = 0; j < pieces; j++) {
            long c0 = j * hop, c1 = std::min(c0 + hop, n_fft);
            frames[{{}, {c0, c1}}] = hops[{{j, j + F}, {0, c1 - c0}}];
        }
        frames *= window.unsqueeze(0);

        Tensor<float, 2> power = OpPower::run(linear(frames, dft_cos), linear(frames, dft_sin));
        Tensor<float, 2> log_mel = OpLog10Clamp::run(linear(power, filters));
        Tensor<float, 2> top = row_max(log_mel.view(Shape<2>{1, F * n_mels}));   // [1, 1]
        return OpWhisperNorm::run(log_mel, top);
    }
};

#endif // MODELS_QWEN3ASR_MEL_HPP
