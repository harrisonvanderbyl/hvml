#ifndef FILE_LOADERS_WAV_HPP
#define FILE_LOADERS_WAV_HPP

//
//  AudioFile: a WAV file of any sample format (8/16/24/32-bit PCM, 16/32/64-bit
//  float), opened on the disk map.
//
//      AudioFile file("speech.wav");                          // read-only (kR); kRW to edit in place
//      file.sample_rate(), file.channels(), file.samples()    // from the header, in the allocation's metadata
//      auto pcm   = file.view<int16_t>();                     // the file's own format: a view of the file
//      auto audio = file.as<AudioSample<float, 16000>>(gpu);  // converted + resampled on the GPU
//
//  The allocation's CPU view starts at the samples (the WAV format put the
//  data offset in its metadata); AudioFile itself is the data as bytes.  When
//  the sample type is known in advance, a typed tensor opens the file
//  directly: Audio<int16_t, 16000> t({0, 0}, "speech.wav").
//

#include <fstream>
#include <stdexcept>
#include <string>
#include "audio/audio.hpp"

// A WAV file of any format: its data as bytes, its header parsed.
struct WavFileFormat : public FileFormat {
    const char* name() const override { return "WAV"; }
    void read(const char* file, size_t size, AllocationMetadata& meta) const override {
        WavHeader h;
        size_t at, len;
        parse_wav(file, size, h, at, len);
        size_t bytes = (size_t)h.samples * h.channels * (h.bits / 8);
        meta.data_offset = at;
        meta.shape = Shape<-1>{(long)(bytes / meta.type_size)};
        meta.byte_size = bytes;
        meta.header = std::make_shared<WavHeader>(h);
    }
    std::vector<char> write(AllocationMetadata&) const override {
        throw std::runtime_error("AudioFile opens existing WAV files; create one with a typed tensor, "
                                 "e.g. Audio<int16_t, 16000> out({samples, channels}, \"out.wav\")");
    }
    static const WavFileFormat* instance() {
        static WavFileFormat format;
        return &format;
    }
};

struct AudioFile : public Tensor<uint8_t, 1> {
    // flags: kR (default) maps the file read-only; kRW lets views of it write
    // into the file.
    explicit AudioFile(const std::string& path, AllocationFlags flags = AllocationFlags::kR)
        : Tensor<uint8_t, 1>(open(path, flags)) {}

    const WavHeader& header() const { return *this->storage_pointer->metadata.header_as<WavHeader>(); }
    int sample_rate() const { return header().sample_rate; }
    int channels() const { return header().channels; }
    long samples() const { return header().samples; }
    int bits_per_sample() const { return header().bits; }

    // Whether the file's samples are F.
    template <typename F>
    bool holds() const { return header().tag == wav_format<F>::tag && header().bits == wav_format<F>::bits; }

    // The file's samples in its own format: a {samples, channels} view.
    template <typename F>
    Tensor<AudioSample<F, 0>, 2> view() const {
        if (!holds<F>()) throw std::runtime_error("AudioFile::view: the file holds " + std::to_string(bits_per_sample()) +
                                                  "-bit " + (header().tag == 3 ? "float" : "integer") + " samples");
        using S = AudioSample<F, 0>;
        return Tensor<S, 2>(Shape<2>{samples(), (long)channels()}, this->data.template reinterpret<S>(),
                            MemoryLocation(*this->device), this->storage_pointer);
    }

    // The samples as Sample (rate 0: the file's).  A view of the file when it
    // already is that; converted in host memory otherwise.
    template <typename Sample>
    Tensor<Sample, 2> as() const { return dispatch<Sample>(nullptr); }

    // Same, on `loc`: the file's samples are moved there as they are, then
    // converted there.
    template <typename Sample>
    Tensor<Sample, 2> as(MemoryLocation loc) const { return dispatch<Sample>(&loc); }

private:
    static Tensor<uint8_t, 1> open(const std::string& path, AllocationFlags flags) {
        if (!std::ifstream(path).good()) throw std::runtime_error("AudioFile: cannot open " + path);
        MemoryLocation disk(path);
        AllocationMetadata meta = AllocationMetadata::create<uint8_t>(Shape<1>{0}, MemoryType::kDISK, ComputeType::kFILE, 0,
                                                                        flags, disk.device_id);
        meta.file_format = WavFileFormat::instance();
        return Tensor<uint8_t, 1>(meta);
    }

    template <typename Sample, typename F>
    Tensor<Sample, 2> from(const MemoryLocation* loc) const {
        // the file's samples, moved as they are
        Tensor<AudioSample<F, 0>, 2> raw = loc ? view<F>().to(*loc) : view<F>();
        int to = Sample::rate ? Sample::rate : sample_rate();
        return audio_resample<Sample>(raw, sample_rate(), to);
    }

    template <typename Sample>
    Tensor<Sample, 2> dispatch(const MemoryLocation* loc) const {
        if (holds<uint8_t>()) return from<Sample, uint8_t>(loc);
        if (holds<int16_t>()) return from<Sample, int16_t>(loc);
        if (holds<pcm24>()) return from<Sample, pcm24>(loc);
        if (holds<int32_t>()) return from<Sample, int32_t>(loc);
        if (holds<float16>()) return from<Sample, float16>(loc);
        if (holds<float>()) return from<Sample, float>(loc);
        if (holds<double>()) return from<Sample, double>(loc);
        throw std::runtime_error("AudioFile: unsupported sample format (tag " + std::to_string(header().tag) + ", " +
                                 std::to_string(bits_per_sample()) + " bits)");
    }
};

// Mono float samples at Rate Hz, on `loc`.
template <int Rate = 16000>
inline Tensor<float, 1> load_audio(const std::string& path, MemoryLocation loc = MemoryLocation(MemoryType::kDDR)) {
    return AudioFile(path).as<AudioSample<float, Rate>>(loc).mono().channel(0);
}

#endif // FILE_LOADERS_WAV_HPP
