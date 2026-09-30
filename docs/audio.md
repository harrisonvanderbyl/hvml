# Audio

Audio is an ordinary tensor: `Tensor<AudioSample<Format, Rate>, 2>`, shape
`{samples, channels}` (interleaved, as in a WAV file). Headers:
`tensor/audio/audio.hpp` (types, WAV format) and `tensor/file_loaders/wav.hpp`
(`AudioFile`).

```cpp
AudioFile file("speech.wav");                                // any WAV, read-only on the disk map
auto audio = file.as<AudioSample<float, 16000>>(gpu);        // converted + resampled on the GPU
auto low   = audio.resample<8000>().convert<AudioSample<int16_t, 8000>>();

Audio<int16_t, 8000> out({low.num_samples(), low.num_channels()}, "out.wav");   // a new WAV file
out = low;                                                   // written through the mapping
low.to(MemoryLocation("copy.wav"));                          // or copied to a new file
```

## Samples

`AudioSample<Format, Rate>` is one sample. Formats: `uint8_t` (8-bit PCM,
centred on 128), `int16_t`, `pcm24` (three bytes), `int32_t`, `float16`,
`float`, `double`. `Rate` is the sample rate in Hz, or 0 for unspecified.

Converting one sample format to another goes through [-1, 1] floats (round to
nearest, clamped), so assigning a tensor of one format to another converts
element-wise on any backend. Converting between two different known rates
does not compile: changing the rate is `resample<To>()`.

## Audio tensors

Every `Tensor<AudioSample<F, Rate>, 2>` has these members (from
`TensorExtensions`, so slices, op results and plain tensors all have them,
and they add no state):

| member | |
|---|---|
| `num_samples()`, `num_channels()`, `sample_rate()`, `seconds()` | shape and rate (`seconds()` needs a rate) |
| `convert<S>()` | another sample format, same rate (a view when only the rate label differs) |
| `resample<To>()` | another rate: Kaiser-windowed sinc, 64 taps, one strided `DotProduct` per phase of the rate ratio; from or to rate 0 it relabels |
| `mono()` | channels averaged, `{samples, 1}` floats |
| `channel(c)` | channel c of float samples as a `Tensor<float, 1>` view |

`AudioTensor<Sample>` is `Tensor<Sample, 2>` and `Audio<F, Rate>` is
`Tensor<AudioSample<F, Rate>, 2>`. Assigning audio that lives on another
device brings it over first, so `file_tensor = gpu_audio` works.

## WAV files

`AudioSample`'s file format is WAV, so on the disk map an audio tensor is a
WAV file:

- `Audio<F, Rate>({samples, channels}, "out.wav")` creates one (replacing an
  existing file of another shape or format): header for F at Rate, samples
  mapped read/write, silent to start.
- `Audio<F, Rate>({0, 0}, "in.wav")` opens an existing one of exactly that
  format and rate (and the shape it has); another format or rate throws.
- `audio.to(MemoryLocation("out.wav"))` writes a copy.
- `AudioFile("in.wav")` opens a WAV file of any format: `sample_rate()`,
  `channels()`, `samples()`, `bits_per_sample()`, `header()`;
  `view<F>()` is the samples in the file's own format (a view of the file);
  `as<Sample>()` is a view when the file already is that, a conversion
  otherwise; `as<Sample>(loc)` moves the raw samples to `loc` and converts
  there. `AudioFile(path, AllocationFlags::kRW)` edits the file in place.
- `load_audio<Rate>(path, loc)` gives mono floats as a `Tensor<float, 1>`.

## File formats on the disk map

A disk allocation can be a file with a header. `AllocationMetadata` carries
`file_format` (a `FileFormat`: parse a header, write one), `header` (the
parsed header) and `data_offset`. The disk allocator calls the format to read
or write the header, and the CPU view of the allocation starts at
`data_offset`, so the tensor's data is the file's data:

```cpp
const WavHeader* h = tensor.storage_pointer->metadata.header_as<WavHeader>();
```

An element type picks the format of its files with
`static const FileFormat* file_format()` (AudioSample: WAV); a loader can set
`meta.file_format` itself (AudioFile for any WAV, `safetensors` for its
8-byte length + JSON header). Flags: `kR` opens an existing file read-only
(`rb`, mapped `PROT_READ`); `kRW` opens it for writing, creating the file when
it does not exist or does not match the requested shape. A shape with no
elements means "open, shape from the file".
