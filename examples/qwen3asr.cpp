// qwen3asr.cpp — speech recognition with Qwen3-ASR in hvml.
//
//   huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B
//
//   # CPU
//   g++ -std=c++20 -O3 -fopenmp -I./tensor examples/qwen3asr.cpp -o qwen3asr -ldl -rdynamic
//   # GPU (HIP + CUDA kernels, as for the other examples)
//   make BASEFILE=examples/qwen3asr.cpp OUTPUT=qwen3asr
//   # Vulkan (any GPU with a Vulkan driver)
//   ./vulkcc examples/qwen3asr.cpp -o qwen3asr -I./tensor -std=c++20 -O3 -fopenmp
//
//   DEVICE_PLUGIN_DIR=tensor/device/plugins ./qwen3asr Qwen3-ASR-0.6B speech.wav --device hip
//
// Options:
//   --device cpu|cuda|hip|vulkan   where the weights and activations live (default cpu)
//   --language <Name>       force the output language (e.g. English, Chinese)
//   --context <text>        system-prompt context (names, jargon, ...)
//   --max-tokens <n>        generation limit (default 512)
//   --full-attention        encoder attends over the whole clip instead of
//                           n_window_infer windows (matches transformers' sdpa path)
//   --profile               print time and operation count per stage

#include "models/qwen3asr/qwen3asr.hpp"
#include "file_loaders/wav.hpp"
#include <chrono>

__weak int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model_dir> <audio.wav> [--device cpu|cuda|hip|vulkan] "
                     "[--language L] [--context text] [--max-tokens n] [--full-attention] [--profile]\n";
        return 1;
    }
    std::string model_dir = argv[1], audio_path = argv[2], device = "cpu";
    qwen3asr::TranscribeOptions opt;
    bool full_attention = false;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--device") device = next();
        else if (a == "--language") opt.language = next();
        else if (a == "--context") opt.context = next();
        else if (a == "--max-tokens") opt.max_new_tokens = std::stol(next());
        else if (a == "--full-attention") full_attention = true;
        else if (a == "--profile") opt.profile = true;
    }

    // --device vulkan: the vulkan plugin's compute device (build with vulkcc)
    MemoryLocation loc = device == "vulkan" ? MemoryLocation(global_device_manager.get_compute_device(ComputeType::kVULKAN, 0))
                       : MemoryLocation(device == "cuda" ? MemoryType::kCUDA_VRAM
                                      : device == "hip"  ? MemoryType::kHIP_VRAM
                                                         : MemoryType::kDDR);

    auto t0 = std::chrono::steady_clock::now();
    auto model = qwen3asr::Qwen3ASR::from_pretrained(model_dir, loc);
    model->set_windowed_encoder_attention(!full_attention);
    // the WAV file (a disk tensor) → 16 kHz float samples on the model's device
    AudioFile file(audio_path);
    auto audio = file.as<AudioSample<float, 16000>>(loc);
    auto t1 = std::chrono::steady_clock::now();

    size_t printed = 0;
    opt.on_text = [&](const std::string& text) {
        // stream the raw decode (language tag included) as it grows
        if (text.size() > printed) {
            std::cerr << text.substr(printed) << std::flush;
            printed = text.size();
        }
    };
    auto result = model->transcribe(audio, opt);
    auto t2 = std::chrono::steady_clock::now();

    std::cerr << "\n";
    std::cout << "language: " << (result.language.empty() ? "(none)" : result.language) << "\n";
    std::cout << "text: " << result.text << "\n";
    std::cerr << "load " << std::chrono::duration<double>(t1 - t0).count() << "s, "
              << audio.seconds() << "s of audio transcribed in "
              << std::chrono::duration<double>(t2 - t1).count() << "s ("
              << result.tokens.size() << " tokens)\n";
    return 0;
}
