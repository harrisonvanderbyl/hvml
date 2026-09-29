
### Modules

Modules allow for the reuse of code, specifically around saving and loading weights.

```cpp


#include "tensor.hpp"
#include "kernels/interface.hpp"
#include "module/base/module.hpp"

struct SubModule: public Module<Tensor<float, 1>> {
public:
    Tensor<float,1> weight;
    SubModule(int size): weight({size}, kCPU), Module({
        weight, "weight"
    }) {
        weight = 1.0f;
    }

    Tensor<float,1> forward(Tensor<float,1> input) {
        return input * weight;
    }
};

struct ModuleExample: public Module<SubModule, Tensor<float,1>> {
public:
    SubModule submod;
    Tensor<float,1> bias;
    ModuleExample(int size)
    : submod(size),
        bias({size}, kCPU),
        Module(
        {
            submod, "submod"
        }, {
            bias, "bias"
        }) {
            bias = 0.5f;
        }

    Tensor<float,1> forward(Tensor<float,1> input) {
        return submod.forward(input) + bias;
    }
};

int main() {
    ModuleExample mod(4);
    std::cout << mod << std::endl;

    Tensor<float,1> input({4}, kCPU);
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    input[3] = 4.0f;

    auto output = mod.forward(input);
    std::cout << "Output: " << output << std::endl;

    safetensors st = mod.to_safetensors();
    st.save("module_example.safetensors");

    std::cout << "Saved module to module_example.safetensors" << std::endl;

    mod.load_from_safetensors("module_example.safetensors");
    std::cout << "Loaded module from module_example.safetensors" << std::endl;

    /*
        (
        submod: (
        weight: (dtype=32bit float, shape=[4], strides=[1], device_type=CPU)[1, 1, 1, 1]
        )
        bias: (dtype=32bit float, shape=[4], strides=[1], device_type=CPU)[0.5, 0.5, 0.5, 0.5]
        )
        Output: (dtype=32bit float, shape=[4], strides=[1], device_type=CPU)[1.5, 2.5, 3.5, 4.5]
        Saved submod.weight to safetensors
        Saved bias to safetensors
        Saved module to module_example.safetensors
        Loaded module from module_example.safetensors
    */

    return 0;
}
```
### Layer lists and devices

`ModuleList<T>` holds numbered submodules, loaded from `name.0.`, `name.1.`,
…  The items are heap-allocated because a module keeps pointers to its own
members:

```cpp
ModuleList<DecoderLayer> layers(28, [&](size_t i) { return new DecoderLayer(config); });
```

`move_to(MemoryLocation)` moves every tensor of a module (recursively) to a
device, e.g. after loading weights from disk:

```cpp
model.load_from_safetensors(safetensors("model.safetensors"));
model.move_to(MemoryType::kHIP_VRAM);
```

Reusable modules: `Linear<W, has_bias>` (`module/linear`), `LayerNorm<W>`,
`RMSNorm<W>`, `Embedding<W>` and `Conv2d<W>`. `W` is the weight dtype as
stored (e.g. `bfloat16`); activations are `float`. See `docs/kernel.md` for
the operations they are built on, and `docs/qwen3asr.md` for a complete model.
