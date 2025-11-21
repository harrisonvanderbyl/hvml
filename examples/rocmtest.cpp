#include "shape.hpp"
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "dtypes/complex32.hpp"
#include "file_loaders/safetensors.hpp"
// #include "file_loaders/gltf.hpp"
#include <ops/ops.h>
#include <string>
#include "display/display.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/rwkv7.hpp"
#include "module/layernorm/layernorm.hpp"


int main(){


    

    auto layernorm = LayerNorm<float>(1024, DeviceType::kHIP);
    layernorm.weight = 1.0f;
    layernorm.bias = 0.0f;
    // std::cout << layernorm << std::endl;

    Tensor<float,2> input({2,1024}, DeviceType::kCPU);
    for(int i = 0; i < input.total_size; i++){
        input.flatget(i) = (rand() % 10000) / 10000.0f;
    }

    auto input_cuda = input.to(DeviceType::kHIP);
    auto output_cuda = layernorm.forward(input_cuda);
    auto output = output_cuda.to(DeviceType::kCPU);
    std::cout << "Input: " << input << std::endl;
    std::cout << "Output: " << output << std::endl;


    auto layernorm_cpu = LayerNorm<float>(1024, DeviceType::kCPU);
    layernorm_cpu.weight = 1.0f;
    layernorm_cpu.bias = 0.0f;
    auto output_cpu = layernorm_cpu.forward(input);
    std::cout << "Output CPU: " << output_cpu << std::endl;
    
    

}