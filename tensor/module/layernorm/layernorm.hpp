#ifndef MODULE_LAYERNORM_HPP
#define MODULE_LAYERNORM_HPP
#include "module/base/module.hpp"
#include "models/rwkv7/layernorm/layernorm.cuh"


template <typename T = float>
struct LayerNorm : public Module<Tensor<T,1>,Tensor<T,1>>
{

    public:
    Tensor<T,1> weight;
    Tensor<T,1> bias;
    
    
    LayerNorm(
        size_t features,
        DeviceType device_type = MemoryType::kDDR
    ):
    weight({features}, device_type),
    bias({features}, device_type), 
    Module<Tensor<T,1>,Tensor<T,1>>({weight, "weight"}, {bias, "bias"})
    {
        
    }
     
    template <int rank = 2>
    Tensor <T,rank> forward(Tensor<T,rank> input)
    {
        Tensor<T, rank> output(input.shape, input.device_type);
        
        if(input.device_type == kHIP){
            normalize_hip(
                input.data,
                weight.data,
                bias.data,
                output.data,
                1e-5,
                input.shape[input.shape.ndim()-1],
                input.shape[input.shape.ndim()-1],
                input.shape.total_size()
            );
        
        }
        else if(input.device_type == kCUDA){
            normalize_cuda(
                input.data,
                weight.data,
                bias.data,
                output.data,
                1e-5,
                input.shape[input.shape.ndim()-1],
                input.shape[input.shape.ndim()-1],
                input.shape.total_size()
            );
        }
        else if (input.device_type == kCPU){
            // CPU implementation
            normalize_cpu(
                input.data,
                weight.data,
                bias.data,
                output.data,
                1e-5,
                input.shape[input.shape.ndim()-1],
                input.shape[input.shape.ndim()-1],
                input.shape.total_size()
            );
        }
        else{
            std::cerr << "LayerNorm only implemented for CUDA and HIP devices" << std::endl;
            throw std::runtime_error("LayerNorm only implemented for CUDA and HIP devices");
        }
        
        return output;
    }
    
};


#endif //MODULE_LAYERNORM_HPP