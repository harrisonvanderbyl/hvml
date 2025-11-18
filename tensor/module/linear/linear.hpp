#ifndef MODULE_LINEAR_HPP
#define MODULE_LINEAR_HPP
#include "module/base/module.hpp"
#include "models/rwkv7/linear/kernels.hpp"

enum ActivationFunction
{
    NONE,
    RELU,
    RELU_SQUARED,
    SIGMOID,
    TANH
};

template <typename T = float, ActivationFunction ACT = NONE>
struct Linear : public Module<Tensor<T,2>>
{

    public:
    Tensor<T,2> weight;
    
    
    Linear(
        size_t in_features,
        size_t out_features,
        DeviceType device_type = DeviceType::kCPU
    ):weight({in_features,out_features}, device_type), Module<Tensor<T,2>>({weight, "weight"})
    {
        
    }
     
    template <int rank = 2>
    Tensor <T,rank> forward(Tensor<T,rank> input)
    {
        Tensor<T,2> reshaped_input = input.view(Shape<2>{-1, this->weight.shape[0]});

        Shape<rank> out_shape = input.shape;
        out_shape[out_shape.ndim()-1] = this->weight.shape[1];
        Tensor<T, rank> output(out_shape, input.device_type);
        
        if constexpr (ACT == RELU_SQUARED){
            linear_relu_squared_cpu(
                input.data,
                weight.data,
                weight.shape[0],
                weight.shape[1],
                output.data,
                reshaped_input.shape[0]
            );
        }else{
            linear_cpu(
                input.data,
                weight.data,
                weight.shape[0],
                weight.shape[1],
                output.data,
                reshaped_input.shape[0]
            );
        }
        
        return output;
    }
    
};


#endif //MODULE_LINEAR_HPP