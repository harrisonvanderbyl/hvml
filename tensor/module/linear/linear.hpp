#ifndef MODULE_LINEAR_HPP
#define MODULE_LINEAR_HPP
#include "module/base/module.hpp"
#include "ops/ops.h"
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
    auto forward(Tensor<T,rank> input)
    {
        Tensor<T,3> reshaped_input = input.view(Shape<3>{input.shape[0], input.shape[1], 1});
        std::cout << "Reshaped input shape: " << reshaped_input.shape << std::endl;
        Tensor<T,3> reshaped_matrix = this->weight.view(Shape<3>{1, weight.shape[0], weight.shape[1]});


        auto output = DotProduct<-2>::run(reshaped_matrix,reshaped_input);
        
        
        return output;
    }
    
};


#endif //MODULE_LINEAR_HPP