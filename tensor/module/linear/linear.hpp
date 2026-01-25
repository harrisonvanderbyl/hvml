#ifndef MODULE_LINEAR_HPP
#define MODULE_LINEAR_HPP
#include "module/base/module.hpp"
#include "ops/ops.hpp"





template <typename T = float>
struct Linear : public Module<Tensor<T,2>>
{

    public:
    Tensor<T,2> weight;
    
    
    Linear(
        size_t in_features,
        size_t out_features,
        MemoryType device_type = MemoryType::kDDR
    ):weight({out_features,in_features}, device_type), Module<Tensor<T,2>>({weight, "weight"})
    {
        
    }

    Linear(
        Tensor<T,2> weight_tensor
    ):weight(weight_tensor.shape,weight_tensor.data,weight_tensor.device_type, weight_tensor.storage_pointer, weight_tensor.original_device_type), Module<Tensor<T,2>>({weight, "weight"})
    {
        
    }
     
    template <int rank = 2>
    auto forward(Tensor<T,rank> input)
    {
        Tensor<T,3> reshaped_input = input.view(Shape<3>{input.shape[0], 1, input.shape[1]});
        Tensor<T,3> reshaped_matrix = this->weight.view(Shape<3>{1, weight.shape[0], weight.shape[1]});


        auto output = DotProduct<-1>::run(reshaped_matrix,reshaped_input);
        
        
        return output;
    }
    
};


#endif //MODULE_LINEAR_HPP