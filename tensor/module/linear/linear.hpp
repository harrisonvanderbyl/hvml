#ifndef MODULE_LINEAR_HPP
#define MODULE_LINEAR_HPP
#include "module/base/module.hpp"

template <typename T = float>
struct Linear : public Module<Tensor<T>>
{

    public:
    Tensor<T> weight;
    
    
    Linear(
        size_t in_features,
        size_t out_features,
        DeviceType device_type = DeviceType::kCPU
    ):weight({in_features,out_features}, device_type), Module<Tensor<T>>({weight, "weight"})
    {
        
    }
     
    Tensor <T> forward(Tensor<T> input)
    {
        return input;
    }
    
};

template <typename T = float>
struct FFN : public Module<Linear<T>,Linear<T>>
{
    public:
    Linear<T> key;
    Linear<T> value;
    
    FFN(
        size_t n_dim,
        size_t ffn_dim,
        DeviceType device_type = DeviceType::kCPU
    ):key(n_dim, ffn_dim, device_type), value(ffn_dim, n_dim, device_type), Module<Linear<T>,Linear<T>>({key, "key"}, {value, "value"})
    {
        
    }
     
    Tensor <T> forward(Tensor<T> input)
    {
        return value(key(input));
    }
    
};

#endif //MODULE_LINEAR_HPP