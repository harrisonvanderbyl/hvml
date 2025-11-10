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



int main(){
    float4 a (0,0,0,0);
    std::cout << a << "\n";
    std::cout << a+1 << "\n";
    a+=1;
    std::cout << a << "\n";
    std::cout << a+a << "\n";
    a+=a;
    std::cout << a*a << "\n";
    std::cout << a/a << "\n";

    auto aa = Tensor<float4,1>({10000},kCPU);
    auto b = Tensor<float4,1>({10000},kCPU);

    auto t = time(NULL);
    for (int i = 0; i < 10000; i++){
        auto c = aa + b;
    }

    auto r = time(NULL) - t;
    std::cout << r << ": seconds";
    std::cout << t;


}