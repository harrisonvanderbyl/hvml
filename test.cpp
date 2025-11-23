#include <iostream>
#include "kernels/interface.hpp"
#include "tensor.hpp"
#include "ops/ops.h"




int main(){

    Tensor<double,2> a({1,3}, DeviceType::kCPU);
    Tensor<float,2> b({3,1}, DeviceType::kCPU);
    for(int i = 0; i < 3; i++){
        a[{0,i}] = i+1;
        b[{i,0}] = (i+1)*10;
    }
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << a+b << std::endl;
    std::cout << a*b << std::endl;
    std::cout << a/b << std::endl;
    std::cout << a-b << std::endl;

}