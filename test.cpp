#include <iostream>
#include "kernels/interface.hpp"



class AddKernel:public Kernel<DeviceType::kCPU, float*, int, int*, int*, float*, int*, float*> {
    public:

    void call(float* a, int a_ndim,  int* a_shape, int* a_strides, float* b, int* b_strides, float* out) override {
        
        int offseta = 0;
        int offsetb = 0;

        for(int i = a_ndim - 1; i >=0; i--){
            for(int j = 0; j < a_shape[i]; j++){
                out[offseta] = a[offseta] + b[offsetb];
                offseta += a_strides[i];
                offsetb += b_strides[i];
            }
            offseta -= a_shape[i] * a_strides[i];
            offsetb -= a_shape[i] * b_strides[i];
        }
    }

    
};




int main(){

    KernelBufferAllocator<DeviceType::kCPU, int, float> allocator({2,3});

    auto r = allocator.getPointer<1>();

    std::cout << "Pointer at index 1: " << r << std::endl;
    // allocator.getPointer<1>() = new float[3]{1.0f, 2.0f, 3.0f};
    std::cout << "Values at pointer 1: ";
    allocator.Allocate();
    for(int i = 0; i < 3; i++){
        std::cout << allocator.getPointer<1>()[i] << " ";
    }
    std::cout << std::endl;

    float* a = new float[3]{0,1,2};
    float* b = new float[3]{0,1,2};
    int* ashape = new int[2]{3,3};
    int* astrides = new int[2]{3,1};
    int* bstrides = new int[2]{1,0};
    
    AddKernel kernel;

    float* c = new float[9];
    kernel(a, 1, ashape, astrides, b, bstrides, c);

    std::cout << "Result of addition: ";
    for(int i = 0; i < 9; i++){
        std::cout << c[i] << " ";
    }
    std::cout << std::endl; 
}