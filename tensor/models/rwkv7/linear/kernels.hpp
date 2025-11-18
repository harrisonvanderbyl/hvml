

#ifndef LINEAR_KERNELS
#define LINEAR_KERNELS
#include "tensor.hpp"



template <typename R = float> 
void linear_cpu(R* input, R* weight, int in_features, int out_features, R* output, int B)
{
    for(int b = 0; b < B; b++){
        for(int o = 0; o < out_features; o++){
            R sum = 0;
            for(int i = 0; i < in_features; i++){
                sum += input[b * in_features + i] * weight[i * out_features + o];
            }
            output[b * out_features + o] = sum;
        }
    }
}

template <typename R = float> 
void linear_relu_squared_cpu(R* input, R* weight, int in_features, int out_features, R* output, int B)
{
    for(int b = 0; b < B; b++){
        for(int o = 0; o < out_features; o++){
            R sum = 0;
            for(int i = 0; i < in_features; i++){
                sum += input[b * in_features + i] * weight[i * out_features + o];
            }
            if(sum > 0){
                sum = sum * sum;
            } else {
                sum = 0;
            }
            output[b * out_features + o] = sum;
        }
    }
}

// weak linear cuda, so cuda compile can add it to the kernel
#define WEAK __attribute__((weak))
template <typename R = float>
WEAK void linear_cuda(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    throw std::runtime_error("Not compiled with CUDA support");
}

template <typename R = float>
WEAK void linear_hip(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    throw std::runtime_error("Not compiled with HIP support");
}

// #ifdef __CUDACC__
// // just use cuda memmove
// template <typename R = float>
// __global__ void linear_cuda_kernel(R* data, R* state, R* bufferdata, int B, int T, int C)
// {
//     int skipsize = (T-1)*C;
//     for (int i = 0; i < B; i++)
//     {
//         cudaMemcpyAsync(data+i*skipsize, bufferdata+i*skipsize, C*sizeof(R), cudaMemcpyDeviceToDevice);
//         cudaMemcpyAsync(data+i*skipsize-C, state+i*C, C*sizeof(R), cudaMemcpyDeviceToDevice);
//         cudaMemcpyAsync(state+i*C, bufferdata+i*skipsize, C*sizeof(R), cudaMemcpyDeviceToDevice);
//     }
// }
// #endif

// #ifdef __HIPCC__
// // just use hip memmove
// template <typename R = float>
// __global__ void linear_hip_kernel(R* data, R* state, R* bufferdata, int B, int T, int C)
// {
//     int skipsize = (T-1)*C;
//     for (int i = 0; i < B; i++)
//     {
//         hipMemcpyAsync(data+i*skipsize, bufferdata+i*skipsize, C*sizeof(R), hipMemcpyDeviceToDevice);
//         hipMemcpyAsync(data+i*skipsize-C, state+i*C, C*sizeof(R), hipMemcpyDeviceToDevice);
//         hipMemcpyAsync(state+i*C, bufferdata+i*skipsize, C*sizeof(R), hipMemcpyDeviceToDevice);
//     }
// }

// #endif



#endif