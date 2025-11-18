#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define __shfl_xor_sync(mask, var, laneMask) __shfl_xor(var, laneMask)

#endif
#include <cmath>

#define CUNORMTHREADS 32
#define CUNORMBLOCKS 1

#if defined(__CUDACC__) || defined(__HIPCC__)

__global__ void layernorm(float* input, float* output, float* weight, float* bias, unsigned long size, unsigned long lastshape, unsigned long headshape, float eps = 1e-5){
    unsigned long thread = threadIdx.x;
    unsigned long hh = blockIdx.x;

    auto start = hh * headshape;
    auto wb = start%lastshape;
    weight += wb;
    bias += wb;
    input += start;
    output += start;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (unsigned long i = thread%32; i < headshape; i+=32){
        sum += input[i];
        sumsq += input[i] * input[i];
    }

    for (int i = 1; i < warpSize; i *= 2)
    {
        sum += __shfl_xor_sync(0xffffffff, sum, i);
        sumsq += __shfl_xor_sync(0xffffffff, sumsq, i);
    }
    
    
   
    float mean = sum / headshape;
    float var = sumsq / headshape - mean * mean;

    float invstd = 1.0f / sqrt(var + eps);

    for (unsigned long i = thread; i < headshape; i+=blockDim.x){
        output[i] = (input[i] - mean) * invstd * weight[i] + bias[i];
    }
}
#endif

#define WEAK __attribute__((weak))
template <typename R = float>
WEAK void normalize_cuda(float* input, float* weight, float* bias, float* output, float eps, unsigned long lastshape, unsigned long headshape, unsigned long size)
{
    throw std::runtime_error("Not compiled with CUDA support");
}

template <typename R = float>
WEAK void normalize_hip(float* input, float* weight, float* bias, float* output, float eps, unsigned long lastshape, unsigned long headshape, unsigned long size)
{
    throw std::runtime_error("Not compiled with HIP support");
}

void normalize_cpu(float* input, float* weight, float* bias, float* output, float eps, unsigned long lastshape, unsigned long headshape, unsigned long size)
{
    unsigned long batches = size / headshape;
    for(unsigned long b = 0; b < batches; b++){
        auto start = b * headshape;
        auto wb = start % lastshape;
        weight += wb;
        bias += wb;
        input += start;
        output += start;

        float sum = 0.0f;
        float sumsq = 0.0f;
        for(unsigned long i = 0; i < headshape; i++){
            sum += input[i];
            sumsq += input[i] * input[i];
        }

        float mean = sum / headshape;
        float var = sumsq / headshape - mean * mean;

        float invstd = 1.0f / sqrt(var + eps);

        for(unsigned long i = 0; i < headshape; i++){
            output[i] = (input[i] - mean) * invstd * weight[i] + bias[i];
        }
    }
}

#if defined(__CUDACC__)
void normalize_cuda(float* input, float* weight, float* bias, float* output, float eps, unsigned long lastshape, unsigned long headshape, unsigned long size){

        

        // batchsize
        unsigned long blocks = size/headshape;
        auto gridsize = dim3(blocks,1,1);
        auto headsperblock = dim3(std::min(headshape,(unsigned long)1024),1,1);

        layernorm<<<gridsize, headsperblock>>>(input, output, weight, bias, size, lastshape, headshape, eps);
       
}
#endif // NORMALIZE_CUH
#if defined(__HIPCC__)
void normalize_hip(float* input, float* weight, float* bias, float* output
, float eps, unsigned long lastshape, unsigned long headshape, unsigned long size){

        

        // batchsize
        unsigned long blocks = size/headshape;
        auto gridsize = dim3(blocks,1,1);
        auto headsperblock = dim3(std::min(headshape,(unsigned long)1024),1,1);

        layernorm<<<gridsize, headsperblock>>>(input, output, weight, bias, size, lastshape, headshape, eps);
       
}
#endif // NORMALIZE_CUH

#endif // NORMALIZE_CUH