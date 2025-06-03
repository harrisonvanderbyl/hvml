

#ifndef TIMESHIFT_KERNELS
#define TIMESHIFT_KERNELS
#include "tensor.hpp"



template <typename R = float> 
void timeshift_cpu(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    // For each batch
    for (int b = 0; b < B; b++) {
        R* batch_data = data + b * T * C;
        R* batch_state = state + b * C;
        
        // Save the last timestep to buffer
        std::copy(batch_data + (T-1) * C, batch_data + T * C, bufferdata);
        
        // Shift all timesteps forward by 1
        for (int t = T-1; t > 0; t--) {
            std::copy(batch_data + (t-1) * C, batch_data + t * C, batch_data + t * C);
        }
        
        // Copy state to first timestep
        std::copy(batch_state, batch_state + C, batch_data);
        
        // Update state with the saved last timestep
        std::copy(bufferdata, bufferdata + C, batch_state);
    }
}

// weak timeshift cuda, so cuda compile can add it to the kernel
#define WEAK __attribute__((weak))
template <typename R = float>
WEAK void timeshift_cuda(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    throw std::runtime_error("Not compiled with CUDA support");
}

template <typename R = float>
WEAK void timeshift_hip(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    throw std::runtime_error("Not compiled with HIP support");
}

#ifdef __CUDACC__
// just use cuda memmove
template <typename R = float>
__global__ void timeshift_cuda_kernel(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    int skipsize = (T-1)*C;
    for (int i = 0; i < B; i++)
    {
        cudaMemcpyAsync(data+i*skipsize, bufferdata+i*skipsize, C*sizeof(R), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(data+i*skipsize-C, state+i*C, C*sizeof(R), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(state+i*C, bufferdata+i*skipsize, C*sizeof(R), cudaMemcpyDeviceToDevice);
    }
}
#endif

#ifdef __HIPCC__
// just use hip memmove
template <typename R = float>
__global__ void timeshift_hip_kernel(R* data, R* state, R* bufferdata, int B, int T, int C)
{
    int skipsize = (T-1)*C;
    for (int i = 0; i < B; i++)
    {
        hipMemcpyAsync(data+i*skipsize, bufferdata+i*skipsize, C*sizeof(R), hipMemcpyDeviceToDevice);
        hipMemcpyAsync(data+i*skipsize-C, state+i*C, C*sizeof(R), hipMemcpyDeviceToDevice);
        hipMemcpyAsync(state+i*C, bufferdata+i*skipsize, C*sizeof(R), hipMemcpyDeviceToDevice);
    }
}

#endif



#endif
