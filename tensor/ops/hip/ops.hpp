
#include "ops/common.hpp"
#include <hip/hip_runtime.h>

template <typename A, typename B> 
__host__ __device__ void atomicAddCuda(A* a, const B& b){
    atomicAdd(a,b);
}

template <typename OP, typename OutputType, typename... Args>
__global__ void OPKERNEL_HIP(
    unsigned long total_size,
    int loopsize, 
    Parameter<OutputType> output, 
    Parameter<Args>... params) {
    int idx = blockIdx.x * blockDim.x * loopsize + threadIdx.x * loopsize;
    
    for (int i = 0; i < loopsize; i++)
    {
        unsigned long global_idx = idx + i;
        if (global_idx >= total_size) return;

        if constexpr (std::is_same<OutputType, void>::value) {
            OP::apply(
                params.get_index(global_idx)...
            );
        }
        else {
          
           AssignmentHelper<OP::assignment_type,kCUDA>::assignOperation(output.get_index(global_idx) , OP::apply(
                params.get_index(global_idx)...
            ));
        }
    }
}

template <typename OP, typename... Args>
void call_hip(
        unsigned long total_size,
        Parameter<typename OutputTypeSelector<OP,Args...>::type> output,
        Parameter<Args>... params
    ) 
    {

        int threadsPerBlock = 256;
        auto firstParamShape = std::get<0>(std::tuple<Parameter<Args>...>(params...)).shape;
        int loopsize = 32;//(firstParamShape[-1]+32-1)/32; // adjust loopsize for performance/memory tradeoff, 32 threads
        
        int numBlocks = (total_size + (threadsPerBlock*loopsize) - 1) / (threadsPerBlock*loopsize);

        // void* inputs[3+sizeof...(params)] = {
        //     &total_size,
        //     &loopsize,
        //     &output,
        //     &params...
        // };

        
            OPKERNEL_HIP<OP, typename OutputTypeSelector<OP,Args...>::type, Args...><<<dim3(numBlocks), dim3(threadsPerBlock)>>>(
                total_size,
                loopsize,
                output,
                params...
            );

      
    }
