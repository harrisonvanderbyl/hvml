
#include "ops/common.hpp"
#include <cuda_runtime.h>
// bfloat16
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

void cuda_thrust_sort(float* keys, int* indices, size_t size) {
    thrust::device_ptr<float> keys_ptr(keys);
    thrust::device_ptr<int> indices_ptr(indices);

    thrust::sort_by_key(
        keys_ptr,
        keys_ptr + size,
        indices_ptr
    );
}

template <typename A, typename B> 
__host__ __device__ void atomicAddCuda(A* a, const B& b){
    atomicAdd(a,b);
}


// ================================================================
// Element-wise kernel: block b covers elements [b·blockDim·loopsize, ...);
// in each of the `loopsize` steps the block's threads take consecutive
// elements, so a warp reads and writes contiguous memory.
// ================================================================
template <typename OP, typename OutputType, typename... Args>
__global__ void OPKERNEL(
    unsigned long total_size,
    int loopsize, 
    Parameter<OutputType> output, 
    Parameter<Args>... params) {
    unsigned long start = (unsigned long)blockIdx.x * blockDim.x * loopsize + threadIdx.x;

    for (int i = 0; i < loopsize; i++)
    {
        unsigned long global_idx = start + (unsigned long)i * blockDim.x;
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

// ================================================================
// Reduction kernel: a group of `lanes` threads (≤ a warp) computes one
// output.  Lane l sums inputs l, l + lanes, ... along the reduced dim
// (contiguous in memory when that dim has stride 1), then the group adds
// its partial sums with warp shuffles.  No atomics.
// ================================================================
template <typename OP, typename Out, typename... C>
__device__ inline Out reduce_lane(int lane, int lanes, long len, C... c) {
    Out acc = OP::template identity<Out>();
    for (long r = lane; r < len; r += lanes) OP::combine(acc, OP::apply(c(r)...));
    return acc;
}

template <typename OP, typename Out, typename... Args>
__global__ void REDUCEKERNEL(
    long outer, long len, long inner, int dim, int lanes,
    Parameter<Out> output,
    Parameter<Args>... params) {
    long thread = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long o = thread / lanes;
    int lane = (int)(thread % lanes);
    bool valid = o < outer;          // every thread stays for the shuffles
    long base = valid ? (o / inner) * len * inner + (o % inner) : 0;

    Out acc = OP::template identity<Out>();
    if (valid) acc = reduce_lane<OP, Out>(lane, lanes, len, ReductionCursor<Args>(params, base, inner, dim)...);

    if constexpr (std::is_arithmetic<Out>::value) {
        for (int off = lanes / 2; off > 0; off >>= 1) {
            Out other = __shfl_down_sync(0xffffffffu, acc, off, lanes);
            OP::combine(acc, other);
        }
        if (valid && lane == 0) output.get_index(base) = acc;
    } else {
        // (sums only) each lane adds its part atomically into the zeroed output
        if (valid) AssignmentHelper<AssignmentType::InplaceAdd,kCUDA>::assignOperation(output.get_index(base), acc);
    }
}

template<class T, typename dim3, typename cudaStream_t, typename cudaError_t>
static __inline__ __attribute__((host)) cudaError_t cudaLaunchKernel(
  T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
);

template <typename OP, typename... Args>
void call_cuda(
    int device_id,
    unsigned long total_size,
    Parameter<typename OutputTypeSelector<OP,Args...>::type> output,
    Parameter<Args>... params
)  
{

    CUDA_ERROR_CHECK(cudaSetDevice(device_id));

    using Out = typename OutputTypeSelector<OP,Args...>::type;
    if constexpr (!std::is_same<Out, void>::value && OP::assignment_type == AssignmentType::InplaceAdd) {
        ReductionShape rs = reduction_shape<OP>(output, total_size);
        if (rs.outer == 0) return;
        int lanes = reduction_lanes(rs.len);
        int threads = 256;
        int blocks = (int)((rs.outer * lanes + threads - 1) / threads);
        void* args[6+sizeof...(params)] = {
            &rs.outer, &rs.len, &rs.inner, &rs.dim, &lanes,
            &output,
            &params...
        };
        cudaLaunchKernel(
            REDUCEKERNEL<OP, Out, Args...>,
            blocks, threads,
            args
        );
        return;
    } else {

    int threadsPerBlock = 256;
    auto firstParamShape = std::get<0>(std::tuple<Parameter<Args>...>(params...)).shape;
    // elements per thread: up to 256, but keep ~1024+ blocks so every SM has work
    long per_thread = (long)total_size / ((long)threadsPerBlock * 1024);
    int loopsize = (int)std::max(1L, std::min(256L, per_thread));
    
    int numBlocks = (total_size + (threadsPerBlock*loopsize) - 1) / (threadsPerBlock*loopsize);

    void* inputs[3+sizeof...(params)] = {
        &total_size,
        &loopsize,
        &output,
        &params...
    };

    cudaLaunchKernel(
        OPKERNEL<OP, typename OutputTypeSelector<OP,Args...>::type, Args...>,
        numBlocks, threadsPerBlock,
        inputs
    );
    }
}


