#pragma once
#include "tensor.hpp"
#include "kernels/interface.hpp"

#ifndef __host__
#define __host__
#define __device__
#endif
// ================================================================
// Operation Enum
// ================================================================
enum class Operation {
    Add,
    Sub,
    Mul,
    Div
};

// ================================================================
// Compile-time Operation Selector
// ================================================================
template <typename A, typename B, Operation OP>
struct BinaryOp {
    static inline
    auto applyCPU(A a, B b) {
        if constexpr (OP == Operation::Add)      return a + b;
        if constexpr (OP == Operation::Sub)      return a - b;
        if constexpr (OP == Operation::Mul)      return a * b;
        if constexpr (OP == Operation::Div)      return a / b;
    }

    #ifdef __CUDACC__
    __host__ __device__
    static inline
    auto apply(A a, B b) {
        if constexpr (OP == Operation::Add)      return a + b;
        if constexpr (OP == Operation::Sub)      return a - b;
        if constexpr (OP == Operation::Mul)      return a * b;
        if constexpr (OP == Operation::Div)      return a / b;
    }
    #endif
};

// ================================================================
// CPU Kernel Implementation
// ================================================================
template <DeviceType device, Operation OP,
          typename A, typename B, typename Out>
class BinaryKernel;

template <Operation OP, typename A, typename B, typename Out>
class BinaryKernel<DeviceType::kCPU, OP, A, B, Out>
    : public Kernel<DeviceType::kCPU, A*, long, long*, long*, B*, long*, Out*> 
{
public:

    void call(A* a_data,
              long ndim,
              long* shape,
              long* a_strides,
              B* b_data,
              long* b_strides,
              Out* out) override 
    {
        long total_size = 1;
        for (long i = 0; i < ndim; i++)
            total_size *= shape[i];

        for (long linear_idx = 0; linear_idx < total_size; linear_idx++) {
            long rem = linear_idx;
            long offset_a = 0;
            long offset_b = 0;

            for (long dim = ndim - 1; dim >= 0; dim--) {
                long coord = rem % shape[dim];
                rem /= shape[dim];

                offset_a += coord * a_strides[dim];
                offset_b += coord * b_strides[dim];
            }

            out[linear_idx] = BinaryOp<A, B, OP>::applyCPU(
                a_data[offset_a], 
                b_data[offset_b]
            );
        }
    }
};



// ================================================================
// CUDA Kernel Implementation
// ================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>

template <typename A, typename B, typename Out, Operation OP>
__global__ void binaryOpCudaKernel(A* a_data, long ndim, const long* shape,
                                   const long* a_strides,
                                   B* b_data, const long* b_strides,
                                   Out* out, long total_size)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    long rem = idx;
    
    long offset_a = 0;
    long offset_b = 0;

    for (long dim = ndim - 1; dim >= 0; dim--) {
        long coord = rem % shape[dim];
        rem /= shape[dim];

        offset_a += coord * a_strides[dim];
        offset_b += coord * b_strides[dim];
    }

    out[idx] = BinaryOp<A, B, OP>::apply(
        a_data[offset_a], 
        b_data[offset_b]
    );
}

// CUDA kernel wrapper
template <Operation OP, typename A, typename B, typename Out>
class BinaryKernel<DeviceType::kCUDA, OP, A, B, Out>
    : public Kernel<DeviceType::kCUDA, A*, long, long*, long*, B*, long*, Out*> 
{
public:

    void call(A* a_data,
              long ndim,
              long* shape,
              long* a_strides,
              B* b_data,
              long* b_strides,
              Out* out) override 
    {
        long total_size = 1;
        for (long i = 0; i < ndim; i++)
            total_size *= shape[i];

        int threads = 256;
        int blocks = (total_size + threads - 1) / threads;

        long* strides = (long*)DeviceAllocator<DeviceType::kCUDA>::allocate(sizeof(long) * ndim * 3);
        cudaMemcpy(strides, a_strides, sizeof(long) * ndim, cudaMemcpyHostToDevice);
        cudaMemcpy(strides + ndim, b_strides, sizeof(long) * ndim, cudaMemcpyHostToDevice);
        cudaMemcpy(strides + 2*ndim, shape, sizeof(long) * ndim, cudaMemcpyHostToDevice);

        binaryOpCudaKernel<A, B, Out, OP>
            <<<blocks, threads>>>(
                a_data, ndim, strides + 2*ndim,
                strides,
                b_data, strides + ndim,
                out, total_size
            );
        cudaDeviceSynchronize();
    }
};

#endif // USE_CUDA



// ================================================================
// Operation Application Helper
// ================================================================
template <typename A, typename B, int AD, int BD,
          typename Out = typename std::common_type<A,B>::type>
class ApplyKernelOperationHelper {
public:

    template <Operation OP, DeviceType device = DeviceType::kCPU>
    static Tensor<Out, std::max(AD, BD)>
    apply(Tensor<A, AD> a, Tensor<B, BD> b)
    {
        constexpr int out_dims = std::max(AD, BD);
        Shape<out_dims> out_shape;

        int final_ndim = out_dims;
        for (int i = 0; i < final_ndim; i++) {
            long a_dim = (a.shape.ndim() - final_ndim + i < 0)
                          ? 1 : a.shape[a.shape.ndim() - final_ndim + i];
            long b_dim = (b.shape.ndim() - final_ndim + i < 0)
                          ? 1 : b.shape[b.shape.ndim() - final_ndim + i];

            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                throw std::runtime_error("Shape mismatch in binary op.");
            }

            out_shape[i] = std::max(a_dim, b_dim);
        }

        auto a_bcast = a.broadcast(out_shape);
        auto b_bcast = b.broadcast(out_shape);

        using KernelT = BinaryKernel<device, OP, A, B, Out>; // or select device
        KernelT kernel;

        long total = out_shape.total_size();
        auto buffers = kernel.template CreateBuffers<Out>({total});
        buffers.Allocate();

        Out* out_ptr = buffers.template getPointer<0>();

        kernel(a_bcast.data, out_shape.ndim(),
               (long*)(&out_shape), (long*)(&a_bcast.strides),
               b_bcast.data, (long*)(&b_bcast.strides),
               out_ptr);

        return Tensor<Out, out_dims>(out_shape, out_ptr, a.device_type);
    }
};



// ================================================================
// Operator Generation Macro
// ================================================================
#define CREATE_BINARY_OP(func_name, OPERATION)                                     \
template <typename A, typename B, int AD, int BD,                                  \
          typename Out = typename std::common_type<A,B>::type>                     \
Tensor<Out, std::max(AD,BD)> func_name(Tensor<A, AD> a, Tensor<B, BD> b)           \
{                                                                                  \
    if(a.device_type == DeviceType::kCPU && b.device_type == DeviceType::kCPU)            \
        return ApplyKernelOperationHelper<A,B,AD,BD,Out>                           \
            ::template apply<Operation::OPERATION, DeviceType::kCPU>(a, b);        \
    else if(a.device_type == DeviceType::kCUDA && b.device_type == DeviceType::kCUDA) \
    return ApplyKernelOperationHelper<A,B,AD,BD,Out>                               \
            ::template apply<Operation::OPERATION, kCUDA>(a, b);                          \
    else{                                                                           \
        std::cerr << "Mixed device types not supported yet in binary operations." << std::endl; \
        throw std::runtime_error("Mixed device types not supported yet in binary operations."); \
    }                                                                              \
}

// ================================================================
// User-Facing Tensor Operators
// ================================================================
CREATE_BINARY_OP(operator+, Add);
CREATE_BINARY_OP(operator-, Sub);
CREATE_BINARY_OP(operator*, Mul);
CREATE_BINARY_OP(operator/, Div);

