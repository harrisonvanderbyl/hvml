#pragma once
#include "tensor.hpp"
#include "kernels/interface.hpp"

// THESE ARE INEFFICIENT KERNELS FOR SIMPLE BINARY OPERATIONS

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
    Div,
    Pow,
    Set,
    Min,
    Max
};



// ================================================================
// Compile-time Operation Selector
// ================================================================
template <typename A, typename B, Operation OP>
struct BinaryOp {
    static inline
    auto applyCPU(A& a, B& b)  {
        if constexpr (OP == Operation::Add)      return a + b;
        if constexpr (OP == Operation::Sub)      return a - b;
        if constexpr (OP == Operation::Mul)      return a * b;
        if constexpr (OP == Operation::Div)      return a / b;
        if constexpr (OP == Operation::Pow)      return pow(a,b);
        if constexpr (OP == Operation::Set)      return b;
        if constexpr (OP == Operation::Min)      return a < b ? a : b;
        if constexpr (OP == Operation::Max)      return a > b ? a : b;
    }

    #ifdef __CUDACC__
    __host__ __device__
    static inline
    auto apply(A& a, B& b)  {
        if constexpr (OP == Operation::Add)      return a + b;
        if constexpr (OP == Operation::Sub)      return a - b;
        if constexpr (OP == Operation::Mul)      return a * b;
        if constexpr (OP == Operation::Div)      return a / b;
        if constexpr (OP == Operation::Pow)      return pow(a,b);
        if constexpr (OP == Operation::Set)      return b;
        if constexpr (OP == Operation::Min)      return a < b ? a : b;
        if constexpr (OP == Operation::Max)      return a > b ? a : b;
    }
    #endif
};

// ================================================================
// CPU Kernel Implementation
// ================================================================
template <DeviceType device, Operation OP,
          typename A, typename B, typename Out>
class BinaryKernel: public Kernel<device, A*, long, long*, long*, B*, long*, Out*, long*> 
{
    // Specializations will be defined below
    void call(A* a_data,
              long ndim,
              long* shape,
              long* a_strides,
              B* b_data,
              long* b_strides,
              Out* out, long* out_strides) override 
    {
        throw std::runtime_error("BinaryKernel not implemented for this device type");
    }
};

template <Operation OP, typename A, typename B, typename Out>
class BinaryKernel<DeviceType::kCPU, OP, A, B, Out>
    : public Kernel<DeviceType::kCPU, A*, long, long*, long*, B*, long*, Out*, long*> 
{
public:

    void call(A* a_data,
              long ndim,
              long* shape,
              long* a_strides,
              B* b_data,
              long* b_strides,
              Out* out, long* out_strides) override 
    {
        long total_size = 1;
        for (long i = 0; i < ndim; i++)
            total_size *= shape[i];

        for (long linear_idx = 0; linear_idx < total_size; linear_idx++) {
            long rem = linear_idx;
            long offset_a = 0;
            long offset_b = 0;
            long offset_out = 0;

            for (long dim = ndim - 1; dim >= 0; dim--) {
                long coord = rem % shape[dim];
                rem /= shape[dim];

                offset_a += coord * a_strides[dim];
                offset_b += coord * b_strides[dim];
                offset_out += coord * out_strides[dim];
            }

            // if (offset_a % 100 == 0)
            // std::cout << "Offsets: " << offset_a << ", " << offset_b << ", " << offset_out << std::endl;
            
            // std::cout << "Offsets: " << offset_a << ", " << offset_b << ", " << offset_out << std::endl;
            out[offset_out] = BinaryOp<A, B, OP>::applyCPU(
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
                                   Out* out, const long* out_strides, long total_size)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    long rem = idx;
    
    long offset_a = 0;
    long offset_b = 0;
    long offset_out = 0;

    for (long dim = ndim - 1; dim >= 0; dim--) {
        long coord = rem % shape[dim];
        rem /= shape[dim];

        offset_a += coord * a_strides[dim];
        offset_b += coord * b_strides[dim];
        offset_out += coord * out_strides[dim];
    }

    out[offset_out] = BinaryOp<A, B, OP>::apply(
        a_data[offset_a], 
        b_data[offset_b]
    );
}

// CUDA kernel wrapper
template <Operation OP, typename A, typename B, typename Out>
class BinaryKernel<DeviceType::kCUDA, OP, A, B, Out>
    : public Kernel<DeviceType::kCUDA, A*, long, long*, long*, B*, long*, Out*, long*> 
{
public:

    void call(A* a_data,
              long ndim,
              long* shape,
              long* a_strides,
              B* b_data,
              long* b_strides,
              Out* out,
              long* out_strides
            ) override 
    {
        long total_size = 1;
        for (long i = 0; i < ndim; i++)
            total_size *= shape[i];

        int threads = 256;
        int blocks = (total_size + threads - 1) / threads;

        long* strides = (long*)DeviceAllocator<DeviceType::kCUDA>::allocate(sizeof(long) * ndim * 4);
        cudaMemcpy(strides, a_strides, sizeof(long) * ndim, cudaMemcpyHostToDevice);
        cudaMemcpy(strides + ndim, b_strides, sizeof(long) * ndim, cudaMemcpyHostToDevice);
        cudaMemcpy(strides + 2*ndim, shape, sizeof(long) * ndim, cudaMemcpyHostToDevice);
        cudaMemcpy(strides + 3*ndim, out_strides, sizeof(long) * ndim, cudaMemcpyHostToDevice);

        binaryOpCudaKernel<A, B, Out, OP>
            <<<blocks, threads>>>(
                a_data, ndim, strides + 2*ndim,
                strides,
                b_data, strides + ndim,
                out, strides + 3*ndim, total_size
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
    apply(const Tensor<A, AD>& a, const Tensor<B, BD>& b)
    {
        constexpr int out_dims = std::max(AD, BD);
        Shape<out_dims> out_shape;

        int final_ndim = out_dims;
        for (int i = final_ndim - 1; i >= 0; i--) {
            long a_dim = (i >= a.shape.ndim())
                          ? 1 : a.shape[a.shape.ndim() - final_ndim + i];
            long b_dim = (i >= b.shape.ndim())
                          ? 1 : b.shape[b.shape.ndim() - final_ndim + i];

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

        auto ret = Tensor<Out, out_dims>(out_shape, out_ptr, a.device_type);

        kernel(a_bcast.data, out_shape.ndim(),
               (long*)(&out_shape), (long*)(&a_bcast.strides),
               b_bcast.data, (long*)(&b_bcast.strides),
               out_ptr, (long*)(&ret.strides));

        return ret;
    }
};

template <typename A, typename B, int AD, int BD, typename OUT>
class ApplyUnitaryKernelOperationHelper {
public:

    template <Operation OP, DeviceType device = DeviceType::kCPU>
    static auto
    apply(const Tensor<A, AD>& a, const Tensor<B, BD>& b)
    {
        constexpr int out_dims = std::max(AD, BD);
        Shape<out_dims> out_shape;

        int final_ndim = std::max(a.shape.ndim(), b.shape.ndim());
        for (int i = final_ndim - 1; i >= 0; i--) {
            long a_dim = (i >= a.shape.ndim())
                          ? 1 : a.shape[a.shape.ndim() - final_ndim + i];
            long b_dim = (i >= b.shape.ndim())
                          ? 1 : b.shape[b.shape.ndim() - final_ndim + i];
            
            out_shape[i] = std::max(a_dim, b_dim);
        }



        auto a_bcast = a.broadcast(out_shape);
        auto b_bcast = b.broadcast(out_shape);


        // std::cout << b_bcast.strides << std::endl;
        // std::cout << a_bcast.strides << std::endl;



        using KernelT = BinaryKernel<device, OP, A, B, A>; // or select device
        KernelT kernel;

        long total = out_shape.total_size();

        kernel(a_bcast.data, out_shape.ndim(),
               (long*)(&out_shape), (long*)(&a_bcast.strides),
               b_bcast.data, (long*)(&b_bcast.strides),
               a_bcast.data, (long*)(&a_bcast.strides));

        return a_bcast;
    }
};

// ================================================================
// Operator Generation Macro
// ================================================================
#define CREATE_BINARY_OP(func_name, OPERATION, KERNELTYPE, OUTTYPE)                                     \
template <typename A, typename B, int AD, int BD,                                  \
          typename Out = typename std::conditional<OUTTYPE,typename std::common_type<A,B>::type, A>::type>                     \
Tensor<Out, std::max(AD,BD)> func_name(Tensor<A, AD> a, Tensor<B, BD> b)           \
{                                                                                  \
    if(a.device_type == DeviceType::kCPU && b.device_type == DeviceType::kCPU)            \
        return KERNELTYPE<A,B,AD,BD,Out>                           \
            ::template apply<Operation::OPERATION, DeviceType::kCPU>(a, b);        \
    else if(a.device_type == DeviceType::kCUDA && b.device_type == DeviceType::kCUDA) \
    return KERNELTYPE<A,B,AD,BD,Out>                               \
            ::template apply<Operation::OPERATION, kCUDA>(a, b);                          \
    else{                                                                           \
        std::cerr << "Mixed device types not supported yet in binary operations." << std::endl; \
        throw std::runtime_error("Mixed device types not supported yet in binary operations."); \
    }                                                                              \
}

#define CREATE_SINGULAR_OP(func_name, OPERATION, KERNELTYPE, OUTTYPE)                                     \
template <typename A, typename B, int AD,                                  \
          typename Out = typename std::conditional<OUTTYPE,typename std::common_type<A,B>::type, A>::type>                     \
auto func_name(Tensor<A, AD> a, B b)           \
{                                                                                  \
    Tensor<B,AD==-1?-1:1> ba(a.device_type, b); \
                                                                                  \
    if(a.device_type == DeviceType::kCPU)            \
        return KERNELTYPE<A,B,AD,AD==-1?-1:1,typename std::conditional<OUTTYPE,typename std::common_type<A,B>::type, A>::type>                           \
            ::template apply<Operation::OPERATION, DeviceType::kCPU>(a, ba);        \
    else if(a.device_type == DeviceType::kCUDA) \
    return KERNELTYPE<A,B,AD,AD==-1?-1:1,typename std::conditional<OUTTYPE,typename std::common_type<A,B>::type, A>::type>                               \
            ::template apply<Operation::OPERATION, kCUDA>(a, ba);                          \
    else{                                                                           \
        std::cerr << "Mixed device types not supported yet in binary operations." << std::endl; \
        std::cout << "A.device_type: " << int(a.device_type) << " B.device_type: " << int(ba.device_type) << std::endl; \
        throw std::runtime_error("Mixed device types not supported yet in binary operations."); \
    }                                                                              \
}



// ================================================================
// User-Facing Tensor Operators
// ================================================================
CREATE_BINARY_OP(operator+, Add, ApplyKernelOperationHelper,1);
CREATE_BINARY_OP(operator-, Sub, ApplyKernelOperationHelper,1);
CREATE_BINARY_OP(operator*, Mul, ApplyKernelOperationHelper,1);
CREATE_BINARY_OP(operator/, Div, ApplyKernelOperationHelper,1);
CREATE_BINARY_OP(pow, Pow, ApplyKernelOperationHelper,1);
CREATE_BINARY_OP(min, Min, ApplyKernelOperationHelper,1);
CREATE_BINARY_OP(max, Max, ApplyKernelOperationHelper,1);
CREATE_SINGULAR_OP(operator+, Add, ApplyKernelOperationHelper,1);
CREATE_SINGULAR_OP(operator-, Sub, ApplyKernelOperationHelper,1);
CREATE_SINGULAR_OP(operator*, Mul, ApplyKernelOperationHelper,1);
CREATE_SINGULAR_OP(operator/, Div, ApplyKernelOperationHelper,1);
CREATE_SINGULAR_OP(pow, Pow, ApplyKernelOperationHelper,1)
CREATE_SINGULAR_OP(min, Min, ApplyKernelOperationHelper,1);
CREATE_SINGULAR_OP(max, Max, ApplyKernelOperationHelper,1);

CREATE_BINARY_OP(operator+=, Add, ApplyUnitaryKernelOperationHelper,0);
CREATE_BINARY_OP(operator-=, Sub, ApplyUnitaryKernelOperationHelper,0);
CREATE_BINARY_OP(operator*=, Mul, ApplyUnitaryKernelOperationHelper,0);
CREATE_BINARY_OP(operator/=, Div, ApplyUnitaryKernelOperationHelper,0);
CREATE_SINGULAR_OP(operator+=, Add, ApplyUnitaryKernelOperationHelper,0);
CREATE_SINGULAR_OP(operator-=, Sub, ApplyUnitaryKernelOperationHelper,0);
CREATE_SINGULAR_OP(operator*=, Mul, ApplyUnitaryKernelOperationHelper,0);
CREATE_SINGULAR_OP(operator/=, Div, ApplyUnitaryKernelOperationHelper,0);

CREATE_BINARY_OP(tensor_copy, Set, ApplyUnitaryKernelOperationHelper,0);
CREATE_SINGULAR_OP(tensor_copy, Set, ApplyUnitaryKernelOperationHelper,0);
