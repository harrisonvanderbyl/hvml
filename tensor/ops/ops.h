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
    Hardamard, // element-wise operation
    Add,
    Sub,
    Mul,
    Div,
    AddEq,
    SubEq,
    MulEq,
    DivEq,
    Pow,
    Set,
    Min,
    Max,
    Relu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Negate,
    Abs,
    ToUnsignedLong,
    VelocitySpread
};



template <typename T>
struct Parameter {
    T* data = nullptr;
    unsigned long ndim = 0;
    Shape<-1> shape ;
    Shape<-1> strides;
    unsigned long* indexer = nullptr;

    Parameter() {}
   
    template <int dim>
    Parameter(Tensor<T, dim> tensor) {
        data = tensor.data;
        shape = tensor.shape;
        ndim = tensor.shape.ndim();
        strides = tensor.strides;
        indexer = tensor.indexer;
    }

    __host__ __device__ T& get_index(long index_flat) {
        int rem = index_flat;
        int offset = 0;
        for (long adim = ndim - 1; adim >= 0; adim--) {
            long coord = rem % shape[adim];
            rem /= shape[adim];
            offset += coord * strides[adim];
        }
        if(indexer != nullptr){
            offset = indexer[offset];
        }
        return data[offset];

    };

    __host__ __device__ int get_total_size() {
        int total = 1;
        for (long i = 0; i < ndim; i++) {
            total *= shape[i];
        }
        return total;
    }

    template <int Z = -1>
    operator Tensor<T, Z>() {
        Tensor<T, Z> tensor{shape, data};
        tensor.strides = strides;
        tensor.indexer = indexer;
        return tensor;
    }
};

// void parameter


template <>
struct Parameter<void> {
    
    Parameter() {}
    template <int dim>
    Parameter(Tensor<void, dim> tensor) {
    }

    __host__ __device__ void get_index(long index_flat) {
       

    };

    __host__ __device__ int get_total_size() {
        return 0;
    }

};

// ================================================================
// Compile-time Operation Selector
// ================================================================
template <Operation OP>
struct OperationSelector {
    template <typename... Params>
    __host__ __device__ static inline
    auto apply(const Params&... params);
};

template <Operation OP, typename... Params>
struct OutputTypeSelector {
    using type = decltype(OperationSelector<OP>::apply(std::declval<Params&>()...));
};

template <>
struct OperationSelector<Operation::Hardamard> {
    template <typename... Types>
    __host__ __device__ static inline
    Shape<-1> get_output_shape(const Parameter<Types>&... params) {
        Shape<-1> output_shape;

        auto update_shape = [&](const auto& param) {
            
            for (int j = 0; j < param.shape.ndim(); j++) {
                if (output_shape[j] == INT64MAX) {
                    output_shape[j] = param.shape[j];
                } else if (param.shape[j] != INT64MAX && output_shape[j] != param.shape[j]) {
                    if (output_shape[j] == 1) {
                        output_shape[j] = param.shape[j];
                    } else if (param.shape[j] != 1) {
                        // Incompatible shapes
                        // throw std::runtime_error("Incompatible shapes for Hardamard");
                    }
                }
            }
        };

        // Apply to all parameters
        (update_shape(params), ...);  // fold expression over the pack

        return output_shape;
    }
};


template <>
struct OperationSelector<Operation::Add>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a + b;
    }
};


template <>
struct OperationSelector<Operation::Sub>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a - b;
    }
};

template <>
struct OperationSelector<Operation::Mul>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a * b;
    }
};

template <>
struct OperationSelector<Operation::Div>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a / b;
    }
};


template <>
struct OperationSelector<Operation::AddEq>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a += b;
    }
};


template <>
struct OperationSelector<Operation::SubEq>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a -= b;
    }
};

template <>
struct OperationSelector<Operation::MulEq>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a *= b;
    }
};

template <>
struct OperationSelector<Operation::DivEq>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a /= b;
    }
};

template <>
struct OperationSelector<Operation::Pow>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return pow(a, b);
    }
};

template <>
struct OperationSelector<Operation::Set>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a = b;
    }
};

template <>
struct OperationSelector<Operation::Min>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a < b ? a : b;
    }
};

template <>
struct OperationSelector<Operation::Max>:public OperationSelector<Operation::Hardamard> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a > b ? a : b;
    }
};

template <>
struct OperationSelector<Operation::Relu>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return a > 0 ? a : 0;
    }
};

template <>
struct OperationSelector<Operation::Sigmoid>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return 1 / (1 + exp(-a));
    }
};

template <>
struct OperationSelector<Operation::Tanh>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return tanh(a);
    }
};

template <>
struct OperationSelector<Operation::Exp>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return exp(a);
    }
};

template <>
struct OperationSelector<Operation::Log>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return log(a);
    }
};

template <>
struct OperationSelector<Operation::Sqrt>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return sqrt(a);
    }
};

template <>
struct OperationSelector<Operation::Negate>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return -a;
    }
};

template <>
struct OperationSelector<Operation::Abs>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return abs(a);
    }
};

template <>
struct OperationSelector<Operation::ToUnsignedLong>:public OperationSelector<Operation::Hardamard> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return (unsigned long)(a);
    }
};

// template <>
// struct OperationSelector<Operation::VelocitySpread>:public OperationSelector<Operation::Hardamard> {
//     __host__ __device__ static inline
//     void apply(float32x4& velocityField, float32x4& particle, int32x2& screenDims) {
//         // Simple example: add spread to velocity
//         velocityField.x += 1.0f;
//     }
// };
// ================================================================


// ================================================================
// CPU Kernel Implementation
// ================================================================
template <DeviceType device, Operation OP,
          typename... Args>
class BinaryKernel: public Kernel<device,Parameter<typename OutputTypeSelector<OP, Args...>::type>, Parameter<Args>...> 
{
    // Specializations will be defined below
    void call(Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...) override 
    {
        throw std::runtime_error("BinaryKernel not implemented for this device type");
    }
};

template <Operation OP, typename... Args>
class BinaryKernel<DeviceType::kCPU, OP, Args...>
    : public Kernel<DeviceType::kCPU, unsigned long, Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...> 
{
public:

    void call(
        unsigned long total_size,
        Parameter<typename OutputTypeSelector<OP, Args...>::type> output,
        Parameter<Args>... params
    ) override 
    {
        // get first argument and use .get_total_size() to get total size

        for (long linear_idx = 0; linear_idx < total_size; linear_idx++) {  
            if constexpr (std::is_same<typename OutputTypeSelector<OP, Args...>::type, void>::value) {
                OperationSelector<OP>::apply(
                    params.get_index(linear_idx)...
                );
                continue;
            }
            else{ 
                output.get_index(linear_idx) = OperationSelector<OP>::apply(
                    params.get_index(linear_idx)...
                );
            }
        }
    }
};


// ================================================================
// CUDA Kernel Implementation
// ================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>

template <Operation OP, typename OutputType, typename... Args>
__global__ void binaryOpCudaKernel(unsigned long total_size, Parameter<OutputType> output, Parameter<Args>... params) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    if constexpr (std::is_same<OutputType, void>::value) {
        OperationSelector<OP>::apply(
            params.get_index(idx)...
        );
        return;
    }
    else{
        output.get_index(idx) = OperationSelector<OP>::apply(
            params.get_index(idx)...
        );
    }    
}


// CUDA kernel wrapper
template <Operation OP, typename... Args>
class BinaryKernel<DeviceType::kCUDA, OP, Args...>
    : public Kernel<DeviceType::kCUDA, unsigned long, Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...>
{
public:

    void call(
        unsigned long total_size,
        Parameter<typename OutputTypeSelector<OP,Args...>::type> output,
        Parameter<Args>... params
    ) override 
    {

        int threads = 256;
        int blocks = (total_size + threads - 1) / threads;
        

        binaryOpCudaKernel<OP>
            <<<blocks, threads>>>(
                total_size,
                output,
                params...
            );
    }
};

#endif // USE_CUDA



// ================================================================
// Operation Application Helper
// ================================================================
template <typename A, typename B, int AD, int BD>
class ApplyKernelOperationHelper {
public:

    template <Operation OP, DeviceType device = DeviceType::kCPU>
    static auto
    apply(const Tensor<A, AD>& a, const Tensor<B, BD>& b)
    {
        // constexpr int out_dims = std::max(AD, BD);
        Shape<-1> out_shape = OperationSelector<OP>::get_output_shape(
            Parameter<A>(a),
            Parameter<B>(b)
        );

        using Out = typename OutputTypeSelector<OP, A, B>::type;

        auto a_bcast = a.broadcast(out_shape);
        auto b_bcast = b.broadcast(out_shape);
        a_bcast.indexer = a.indexer;
        b_bcast.indexer = b.indexer;

        // std::cout << "out_shape: " << std::string(out_shape) << std::endl;


        using KernelT = BinaryKernel<device, OP, A, B>; // or select device
        KernelT kernel;

        unsigned long total = out_shape.total_size();
        Out* out_ptr = nullptr;


        // if Out type is void, dont allocate output
        if constexpr (!std::is_same<Out, void>::value) {

            auto buffers = kernel.template CreateBuffers<Out>({total});
            buffers.Allocate();
            out_ptr = buffers.template getPointer<0>();

            auto out_param = Tensor<Out, -1>(out_shape, out_ptr, a.device_type);

            kernel(
                total,
                out_param,
                a_bcast,
                b_bcast
            );

            return out_param;
        }
        
        else {
            kernel(
                total,
                Parameter<void>(),
                a_bcast,
                b_bcast
            );
        }
        
    }
};


// ================================================================
// Operator Generation Macro
// ================================================================
#define CREATE_BINARY_OP(func_name, OPERATION)                                     \
template <typename A, typename B, int AD, int BD>                     \
auto func_name(Tensor<A, AD> a, Tensor<B, BD> b)           \
{                                                                                  \
    if(a.device_type == DeviceType::kCPU && b.device_type == DeviceType::kCPU)            \
        return ApplyKernelOperationHelper<A,B,AD,BD>                           \
            ::template apply<Operation::OPERATION, kCPU>(a, b);        \
    else if(a.device_type == DeviceType::kCUDA && b.device_type == DeviceType::kCUDA) \
    return ApplyKernelOperationHelper<A,B,AD,BD>                               \
            ::template apply<Operation::OPERATION, kCUDA>(a, b);                          \
    else{                                                                           \
        std::cerr << "Mixed device types not supported yet in binary operations." << std::endl; \
        throw std::runtime_error("Mixed device types not supported yet in binary operations."); \
    }                                                                              \
}

#define CREATE_SINGULAR_OP(func_name, OPERATION)                                     \
template <typename A, typename B, int AD>                     \
auto func_name(Tensor<A, AD> a, B b)           \
{                                                                                  \
    Tensor<B,AD==-1?-1:1> ba(a.device_type, b); \
                                                                                  \
    if(a.device_type == DeviceType::kCPU)            \
        return ApplyKernelOperationHelper<A,B,AD,AD==-1?-1:1>                           \
            ::template apply<Operation::OPERATION, kCPU>(a, ba);        \
    else if(a.device_type == DeviceType::kCUDA) \
    return ApplyKernelOperationHelper<A,B,AD,AD==-1?-1:1>                               \
            ::template apply<Operation::OPERATION, kCUDA>(a, ba);                          \
    else{                                                                           \
        std::cerr << "Mixed device types not supported yet in binary operations." << std::endl; \
        std::cout << "A.device_type: " << int(a.device_type) << " B.device_type: " << int(ba.device_type) << std::endl; \
        throw std::runtime_error("Mixed device types not supported yet in binary operations."); \
    }                                                                              \
}

#define CREATE_KERNEL_OP(func_name, OPERATION)                                    \
template <typename Out, typename... Args>                     \
auto func_name(Args... args)           \
{                                                                                  \
    auto device = std::get<0>(std::tuple<Args...>(args...)).device_type; \
    if(device == DeviceType::kCPU)            \
        return BinaryKernel<DeviceType::kCPU, Operation::OPERATION, Args...>()(args...);        \
    else if(device == DeviceType::kCUDA) \
    return BinaryKernel<DeviceType::kCUDA, Operation::OPERATION, Args...>()(args...);                          \
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
CREATE_BINARY_OP(pow, Pow);
CREATE_BINARY_OP(min, Min);
CREATE_BINARY_OP(max, Max);
CREATE_SINGULAR_OP(operator+, Add);
CREATE_SINGULAR_OP(operator-, Sub);
CREATE_SINGULAR_OP(operator*, Mul);
CREATE_SINGULAR_OP(operator/, Div);
CREATE_SINGULAR_OP(pow, Pow)
CREATE_SINGULAR_OP(min, Min);
CREATE_SINGULAR_OP(max, Max);

CREATE_BINARY_OP(operator+=, AddEq);
CREATE_BINARY_OP(operator-=, SubEq);
CREATE_BINARY_OP(operator*=, MulEq);
CREATE_BINARY_OP(operator/=, DivEq);
CREATE_SINGULAR_OP(operator+=, AddEq);
CREATE_SINGULAR_OP(operator-=, SubEq);
CREATE_SINGULAR_OP(operator*=, MulEq);
CREATE_SINGULAR_OP(operator/=, DivEq);

CREATE_BINARY_OP(tensor_copy, Set);
CREATE_SINGULAR_OP(tensor_copy, Set);


// CREATE_KERNEL_OP(tensor_hardamard, Hardamard);
// CREATE_KERNEL_OP(tensor_velocity_spread, VelocitySpread);