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




template <typename T>
struct Parameter {
    T* data = 0;
    unsigned long ndim = 0;
    Shape<-1> shape ;
    Shape<-1> strides;
    unsigned long* indexer = nullptr;
    bool istensor = true;
    T data_scalar ;

    Parameter() {}
   
    template <int dim>
    Parameter(const Tensor<T, dim>& tensor) {
        data = tensor.data;
        shape = tensor.shape;
        ndim = tensor.shape.ndim();
        strides = tensor.strides;
        indexer = tensor.indexer;
        istensor = true;
    }

    
    Parameter(const T& indata) {
        ndim = 1;
        shape = Shape<-1>({1});
        strides = Shape<-1>({1});
        istensor = false;
        indexer = nullptr;
        data_scalar = indata;

    }
    
    

    __host__ __device__ T& get_index(long index_flat) {

        if(!istensor){
            return data_scalar;
        }

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

    __host__ __device__ unsigned long get_total_size() {
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
        tensor.total_size = shape.total_size();
        return tensor;
    }

    Parameter broadcast(const Shape<-1>& a) const
    {

        Parameter b;
        b.data = data;
        b.shape = a;
        b.strides = a.calc_strides();
        b.indexer = indexer;
        b.data_scalar = data_scalar;
        b.istensor = istensor;
        b.ndim = a.ndim();
        // std::cout << "From shape: " << shape << " to broadcast shape: " << a << std::endl;
        if (shape == a)
        {
            b.strides = strides;
            return b;
        }

        for (size_t i = 1; i < a.ndim()+1; i++)
        {
            if(shape.ndim() > i && a[-i%a.ndim()] != shape[-i%shape.ndim()] && shape[-i%shape.ndim()] != 1){
                std::cerr << "Incompatible shapes for broadcast" << std::endl;
                std::cerr << i << "\n";
                std::cerr << "Shape: " << shape << " Broadcast shape: " << a << std::endl;
                std::cerr << "Shape: " << shape[-i%shape.ndim()] << " Broadcast shape: " << a[-i%a.ndim()] << std::endl;
                throw std::runtime_error("Incompatible shapes for broadcast");
            }
            if (shape.ndim() < i || shape[-i] == 1)
            {
                b.strides[-i] = 0;
                if(i < shape.ndim()){
                    for (size_t j = i+1; j < a.ndim()+1; j++)
                    {
                        b.strides[-j] = b.strides[-j]/ shape[-i-1];
                    }
                }
            }
        }
        // std::cout << "broadcasted parameter strides: " << b.strides << std::endl;
        return b;
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



template <typename OP, typename... Params>
struct OutputTypeSelector {
    using type = decltype(OP::apply(std::declval<Params&>()...));
};
// ================================================================
// Compile-time Operation Selector
// ================================================================




// ================================================================


// ================================================================
// CPU Kernel Implementation
// ================================================================
template <DeviceType device, typename OP,
          typename... Args>
class BinaryKernel: public Kernel<device,Parameter<typename OutputTypeSelector<OP, Args...>::type>, Parameter<Args>...> 
{
    // Specializations will be defined below
    void call(Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...) override 
    {
        throw std::runtime_error("BinaryKernel not implemented for this device type");
    }
};

// template <typename OP, DeviceType device>
// class ExtensionHelpers : public OP {
//     public:
//         template <typename A>
//         __host__ __device__ static inline
//         auto reduceAdd(A& input){
//             if constexpr (device == DeviceType::kCPU){
//                 using T = decltype(input[0]);
//                 T sum = 0;
//                 for(long i = 0; i < input.get_total_size(); i++){
//                     sum += input.get_index(i);
//                 }
//                 return sum;
//             }
//             else{
//                 // GPU reduction can be implemented here
//                 // 32 thread tree reduction
//                 for (int i = 0; i < 5; i++){
//                     // warp
//                 }
//             }
        
//         }
// };

template <typename OP, typename... Args>
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
                OP::apply(
                    params.get_index(linear_idx)...
                );
                continue;
            }
            else{ 
                OP::assignOperation(output.get_index(linear_idx) , OP::apply(
                    params.get_index(linear_idx)...
                ));
            }
        }
    }
};


// ================================================================
// CUDA Kernel Implementation
// ================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>

template <typename OP, typename OutputType, typename... Args>
__global__ void OPKERNEL(
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
           OP::assignOperation(output.get_index(global_idx) , OP::apply(
                params.get_index(global_idx)...
            ));
        }
    }
}


// CUDA kernel wrapper
template <typename OP, typename... Args>
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
        int loopsize = 16; // adjust loopsize for performance/memory tradeoff, 32 threads
        int threadsPerBlock = 256;
        
        int numBlocks = (total_size + (threadsPerBlock*loopsize) - 1) / (threadsPerBlock*loopsize);

        OPKERNEL<OP, typename OutputTypeSelector<OP,Args...>::type, Args...>
            <<<numBlocks, threadsPerBlock>>>(
                total_size,
                loopsize,
                output,
                params...
            );
    }
};

#endif // USE_CUDA



// ================================================================
// Operation Application Helper
// ================================================================
template <typename OP, DeviceType device = DeviceType::kCPU>
class ApplyKernelOperationHelper {
public:

    template < typename... types>
    static auto
    apply(const Parameter<types>&... params)
    {
        // constexpr int out_dims = std::max(AD, BD);
        Shape<-1> out_shape = OP::get_output_shape(
            params...
        );

        using Out = typename OutputTypeSelector<OP, types...>::type;


        using KernelT = BinaryKernel<device, OP, types...>; // or select device
        KernelT kernel;

        size_t total = out_shape.total_size();
        

        // if Out type is void, dont allocate output
        if constexpr (!std::is_same<Out, void>::value) {

            size_t allocate_size = OP::get_allocate_size(out_shape);

            auto out_param = Tensor<Out, -1>({allocate_size}, device);
            out_param = 0;

            out_param.shape = out_shape;

            out_param.strides = OP::broadcast_output_strides(out_shape);
            kernel(
                total,
                out_param,
                OP::broadcast_param(params,out_shape)...
            );

            // out_param.strides = out_shape.calc_strides();

            return OP::broadcast_output(out_param);
        }
        
        else {
            kernel(
                total,
                Parameter<void>(),
                OP::broadcast_param(params,out_shape)...
            );
        }
        
    }
};

template <typename OP>
struct OperationSelector {
    template <typename... Params>
    __host__ __device__ static inline
    auto apply(const Params&... params);

    template <int AD, typename A, typename... B>                     
    inline static auto run(Tensor<A, AD> a, B... b)           
    {                                   
        // OPERATION is the current class calling this operator(), including its decendents, so if a decendent calls this function, it will use its own apply function                                               
                             
        if(a.device_type == DeviceType::kCPU)            
            return ApplyKernelOperationHelper<OP, kCPU>::apply(Parameter(a), Parameter(b)...);        
        else if(a.device_type == DeviceType::kCUDA) 
            return ApplyKernelOperationHelper<OP, kCUDA>::apply(Parameter(a), Parameter(b)...);                          
        else{                                                                           
            std::cerr << "Mixed device types not supported yet in binary operations." << std::endl; \
            throw std::runtime_error("Mixed device types not supported yet in binary operations."); \
        }                                                                              
    }

    
   
};



template <typename OP>
struct HardamardOperation: public OperationSelector<OP> {
    template <typename... Types>
    static inline
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
                        std::cout << "Incompatible shapes for Hardamard" << std::endl;
                        std::cout << "Shape: " << output_shape << " Param shape: " << param.shape << std::endl;
                        throw std::runtime_error("Incompatible shapes for Hardamard");
                    }
                }
            }
        };

        // Apply to all parameters
        (update_shape(params), ...);  // fold expression over the pack

        return output_shape;
    };

    static inline size_t get_allocate_size(Shape<-1> output_shape){
        return output_shape.total_size();
    }

    template <typename T>
    static inline auto broadcast_param(const Parameter<T>& input, Shape<-1> output_shape){
        return input.broadcast(output_shape);
    }

    static inline auto broadcast_output_strides(Shape<-1> output_shape){
        return output_shape.calc_strides();
    }

    template <typename T, typename TT>
    __host__ __device__ static void assignOperation(T& output, const TT& input){
        output = input;
    }

    template <typename T>
    static inline auto broadcast_output(Parameter<T>& input){
        return input;
    }

};
// eg, 
// eg, 3,2 @ 4,2 -> 3,4 

// eg, 3,1,2 @ 1,4,2 -> 3,4,1(2) 

template <typename OP, int reductionDim>
struct ReductionOperation: public OperationSelector<OP> {
    template <typename... Types>
    static inline
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
        // output_shape[reductionDim] = 1;
        return output_shape;
    };

    template <typename T>
    static inline auto broadcast_param(const Parameter<T>& input, Shape<-1> output_shape){
        return input.broadcast(output_shape);
        
    }

    static inline auto broadcast_output_strides(Shape<-1> output_shape){
        auto newStrides = output_shape.calc_strides();
        int fixedi = (reductionDim + int(output_shape.ndim()))%output_shape.ndim();
        for (int i = 0; i < fixedi; i++){
            newStrides[i] /= output_shape[fixedi];
        }
        newStrides[fixedi] = 0;
        return newStrides;
    }

    static inline size_t get_allocate_size(Shape<-1> output_shape){
        auto a = output_shape.total_size()/output_shape[reductionDim];
        return a;
    }

    template <typename T, typename TT>
    __host__ __device__ static void assignOperation(T& output, const TT& input){
        output += input;
    }

    template <typename T>
    static inline auto broadcast_output(T input){
        Shape<-1> outshape = input.shape;
        int fixedi = (reductionDim + int(outshape.ndim()))%int(outshape.ndim());
        for(int i = 0; i <= outshape.ndim(); i++){
            if (i >= fixedi){
                outshape[i] = outshape[i+1];
            };
        }

        input.shape = outshape;
        input.strides = outshape.calc_strides();
        return input;
    }
};

template <int dim = -1>
struct ReduceSum: public ReductionOperation<ReduceSum<dim>,dim> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return a;
    }
};

template <int dim = -1>
struct DotProduct: public ReductionOperation<DotProduct<dim>,dim> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a, const A& b) {
        return a * b;
    }
};

// ================================================================
// Operator Generation Macro
// ================================================================


#define CREATE_OP(func_name, OPERATION)                                     \
template <int AD, typename A, typename... B>                     \
auto func_name(Tensor<A, AD> a, B... b)           \
{                                                                                  \
    return OPERATION::run(a, b...);                                                                           \
}




struct OperationAdd:public HardamardOperation<OperationAdd> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a + b;
    }
};



struct OperationSub:public HardamardOperation<OperationSub> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a - b;
    }
};


struct OperationMul:public HardamardOperation<OperationMul> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a * b;
    }
};


struct OperationDiv:public HardamardOperation<OperationDiv> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a / b;
    }
};



struct OperationAddEq:public HardamardOperation<OperationAddEq> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a += b;
    }
};



struct OperationSubEq:public HardamardOperation<OperationSubEq> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a -= b;
    }
};


struct OperationMulEq:public HardamardOperation<OperationMulEq> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a *= b;
    }
};


struct OperationDivEq:public HardamardOperation<OperationDivEq> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a /= b;
    }
};


struct OperationPow:public HardamardOperation<OperationPow> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return pow(a, b);
    }
};


struct OperationSet:public HardamardOperation<OperationSet> {
    template <typename A, typename B>
    __host__ __device__ static inline
    void apply(A& a, const B& b) {
        a = b;
    }
};


struct OperationMin:public HardamardOperation<OperationMin> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a < b ? a : b;
    }
};


struct OperationMax:public HardamardOperation<OperationMax> {
    template <typename A, typename B>
    __host__ __device__ static inline
    auto apply(const A& a, const B& b) {
        return a > b ? a : b;
    }
};


struct OperationRelu:public HardamardOperation<OperationRelu> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return a > 0 ? a : 0;
    }
};


struct OperationSigmoid:public HardamardOperation<OperationSigmoid> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return 1 / (1 + exp(-a));
    }
};


struct OperationTanh:public HardamardOperation<OperationTanh> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return tanh(a);
    }
};


struct OperationExp:public HardamardOperation<OperationExp> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return exp(a);
    }
};


struct OperationLog:public HardamardOperation<OperationLog> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return log(a);
    }
};


struct OperationSqrt:public HardamardOperation<OperationSqrt> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return sqrt(a);
    }
};


struct OperationNegate:public HardamardOperation<OperationNegate> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return -a;
    }
};


struct OperationAbs:public HardamardOperation<OperationAbs> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return abs(a);
    }
};


struct OperationToUnsignedLong:public HardamardOperation<OperationToUnsignedLong> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return (unsigned long)(a);
    }
};


struct OperationRound:public HardamardOperation<OperationRound> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return round(a);
    }
};


struct OperationFloor:public HardamardOperation<OperationFloor> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return floor(a);
    }
};



struct OperationCeil:public HardamardOperation<OperationCeil>
    {
        template <typename A>
        __host__ __device__ static inline
        auto apply(const A& a) {
            return ceil(a);
        }
    };

struct OperationSin:public HardamardOperation<OperationSin>
{
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return sin(a);
    }
};


struct OperationCos:public HardamardOperation<OperationCos>
{
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return cos(a);
    }
};


struct OperationTan:public HardamardOperation<OperationTan>
{
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return tan(a);
    }
}; 

// ================================================================
// User-Facing Tensor Operators
// ================================================================

CREATE_OP(operator+, OperationAdd);
CREATE_OP(operator-, OperationSub);
CREATE_OP(operator*, OperationMul);
CREATE_OP(operator/, OperationDiv);
CREATE_OP(pow, OperationPow);
CREATE_OP(min, OperationMin);
CREATE_OP(max, OperationMax);

CREATE_OP(operator+=, OperationAddEq);
CREATE_OP(operator-=, OperationSubEq);
CREATE_OP(operator*=, OperationMulEq);
CREATE_OP(operator/=, OperationDivEq);

CREATE_OP(tensor_copy, OperationSet);

CREATE_OP(round, OperationRound);
CREATE_OP(floor, OperationFloor);
CREATE_OP(ceil, OperationCeil);
CREATE_OP(sin, OperationSin);
CREATE_OP(cos, OperationCos);
CREATE_OP(tan, OperationTan);


