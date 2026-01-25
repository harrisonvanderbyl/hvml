
#pragma once
#include "tensor.hpp"
#include "kernels/interface.hpp"

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
        shape = Shape<-1>(1);
        strides = Shape<-1>(1);
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
                    for (int j = i+1; j < a.ndim()+1; j++)
                    {
                        // b.strides[-j] = b.strides[-j]/ shape[-i];
                        // std::cout << "Adjusting stride at dimension " << -j << " from " << b.strides[-j];
                        b.strides[-j] = b.strides[-j] / strides[-j];
                        // std::cout << " to " << b.strides[-j] << std::endl;
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

template <typename A, typename B> 
__host__ __device__ void atomicAddHip(A* a, const B& b);

template <typename A, typename B> 
__host__ __device__ void atomicAddCuda(A* a, const B& b);
// ================================================================
// CPU Kernel Implementation
// ================================================================


template < AssignmentType atype,ComputeType device_type>
struct AssignmentHelper {
    
};

template <>
struct AssignmentHelper<AssignmentType::Direct,ComputeType::kCPU> {
    template <typename T>
    static __host__ __device__ inline void assignOperation(T& dest, const T& src) {
        dest = src;
    }
};

template <>
struct AssignmentHelper<AssignmentType::InplaceAdd,ComputeType::kCPU> {
    template <typename T>
    static __host__ __device__ inline void assignOperation(T& dest, const T& src) {
        dest += src;
    }
};

template<>
struct AssignmentHelper<AssignmentType::Direct,ComputeType::kCUDA> {
    template <typename T>
    static __host__ __device__ inline void assignOperation(T& dest, const T& src) {
        dest = src;
    }
};

template <>
struct AssignmentHelper<AssignmentType::InplaceAdd,ComputeType::kCUDA> {
    template <typename T>
    static __host__ __device__ inline void assignOperation(T& dest, const T& src) {
        atomicAddCuda(&dest, src);
    }
};

template <>
struct AssignmentHelper<AssignmentType::Direct,ComputeType::kHIP> {
    template <typename T>
    static __host__ __device__ inline void assignOperation(T& dest, const T& src) {
        dest = src;
    }   
};

template <>
struct AssignmentHelper<AssignmentType::InplaceAdd,ComputeType::kHIP> {
    template <typename T>
    static __host__ __device__ inline void assignOperation(T& dest, const T&src) {
        atomicAddHip(&dest, src);
    }
};




     

template <ComputeType device, typename OP,
          typename... Args>
class BinaryKernel: public Kernel<device, unsigned long, Parameter<typename OutputTypeSelector<OP, Args...>::type>, Parameter<Args>...> 
{
    // Specializations will be defined below
    void call( unsigned long total_size,
        Parameter<typename OutputTypeSelector<OP, Args...>::type> output,
        Parameter<Args>... params
    )
    {
        throw std::runtime_error("BinaryKernel not implemented for this device type");
    }
};

// void parameter


template <typename OP, typename... Args>
__weak void call_cuda(
    unsigned long total_size,
    Parameter<typename OutputTypeSelector<OP,Args...>::type> output,
    Parameter<Args>... params
);


template <typename OP, typename... Args>
__weak void call_hip(
    unsigned long total_size,
    Parameter<typename OutputTypeSelector<OP,Args...>::type> output,
    Parameter<Args>... params
);

template <typename OP, typename... Args>
struct BinaryKernel<ComputeType::kCUDA, OP, Args...>
    : public Kernel<ComputeType::kCUDA, unsigned long, Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...> 
{
public:

    void call(
        unsigned long total_size,
        Parameter<typename OutputTypeSelector<OP, Args...>::type> output,
        Parameter<Args>... params
    ) override 
    {
        call_cuda<OP, Args...>(
            total_size,
            output,
            params...
        );
    }
};

template <typename OP, typename... Args>
struct BinaryKernel<ComputeType::kHIP, OP, Args...>
    : public Kernel<ComputeType::kHIP, unsigned long, Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...> 
{
public:

    void call(
        unsigned long total_size,
        Parameter<typename OutputTypeSelector<OP, Args...>::type> output,
        Parameter<Args>... params
    ) override 
    {
        call_hip<OP, Args...>(
            total_size,
            output,
            params...
        );
    }
};


template <typename OP, typename... Args>
struct BinaryKernel<ComputeType::kCPU, OP, Args...>
    : public Kernel<ComputeType::kCPU, unsigned long, Parameter<typename OutputTypeSelector<OP, Args...>::type>,Parameter<Args>...> 
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
                
                AssignmentHelper<OP::assignment_type,kCPU>::assignOperation(output.get_index(linear_idx) , OP::apply(
                        params.get_index(linear_idx)...
                ));
            }
        }
    }
};




// ================================================================
// Operation Application Helper
// ================================================================
template <typename OP, ComputeType device = ComputeType::kCPU>
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

            auto out_param = Tensor<Out, -1>({allocate_size}, global_device_manager.get_compute_device(device,0).default_memory_type);

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

template <typename OP, int alignDim = -1>
struct OperationSelector {
    constexpr static int AlignDim = alignDim;

    template <typename... Params>
    __host__ __device__ static inline
    auto apply(const Params&... params);

    template <int AD, typename A, typename... B>                     
    inline static auto run(Tensor<A, AD> a, B... b)           
    {                                   
        // OPERATION is the current class calling this operator(), including its decendents, so if a decendent calls this function, it will use its own apply function                                               
        auto& mem_device = *a.device;
        ComputeType compute_type = mem_device.default_compute_type;
        switch (compute_type) {
            case ComputeType::kCPU:
                return ApplyKernelOperationHelper<OP, ComputeType::kCPU>::apply(Parameter(a), Parameter(b)...);
            case ComputeType::kCUDA:
                return ApplyKernelOperationHelper<OP, ComputeType::kCUDA>::apply(Parameter(a), Parameter(b)...);
            case ComputeType::kHIP:
                return ApplyKernelOperationHelper<OP, ComputeType::kHIP>::apply(Parameter(a), Parameter(b)...);
            default:
                throw std::runtime_error("Unsupported compute type");
        }                                                                          
    }

    
   
};



template <typename OP, int alignDim = -1>
struct HardamardOperation: public OperationSelector<OP, alignDim> {
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

    static constexpr AssignmentType assignment_type = AssignmentType::Direct; 

    template <typename T>
    static inline auto broadcast_output(T& input){
        return input;
    }

};
// eg, 
// eg, 3,2 @ 4,2 -> 3,4 

// eg, 3,1,2 @ 1,4,2 -> 3,4,1(2) 

template <typename OP, int reductionDim>
struct ReductionOperation: public OperationSelector<OP, reductionDim> {
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


    static constexpr AssignmentType assignment_type = AssignmentType::InplaceAdd;

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
        #if defined(__CUDACC__) || defined(__HIPCC__)
        atomicAdd(&output, input);
        #else
        output += input;
        #endif
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
template <typename... OPlist>
struct ChainOperations: public HardamardOperation<ChainOperations<OPlist...>> {
    template <typename... Args>
    __host__ __device__ static inline
    auto apply(Args&&... args) {
        return apply_impl<OPlist...>(std::forward<Args>(args)...);
    }

private:
    // Base case: single operation
    template <typename First>
    __host__ __device__ static inline
    auto apply_impl(auto&&... args) {
        return First::apply(std::forward<decltype(args)>(args)...);
    }

    // Recursive case: chain operations
    template <typename First, typename Second, typename... Rest>
    __host__ __device__ static inline
    auto apply_impl(auto&&... args) {
        auto result = First::apply(std::forward<decltype(args)>(args)...);
        return apply_impl<Second, Rest...>(std::forward<decltype(result)>(result));
    }
};

