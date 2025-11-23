#include "tensor.hpp"
#include "kernels/interface.hpp"



template <DeviceType device_type, typename inputA, typename inputB, typename outputType>
class AddKernel:public Kernel<device_type, inputA*, long, long*, long*, inputB*, long*, outputType*> {
    public:

    void call(inputA* a, long a_ndim,  long* a_shape, long* a_strides, inputB* b, long* b_strides, outputType* out) override {
        
        long offseta = 0;
        long offsetb = 0;
        long totalsize = 1;
        for(long i = 0; i < a_ndim; i++){
            totalsize *= a_shape[i];
        };

        for (long index = 0; index < totalsize; index++) {
            long remainder = index;
            offseta = 0;
            offsetb = 0;
            for(long dim = a_ndim - 1; dim >= 0; dim--) {
                long coord = remainder % a_shape[dim];
                remainder = remainder / a_shape[dim];
                offseta += coord * a_strides[dim];
                offsetb += coord * b_strides[dim];
            }
            out[index] = a[offseta] + b[offsetb];
        }
    }    
};

template <DeviceType device_type, typename inputA, typename inputB, typename outputType>
class MulKernel:public Kernel<device_type, inputA*, long, long*, long*, inputB*, long*, outputType*> {
    public:

    void call(inputA* a, long a_ndim,  long* a_shape, long* a_strides, inputB* b, long* b_strides, outputType* out) override {
        
        long offseta = 0;
        long offsetb = 0;
        long totalsize = 1;
        for(long i = 0; i < a_ndim; i++){
            totalsize *= a_shape[i];
        };

        for (long index = 0; index < totalsize; index++) {
            long remainder = index;
            offseta = 0;
            offsetb = 0;
            for(long dim = a_ndim - 1; dim >= 0; dim--) {
                long coord = remainder % a_shape[dim];
                remainder = remainder / a_shape[dim];
                offseta += coord * a_strides[dim];
                offsetb += coord * b_strides[dim];
            }
            out[index] = a[offseta] * b[offsetb];
        }
    }    
};

template <DeviceType device_type, typename inputA, typename inputB, typename outputType>
class SubKernel:public Kernel<device_type, inputA*, long, long*, long*, inputB*, long*, outputType*> {
    public:

    void call(inputA* a, long a_ndim,  long* a_shape, long* a_strides, inputB* b, long* b_strides, outputType* out) override {
        
        long offseta = 0;
        long offsetb = 0;
        long totalsize = 1;
        for(long i = 0; i < a_ndim; i++){
            totalsize *= a_shape[i];
        };

        for (long index = 0; index < totalsize; index++) {
            long remainder = index;
            offseta = 0;
            offsetb = 0;
            for(long dim = a_ndim - 1; dim >= 0; dim--) {
                long coord = remainder % a_shape[dim];
                remainder = remainder / a_shape[dim];
                offseta += coord * a_strides[dim];
                offsetb += coord * b_strides[dim];
            }
            out[index] = a[offseta] - b[offsetb];
        }
    }    
};

template <DeviceType device_type, typename inputA, typename inputB, typename outputType>
class DivKernel:public Kernel<device_type, inputA*, long, long*, long*, inputB*, long*, outputType*> {
    public:

    void call(inputA* a, long a_ndim,  long* a_shape, long* a_strides, inputB* b, long* b_strides, outputType* out) override {
        
        long offseta = 0;
        long offsetb = 0;
        long totalsize = 1;
        for(long i = 0; i < a_ndim; i++){
            totalsize *= a_shape[i];
        };

        for (long index = 0; index < totalsize; index++) {
            long remainder = index;
            offseta = 0;
            offsetb = 0;
            for(long dim = a_ndim - 1; dim >= 0; dim--) {
                long coord = remainder % a_shape[dim];
                remainder = remainder / a_shape[dim];
                offseta += coord * a_strides[dim];
                offsetb += coord * b_strides[dim];
            }
            out[index] = a[offseta] / b[offsetb];
        }
    }    
};



#define applyKernelOperation(opKernel) {\
    Shape<std::max(D,E)> output;\
    auto longestOutputDims = std::max(a.shape.ndim(), b.shape.ndim());\
    \
    for (size_t i = 0; i < longestOutputDims; i++)\
    {\
        long ashape = ((a.shape.ndim() - longestOutputDims + i) < 0) ? 1 : a.shape[a.shape.ndim() - longestOutputDims + i];\
        long bshape = ((b.shape.ndim() - longestOutputDims + i) < 0) ? 1 : b.shape[b.shape.ndim() - longestOutputDims + i];\
        if (ashape != bshape && ashape != 1 && bshape != 1)\
        {\
            std::cout << "Error: Incompatible shapes for multiplication\n"\
                      << ashape << " " << bshape << "\n";\
            exit(0);\
        }\
        ((unsigned long*)&output)[i] = std::max(ashape, bshape);\
    }\
\
    auto aa = a.broadcast(output);\
    auto bb = b.broadcast(output);\
\
    opKernel A = opKernel<kCPU, U,V,O>();\
    auto totalOutputSize = output.total_size();\
    auto buffers = A.template CreateBuffers<O>({totalOutputSize});\
    buffers.Allocate();\
    auto outPtr = buffers.template getPointer<0>();\
    A(aa.data, output.ndim(), (long*)(&output), (long*)(&aa.strides), bb.data, (long*)(&bb.strides), outPtr);\
\
    return Tensor<O, std::max(D,E)>(output, outPtr, a.device_type);\
}\

#define createOp(name, op) \
template <typename U, typename V, int D, int E, typename O = typename std::common_type<U,V>::type> \
Tensor<O, std::max(D,E)> name(Tensor<U, D> a, Tensor<V, E> b) \
{\
    applyKernelOperation(op)\
} 



createOp(operator-, SubKernel)

createOp(operator+, AddKernel)

createOp(operator*, MulKernel)

createOp(operator/, DivKernel)
