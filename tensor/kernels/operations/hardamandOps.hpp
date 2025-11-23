#include "tensor.hpp"
#include "../interface.hpp"





#define applyKernelOperation(op) {\
    Shape<std::max(D,E)> output;\
    auto longestOutputDims = std::max(a.shape.ndim(), b.shape.ndim());\
    \
    for (size_t i = 0; i < longestOutputDims; i++)\
    {\
        int ashape = ((a.shape.ndim() - longestOutputDims + i) < 0) ? 1 : a.shape[a.shape.ndim() - longestOutputDims + i];\
        int bshape = ((b.shape.ndim() - longestOutputDims + i) < 0) ? 1 : b.shape[b.shape.ndim() - longestOutputDims + i];\
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
    AddKernel(a.device_type, output).call(a.data,b.data)\
\
    return Tensor<O, std::max(D,E)>(output, a.device_type);\
}\

#define createOp(name, op) \
template <typename U, typename V, int D, int E, typename O = typename std::common_type<U,V>::type> \
Tensor<O, std::max(D,E)> name(Tensor<U, D> a, Tensor<V, E> b) \
{\
    applyKernelOperation(op)\
} 



createOp(operator-, sub)

createOp(operator+, add)

createOp(operator*, mul)

createOp(operator/, div)
