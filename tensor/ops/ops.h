
#include <iostream>
#include <map>
#include "enums/dtype.hpp"
#include "enums/device.hpp"
#include "tensor.hpp"
#include <stdarg.h>
// ops is an array of function pointers for dynamic library inclusion
#ifndef OPS
#define OPS

typedef void (*funcpointer)(void*, void*, void*, size_t&);

// template <typename U, typename V, typename O, int D>
// Tensor<O> applyOperation(Tensor<U> a, Tensor<V> b, void (*op)(U&, V&, O&, size_t&) )
#define applyOperation(op) {\
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
    Tensor<O, std::max(D,E)> out = Tensor<O, std::max(D,E)>(output, a.device_type);\
\
    auto aa = a.broadcast(output);\
    auto bb = b.broadcast(output);\
\
    for (size_t i = 0; i < out.total_size; )\
    {\
         op(aa.flatget(i) , bb.flatget(i), out.flatget(i), i);\
    }\
\
    return out;\
}\

#define applyAlterOperation(op) {\
    auto bb = b.broadcast(a.shape);\
\
    for (size_t i = 0; i < bb.total_size; )\
    {\
         op(a.flatget(i) , bb.flatget(i), i);\
    }\
\
    return a;\
}\

template <typename U, typename V, typename O>
void add(U& a, V& b, O& o, size_t& i)
{
    
    i++;
    o = a + b;
}

template <typename U, typename V, typename O>
void sub(U& a, V& b, O& o, size_t& i)
{
    o = a - b;
    i++;
}

template <typename U, typename V, typename O>
void mul(U& a, V& b, O& o, size_t& i)
{
    o = a * b;
    i++;
}

template <typename U, typename V, typename O>
void div(U& a, V& b, O& o, size_t& i)
{
    o = a / b;
    i++;
}

template <typename U, typename V>
void plusequals(U& a, V& b, size_t& i)
{
    a = a + b;
    i++;
}
// template <typename T>
// void pow(T& a, T& b, T& out, size_t& i)
// {
//     out = pow(a, b);
//     i++;
// }

#define createOp(name, op) \
template <typename U, typename V, int D, int E, typename O = typename std::common_type<U,V>::type> \
Tensor<O, std::max(D,E)> name(Tensor<U, D> a, Tensor<V, E> b) \
{\
    applyOperation(op)\
} 

#define createAlterOp(name, op) \
template <typename U, typename V, int D, int E> \
Tensor<U, D> name(Tensor<U, D> a, Tensor<V, E> b) \
{\
    applyAlterOperation(op)\
} 

createOp(operator-, sub)

createOp(operator+, add)

createOp(operator*, mul)

createOp(operator/, div)

createAlterOp(operator+=, plusequals)

// template <typename T>
// Tensor<T> operator^(Ten<T>sor a, Ten<T>sor b)
// {
//     applyOperation(a,b,pow)
// }


#endif