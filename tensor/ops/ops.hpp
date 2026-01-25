#ifndef OPMASTER
#define OPMASTER
#include "ops/common.hpp"
#if defined(__CUDACC__)
#include "ops/nvidia/ops.cu"
#endif
#if defined(__HIPCC__)
#include "ops/hip/ops.hpp"
#endif

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

struct OperationSqr:public HardamardOperation<OperationSqr> {
    template <typename A>
    __host__ __device__ static inline
    auto apply(const A& a) {
        return a*a;
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

#endif