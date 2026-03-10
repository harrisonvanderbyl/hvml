#include <iostream>
#include <cstdint>
#ifndef Hvec_HPP
#define Hvec_HPP
#if !defined(__host__)
#define __host__
#define __device__
#endif

enum Intrinsics
{
    None,
    // cpu intrinsics
    SSE2,
    AVX,
    AVX2,
    AVX512,
    // arm intrinsics
    NEON,
    // gpu intrinsics
    CUDA,
    HIP
};

// swizzle bit
// 0 = xxxx
// 1 = xxxy
// 2 = xxxz
// 3 = xxxw
// 4 = xxyx
// ...
// 255 = wwww


template <int numitems = 4>
static constexpr __device__ __host__ std::conditional_t<numitems <= 4, uint8_t, size_t> get_swizzle_component(uint8_t swizzle, int index)
{
    if constexpr (numitems <= 4)
    {
        return (swizzle >> ((3 - index) * 2)) & 0b11;
    }
    else
    {
        return index;
    }
}


template <typename T, uint8_t swizzle, int numitems>
struct constIfSwizzleRepeats;

// Specialization for 4 components
template <typename T, uint8_t swizzle>
struct constIfSwizzleRepeats<T, swizzle, 4>
{
    static constexpr bool has_duplicates = 
        // Compare all pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        (get_swizzle_component(swizzle, 0) == get_swizzle_component(swizzle, 1)) ||
        (get_swizzle_component(swizzle, 0) == get_swizzle_component(swizzle, 2)) ||
        (get_swizzle_component(swizzle, 0) == get_swizzle_component(swizzle, 3)) ||
        (get_swizzle_component(swizzle, 1) == get_swizzle_component(swizzle, 2)) ||
        (get_swizzle_component(swizzle, 1) == get_swizzle_component(swizzle, 3)) ||
        (get_swizzle_component(swizzle, 2) == get_swizzle_component(swizzle, 3));
    
    using type = std::conditional_t<has_duplicates, const T, T>;
};

// Specialization for 3 components
template <typename T, uint8_t swizzle>
struct constIfSwizzleRepeats<T, swizzle, 3>
{
    static constexpr bool has_duplicates = 
        (get_swizzle_component(swizzle, 0) == get_swizzle_component(swizzle, 1)) ||
        (get_swizzle_component(swizzle, 0) == get_swizzle_component(swizzle, 2)) ||
        (get_swizzle_component(swizzle, 1) == get_swizzle_component(swizzle, 2));
    
    using type = std::conditional_t<has_duplicates, const T, T>;
};

// Specialization for 2 components
template <typename T, uint8_t swizzle>
struct constIfSwizzleRepeats<T, swizzle, 2>
{
    static constexpr bool has_duplicates = 
        (get_swizzle_component(swizzle, 0) == get_swizzle_component(swizzle, 1));
    
    using type = std::conditional_t<has_duplicates, const T, T>;
};

// Specialization for 1 component (never has duplicates)
template <typename T, uint8_t swizzle>
struct constIfSwizzleRepeats<T, swizzle, 1>
{
    static constexpr bool has_duplicates = false;
    
    using type = T;
};

template <uint8_t swizzle1, uint8_t swizzle2>
struct combine_swizzles // xxyy swizzled by xy = xx
{
    // so, xxxx swizzled by xyzw = xxxx
    static constexpr uint8_t swiz = 
        (get_swizzle_component(swizzle1,get_swizzle_component(swizzle2,0)) << 6) |
        (get_swizzle_component(swizzle1,get_swizzle_component(swizzle2,1)) << 4) |
        (get_swizzle_component(swizzle1,get_swizzle_component(swizzle2,2)) << 2) |
        (get_swizzle_component(swizzle1,get_swizzle_component(swizzle2,3)) << 0);
};


#define createDuoOperation(opname, oper)                                      \
    template <uint8_t otherswizzle, typename U, typename shared = std::common_type_t<T,U>> \
    __host__ __device__ auto opname(const Hvec<U, numitems, otherswizzle>& other) const \
    {                                                                        \
        Hvec<shared, numitems> out;                                         \
        for (int i = 0; i < numitems; i++)                                  \
        {                                                                    \
            out.data[i] = this->data[get_swizzle_component<numitems>(swizzle,i)] oper \
                              other.data[get_swizzle_component<numitems>(otherswizzle,i)]; \
        }                                                                    \
        return out;                                                         \
    };

#define createSingleOperation(opname, oper)                                   \
    __host__ __device__ auto opname() const                                 \
    {                                                                        \
        Hvec<T, numitems> out;                                         \
        for (int i = 0; i < numitems; i++)                                  \
        {                                                                    \
            out.data[i] = oper(this->data[get_swizzle_component<numitems>(swizzle,i)]); \
        }                                                                    \
        return out;                                                         \
    };

#define createScalarOperation(opname, oper)                                   \
    template <typename U>                                                    \
    __host__ __device__ auto opname(const U& scalar) const                 \
    {                                                                        \
        Hvec<T, numitems> out;                                         \
        for (int i = 0; i < numitems; i++)                                  \
        {                                                                    \
            out.data[i] = this->data[get_swizzle_component<numitems>(swizzle,i)] oper scalar; \
        }                                                                    \
        return out;                                                         \
    };

#define createDuoAssignmentOperation(opname, oper)                           \
    template <uint8_t otherswizzle, typename U> \
    __host__ __device__ Hvec<T, numitems, swizzle>& opname(const Hvec<U, numitems, otherswizzle>& other) \
    {                                                                        \
        T temp[numitems];                                                   \
        \
        for (int i = 0; i < numitems; i++)                                  \
        {                                                                    \
            temp[i] = this->data[get_swizzle_component<numitems>(swizzle,i)];         \
        }                                                                    \
                                                                             \
        for (int i = 0; i < numitems; i++)                                  \
        {                                                                    \
            this->data[get_swizzle_component<numitems>(swizzle,i)] = temp[i] oper     \
                other.data[get_swizzle_component<numitems>(otherswizzle,i)];          \
        }                                                                    \
        return *this;                                                       \
    };
    
#define createScalarAssignmentOperation(opname, oper)                           \
    __host__ __device__ Hvec<T, numitems, swizzle>& opname(const T& scalar) \
    {                                                                        \
        for (int i = 0; i < numitems; i++)                                  \
        {                                                                    \
            this->data[get_swizzle_component<numitems>(swizzle,i)] oper scalar;                                                     \
        }                                                                    \
        return *this;                                                       \
    };

#define createSwizzleReference(swizzeleName, swizzleBits, numout, maxvalues)  \
    template <int numitemss = numitems, \
              typename = std::enable_if_t<(numitemss >= maxvalues && numitemss <= 4)>> \
    __host__ __device__ inline \
    typename constIfSwizzleRepeats< \
        Hvec<T, numout, combine_swizzles<swizzle, swizzleBits>::swiz>, \
        combine_swizzles<swizzle, swizzleBits>::swiz, numout \
    >::type& swizzeleName() { \
        return *reinterpret_cast< \
            typename constIfSwizzleRepeats< \
                Hvec<T, numout, combine_swizzles<swizzle, swizzleBits>::swiz>, \
                combine_swizzles<swizzle, swizzleBits>::swiz, numout \
            >::type * \
        >(this); \
    } \
    \
    template <int numitemss = numitems, \
              typename = std::enable_if_t<(numitemss >= maxvalues && numitemss <= 4)>> \
    __host__ __device__ inline \
    const Hvec<T, numout, combine_swizzles<swizzle, swizzleBits>::swiz>& swizzeleName() const { \
        return *reinterpret_cast< \
            const Hvec<T, numout, combine_swizzles<swizzle, swizzleBits>::swiz> * \
        >(this); \
    }

#define createSwizzleSingle(swizzeleName, number, maxvalues)  \
    template <int numitemss = numitems, \
              typename = std::enable_if_t<(numitemss >= maxvalues && numitemss <= 4)>> \
    __host__ __device__ inline T& swizzeleName() { \
        return this->data[get_swizzle_component<numitems>(swizzle, number)]; \
    }

// 0 -> x, 1 -> y, 2 -> z, 3 -> w
#define createSwizzle4(firstThreeletters, firstthreedigits, maxvalues) \
    createSwizzleReference(x##firstThreeletters, 0b00##firstthreedigits,4, std::max(maxvalues, 1)) \
    createSwizzleReference(y##firstThreeletters, 0b01##firstthreedigits,4, std::max(maxvalues, 2)) \
    createSwizzleReference(z##firstThreeletters, 0b10##firstthreedigits,4, std::max(maxvalues, 3)) \
    createSwizzleReference(w##firstThreeletters, 0b11##firstthreedigits,4, std::max(maxvalues, 4)) \
    createSwizzleReference(firstThreeletters, 0b##firstthreedigits##11,3, maxvalues)

#define createSwizzle3(firstTwoletters, firsttwodigits, maxvalues) \
    createSwizzle4(x##firstTwoletters, 00##firsttwodigits, std::max(maxvalues, 1)) \
    createSwizzle4(y##firstTwoletters, 01##firsttwodigits, std::max(maxvalues, 2)) \
    createSwizzle4(z##firstTwoletters, 10##firsttwodigits, std::max(maxvalues, 3)) \
    createSwizzle4(w##firstTwoletters, 11##firsttwodigits, std::max(maxvalues, 4)) \
    createSwizzleReference(firstTwoletters, 0b##firsttwodigits##1011,2, maxvalues)

#define createSwizzle2(firstletter, firstdigit, maxvalues) \
    createSwizzle3(x##firstletter, 00##firstdigit, std::max(maxvalues, 1)) \
    createSwizzle3(y##firstletter, 01##firstdigit, std::max(maxvalues, 2)) \
    createSwizzle3(z##firstletter, 10##firstdigit, std::max(maxvalues, 3)) \
    createSwizzle3(w##firstletter, 11##firstdigit, std::max(maxvalues, 4)) \
    createSwizzleSingle(firstletter, 0b##firstdigit, maxvalues)

#define createSwizzle1() \
    createSwizzle2(x, 00, 1) \
    createSwizzle2(y, 01, 2) \
    createSwizzle2(z, 10, 3) \
    createSwizzle2(w, 11, 4) \
    // createSwizzleReference(all, 0b00000000)

    


template <typename T, int numitems = 1, uint8_t swizzle = 0b00011011>
struct Hvec
{
    T data[numitems];

    __host__ __device__ Hvec()
    {
    };

    __host__ __device__ Hvec(const T& val)
    {
        for (int i = 0; i < numitems; i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i)] = val;
        }
    };
    

    // __host__ __device__ operator T&()
    // {
    //     static_assert(numitems == 1, "Can only convert to T& if numitems is 1");
    //     return *((T*)data + get_swizzle_component<numitems>(swizzle,0));
    // };

    template <typename U, uint8_t oswiz, typename... Args>
    __host__ __device__ Hvec(const Hvec<U, numitems - sizeof...(Args), oswiz>& other, Args... args)
    {
        for (int i = 0; i < numitems - sizeof...(Args); i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i)] = other.data[get_swizzle_component<numitems>(oswiz,i)];
        }
        T vals[] = {static_cast<T>(args)...};
        for (int i = 0; i < sizeof...(Args); i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i + numitems - sizeof...(Args))] = vals[i];
        }
        
    };

    // equality operator
    template <uint8_t oswiz>
    __host__ __device__ bool operator==(const Hvec<T, numitems, oswiz>& other) const
    {
        for (int i = 0; i < numitems; i++)
        {
            if (this->data[get_swizzle_component<numitems>(swizzle,i)] != other.data[get_swizzle_component<numitems>(oswiz,i)])
            {
                return false;
            }
        }
        return true;
    };

     template <typename U, uint8_t oswiz>
    __host__ __device__ Hvec(const Hvec<U, numitems, oswiz>& other)
    {
        for (int i = 0; i < numitems; i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i)] = other.data[get_swizzle_component<numitems>(oswiz,i)];
        }
    };

    template <typename... Args>
    __host__ __device__ Hvec(const Args&... args)
    {
        T vals[] = {static_cast<T>(args)...};
        for (int i = 0; i < numitems; i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i)] = vals[i];
        }
    };

    

    template <uint8_t otherSwizzle>
    __host__ __device__ Hvec(const Hvec<T, numitems, otherSwizzle>& other)
    {
        for (int i = 0; i < numitems; i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i)] = other.data[get_swizzle_component<numitems>(otherSwizzle,i)];
        }
    };

    template <uint8_t otherSwizzle>
    __host__ __device__ Hvec& operator=(const Hvec<T, numitems, otherSwizzle>& other)
    {

        T temp[numitems];
        for (int i = 0; i < numitems; i++)
        {
            temp[i] = other.data[get_swizzle_component<numitems>(otherSwizzle,i)];
        }

        for (int i = 0; i < numitems; i++)
        {
            data[get_swizzle_component<numitems>(swizzle,i)] = temp[i];
        }
        return *this;
    };

    __host__ __device__ T &operator[](int index)
    {
        return data[get_swizzle_component<numitems>(swizzle,index)];
    };

    __host__ __device__ const T &operator[](int index) const
    {
        return data[get_swizzle_component<numitems>(swizzle,index)];
    };

    // operator dot and operator length
    template <uint8_t swizzle2>
    __host__ __device__ T dot(const Hvec<T, numitems, swizzle2>& other) const
    {
        T result = T(0);
        for (int i = 0; i < numitems; i++)
        {
            result += this->data[get_swizzle_component<numitems>(swizzle,i)] * other.data[get_swizzle_component<numitems>(swizzle2,i)];
        }
        return result;
    };

    __host__ __device__ T length() const
    {
        return sqrt(this->dot(*this));
    };

    __host__ __device__ Hvec<T, numitems> normalized() const
    {
        T len = length();
        Hvec<T, numitems> out;
        for (int i = 0; i < numitems; i++)
        {
            out.data[i] = this->data[get_swizzle_component<numitems>(swizzle,i)] / len;
        }
        return out;
    };

    // cross product for 3D vectors
    __host__ __device__ Hvec<T, 3> cross(const Hvec<T, 3, swizzle>& other) const
    {
        assert(numitems == 3 && "Cross product is only defined for 3D vectors.");
        Hvec<T, 3> result;
        result.data[0] = this->data[get_swizzle_component<numitems>(swizzle,1)] * other.data[get_swizzle_component<numitems>(swizzle,2)] -
                         this->data[get_swizzle_component<numitems>(swizzle,2)] * other.data[get_swizzle_component<numitems>(swizzle,1)];
        result.data[1] = this->data[get_swizzle_component<numitems>(swizzle,2)] * other.data[get_swizzle_component<numitems>(swizzle,0)] -
                         this->data[get_swizzle_component<numitems>(swizzle,0)] * other.data[get_swizzle_component<numitems>(swizzle,2)];
        result.data[2] = this->data[get_swizzle_component<numitems>(swizzle,0)] * other.data[get_swizzle_component<numitems>(swizzle,1)] -
                         this->data[get_swizzle_component<numitems>(swizzle,1)] * other.data[get_swizzle_component<numitems>(swizzle,0)];
        return result;
    };


    createDuoOperation(operator+, +)
    createDuoOperation(operator-, -)
    createDuoOperation(operator*, *)
    createDuoOperation(operator/, /)

    createScalarOperation(operator+, +)
    createScalarOperation(operator-, -)
    createScalarOperation(operator*, *)
    createScalarOperation(operator/, /)

    createSingleOperation(operator-, -)

    createDuoAssignmentOperation(operator+=, +=)
    createDuoAssignmentOperation(operator-=, -=)
    createDuoAssignmentOperation(operator*=, *=)
    createDuoAssignmentOperation(operator/=, /=)

    createScalarAssignmentOperation(operator+=, +=)
    createScalarAssignmentOperation(operator-=, -=)
    createScalarAssignmentOperation(operator*=, *=)
    createScalarAssignmentOperation(operator/=, /=)

    createSwizzle1()

    // std cout
    friend std::ostream &operator<<(std::ostream &os, const Hvec<T, numitems, swizzle> &vec)
    {
        os << "(";
        for (int i = 0; i < numitems; i++)
        {
            os << vec.data[get_swizzle_component<numitems>(swizzle,i)];
            if (i < numitems - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    };
};


typedef Hvec<float, 2> float32x2;

typedef Hvec<float, 3> float32x3;

typedef Hvec<float, 4> float32x4;



typedef Hvec<int, 2> int32x2;

typedef Hvec<int, 3> int32x3;

typedef Hvec<int, 4> int32x4;

typedef Hvec<bfloat16,4> bfloat16x4;

typedef Hvec<bfloat16,3> bfloat16x3;

typedef Hvec<float16,4> float16x4;

struct uint84: public Hvec<uint8_t, 4>
{

    using Hvec<uint8_t, 4>::Hvec;

    __host__ __device__ uint84(Hvec<uint8_t, 4> other) : Hvec<uint8_t, 4>(other) {}

    __host__ __device__ uint84(uint32_t v){
        data[0] = (v >> 24) & 0xFF;
        data[1] = (v >> 16) & 0xFF;
        data[2] = (v >> 8) & 0xFF;
        data[3] = v & 0xFF;
    };

    __host__ __device__ operator uint32_t() const {
        return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    };

};

#endif