#ifndef BF16_HPP
#define BF16_HPP

#include <cstdint>
#ifndef __host__
#define __host__
#define __device__
#endif

struct float16
{
    uint16_t fvalue;
    operator uint16_t() const { return fvalue; }
    __host__ __device__ float16(float value)
    {
        uint32_t x = *((uint32_t *)&value);
        fvalue = (uint16_t)((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
    }
    __host__ __device__ float16() { fvalue = 0; }
    __host__ __device__ float16 operator=(float value)
    {
        uint32_t x = *((uint32_t *)&value);
        fvalue = (uint16_t)((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
        return *this;
    }
    __host__ __device__ float16 operator=(float16 value)
    {
        fvalue = value.fvalue;
        return *this;
    }

    template <typename T>
    __host__ __device__ float16 operator+(T value)
    {
        return float16(float(*this) + float(value));
    }

    template <typename T>
    __host__ __device__ float16 operator-(T value)
    {
        return float16(float(*this) - float(value));
    }

    template <typename T>
    __host__ __device__ float16 operator*(T value)
    {
        return float16(float(*this) * float(value));
    }

    template <typename T>
    __host__ __device__ float16 operator/(T value)
    {
        return float16(float(*this) / float(value));
    }

    __host__ __device__ operator float() const
    {
        uint32_t x = ((fvalue & 0x8000) << 16) | (((fvalue & 0x7c00) + 0x1C000) << 13) | ((fvalue & 0x03FF) << 13);
        return *((float *)&x);
    }
    
};

// #if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)


struct bfloat16;
static float bfloat16_to_float32(bfloat16 value);
static bfloat16 float32_to_bfloat16(float value);
struct bfloat16
{
    uint16_t value;
    __host__ __device__ operator float() const { 
        // cast as uint16_t, then cast as float32, then bitshift 16 bits to the left, then cast as float32
            uint32_t inter(uint32_t(value) << 16);
            return *((float *)&inter);
     }
     __host__ __device__ bfloat16() { this->value = 0; };

     __host__ __device__ bfloat16(float value) {
        uint32_t x = *((uint32_t *)&value);
        this->value = (uint16_t)(x >> 16);
    }

    
   
    // bfloat16 operator = (uint16_t value) {this->value = value; return *this;}
    // bfloat16 operator = (double value) {this->value = float32_to_bfloat16((float)value); return *this;}
    bfloat16&  __host__ __device__ operator+=(const bfloat16& valuein)
    {
        *this = *this + valuein;
        return *this;
    }

    bfloat16&  __host__ __device__ operator-=(const bfloat16& valuein)
    {
        *this = *this - valuein;
        return *this;
    }

    bfloat16&  __host__ __device__ operator*=(const bfloat16& valuein)
    {
        *this = *this * valuein;
        return *this;
    }

    template <typename T>
    bfloat16 __host__ __device__  operator-(const T& valuein) const { return bfloat16(float(*this) - float(valuein)); }

    template <typename T>
    bfloat16  __host__ __device__ operator+(const T& valuein) const { return bfloat16(float(*this) + float(valuein)); }

    template <typename T>
    bfloat16  __host__ __device__ operator*(const T& valuein) const { return bfloat16(float(*this) * float(valuein)); }

    template <typename T>
    bfloat16  __host__ __device__ operator/(const T& valuein) const { return bfloat16(float(*this) / float(valuein)); }

    
};



#endif