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

    __host__ __device__ static uint16_t float_to_fp16(float value)
    {
        uint32_t x = *((uint32_t *)&value);
        uint16_t sign    = (x >> 16) & 0x8000;
        int32_t  exp     = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = x & 0x007FFFFF;

        if (exp <= 0) {
            // Flush to zero (includes true zero and very small denormals)
            return sign;
        } else if (exp >= 31) {
            // Overflow -> infinity
            return sign | 0x7C00;
        } else {
            return (uint16_t)(sign | (exp << 10) | (mantissa >> 13));
        }
    }

    __host__ __device__ float16(float value) { fvalue = float_to_fp16(value); }
    __host__ __device__ float16() { fvalue = 0; }

    __host__ __device__ float16 operator=(float value)
    {
        fvalue = float_to_fp16(value);
        return *this;
    }
    __host__ __device__ float16 operator=(float16 value)
    {
        fvalue = value.fvalue;
        return *this;
    }

    template <typename T>
    __host__ __device__ float16 operator+(T value) { return float16(float(*this) + float(value)); }
    template <typename T>
    __host__ __device__ float16 operator-(T value) { return float16(float(*this) - float(value)); }
    template <typename T>
    __host__ __device__ float16 operator*(T value) { return float16(float(*this) * float(value)); }
    template <typename T>
    __host__ __device__ float16 operator/(T value) { return float16(float(*this) / float(value)); }

    __host__ __device__ operator float() const
    {
        uint32_t x = ((fvalue & 0x8000) << 16) | (((fvalue & 0x7c00) + 0x1C000) << 13) | ((fvalue & 0x03FF) << 13);
        return *((float *)&x);
    }
};

struct bfloat16;
static float bfloat16_to_float32(bfloat16 value);
static bfloat16 float32_to_bfloat16(float value);
struct __attribute__((packed)) bfloat16
{
    uint16_t value;
    __host__ __device__ operator float() const {
        uint32_t inter(uint32_t(value) << 16);
        return *((float *)&inter);
    }
    __host__ __device__ bfloat16() { this->value = 0; };
    __host__ __device__ bfloat16(float value) {
        uint32_t x = *((uint32_t *)&value);
        this->value = (uint16_t)(x >> 16);
    }

    bfloat16& __host__ __device__ operator+=(const bfloat16& valuein) { *this = *this + valuein; return *this; }
    bfloat16& __host__ __device__ operator-=(const bfloat16& valuein) { *this = *this - valuein; return *this; }
    bfloat16& __host__ __device__ operator*=(const bfloat16& valuein) { *this = *this * valuein; return *this; }

    template <typename T>
    bfloat16 __host__ __device__ operator-(const T& valuein) const { return bfloat16(float(*this) - float(valuein)); }
    template <typename T>
    bfloat16 __host__ __device__ operator+(const T& valuein) const { return bfloat16(float(*this) + float(valuein)); }
    template <typename T>
    bfloat16 __host__ __device__ operator*(const T& valuein) const { return bfloat16(float(*this) * float(valuein)); }
    template <typename T>
    bfloat16 __host__ __device__ operator/(const T& valuein) const { return bfloat16(float(*this) / float(valuein)); }
};

// int24 for depth buffer and other stuff
struct __attribute__((packed)) int24
{
    uint8_t bytes[3];
    __host__ __device__ operator uint32_t() const {
        return (uint32_t(bytes[0]) << 16) | (uint32_t(bytes[1]) << 8) | uint32_t(bytes[2]);
    }
    __host__ __device__ int24() { bytes[0] = bytes[1] = bytes[2] = 0; };
    __host__ __device__ int24(uint32_t value) {
        bytes[0] = (value >> 16) & 0xFF;
        bytes[1] = (value >> 8) & 0xFF;
        bytes[2] = value & 0xFF;
    }
};
#endif