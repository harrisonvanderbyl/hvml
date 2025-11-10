#include <arm_neon.h>    // NEON for ARM
#include <iostream>
#include "../float4.hpp"

float4 float4::operator+(const float4& other) const {
        float4 n;
        vst1q_f32(&n.x, vaddq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return n;
}

float4 float4::operator-(const float4& other) const {
        float4 n;
        vst1q_f32(&n.x, vsubq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return n;
}

float4 float4::operator*(const float4& other) const {
        float4 n;
        vst1q_f32(&n.x, vmulq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return n;
}

float4 float4::operator/(const float4& other) const {
        float4 n;
        vst1q_f32(&n.x, vdivq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return n;
}

float4& float4::operator+=(const float4& other) {
        vst1q_f32(&(this->x), vaddq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return *this;
}

float4& float4::operator-=(const float4& other) {
        vst1q_f32(&(this->x), vsubq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return *this;
}

float4& float4::operator*=(const float4& other) {
        vst1q_f32(&(this->x), vmulq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return *this;
}

float4& float4::operator/=(const float4& other) {
        vst1q_f32(&(this->x), vdivq_f32(*(float32x4_t*)this, *(float32x4_t*)&other));
        return *this;
}