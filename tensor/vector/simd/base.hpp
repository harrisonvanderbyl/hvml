#include "../float4.hpp"

__attribute__((weak))
float32x4 float32x4::operator+(const float32x4& other) const {
    return float32x4(this->x + other.x, this->y + other.y, this->z + other.z, this->w + other.z);
};

__attribute__((weak))
float32x4 float32x4::operator-(const float32x4& other) const {
    return float32x4(this->x - other.x, this->y - other.y, this->z - other.z, this->w - other.z);
};

__attribute__((weak))
float32x4 float32x4::operator*(const float32x4& other) const {
    return float32x4(this->x * other.x, this->y * other.y, this->z * other.z, this->w * other.z);
};

__attribute__((weak))
float32x4 float32x4::operator/(const float32x4& other) const {
    return float32x4(this->x / other.x, this->y / other.y, this->z / other.z, this->w / other.z);
};

__attribute__((weak))
float32x4& float32x4::operator+=(const float32x4& other) {
    *this = *this + other;
    return *this;
};

__attribute__((weak))
float32x4& float32x4::operator-=(const float32x4& other) {
    *this = *this - other;
    return *this;
};

__attribute__((weak))
float32x4& float32x4::operator*=(const float32x4& other) {
    *this = *this * other;
    return *this;
};

__attribute__((weak))
float32x4& float32x4::operator/=(const float32x4& other) {
    *this = *this / other;
    return *this;
};