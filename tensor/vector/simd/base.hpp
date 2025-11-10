#include "../float4.hpp"

__attribute__((weak))
float4 float4::operator+(const float4& other) const {
    return float4(this->x + other.x, this->y + other.y, this->z + other.z, this->w + other.z);
};

__attribute__((weak))
float4 float4::operator-(const float4& other) const {
    return float4(this->x - other.x, this->y - other.y, this->z - other.z, this->w - other.z);
};

__attribute__((weak))
float4 float4::operator*(const float4& other) const {
    return float4(this->x * other.x, this->y * other.y, this->z * other.z, this->w * other.z);
};

__attribute__((weak))
float4 float4::operator/(const float4& other) const {
    return float4(this->x / other.x, this->y / other.y, this->z / other.z, this->w / other.z);
};

__attribute__((weak))
float4& float4::operator+=(const float4& other) {
    *this = *this + other;
    return *this;
};

__attribute__((weak))
float4& float4::operator-=(const float4& other) {
    *this = *this - other;
    return *this;
};

__attribute__((weak))
float4& float4::operator*=(const float4& other) {
    *this = *this * other;
    return *this;
};

__attribute__((weak))
float4& float4::operator/=(const float4& other) {
    *this = *this / other;
    return *this;
};