#include <ostream>

struct complex32
{
    float real;
    float imag;

    complex32() : real(0.0f), imag(0.0f) {}
    complex32(float r, float i) : real(r), imag(i) {}
    complex32(float r) : real(r), imag(0.0f) {}

    complex32 operator+(const complex32& other) const
    {
        return complex32(real + other.real, imag + other.imag);
    }

    complex32 operator-(const complex32& other) const
    {
        return complex32(real - other.real, imag - other.imag);
    }

    complex32 operator*(const complex32& other) const
    {
        return complex32(real * other.real - imag * other.imag,
                         real * other.imag + imag * other.real);
    }

    complex32 operator/(const complex32& other) const
    {
        float denom = other.real * other.real + other.imag * other.imag;
        return complex32((real * other.real + imag * other.imag) / denom,
                         (imag * other.real - real * other.imag) / denom);
    }

    // print complex number
    friend std::ostream& operator<<(std::ostream& os, const complex32& c)
    {
        os << c.real << " + " << c.imag << "i";
        return os;
    }
};

struct quaternion
{
    float w;
    float x;
    float y;
    float z;

    quaternion() : w(0.0f), x(0.0f), y(0.0f), z(0.0f) {}
    quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
    quaternion(float w) : w(w), x(0.0f), y(0.0f), z(0.0f) {}

    quaternion operator+(const quaternion& other) const
    {
        return quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }

    quaternion operator-(const quaternion& other) const
    {
        return quaternion(w - other.w, x - other.x, y - other.y, z - other.z);
    }

    quaternion operator*(const quaternion& other) const
    {
        return quaternion(w * other.w - x * other.x - y * other.y - z * other.z,
                         w * other.x + x * other.w + y * other.z - z * other.y,
                         w * other.y - x * other.z + y * other.w + z * other.x,
                         w * other.z + x * other.y - y * other.x + z * other.w);
    }

    quaternion operator/(const quaternion& other) const
    {
        float denom = other.w * other.w + other.x * other.x + other.y * other.y + other.z * other.z;
        return quaternion((w * other.w + x * other.x + y * other.y + z * other.z) / denom,
                         (x * other.w - w * other.x - y * other.z + z * other.y) / denom,
                         (y * other.w - w * other.y + x * other.z - z * other.x) / denom,
                         (z * other.w - w * other.z + x * other.y - y * other.x) / denom);
    }

    quaternion operator*(float scalar) const
    {
        return quaternion(w * scalar, x * scalar, y * scalar, z * scalar);
    }

    quaternion operator/(float scalar) const
    {
        return quaternion(w / scalar, x / scalar, y / scalar, z / scalar);
    }

    quaternion operator+(float scalar) const
    {
        return quaternion(w + scalar, x, y, z);
    }

    quaternion operator-(float scalar) const
    {
        return quaternion(w - scalar, x, y, z);
    }

    quaternion normalize() const
    {
        float norm = sqrt(w * w + x * x + y * y + z * z);
        return quaternion(w / norm, x / norm, y / norm, z / norm);
    }

    // print quaternion
    friend std::ostream& operator<<(std::ostream& os, const quaternion& q)
    {
        os << q.w << " + " << q.x << "i + " << q.y << "j + " << q.z << "k";
        return os;
    }
};