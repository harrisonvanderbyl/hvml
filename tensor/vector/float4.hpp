
#define pows(x) (pow(float(x), 2))
#ifndef float32x4_HPP
#define float32x4_HPP
#include "float3.hpp"

struct float32x4
{
    float x;
    float y;
    float z;
    float w;

    template <typename T,typename TT,typename TTT,typename TTTT>
    float32x4(T x, TT y, TTT z, TTTT w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };

    template <typename T>
    float32x4(T x){
        this->x = x;
        this->y = x;
        this->z = x;
        this->w = x;
    }

    float32x4()
    {
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };

    float32x32x2& xy()
    {
        return *(float32x32x2 *)this;
    };

    float32x32x2& yz()
    {
        return *(float32x32x2 *)&y;
    };

    float32x32x2& zw()
    {
        return *(float32x32x2 *)&z;
    };

    float32x3& xyz()
    {
        return *(float32x3 *)this;
    };

    float32x3& yzw()
    {
        return *(float32x3 *)&y;
    };


    float32x4 copy()
    {
        float32x4 out;
        out.x = x;
        out.y = y;
        out.z = z;
        out.w = w;
        return out;
    };

    static float32x4 random()
    {
        float32x4 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        out.z = (rand() % 10000) / 10000.0 - 0.5;
        out.w = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    float32x4 operator=(const float32x4& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };

    float32x4 operator=(const float& other)
    {
        x = other;
        y = other;
        z = other;
        w = other;
        return *this;
    };


    

    float& operator [](int index)
    {
        switch (index)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        default:
            throw "Index out of range";
        }
    };

    // print
    friend std::ostream &operator<<(std::ostream &os, float32x4 a)
    {
        os << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
        return os;
    }


    float32x4 operator+(const float32x4& other) const;
    float32x4 operator-(const float32x4& other) const;
    float32x4 operator*(const float32x4& other) const;
    float32x4 operator/(const float32x4& other) const;
    float32x4& operator+=(const float32x4& other);
    float32x4& operator-=(const float32x4& other);
    float32x4& operator*=(const float32x4& other);
    float32x4& operator/=(const float32x4& other);
    
};








#endif