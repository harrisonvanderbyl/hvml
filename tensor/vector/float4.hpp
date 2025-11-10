
#define pows(x) (pow(float(x), 2))
#ifndef FLOAT4_HPP
#define FLOAT4_HPP
#include "float3.hpp"

struct float4
{
    float x;
    float y;
    float z;
    float w;

    template <typename T,typename TT,typename TTT,typename TTTT>
    float4(T x, TT y, TTT z, TTTT w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };

    template <typename T>
    float4(T x){
        this->x = x;
        this->y = x;
        this->z = x;
        this->w = x;
    }

    float4()
    {
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };

    float2& xy()
    {
        return *(float2 *)this;
    };

    float2& yz()
    {
        return *(float2 *)&y;
    };

    float2& zw()
    {
        return *(float2 *)&z;
    };

    float3& xyz()
    {
        return *(float3 *)this;
    };

    float3& yzw()
    {
        return *(float3 *)&y;
    };


    float4 copy()
    {
        float4 out;
        out.x = x;
        out.y = y;
        out.z = z;
        out.w = w;
        return out;
    };

    static float4 random()
    {
        float4 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        out.z = (rand() % 10000) / 10000.0 - 0.5;
        out.w = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    float4 operator=(const float4& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };

    float4 operator=(const float& other)
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
    friend std::ostream &operator<<(std::ostream &os, float4 a)
    {
        os << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
        return os;
    }


    float4 operator+(const float4& other) const;
    float4 operator-(const float4& other) const;
    float4 operator*(const float4& other) const;
    float4 operator/(const float4& other) const;
    float4& operator+=(const float4& other);
    float4& operator-=(const float4& other);
    float4& operator*=(const float4& other);
    float4& operator/=(const float4& other);
    
};








#endif