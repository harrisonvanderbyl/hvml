
#define pows(x) (pow(float(x), 2))
#include "immintrin.h"
#include "tensor.hpp"

struct float2 {
    float x;
    float y;

    float2(int x, int y)
    {
        this->x = x;
        this->y = y;
    };

    float2(float x, float y)
    {
        this->x = x;
        this->y = y;
    };

    float2()
    {
        this->x = 0;
        this->y = 0;
    };

    
};

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

    float4 operator+(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_add_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator-(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_sub_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator/(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_div_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator*(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_mul_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator*(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_mul_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator/(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_div_ps(_mm_loadu_ps((float *)this),_mm_set1_ps(other)));
        return out;
    };

    float4 operator+(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_add_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator-(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_sub_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator+=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_add_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator-=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_sub_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator/=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_div_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator*=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_mul_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator*=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_mul_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator/=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_div_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator+=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_add_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator-=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_sub_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
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

    float4 operator=(float4 &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };

    float4 operator=(float other)
    {
        x = other;
        y = other;
        z = other;
        w = other;
        return *this;
    };

    float4 operator=(float4 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
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
};


struct mat4
{
    float4 rows[4];

    mat4(float4 row0, float4 row1, float4 row2, float4 row3):
        rows{row0, row1, row2, row3}
    {
    };

    mat4():
        rows{float4(), float4(), float4(), float4()}
    {
    };

    // print
    friend std::ostream &operator<<(std::ostream &os, mat4 a)
    {
        os << "[";
        for (int i = 0; i < 4; i++)
        {
            os << a.rows[i];
            if (i != 3)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};