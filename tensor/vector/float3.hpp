
#define pows(x) (pow(float(x), 2))
#ifndef FLOAT3_HPP
#define FLOAT3_HPP
#include "cmath"
struct float2 {
    float x;
    float y;

    template <typename T, typename TT>
    float2(T x, TT y)
    {
        this->x = x;
        this->y = y;
    };

    template <typename T>
    float2(T xy)
    {
        this->x = xy;
        this->y = xy;
    };

    float2()
    {
        this->x = 0;
        this->y = 0;
    };

    float2 operator+(float2 other)
    {
        float2 out = {
            x + other.x,
            y + other.y
        };
        return out;
    };
    float2 operator-(float2 other)
    {
        float2 out = {
            x - other.x,
            y - other.y
        };
        return out;
    };
    float2 operator/(float2 &other)
    {
        float2 out = {
            x / other.x,
            y / other.y
        };
        return out;
    };

    float2 operator -()
    {
        float2 out;
        out.x = -x;
        out.y = -y;
        return out;
    };

    float2 operator*(float2 other)
    {
        float2 out = {
            x * other.x,
            y * other.y
        };
        return out;
    };

    
};

struct float3
{
    float x;
    float y;
    float z;

    template <typename T,typename TT,typename TTT>
    float3(T x, TT y, TTT z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    };

    template <typename T>
    float3(T xyz)
    {
        this->x = xyz;
        this->y = xyz;
        this->z = xyz;
    };

    float3()
    {
        this->x = 0;
        this->y = 0;
        this->z = 0;
    };

    float2& xy()
    {
        return *(float2 *)this;
    };

    float2& yz()
    {
        return *(float2 *)&y;
    };

    


    float3 copy()
    {
        float3 out;
        out.x = x;
        out.y = y;
        out.z = z;
        return out;
    };

    static float3 random()
    {
        float3 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        out.z = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    float3 operator=(float3 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    };

    float3 operator=(float other)
    {
        x = other;
        y = other;
        z = other;
        return *this;
    };


    float3 operator+(float3 other)
    {
        float3 out = {
            x + other.x,
            y + other.y,
            z + other.z
        };
        return out;
    };

    float3 operator-(float3 other)
    {
        float3 out = {
            x - other.x,
            y - other.y,
            z - other.z
        };
        return out;
    };

    float3 operator-()
    {
        float3 out;
        out.x = -x;
        out.y = -y;
        out.z = -z;
        return out;
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
        default:
            throw "Index out of range";
        }
    };

    //math

    float3 operator+(const float3& other) const;

    __attribute__((weak))
    float3 operator-(const float3& other) const {
        return float3(this->x - other.x, this->y - other.y, this->z - other.z);
   
    }

    __attribute__((weak))
    float3 operator*(const float3& other) const {
       return float3(this->x * other.x, this->y * other.y, this->z * other.z);
   
    }

    __attribute__((weak))
    float3 operator/(const float3& other) const {
        return float3(this->x / other.x, this->y / other.y, this->z / other.z);
   
    }

    __attribute__((weak))
    float3 operator*(float scalar) const {
       return float3(this->x *scalar, this->y *scalar, this->z *scalar);
   
    }

    __attribute__((weak))
    float3 operator/(float scalar) const {
       return *this * (1/scalar);
   
    }

    __attribute__((weak))
    float3 operator+=(const float3& other) {
        *this = *this + other;
        return *this;
    }

    __attribute__((weak))
    float3 operator-=(const float3& other) {
         *this = *this - other;
        return *this;
    }

    __attribute__((weak))
    float3 cross(float3 b)
    {
        return float3(
            this->y * b.z - this->z * b.y,
            this->z * b.x - this->x * b.z,
            this->x * b.y - this->y * b.x
        );
    }
    
    __attribute__((weak))
    float3 normalize()
    {
        float length = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
        if (length == 0) return float3(0, 0, 0);
        return *this / length;
}

    // print
    friend std::ostream &operator<<(std::ostream &os, float3 a)
    {
        os << "(" << a.x << ", " << a.y << ", " << a.z << ")";
        return os;
    }
};








#endif