
#define pows(x) (pow(float(x), 2))
#ifndef float32x3_HPP
#define float32x3_HPP
#include "cmath"
struct float32x2 {
    float x;
    float y;

    template <typename T, typename TT>
    float32x2(T x, TT y)
    {
        this->x = x;
        this->y = y;
    };

    template <typename T>
    float32x2(T xy)
    {
        this->x = xy;
        this->y = xy;
    };

    float32x2()
    {
        this->x = 0;
        this->y = 0;
    };

    float32x2 operator+(float32x2 other)
    {
        float32x2 out = {
            x + other.x,
            y + other.y
        };
        return out;
    };
    float32x2 operator-(float32x2 other)
    {
        float32x2 out = {
            x - other.x,
            y - other.y
        };
        return out;
    };
    float32x2 operator/(float32x2 &other)
    {
        float32x2 out = {
            x / other.x,
            y / other.y
        };
        return out;
    };

    float32x2 operator -()
    {
        float32x2 out;
        out.x = -x;
        out.y = -y;
        return out;
    };

    float32x2 operator*(float32x2 other)
    {
        float32x2 out = {
            x * other.x,
            y * other.y
        };
        return out;
    };

    
};

struct float32x3
{
    float x;
    float y;
    float z;

    template <typename T,typename TT,typename TTT>
    float32x3(T x, TT y, TTT z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    };

    template <typename T>
    float32x3(T xyz)
    {
        this->x = xyz;
        this->y = xyz;
        this->z = xyz;
    };

    float32x3()
    {
        this->x = 0;
        this->y = 0;
        this->z = 0;
    };

    float32x2& xy()
    {
        return *(float32x2 *)this;
    };

    float32x2& yz()
    {
        return *(float32x2 *)&y;
    };

    


    float32x3 copy()
    {
        float32x3 out;
        out.x = x;
        out.y = y;
        out.z = z;
        return out;
    };

    static float32x3 random()
    {
        float32x3 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        out.z = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    float32x3 operator=(float32x3 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    };

    float32x3 operator=(float other)
    {
        x = other;
        y = other;
        z = other;
        return *this;
    };


    float32x3 operator+(float32x3 other)
    {
        float32x3 out = {
            x + other.x,
            y + other.y,
            z + other.z
        };
        return out;
    };

    float32x3 operator-(float32x3 other)
    {
        float32x3 out = {
            x - other.x,
            y - other.y,
            z - other.z
        };
        return out;
    };

    float32x3 operator-()
    {
        float32x3 out;
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

    float32x3 operator+(const float32x3& other) const;

    __attribute__((weak))
    float32x3 operator-(const float32x3& other) const {
        return float32x3(this->x - other.x, this->y - other.y, this->z - other.z);
   
    }

    __attribute__((weak))
    float32x3 operator*(const float32x3& other) const {
       return float32x3(this->x * other.x, this->y * other.y, this->z * other.z);
   
    }

    __attribute__((weak))
    float32x3 operator/(const float32x3& other) const {
        return float32x3(this->x / other.x, this->y / other.y, this->z / other.z);
   
    }

    __attribute__((weak))
    float32x3 operator*(float scalar) const {
       return float32x3(this->x *scalar, this->y *scalar, this->z *scalar);
   
    }

    __attribute__((weak))
    float32x3 operator/(float scalar) const {
       return *this * (1/scalar);
   
    }

    __attribute__((weak))
    float32x3 operator+=(const float32x3& other) {
        *this = *this + other;
        return *this;
    }

    __attribute__((weak))
    float32x3 operator-=(const float32x3& other) {
         *this = *this - other;
        return *this;
    }

    __attribute__((weak))
    float32x3 cross(float32x3 b)
    {
        return float32x3(
            this->y * b.z - this->z * b.y,
            this->z * b.x - this->x * b.z,
            this->x * b.y - this->y * b.x
        );
    }
    
    __attribute__((weak))
    float32x3 normalize()
    {
        float length = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
        if (length == 0) return float32x3(0, 0, 0);
        return *this / length;
}

    // print
    friend std::ostream &operator<<(std::ostream &os, float32x3 a)
    {
        os << "(" << a.x << ", " << a.y << ", " << a.z << ")";
        return os;
    }
};








#endif