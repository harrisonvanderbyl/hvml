
#define pows(x) (pow(float(x), 2))
#include "tensor.hpp"
struct float64x2
{
    double x;
    double y;

    float64x2(int x, int y)
    {
        this->x = x;
        this->y = y;
    };

    float64x2()
    {
        this->x = 0;
        this->y = 0;
    };

    


    float64x2 copy()
    {
        float64x2 out;
        out.x = x;
        out.y = y;
        return out;
    };

    static float64x2 random()
    {
        float64x2 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    float64x2 operator=(float64x2 &other)
    {
        x = other.x;
        y = other.y;
        return *this;
    };

    float64x2 operator=(double other)
    {
        x = other;
        y = other;
        return *this;
    };

    float64x2 operator=(float64x2 other)
    {
        x = other.x;
        y = other.y;
        return *this;
    };

    
};

#if 0
float64x2 operator+(float64x2 &other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_add_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator-(float64x2 &other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_sub_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator/(float64x2 &other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_div_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator*(float64x2 &other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_mul_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator*(double other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_mul_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator/(double other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_div_pd(_mm_loadu_pd((double *)this),_mm_set1_pd(other)));
        return out;
    };

    float64x2 operator+(double other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_add_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator-(double other)
    {
        float64x2 out;
        _mm_storeu_pd((double *)&out, _mm_sub_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    float64x2 operator+=(float64x2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_add_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator-=(float64x2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_sub_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator/=(float64x2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_div_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator*=(float64x2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_mul_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator*=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_mul_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator/=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_div_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator+=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_add_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    float64x2 operator-=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_sub_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };
#endif