#include "vector/vectors.hpp"

struct Particle
{
    public:
    float32x3 position; // x, y, z
    float32x2 temperatureopacity = float32x2(0.0f, 1.0f); // density and opacity
    float neighborsfilled = 0;
    uint84 color = uint84(0xff, 0x00, 0x00, 0xff); // Default red color

    Particle() : position(0.0f, 0.0f, 0.0f) {};

    __device__ __host__ Particle(float a, float b, float c):position(a,b,c){}

    __device__ __host__ Particle(float a):position(a,a,a){}

    __device__ __host__ Particle(float32x3 pos):position(pos){}
    

    friend std::ostream &operator<<(std::ostream &os, const Particle& q)
    {
        os << "(" << q.position[0] << ", " << q.position[1] << ", " << q.position[2] << ")";
        return os;
    }

};