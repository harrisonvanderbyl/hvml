#ifndef TENSOR_VECTOR_INT2_HPP
#define TENSOR_VECTOR_INT2_HPP
#ifndef __host__
#define __host__
#define __device__
#endif

struct int32x2 
{
    int x;
    int y;
    __device__ __host__ int32x2(int x, int y){
        this->x = x;
        this->y = y;
    };
    __device__ __host__  int32x2(){
        this->x = 0;
        this->y = 0;
    };
};


struct int32x4 
{
    int x;
    int y;
    int z;
    int w;
    int32x4(int x, int y, int z, int w){
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    int32x4(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };
};

struct uint84 
{
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t w;
    __host__ __device__ uint84(uint8_t x, uint8_t y, uint8_t z, uint8_t w){
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    __host__ __device__ uint84(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };
    __host__ __device__ uint84(uint32_t v){
        this->x = (v >> 24) & 0xFF;
        this->y = (v >> 16) & 0xFF;
        this->z = (v >> 8) & 0xFF;
        this->w = v & 0xFF;
    };

    __host__ __device__ uint84& operator+= (const uint84& other) {
        this->x = min(255, this->x + other.x);
        this->y = min(255, this->y + other.y);
        this->z = min(255, this->z + other.z);
        this->w = min(255, this->w + other.w);
        return *this;
    }
};


// uint8_t& uint84::r = &uint84::x;
#endif // TENSOR_VECTOR_INT2_HPP