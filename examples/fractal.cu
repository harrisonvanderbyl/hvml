
#include "tensor.hpp"
#include "ops/ops.h"
#include "display/display.hpp"
#include <assert.h>

float relu(float a){
    return a > 0 ? a : 0;
}

struct SuperReal{
    float real;
    float alte; // a^2 = 1 // technically negatives hmm

    SuperReal(float a, float b):real(a),alte(b){
        normalize();
    }
    
    SuperReal(float a):real(relu(a)),alte(relu(-a)){
        normalize();
    }

    void normalize(){
        float s = real + alte;
        if(s > 10){
            float reala = relu(real - alte);
            alte = relu(-real + alte);
            real = reala;
        }
    }

    SuperReal operator+(const SuperReal& a) const{
        return SuperReal(a.real+real, a.alte+alte);
    };

    SuperReal operator*(const SuperReal a) const{
        return SuperReal(
            real * a.real + alte * a.alte,
            real * a.alte + alte * a.real
        );
    };

    SuperReal operator-(const SuperReal a) const{
        return SuperReal(real + a.alte, alte + a.real);
    };

    float toFloat() const{
        return real - alte;
    };
};

struct SuperReal2{
    float real;
    float zeta; // zeta^2 = 0
    __device__ __host__ SuperReal2(float a, float b):real(a),zeta(b){}
    __device__ __host__ SuperReal2(float a):real(a),zeta(1){}
    __device__ __host__ SuperReal2 operator+(const SuperReal2& a) const{
        return SuperReal2(a.real+real, a.zeta+zeta);
    };
    __device__ __host__ SuperReal2 operator*(const SuperReal2 a) const{
        return SuperReal2(
            real * a.real,
            real * a.zeta + zeta * a.real
        );
    };
    __device__ __host__ SuperReal2 operator-(const SuperReal2 a) const{
        return SuperReal2(real - a.real, zeta - a.zeta);
    };
    float toFloat() const{
        return real;
    };

    // cout
    friend std::ostream &operator<<(std::ostream &os, const SuperReal2& sr)
    {
        os << "(" << sr.real << " + " << sr.zeta << "ζ)";
        return os;
    }

};

// struct Quaternion
// {
//     public:
//     float real;
//     float imagi; // i^2 = -1
//     float imagj; // j^2 = -1
//     float imagk; // k^2 = -1

//     Quaternion(float a, float b, float c, float d):real(a),imagi(b),imagj(c),imagk(d){}
    
//     Quaternion operator+(Quaternion& a) const{
//         return Quaternion(a.real+real, a.imagi+imagi, a.imagj+imagj, a.imagk+imagk);
//     };

//     Quaternion operator*(Quaternion& a) const{
//         Quaternion r = {
//             real * a.real - imagi * a.imagi - imagj * a.imagj - imagk * a.imagk,
//             real * a.imagi + imagi * a.real + imagj * a.imagk - imagk * a.imagj,
//             real * a.imagj - imagi * a.imagk + imagj * a.real + imagk * a.imagi,
//             real * a.imagk + imagi * a.imagj - imagj * a.imagi + imagk * a.real
//         };
//         return r;
//     };

//     Quaternion operator-(Quaternion& a) const{
//         return Quaternion(real - a.real, imagi - a.imagi, imagj - a.imagj, imagk - a.imagk);
//     };

// };


#include <iostream>
bool test_superreal_multiplication(){
    bool success = true;
    SuperReal a(3,2); // 1
    SuperReal b(4,5); // -1
    SuperReal c = a * b; // 12 + 15 + 8 + 10 = 45, 15 + 8 + 20 + 12 = 55
    success &= (c.toFloat() == -1);
    if (!success) {
        std::cout << "Expected -1, got " << c.toFloat() << std::endl;
    }

    SuperReal d(1,0); // 1
    SuperReal e(0,5); // -1
    SuperReal f = d * e; // 0 + 0 + 5 + 0 = 5, 0 + 5 + 0 + 0 = 5
    success &= (f.toFloat() == -5);
    if (!success) {
        std::cout << "Expected -5, got " << f.toFloat() << std::endl;
    }

    return success;
}

// bool testQuaternianMultiplication(){
//     bool success = true;
//     SuperReal one(1,0); // 1
//     SuperReal neg_one(0,1); // -1
//     SuperReal zero(0,0); // 0

//     Quaternion a(zero, one, zero, zero); // (1 + i + j + k)
//     Quaternion b(zero, one, zero, zero); // (1 - i + j - k)

//     Quaternion c = a * b;

//     success &= (c.real.toFloat() == -1);
//     success &= (c.imagi.toFloat() == 0);
//     success &= (c.imagj.toFloat() == 0);
//     success &= (c.imagk.toFloat() == 0);

//     if (!success) {
//         std::cout << "Quaternion multiplication test failed!" << std::endl;
//         std::cout << "Result: (" << c.real.toFloat() << " + " << c.imagi.toFloat() << "i + " 
//                   << c.imagj.toFloat() << "j + " << c.imagk.toFloat() << "k)" << std::endl;
//     }

//     return success;
// }



#define BENCHMARK(...) { \
    auto start = std::chrono::high_resolution_clock::now(); \
    \
    \
    __VA_ARGS__ \
\
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> duration = end - start; \
    std::cout << "Benchmark: " << " took " << duration.count() << " ms" << std::endl; \
}

template <typename FTYPE>
struct Quaternion
{
    public:
    FTYPE real;
    FTYPE imagi; // i^2 = -1
    FTYPE imagj; // j^2 = -1
    FTYPE imagk; // k^2 = -1

    __device__ __host__ Quaternion(FTYPE a, FTYPE b, FTYPE c, FTYPE d):real(a),imagi(b),imagj(c),imagk(d){}
    
    __device__ __host__ Quaternion operator+(const Quaternion& a) const{
        return Quaternion(a.real+real, a.imagi+imagi, a.imagj+imagj, a.imagk+imagk);
    };

    __device__ __host__ Quaternion operator*(const Quaternion& a) const{
        Quaternion r = {
            real * a.real - imagi * a.imagi - imagj * a.imagj - imagk * a.imagk,
            real * a.imagi + imagi * a.real + imagj * a.imagk - imagk * a.imagj,
            real * a.imagj - imagi * a.imagk + imagj * a.real + imagk * a.imagi,
            real * a.imagk + imagi * a.imagj - imagj * a.imagi + imagk * a.real
        };
        return r;
    };

    __device__ __host__ Quaternion operator-(const Quaternion& a) const{
        return Quaternion(real - a.real, imagi - a.imagi, imagj - a.imagj, imagk - a.imagk);
    };

    // cout
    friend std::ostream &operator<<(std::ostream &os, const Quaternion<FTYPE>& q)
    {
        os << "(" << q.real << " + " << q.imagi << "i + " << q.imagj << "j + " << q.imagk << "k)";
        return os;
    }

};


int main(){
   

    int size = 1024;
    int horizontal_size = 1536;

    VectorDisplay<kCUDA> display({size,horizontal_size});

    using QT = Quaternion<SuperReal2>;

    Tensor<QT,2> tensorCPU({size,horizontal_size}, DeviceType::kCPU);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < horizontal_size; j++) {
            tensorCPU[{i, j}] = QT(
                SuperReal2((float)(i - size / 2) / (size / 2)),
                SuperReal2((float)(j - horizontal_size / 2) / (horizontal_size / 2)),
                SuperReal2(0.0f, 0.0f),
                SuperReal2(0.0f, 0.0f)
            );
        }
    }
    auto tensor = tensorCPU.to(DeviceType::kCUDA);
    Tensor<QT,2> main_tensor({size, horizontal_size}, DeviceType::kCUDA);
    main_tensor = tensorCPU.to(DeviceType::kCUDA);

    // std::cout << "tensor: " << tensor << std::endl;
    // std::cout << "main_tensor: " << main_tensor << std::endl;
    

    
    
    display[{{}}] = 0x00000000; // Black background
    // display.view<uint8_t,3>({size,horizontal_size,4})[{{},{},3}] = (uint8_t)0xff; // Alpha channel
       

    display.add_on_update([&](CurrentScreenInputInfo& info){

        main_tensor = main_tensor + tensor;
        main_tensor = main_tensor * main_tensor;
        

        auto reshaped = main_tensor.view<float,3>({size, horizontal_size, 8}); // real
        auto reshaped_squared = reshaped * reshaped;
        auto distance = min(pow(reshaped_squared[{{}, {}, 1}] + reshaped_squared[{{}, {}, 3}],0.5)/4,1.0);


        display.view<uint8_t,3>({size,horizontal_size,4})[{{},{},0}] = distance*255; // R
        display.view<uint8_t,3>({size,horizontal_size,4})[{{},{},1}] = distance*255; // G
        display.view<uint8_t,3>({size,horizontal_size,4})[{{},{},2}] = distance*255; // B
    })
    ;

    
    

    display.displayLoop();

    return 0;
}