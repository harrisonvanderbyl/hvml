
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

    unsigned long indexerdata[2] = {2,1};
    Tensor<unsigned long, 1> indexertest = Tensor<unsigned long, 1>({2}, (unsigned long *)indexerdata, DeviceType::kCPU);
    std::cout << "indexertest: " << indexertest << std::endl;
    float testdata[12] = {10.0f,1.0f, 20.0f,1.0f, 30.0f,1.0f, 40.0f,1.0f, 50.0f,1.0f, 60.0f,1.0f};
    Tensor<float, 2> testtensor = Tensor<float, 2>({6,2}, (float *)testdata, DeviceType::kCPU);
    std::cout << "testtensor before: " << testtensor << std::endl;
    Tensor<float,1> gathered = testtensor.tensor_index(indexertest);
    std::cout << "gathered: " << gathered << std::endl;

    // move everything to CUDA
    std::cout << "gathered->CUDA:" << std::endl;
    std::cout << "gathered: " << gathered.to(DeviceType::kCUDA) << std::endl;
    auto indexertest_cuda = indexertest.to(DeviceType::kCUDA);
    auto testtensor_cuda = testtensor.to(DeviceType::kCUDA);
    auto gathered_cuda = testtensor_cuda.tensor_index(indexertest_cuda);
    std::cout << "gathered_cuda: " << gathered_cuda << std::endl;


    // using QT = Quaternion<SuperReal2>;

   Tensor<float32x4, 2> patch = Tensor<float32x4, 2>({10,10}, DeviceType::kCPU);
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 10; j++){
            patch[{i,j}] = float32x4(i, j, 0.0f, 1.0f); // 10 by 10 block of particles
        }
    }
    auto patchcuda = patch.view(Shape<1>{-1}).to(DeviceType::kCUDA);
    
    
    VectorDisplay<kCUDA> display({size,horizontal_size});
    Tensor<float32x4,2> Field = Tensor<float32x4,2>({size,horizontal_size}, DeviceType::kCUDA);

    display[{{}}] = 0x00000000; // Black background
    // display.view<uint8_t,3>({size,horizontal_size,4})[{{},{},3}] = (uint8_t)0xff; // Alpha channel
    

    Tensor<float32x4, 1> particles = Tensor<float32x4, 1>({100}, DeviceType::kCUDA);
    particles[{{}}] = patchcuda;
    
    Tensor<unsigned long, 1> particleindex = Tensor<unsigned long, 1>({100}, DeviceType::kCUDA);

    
    
    auto MappedDisplay = display.view<uint84,1>({-1}).tensor_index(particleindex);
    auto FieldMapped = Field.view<float32x4,1>({-1}).tensor_index(particleindex);

    int i = 0;
    display.add_on_update([&](CurrentScreenInputInfo& info){
        i++;
        particles+=float32x4(2.0f, 0.5f, 0.0f, 0.0f); // Move particles right and down

        
        display[{{}}] = 0x00000000; // Black background
        auto particleOffsets = particles.view<float, 2>({-1,4})[{{},0}] * horizontal_size;
        auto particleHeights = particles.view<float, 2>({-1,4})[{{},1}] ;
        particleindex = (particleOffsets + particleHeights);


        MappedDisplay = 0xFF000000; // Red particles at particle positions
    });

    display.displayLoop();

    return 0;
}