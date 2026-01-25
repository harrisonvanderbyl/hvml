
#include "tensor.hpp"
#include "ops/ops.h"
#include "display/display.hpp"
#include <assert.h>

float relu(float a){
    return a > 0 ? a : 0;
}




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

template<typename T>
struct Field: public T {
    using T::T; // Inherit constructors

    __device__ __host__ Field<T>& get_field_relative(int32x2 fieldsize, int32x2 direction){
        // get pointer to this
        Field<T>* ptr = (Field<T>*)this;
        // move pointer
        long int offset = (long int)(direction.y * fieldsize.x + direction.x);
        
        ptr += offset ;
        return *ptr;
    };

    __device__ __host__ auto operator=(const T& a){
        return this->T::operator=(a);
    };
 };




struct Particle
{
    public:
    double x;
    double y; 
    double filled = 0.0;
    uint84 color = uint84(0xff, 0x00, 0x00, 0xff); // Default red color

    Particle() : x(0), y(0) {}

    __device__ __host__ Particle(double a, double b):x(a),y(b){}

    __device__ __host__ Particle(double a):x(a),y(a){}
    
    __device__ __host__ Particle operator+(const Particle& a) const{
        return Particle(a.x+x, a.y+y);
    };

     __device__ __host__ Particle& operator+=(const Particle& a){
        this->x += a.x;
        this->y += a.y;
        return *this;
    };

    __device__ __host__ Particle& atomic_plus_equals(const Particle& a){
        #if defined(__CUDA_ARCH__)
        this->x += a.x;
        this->y += a.y;
        #else
        std::atomic<double>{this->x}.fetch_add(a.x);
        std::atomic<double>{this->y}.fetch_add(a.y);
        #endif
        return *this;
    };

    __device__ __host__ Particle operator*(const Particle& a) const{
        Particle r = {
            x * a.x,
            y * a.y
        };
        return r;
    };
    

    __device__ __host__ void zero(){
        x = 0;
        y = 0;
    };
    
    __device__ __host__ Particle operator-(const Particle& a) const{
        return Particle(x - a.x, y - a.y);
    };

    __device__ __host__ Particle operator/(double a) const{
        return Particle(x / a, y / a);
    };

    __device__ __host__ Particle operator*(double a) const{
        return Particle(x * a, y * a);
    };

    __device__ __host__ Particle operator /(const Particle& a) const{
        return Particle(x / a.x, y / a.y);
    };
    // cout
    friend std::ostream &operator<<(std::ostream &os, const Particle& q)
    {
        os << "(" << q.x << " , " << q.y << ")";
        return os;
    }

    

    __device__ __host__ Particle swap0(){
        // swap with zero, using copy elision
        // 64 bit atomic swap
        #if defined(__CUDA_ARCH__)
        double oldx = this->x;
        double oldy = this->y;
        this->x = 0.0f;
        this->y = 0.0f;
        
        return Particle(oldx, oldy);
        
        #else
        return Particle(std::atomic<double>{this->x}.exchange(0.0f), std::atomic<double>{this->y}.exchange(0.0f));
        #endif
    };

    

};



using ParticleField = Field<Particle>;
using Uint84Field = Field<uint84>;

struct FractalParticle
{
    public:
    FractalParticle* subparticles;
};

constexpr int SIZE = 2;
constexpr int size = SIZE;

#define repulsionFactor(r, distance) r

#define FADEOFF(io,jo) sqrt((io*io+jo*jo + 1.0))
// #define FADEOFF(io,jo) 1
struct VelocitySpread:public HardamardOperation<VelocitySpread> {

    __host__ __device__ static inline
    void apply(ParticleField& velocityField, ParticleField& particle, int32x2& screenDims, Uint84Field& displayPixel) {
        // Simple example: add spread to velocity
        if(particle.x <=0 || particle.y <=0){
            return;
        }

        auto friction = 0.0;
        for (int i = -size; i <= size ; i++)
        {
            for (int j = -size; j <= size ; j++)
            {
                friction += 1 / FADEOFF(i, j);
            }
        }
        friction = 1. / friction;

        size_t width = screenDims.x;
        size_t height = screenDims.y;
        

        ParticleField momentum(0.0,0.4);
        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                momentum += velocityField.get_field_relative(screenDims, int32x2(round(particle.x+i), round(particle.y+j))).swap0();
            }
        }
        velocityField.get_field_relative(screenDims, int32x2(round(particle.x), round(particle.y))).filled = 0.0;
        
                               
        ParticleField oldparticle = particle;
        
        

        
        
        
        particle += momentum;
        

        if ( oldparticle.x + momentum.x < 1 + size || oldparticle.x + momentum.x > width - 1 - size || oldparticle.y + momentum.y < 1 + size || oldparticle.y + momentum.y > height - 1 - size)
        {

            if (oldparticle.y + momentum.y < 1 + size || oldparticle.y + momentum.y > height - 1 - size)
            {
                momentum.y *= -0.5;
                particle.y = max(1.0f + size, min(double(height - (1.0 + size)), oldparticle.y + momentum.y));
            }
            if (oldparticle.x + momentum.x < 1 + size || oldparticle.x + momentum.x > width - 1 - size)
            {
                momentum.x *= -0.5;
                particle.x = max(1.0f + size, min(double(width - (1.0 + size)), oldparticle.x + momentum.x));
            }
            
            
        }
        
        auto relative = int32x2(round(particle.x), round(particle.y));
       
       
        auto& curra = velocityField.get_field_relative(screenDims, relative);
        if(curra.filled > 0.75){
            particle = oldparticle;
            momentum = momentum + curra;
            momentum = momentum / 2.0;
            curra = momentum;
            relative = int32x2(round(particle.x), round(particle.y));
        }   
       

        double repulsion = 0.3f;
        double drag = 1.0f;
        auto aa = momentum * drag * friction;
        auto& curr = velocityField.get_field_relative(screenDims, relative);
        curr.filled += 1.0;


        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                auto& neighbor = velocityField.get_field_relative(screenDims, int32x2(round(particle.x+ i) , round(particle.y+ j) ));
                neighbor.atomic_plus_equals((aa + ParticleField(i, j) * repulsionFactor(repulsion, FADEOFF(i,j))) / FADEOFF(i, j));
            }
        }
        
        // displayPixel = uint84(0xff, 0x00, 0x00, 0xff); // Red particle
        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                auto& neighborPixel = displayPixel.get_field_relative(screenDims, int32x2(i, j));
                neighborPixel = uint84(
                    particle.color.x / FADEOFF(i,j) * float(particle.color.w ) / 255.0f + neighborPixel.x * float(particle.color.w ) / 255.0f,
                    particle.color.y / FADEOFF(i,j) * float(particle.color.w ) / 255.0f + neighborPixel.y * float(particle.color.w ) / 255.0f,
                    particle.color.z / FADEOFF(i,j) * float(particle.color.w ) / 255.0f + neighborPixel.z * float(particle.color.w ) / 255.0f,
                    particle.color.w + neighborPixel.w
                );
            }
        }
    }
};

struct UpdateParticleIndex:public HardamardOperation<UpdateParticleIndex> {

    __host__ __device__ static inline
    void apply(ParticleField& particle, int horizontal_size, unsigned long& index) {
        size_t width = horizontal_size;

        int px = round(particle.x);
        int py = round(particle.y);

        // clamp to screen
        if(px < 0) px = 0;
        if(py < 0) py = 0;
        if(px >= width) px = width - 1;

        index = (unsigned long)(py * width + px);
    }
};


int main(){

    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);
    setenv("SDL_DEBUG", "1", 1);
    /*
    export SDL_VIDEO_DRIVER=x11
export EGL_PLATFORM=x11
    */

    int size = 1800;
    int horizontal_size = 2880;


    VectorDisplay<kCUDA> display({size,horizontal_size}, WP_BORDERLESS | WP_FULLSCREEN | WP_ALPHA_ENABLED | WP_ON_TOP);



    unsigned long indexerdata[2] = {2,1};
    Tensor<unsigned long, 1> indexertest = Tensor<unsigned long, 1>({2}, (unsigned long *)indexerdata, MemoryType::kDDR);
    std::cout << "indexertest: " << indexertest << std::endl;
    float testdata[12] = {10.0f,1.0f, 20.0f,1.0f, 30.0f,1.0f, 40.0f,1.0f, 50.0f,1.0f, 60.0f,1.0f};
    Tensor<float, 2> testtensor = Tensor<float, 2>({6,2}, (float *)testdata, MemoryType::kDDR);
    std::cout << "testtensor before: " << testtensor << std::endl;
    Tensor<float,1> gathered = testtensor.tensor_index(indexertest);
    std::cout << "gathered: " << gathered << std::endl;

    // // move everything to CUDA
    std::cout << "gathered->CUDA:" << std::endl;
    std::cout << "gathered: " << gathered.to(DeviceType::kCUDA) << std::endl;
    auto indexertest_cuda = indexertest.to(DeviceType::kCUDA);
    auto testtensor_cuda = testtensor.to(DeviceType::kCUDA);
    auto gathered_cuda = testtensor_cuda.tensor_index(indexertest_cuda);
    std::cout << "gathered_cuda: " << gathered_cuda << std::endl;


    // using QT = Quaternion<SuperReal2>;

   Tensor<ParticleField, 2> patch = Tensor<ParticleField, 2>({100,100}, MemoryType::kDDR);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            patch[{i,j}] = ParticleField(i+30, j+30); // 10 by 10 block of particles
        }
    }
    auto patchcuda = patch.view(Shape<1>{-1}).to(DeviceType::kCUDA);
    
    
    Tensor<uint84,2> display_cpu = Tensor<uint84,2>({size,horizontal_size}, DeviceType::kCUDA);
    display_cpu[{{}}] = 0xffffff00; // Black background
    Tensor<ParticleField,2> Field = Tensor<ParticleField,2>({size,horizontal_size}, DeviceType::kCUDA);
    Field[{{}}] = ParticleField(0.0f, 0.0f);
    display[{{}}] = 0xffffff00; // Black background
    // display.view<uint8_t,3>({size,horizontal_size,4})[{{},{},3}] = (uint8_t)0xff; // Alpha channel
    

    Tensor<ParticleField, 1> particles = Tensor<ParticleField, 1>({10000*100}, DeviceType::kCUDA);
    particles[{{}}] = ParticleField(-10.0f, -10.0f);
    // particles[{{0,100*100}}] = patchcuda;
    
    Tensor<unsigned long, 1> particleindex = Tensor<unsigned long, 1>({10000*100}, DeviceType::kCUDA);

    
    
    auto MappedDisplay = display.view<Uint84Field,1>({-1}).tensor_index(particleindex);
    auto FieldMapped = Field.view<ParticleField,1>({-1}).tensor_index(particleindex);

    int currentIndex = 100*100;
    int currentFrame = 0;
    // get current time in milliseconds
    auto start_time = std::chrono::high_resolution_clock::now();
    display.add_on_update([&](CurrentScreenInputInfo& info){
        currentFrame++;
        std::cout << "Frame " << currentFrame << std::endl;

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
        float time_seconds = elapsed.count() / 1000.0f;
        start_time = current_time;
        std::cout << "Fps: " << 1.0f / time_seconds << std::endl;
        
        
        // auto particleOffsets = round(particles.view<float, 2>({-1,2})[{{},1}])*horizontal_size;
        // auto particleHeights = round(particles.view<float, 2>({-1,2})[{{},0}]);
        // particleindex = (particleOffsets + particleHeights);
        UpdateParticleIndex::run(
            particles,
            horizontal_size,
            particleindex
        );


        display[{{}}] = 0x00000000; // Black background
        
        VelocitySpread::run(
            Field[{0,{0,1}}],
            particles,
            int32x2(horizontal_size, size),
            MappedDisplay
        );
        


        if(info.just_selected_area){
            auto area = info.getSelectedArea()*float32x4(1,1,0.25,0.25);
            unsigned long count = (unsigned long)(area.z) * (unsigned long)area.w ;
            Tensor<ParticleField, 2> newparticles = Tensor<ParticleField, 2>({area.z,area.w}, MemoryType::kDDR);
            for(int i = 0; i < area.z; i++){
                for(int j = 0; j < area.w; j++){
                    newparticles[{i , j}] = ParticleField(area.x + i*4 , area.y + j*4 );
                    newparticles[{i , j}].color = uint84(
                        (rand() % 256),
                        (rand() % 256),
                        (rand() % 256),
                        0x55
                    );
                }
            }
            auto newparticles_cuda = newparticles.view(Shape<1>{-1}).to(DeviceType::kCUDA);
            if(currentIndex + count > particles.shape[0]){
                std::cout << "Particle limit reached, cannot add more particles." << std::endl;
                return;
            }
            particles[{{currentIndex, currentIndex + count}}] = newparticles_cuda;


            currentIndex += count;
            
        }

        // limit 60fps
        SDL_Delay(16);
    });

    display.displayLoop();

    return 0;
}