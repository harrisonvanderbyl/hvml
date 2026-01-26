
#include "tensor.hpp"
#include "ops/ops.hpp"
#include "display/display.hpp"
#include <assert.h>
#include <atomic>
#include "module/linear/linear.hpp"


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
    float filled = 0.0f; // pointer to filled velocity field
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
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        atomicAdd((double*)&this->x, a.x);
        atomicAdd((double*)&this->y, a.y);
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
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // double oldx = this->x;
        // double oldy = this->y;
        // this->x = 0.0f;
        // this->y = 0.0f;
        unsigned long long oldx = atomicExch((unsigned long long int*)&this->x, __double_as_longlong(0.0));
        unsigned long long oldy = atomicExch((unsigned long long int*)&this->y, __double_as_longlong(0.0));     
        
        return Particle(__longlong_as_double(oldx), __longlong_as_double(oldy));
        
        #else
        return Particle(std::atomic<double>{this->x}.exchange(0.0f), std::atomic<double>{this->y}.exchange(0.0f));
        #endif
    };

    __device__ __host__ float addFilled(float increment = 1.0f){
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return atomicAdd((float*)&this->filled, increment);
        #else
        return std::atomic<float>{this->filled}.fetch_add(increment);
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
constexpr int compression = 1;
constexpr int allowDensity = 5;

#define repulsionFactor(r, distance) r

#define FADEOFF(io,jo) (sqrtf(float(io*io+jo*jo))+1) //sqrt((io*io+jo*jo + 1.0))
// #define FADEOFF(io,jo) 1
struct VelocitySpread:public HardamardOperation<VelocitySpread> {

    __host__ __device__ static inline
    void apply(ParticleField& velocityField, ParticleField& particle, int32x2& screenDims, uint84& displayPixel, int32x2 globalAddVelocity) {
        // Simple example: add spread to velocity
        if(particle.x <=0 || particle.y <=0){
            return;
        }

        ParticleField oldparticle = particle;
        
        
        auto oldrelative = int32x2(round(oldparticle.x), round(oldparticle.y));
        
        if(
        velocityField.get_field_relative(screenDims, oldrelative).filled == 0.0f){
            // not initialized yet
            velocityField.get_field_relative(screenDims, oldrelative).addFilled(1.0f);
        }
        // if(particle.filled == nullptr){
        //     particle.filled = &velocityField.get_field_relative(screenDims, oldrelative);
        //     // particle.filled->swapfilled(particle);
        // }
        

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
        

        ParticleField momentum(0.0,0.0);
        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                momentum += velocityField.get_field_relative(screenDims, int32x2(round(particle.x)+i, round(particle.y)+j)).swap0();
            }
        }
              
        
        
        particle += momentum;
        particle += Particle(-globalAddVelocity.x, -globalAddVelocity.y) * compression;

        momentum.y += 0.5f;

        if ( particle.x < 1 + size || particle.x > width - 1 - size || particle.y < 1 + size || particle.y > height - 1 - size)
        {

            if (particle.y < 1 + size || particle.y > height - 1 - size)
            {
                momentum.y *= -0.5;
                particle.y = max(2.0f + size, min(double(height - (2.0 + size)), particle.y));
            }
            if (particle.x < 1 + size || particle.x > width - 1 - size)
            {
                momentum.x *= -0.5;
                particle.x = max(2.0f + size, min(double(width - (2.0 + size)), particle.x));
            }
            
            
        }
        
        auto relative = int32x2(round(particle.x), round(particle.y));
        
       
        

        if(oldrelative.x != relative.x || oldrelative.y != relative.y){

            // velocityField.get_field_relative(screenDims, oldrelative).filled = 0.0;   
            auto& newpos = velocityField.get_field_relative(screenDims, relative);   
            // should be safe
            float curra = newpos.addFilled(1.0f);

            // // should be safe:
            
            if(curra >= allowDensity){ // collision
                particle = oldparticle; // revert position
                if(curra < allowDensity+1){
                    Particle assumed_oppos = {float(relative.x) , float(relative.y) };
                    auto norm = (particle - assumed_oppos);
                    // reflect momentum along normal
                    float dot = (momentum.x * norm.x + momentum.y * norm.y) / (norm.x * norm.x + norm.y * norm.y);
                    momentum = momentum - norm * (2.0 * dot);
                    momentum = momentum * 0.5; // bounce back
                    newpos.atomic_plus_equals(momentum * -1.0f);
                }else{
                    momentum = momentum * -1.0; // bounce back with half momentum
                }
                newpos.addFilled(-1.0f);
            }
            else{
                // particle.swapfilled(newpos);
                velocityField.get_field_relative(screenDims, oldrelative).addFilled(-1.0f);

            }
        }

        
        // auto a = particle.swapfilled());
        // if(a)->swapfilled(particle);
        
        

        double repulsion = 0.02;
        double drag = 0.99;
        auto aa = momentum * drag ;
        
        // auto& curr = *particle.filled;
        auto& mpos = velocityField.get_field_relative(screenDims, int32x2(round(particle.x) , round(particle.y) ));
        float mfilled = mpos.filled;
        mpos.atomic_plus_equals(aa);

        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                auto& neighbor = velocityField.get_field_relative(screenDims, int32x2(round(particle.x)+ i , round(particle.y)+ j ));
                neighbor.atomic_plus_equals( (ParticleField(i, j) * repulsion*(mfilled+1)) / FADEOFF(i, j));
            }
        }
        
        // displayPixel = uint84(0xff, 0x00, 0x00, 0xff); // Red particle
        for (int i = - SIZE/compression; i <= SIZE/compression; i++) {
            for (int j = - SIZE/compression; j <= SIZE/compression; j++) {
                auto& neighborPixel =  ((Field<uint84>*)&displayPixel)->get_field_relative(int32x2(screenDims.x/compression,screenDims.y/compression), int32x2(particle.x/compression+ i , particle.y/compression + j ));
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

struct UpdateFieldDetails:public HardamardOperation<UpdateFieldDetails> {

    __host__ __device__ static inline
    void apply(ParticleField& particlefield, uint84& displayPixel){
        if(particlefield.filled > 0.0f){
            displayPixel = uint84(0x00, 0xff, 0x00, 0xff); // Green for filled
        }
    }
};



__weak int main(){

    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);
    // setenv("SDL_DEBUG", "1", 1);
    /*
    export SDL_VIDEO_DRIVER=x11
export EGL_PLATFORM=x11
    */

    int size = 768;
    int horizontal_size = 1024;


    VectorDisplay display({size,horizontal_size},  WP_ON_TOP|WP_ALPHA_ENABLED);





//     // using QT = Quaternion<SuperReal2>;

   Tensor<ParticleField, 2> patch = Tensor<ParticleField, 2>({100,100}, MemoryType::kDDR);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            patch[{i,j}] = ParticleField(i+30, j+30); // 10 by 10 block of particles
        }
    }
    auto patchcuda = patch.view(Shape<1>{-1}).to(*display.device);
    
    
    Tensor<ParticleField,2> Field = Tensor<ParticleField,2>({size*compression,horizontal_size*compression}, *display.device);
    Field[{{}}] = ParticleField(0.0f, 0.0f);

    Tensor<ParticleField, 1> particles = Tensor<ParticleField, 1>({5000*100}, *display.device);
    particles[{{}}] = ParticleField(-10.0f, -10.0f);
//     // particles[{{0,100*100}}] = patchcuda;
    
//     Tensor<unsigned long, 1> particleindex = Tensor<unsigned long, 1>({5000*100}, MemoryType::kDDR);

    
    
//     // auto MappedDisplay = display.view<Uint84Field,1>({-1}).tensor_index(particleindex);

    size_t currentIndex = 100*100;
    int currentFrame = 0;
//     // get current time in milliseconds
    auto start_time = std::chrono::high_resolution_clock::now();
    display.add_on_update([&](CurrentScreenInputInfo& info){
        display[{{}}] = 0xffffff00; // White background
        currentFrame++;
        // std::cout << "Frame " << currentFrame << std::endl;

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = current_time - start_time;
        float time_seconds = elapsed.count() / 1000.0f;
        start_time = current_time;
        // std::cout << "Fps: " << 1.0f / time_seconds << std::endl;
        
        
        // auto particleOffsets = round(particles.view<float, 2>({-1,2})[{{},1}])*horizontal_size;
        // auto particleHeights = round(particles.view<float, 2>({-1,2})[{{},0}]);
        // particleindex = (particleOffsets + particleHeights);
        // UpdateParticleIndex::run(
        //     particles,
        //     horizontal_size,
        //     particleindex
        // );


        display[{{}}] = 0x00000000; // Black background

        UpdateFieldDetails::run(
            Field[{{0,{},compression},{0,{},compression}}],//[{0,{0,1}}],
            display[{{}}]//[{0,{0,1}}]
        );
        
        VelocitySpread::run(
            Field[{0,{0,1}}],
            particles,
            int32x2(horizontal_size*compression, size*compression),
            display[{0,{0,1}}],
            info.relativeWindowMove
        );
        


        if(info.just_selected_area){
            int solid = info.isMouseRightButtonPressed();
            auto area = info.getLocalSelectedArea();
            std::cout << "Selected area: " << area.x << "," << area.y << " to " << area.z << "," << area.w << std::endl;
            std::cout << "Adding particles..." << std::endl;
            if(!solid){
                area.z /= SIZE*2+1;
                area.w /= SIZE*2+1;
                area.x = area.x - (int(area.x) % (2*SIZE + 1));
                area.y = area.y - (int(area.y) % (2*SIZE + 1));
                area = int32x4(
                    (area.x)*compression,
                    (area.y)*compression,
                    (area.z)*compression,
                    (area.w)*compression
                );
                unsigned long count = (unsigned long)(area.z) * (unsigned long)area.w ;
                Tensor<ParticleField, 2> newparticles = Tensor<ParticleField, 2>({area.z,area.w}, MemoryType::kDDR);
                // cudaDeviceSynchronize();
                for(int i = 0; i < area.z; i++){
                    for(int j = 0; j < area.w; j++){
                        newparticles[{i , j}] = ParticleField(area.x + i*(2*SIZE + 1) , area.y + j*(2*SIZE + 1) );
                        newparticles[{i , j}].color = uint84(
                            (rand() % 256),
                            (rand() % 256),
                            (rand() % 256),
                            0x99
                        );
                    }
                }
                // cudaDeviceSynchronize();
                auto newparticles_cuda = newparticles.view(Shape<1>{-1}).to(*display.device);
                // cudaDeviceSynchronize();
                if(currentIndex + count > particles.shape[0]){
                    std::cout << "Particle limit reached, cannot add more particles." << std::endl;
                    return;
                }
                particles[{{currentIndex, currentIndex + count}}] = newparticles_cuda;
                // cudaDeviceSynchronize();


                currentIndex += count;
            }
            
        }
        if(info.isMouseRightButtonPressed())
        {
                auto part = ParticleField(
                    0,0
                );
                part.filled = 5.0f;
                Field[
                    {
                        {info.getLocalMousePosition().y*compression, (info.getLocalMousePosition().y + 10)*compression},
                        {info.getLocalMousePosition().x*compression, (info.getLocalMousePosition().x + 10)*compression}
                    }
                ] = part;
            }

        // limit 60fps
        // SDL_Delay(16);
    });

    display.displayLoop();

    return 0;
}