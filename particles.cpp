
#include "tensor.hpp"
#include "ops/ops.hpp"
#include "display/display.hpp"
#include <assert.h>
#include <atomic>
#include "module/linear/linear.hpp"
#include "camera/camera.hpp"
#include "file_loaders/gltf.hpp"
#include "display/materials/materials.hpp"
#include "display/scene/scene.hpp"



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

    __device__ __host__ Field<T>& get_field_relative(int32x4 fieldsize, int32x4 direction){
        // get pointer to this
        Field<T>* ptr = (Field<T>*)this;
        // move pointer
        long int offset = (long int)(direction[2] * fieldsize[0] * fieldsize[1] + 
                             direction[1] * fieldsize[0] + 
                             direction[0]);
        ptr += offset ;
        return *ptr;
    };

    __device__ __host__ auto operator=(const T& a){
        return this->T::operator=(a);
    };
 };

// uint8_t swappoint = 0b000; // 

static inline __host__ __device__ int32x4 tnf(int a, int b, int c, int d){ 
    return int32x4(a, b, c, 1 << d);
}

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


struct FractalField {
    float32x3 velocity;
    FractalField* subfields[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    FractalField* neighboringFields[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    FractalField* parentField = nullptr;
    FractalField* freelistNext = nullptr;
    Particle* occupant = nullptr;

    __device__ __host__ void freeOccupant(){
        occupant = nullptr;
        bool freeself = true;
        for (int i = 0; i < 8; i++){
            if (subfields[i] != nullptr){
                if (subfields[i]->occupant != nullptr){
                    freeself = false;
                }
            }
        };
        if (freeself){
            freeSelf();
        }
    };

    __device__ __host__ void freeSelf(){
        for (int i = 0; i < 8; i++){
            if (subfields[i] != nullptr){
                for (int j = 0; j < 8; j++){
                    if (subfields[i]->subfields[j] == this){
                        subfields[i]->subfields[j] = nullptr;
                    }
                }
            }
        };
        FractalField* attachpoint = parentField->freelistNext;
        while(attachpoint->freelistNext != nullptr){
            attachpoint = attachpoint->freelistNext;
        }
        attachpoint->freelistNext = freelistNext;
        
        freelistNext = parentField->freelistNext;
        parentField->freelistNext = this;
        parentField = nullptr;  
    };

    FractalField* getFreeFromList(){
        if (freelistNext != nullptr){
            FractalField* newfield = freelistNext;
            freelistNext = freelistNext->freelistNext;
            return newfield;
        }
        else{
            if (parentField != nullptr){
                return parentField->getFreeFromList();
            }
            else{
                return nullptr;
            }
        }
    }

    FractalField* get_field_relative(int32x4 direction){
        // x,y,z,depth
        if(direction[3] == 0){
            return this;
        }
        int32x4 reduce = int32x4(direction[0] >> direction[3], direction[1] >> direction[3], direction[2] >> direction[3], 0); // 0-7 -> 0, 8-15 -> 1 => rightshifts by depth to get index of subfield
        int index = reduce[2] * 4 + reduce[1] * 2 + reduce[0];
        if (subfields[index] != nullptr){
            return subfields[index]->get_field_relative(int32x4(direction[0] & ((1 << direction[3]) - 1), direction[1] & ((1 << direction[3]) - 1), direction[2] & ((1 << direction[3]) - 1), direction[3] - 1));
        }
        else{
            if (freelistNext != nullptr){
                FractalField* newfield = freelistNext;
                freelistNext = freelistNext->freelistNext;
                newfield->parentField = this;
                subfields[index] = newfield;
                return newfield->get_field_relative(int32x4(direction[0] & ((1 << direction[3]) - 1), direction[1] & ((1 << direction[3]) - 1), direction[2] & ((1 << direction[3]) - 1), direction[3] - 1));
            }
            else{
                // allocate new field
                FractalField* newfield = new FractalField();
                newfield->parentField = this;
                subfields[index] = newfield;
                return newfield->get_field_relative(int32x4(direction[0] & ((1 << direction[3]) - 1), direction[1] & ((1 << direction[3]) - 1), direction[2] & ((1 << direction[3]) - 1), direction[3] - 1));
            }
        }
    }
};




struct ChunkOfSpace{
    bfloat16x4 velocity;
    Particle* occupant = nullptr;
    // ChunkOfSpace* neighboringChunks[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
     __device__ __host__ Particle* addFilled(Particle* increment = nullptr){
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return (Particle*)(void*)atomicExch((unsigned long long*)&this->occupant, (unsigned long long)increment);
        // auto oldOccupant = this->occupant;
        // this->occupant = increment;
        // return oldOccupant;
        #else
        Particle* oldOccupant = std::atomic<Particle*>{this->occupant}.exchange(increment);
        return oldOccupant;
        #endif
    };

    ChunkOfSpace() : velocity(0.0f, 0.0f, 0.0f, 0.0f) {};




    __device__ __host__ float32x3 swap0(){
        // swap with zero, using copy elision
        // 64 bit atomic swap
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // bfloat16x4 is equivalent to uint64_t for atomic operations
        // auto oldvelocity = this->velocity;
        // this->velocity = bfloat16x4(0.0f, 0.0f, 0.0f, 0.0f);
        auto oldvelocity = atomicExch((unsigned long long*)&this->velocity, 0);
        bfloat16x4 oldVelocity = *(bfloat16x4*)&oldvelocity;
        return float32x3(oldVelocity[0], oldVelocity[1], oldVelocity[2]);
        
        #else
        return float32x3(std::atomic<bfloat16>{this->velocity[0]}.exchange(0.0f),
                        std::atomic<bfloat16>{this->velocity[1]}.exchange(0.0f),
                        std::atomic<bfloat16>{this->velocity[2]}.exchange(0.0f));
        #endif
    };

    __device__ __host__ void atomic_plus_equals(const float32x3& a){
        #if defined(__CUDA_ARCH__)  
        atomicAdd((__nv_bfloat162*)&this->velocity[0], __float22bfloat162_rn(*(float2*)&a[0]));
        atomicAdd((__nv_bfloat16*)&this->velocity[2], a[2]);
        // this->velocity += bfloat16x4(a[0], a[1], a[2], 0);
        #elif defined(__HIP_DEVICE_COMPILE__)
        // atomicAdd((__hip_bfloat16*)&this->velocity[0], a[0]);
        // atomicAdd((__hip_bfloat16*)&this->velocity[1], a[1]);
        // atomicAdd((__hip_bfloat16*)&this->velocity[2], a[2]);
        #else
        // std::atomic<bfloat16>{this->velocity[0]}.fetch_add(a[0]);
        // std::atomic<bfloat16>{this->velocity[1]}.fetch_add(a[1]);
        // std::atomic<bfloat16>{this->velocity[2]}.fetch_add(a[2]);
        #endif
    };

    // cout
    friend std::ostream &operator<<(std::ostream &os, const ChunkOfSpace& q)
    {
        os << "Velocity: (" << q.velocity[0] << ", " << q.velocity[1] << ", " << q.velocity[2] << ")";
        if (q.occupant != nullptr) {
            os << ", Occupant: " << *(q.occupant);
        } else {
            os << ", Occupant: None";
        }
        return os;
    }
};




using SpaceField = Field<ChunkOfSpace>;
using Uint84Field = Field<uint84>;

struct FractalParticle
{
    public:
    FractalParticle* subparticles;
};

constexpr int SIZE = 1;
constexpr int size = SIZE;
constexpr int compression = 1;
constexpr int allowDensity = 1;
constexpr __device__ float surfaceTension = -1.5;


#define FADEOFF(io,jo, ko) 1 //sqrt((io*io+jo*jo + 1.0))

static __host__ __device__ float maxx(float a, float b){
    return a > b ? a : b;
}

static __host__ __device__ float minn(float a, float b){
    return a < b ? a : b;
}



struct VelocitySpread:public HardamardOperation<VelocitySpread> {

    __host__ __device__ static inline
    void apply(SpaceField& velocityField, Particle& particlea, int32x4& screenDims, float32x3 globalAddVelocity, float& depthsortkeya, float32x3 cameraPos, size_t framecount, int& index, int neighborhoodsize) {
        // Simple example: add spread to velocity
        Particle& particle = *(&particlea + index);
        float& depthsortkey = *(&depthsortkeya + index);


        if(globalAddVelocity.dot(globalAddVelocity) > 0.0f){
            particle.temperatureopacity[1] = 1.0f;
        }

        if(particle.temperatureopacity[1] < 1.0f){
            particle.temperatureopacity[1] += 1.0f/60.0f; // fade in over 1 second
            return;
        }

        if(particle.position[0] <=0 || particle.position[1] <=0 || particle.position[2] <=0){
            return;
        }


        int neighborsize = neighborhoodsize;

        float32x3 tocam = cameraPos - particle.position;
        depthsortkey = sqrt((tocam).dot(tocam));

        // if (depthsortkey < 200.0f){
        //     neighborsize = 7;
        // }

        int skipframe = (depthsortkey / 100 + 1);
        if (framecount % skipframe != 0){
            return;
        }


        float32x3 oldparticle = particle.position;
        
        auto oldrelative = int32x4(round(oldparticle[0]), round(oldparticle[1]), round(oldparticle[2]), 0);
        
        if(
        velocityField.get_field_relative(screenDims, oldrelative).occupant == nullptr){
            // not initialized yet
            velocityField.get_field_relative(screenDims, oldrelative).addFilled(&particle);
        }

        
        // if(particle.filled == nullptr){
        //     particle.filled = &velocityField.get_field_relative(screenDims, oldrelative);
        //     // particle.filled->swapfilled(particle);
        // }
        

        
        size_t width = screenDims[0];

        size_t height = screenDims[1];
        size_t depth = screenDims[2];


        const int32x4 neighborIndexOffsets[15] = {
            tnf(-1, 0, 0, 0), // left
            tnf(1, 0, 0, 1), // right
            tnf(0, -1, 0, 2), // down
            tnf(0, 1, 0, 3), // up
            tnf(0, 0, -1, 4), // back
            tnf(0, 0, 1, 5), // front
            tnf(0, 0, 0, 6) // middle
            ,tnf(1, -1, 1, 7), // right-down
            tnf(-1, -1, 1, 8), // left-back
            tnf(1, -1, -1, 9), // right-front
            tnf(-1, -1, -1, 10), // left-front
            tnf(1, 1, 1, 11), // right-back
            tnf(1, 1, -1, 12),  // right-front
            tnf(-1, 1, -1, 13),  // left-front
            tnf(-1, 1, 1, 14)  // left-back
        };




        int neighborallowance = neighborsize == 15 ? 2 : 0; // if we have diagonal neighbors, allow 2 empty neighbors for movement, otherwise require all neighbors to be filled for movement

        float friction = 1.0f/float(neighborsize);
        

        float32x3 momentum(0.0, 0.0, 0.0);
        int allfilled = 0;
        // float32x3 normals = float32x3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < neighborsize; i++) {
            auto& part = velocityField.get_field_relative(screenDims, neighborIndexOffsets[i] + int32x4(round(particle.position[0]), round(particle.position[1]), round(particle.position[2]), 0));
            momentum += part.swap0();
            if (part.occupant != nullptr) {
                allfilled += ((part.occupant->color[3] > 200) ^ (particle.color[3] > 200))? 0 : 1;
            }
            momentum += neighborIndexOffsets[i].xyz() * (part.occupant != nullptr ? 0.0f : surfaceTension); // add density-based repulsion
        }

        // particle.normal = normals;
        float mix = 0.98f;
        
        if(allfilled < neighborsize - neighborallowance ){
            // no velocity
            // particle.color = uint84(particle.color.xyz(), 0xff);
            // particle.temperatureopacity[1] = 1.0f;
            particle.neighborsfilled = (float(allfilled*15)/(neighborsize)*(1.0f-mix) + particle.neighborsfilled*mix);

            particle.position += (momentum - globalAddVelocity) * skipframe;
            
        }else{
            // particle.color = uint84(particle.color.xyz(), 0x00);
            particle.temperatureopacity[1] = 0.0f;
            particle.neighborsfilled = 0.0f;
            // momentum = momentum * 0.95;
            
        }
              

        momentum[1] -= 0.02f;
      
        if (particle.position[1] < 1 + size || particle.position[1] > height - 1 - size)
        {
            particle.position[1] = maxx(2.0f + size, minn(float(height - (2.0 + size)), particle.position[1]));
            momentum[1] *= -0.125f;
        }
        if (particle.position[0] < 1 + size || particle.position[0] > width - 1 - size)
        {
            particle.position[0] = maxx(2.0f + size, minn(float(width - (2.0 + size)), particle.position[0]));
            momentum[0] *= -0.125f;
        }
        if (particle.position[2] < 1 + size || particle.position[2] > depth - 1 - size)
        {
            particle.position[2] = maxx(2.0f + size, minn(float(depth - (2.0 + size)), particle.position[2]));
            momentum[2] *= -0.125f;
        }
        
        auto relative = int32x4(round(particle.position[0]), round(particle.position[1]), round(particle.position[2]), 0);
        
        

        if(oldrelative[0] != relative[0] || oldrelative[1] != relative[1] || oldrelative[2] != relative[2]){

            // velocityField.get_field_relative(screenDims, oldrelative).filled = 0.0;   
            auto& newpos = velocityField.get_field_relative(screenDims, relative);   
            // should be safe
            auto curra = newpos.addFilled(&particle);

            // // should be safe:
            
            if(curra != nullptr){ // collision
                particle.position = oldparticle; // revert position
                momentum = momentum * 0.5; // bounce back
                newpos.atomic_plus_equals(momentum*0.5f); // transfer momentum to new cell
            
                Particle* p = newpos.addFilled(curra);
                if (p == nullptr){
                    // particle.swapfilled(curra);
                    newpos.addFilled(nullptr);
                }
            }
            else{
                // particle.swapfilled(newpos);
                velocityField.get_field_relative(screenDims, oldrelative).addFilled(nullptr);

            }
        }

        
        
\
        auto aa = momentum * friction;
        
        for (int i = 0; i < neighborsize; i++) {
            auto& neighbor = velocityField.get_field_relative(screenDims, neighborIndexOffsets[i] + int32x4(round(particle.position[0]) , round(particle.position[1]) , round(particle.position[2]) , 0));
            neighbor.atomic_plus_equals( aa + (float32x3(neighborIndexOffsets[i].xyz()) * particle.temperatureopacity[0]));
        }
    }
};

__weak float randomfloat(){
    return ((float)(rand() % 1000)) / 1000.0f;
}

constexpr float lavaviscosity = 0.005f;
constexpr float waterviscosity = 0.02f;



__weak int main(){

    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);

    // test float32x3

    // setenv("SDL_DEBUG", "1", 1);
    /*
    export SDL_VIDEO_DRIVER=x11
export EGL_PLATFORM=x11
    */

    int size = 256;
    int horizontal_size = 256;
    int depth_size = 256;
    std::cout << "sizeof(size_t): " << sizeof(size_t) << std::endl;
    std::cout << "sizeof(unsigned long): " << sizeof(unsigned long) << std::endl;
    std::cout << "sizeof(unsigned long long): " << sizeof(unsigned long long) << std::endl;
    std::cout << "sizeof(void*): " << sizeof(void*) << std::endl;
    int testint = 42;
    unsigned long testptr = *(unsigned long*)&testint;
    std::cout << "testptr: " << testptr << std::endl;


    OpenGLDisplay window({1024,1024},  WP_ON_TOP);
    Scene scene(&window);

    Tensor<uint84,2> mypointer = Tensor<uint84,2>({1024,768}, window.device->default_memory_type, kOPENGL);
 
    VectorDisplay display(mypointer);

    global_device_manager.get_device(MemoryType::kDDR,0).default_compute_type = ComputeType::kCPU;

    scene.setCamera({0, 0, -5.0f}, {0, 0, 1.0f}, {0, 1, 0});

    Tensor<float32x4, 1> pointlist = Tensor<float32x4, 1>({4}, MemoryType::kDDR);
    pointlist[{0}] = float32x4(0.0f, 0.0f, 0.0f, 1.0);
    pointlist[{1}] = float32x4(128.0f, 0.0f, 0.0f, 1.0);
    pointlist[{2}] = float32x4(0.0f, 128.0f, 0.0f, 1.0);
    pointlist[{3}] = float32x4(128.0f, 128.0f, 0.0f, 1.0);


    gltf model = gltf("examples/porygon/","scene.gltf");
    std::cout << model << std::endl;
    scene.loadGLTF(model);

//     // using QT = Quaternion<SuperReal2>;

   Tensor<Particle, 3> patch = Tensor<Particle, 3>({100,100,100}, MemoryType::kDDR);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            for(int k = 0; k < 100; k++){
            patch[{j,k,i}] = Particle(i+2, j+2, k+2);
            patch[{j,k,i}].color = uint84(0xff,0xff,0xff, 0xff); // Green color
            float randomfactor = ((float)(rand() % 1000)) / 1000.0f * 0.1f;
            patch[{j,k,i}].temperatureopacity = float32x2(0.01, 1.0f); // random density and full opacity
            if (j >= 90){
                patch[{j,k,i}].color = uint84(0x00, 0x00, 0xff, 0x11); // Red color
                // repulsion at 0.01
                patch[{j,k,i}].temperatureopacity = float32x2(waterviscosity, 1.0f); // low density and half opacity

            }
            }

        }

    }

    auto patchcuda = patch.view(Shape<1>{-1}).to(window.device->default_memory_type);

    size_t rco = 100*100*90;
    Tensor<SpaceField,3> Field = Tensor<SpaceField,3>({size*compression,horizontal_size*compression, depth_size*compression}, window.device->default_memory_type);
    
    Field[{{}}] = SpaceField();

    Tensor<int, 1> particleindex = Tensor<int, 1>({rco}, MemoryType::kDDR);
    for (int i = 0; i < rco; i++){
        particleindex[{i}] = i;
    }

    Tensor<int, 1> particleindexliquid = Tensor<int, 1>({patch.shape.total_size() - rco}, MemoryType::kDDR);
    for (int i = 0; i < patch.shape.total_size() - rco; i++){
        particleindexliquid[{i}] = rco + i;
    }

    Tensor<float,1> depthsortkey = Tensor<float,1>({patch.shape.total_size()}, window.device->default_memory_type);

    // auto particleindexcuda = particleindex.to(window.device->default_memory_type, kOPENGL);

    RenderStruct<float32x3, float32x2, float, uint84> particles_renderable(
        Shape<1>{patch.shape.total_size()},
        particleindex
    );


    RenderStruct<float32x3, float32x2, float, uint84> particles_renderable_liquid(
       particles_renderable,
       particleindexliquid
    );

    

    sampler2D rock = load_texture("./image.png");
    auto rockdevice = rock.to(window.device->default_memory_type, kOPENGLTEXTURE);
    Tensor<Particle, 1> particles = particles_renderable.view<Particle,1>({-1});


    // // particles[{{}}] = Particle(-10.0f, -10.0f, -10.0f); // Initialize all particles off-screen
    particles[{{}}] = patchcuda;
    particles_renderable.material = new Shader<ParticleShader>();
    particles_renderable.material->textures_ids["texture1"] = (unsigned long long)rockdevice.storage_pointer - 0x10000;
    particles_renderable_liquid.material = particles_renderable.material;
    particles_renderable_liquid.material->textures_ids["screentexture"] = (unsigned long long)window.solidParticlesTexture.storage_pointer - 0x10000;

//     Tensor<unsigned long, 1> particleindex = Tensor<unsigned long, 1>({5000*100}, MemoryType::kDDR);

    window.setMouseGrab(true);
// //     // auto MappedDisplay = display.view<Uint84Field,1>({-1}).tensor_index(particleindex);
//     size_t currentIndex = 100*100;
//     int currentFrame = 0;
//     // get current time in milliseconds
    glEnable(GL_PROGRAM_POINT_SIZE);
    // mypointer[{{}}] = 0xffffff00; // White background
    // auto start_time = std::chrono::high_resolution_clock::now();
    Camera& camera = scene.getCamera();
    size_t last_frame_time = 0;
    size_t total_frames = 0;
    window.add_on_update([&](CurrentScreenInputInfo& info){
        size_t current_time = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
        size_t time_since_last_frame = current_time - last_frame_time;
        total_frames++;
        last_frame_time = current_time;
        float32x3 forward = camera.forward.normalized();
        float32x3 right = forward.cross(float32x3(0.0,1.0,0.0)).normalized();
        if(info.isKeyPressed(SDLK_A)){
            camera.position -= right * 1.0f;
            // camera.target -= right * 1.0f;
        }
        if(info.isKeyPressed(SDLK_D)){
            camera.position += right * 1.0f;
            // camera.target += right * 1.0f;
        }
        if(info.isKeyPressed(SDLK_W)){
            camera.position += forward * 1.0f;
            // camera.target += forward * 1.0f;
        }
        if(info.isKeyPressed(SDLK_S)){
            camera.position -= forward * 1.0f;
            // camera.target -= forward * 1.0f;
        }
        if (info.isKeyPressed(SDLK_SPACE)) {
            camera.position[1] += 1.0f;
            // camera.target[1] += 1.0f;
        }
        if (info.isKeyPressed(SDLK_LCTRL)) {
            camera.position[1] -= 1.0f;
            // camera.target[1] -= 1.0f;
        }

        if (info.isMouseGrabbed()) {
            camera.forward = mat4::identity().rotated(-info.getMouseRel().first * 0.005f,float32x3(0.0f, 1.0f, 0.0f)) * (camera.forward );
            camera.forward = mat4::identity().rotated(-info.getMouseRel().second * 0.005f, right) * (camera.forward );
            camera.up = -(camera.forward).cross(right);
        }

        if(info.isKeyPressed(SDLK_TAB)){
            window.setMouseGrab(!info.isMouseGrabbed());
        }
 

       
        // // print aly gl errors
        
        // if still rendering, skip rendering this frame to avoid stuttering
        
        VelocitySpread::run(
                Field[{0,{0,1},{0,1}}],
                particles[{{0,1}}],
                int32x4(horizontal_size*compression, size*compression, depth_size*compression, 0),
                info.relativeWindowMove.xxx() * right + info.relativeWindowMove.yyy() * forward.cross(right),
                depthsortkey[{{0,1}}],
                camera.position,
                total_frames,
                particles_renderable.indices,
                7
            );

            VelocitySpread::run(
                Field[{0,{0,1},{0,1}}],
                particles[{{0,1}}],
                int32x4(horizontal_size*compression, size*compression, depth_size*compression, 0),
                info.relativeWindowMove.xxx() * right + info.relativeWindowMove.yyy() * forward.cross(right),
                depthsortkey[{{0,1}}],
                camera.position,
                total_frames,
                particles_renderable_liquid.indices,
                15
            );
        
        //
        if (total_frames % 5*60==0){
            call_sort(
                depthsortkey.data,
                particles_renderable.indices.data,
                rco,
                kCUDA
            );
        }


        particles_renderable.bind();
        camera.bind(particles_renderable.material->shader_program);
        // // // blend mode additive
        // glDisable(GL_BLEND);
        // // // add blend mode, source + destination
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        mat4::identity().bind(particles_renderable.material->shader_program, "model");

        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        // set depthOnly to 1.0f
        GLuint depthOnlyLocation = GLFuncs->glGetUniformLocation(particles_renderable.material->shader_program, "depthOnly");
        GLFuncs->glUniform1f(depthOnlyLocation, 1.0f);
        
        particles_renderable.draw(); // cheap depth-only

        // Main pass: full shading with early-z rejection
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        GLFuncs->glUniform1f(depthOnlyLocation, 0.0f);
        particles_renderable.draw();
        // particles_renderable.draw();
 
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }

        // water, have it glow by having it be additive blended with itself
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        // glDepthMask(false); // disable depth writing for water to prevent z-fighting

        window.activateBackBuffer();

        particles_renderable_liquid.bind();
        camera.bind(particles_renderable_liquid.material->shader_program);
        particles_renderable_liquid.draw();

        // glDepthMask(true); // re-enable depth writing
        // 


        particles.device->synchronize_function(); // wait until physics is done before rendering? not really needed

        // 60fps
        // SDL_Delay(16);
        // print fps using current_time and last_frame_time
        
        size_t frame_time = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::milliseconds(1) - current_time;
        if(total_frames % 2 == 0){
        std::cout << "Frame time: " << frame_time << " ms, FPS: " << 1000.0f / frame_time << "\r" << std::flush;
        }
        // limit to 60fps by sleeping for the remaining time
        // if (frame_time < 16) {
        //     SDL_Delay(16 - frame_time);
        // }
        
    });
    // std::cout << Field << std::endl;
    window.displayLoop();

    return 0;
}