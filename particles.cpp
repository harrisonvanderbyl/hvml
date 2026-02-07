
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



struct Particle
{
    public:
    float32x3 position; // x, y, z
    float32x2 temperatureopacity = float32x2(0.0f, 1.0f); // density and opacity
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
    float32x3 velocity;
    Particle* occupant = nullptr;
    // ChunkOfSpace* neighboringChunks[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
     __device__ __host__ Particle* addFilled(Particle* increment = nullptr){
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return (Particle*)(void*)atomicExch((unsigned long long*)&this->occupant, (unsigned long long)increment);
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
        
        return float32x3(atomicExch((float*)&this->velocity[0], 0.0f), atomicExch((float*)&this->velocity[1], 0.0f), atomicExch((float*)&this->velocity[2], 0.0f));
        
        #else
        return float32x3(std::atomic<float>{this->velocity[0]}.exchange(0.0f),
                        std::atomic<float>{this->velocity[1]}.exchange(0.0f),
                        std::atomic<float>{this->velocity[2]}.exchange(0.0f));
        #endif
    };
    __device__ __host__ float32x3& atomic_plus_equals(const float32x3& a){
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        atomicAdd((float*)&this->velocity[0], a[0]);
        atomicAdd((float*)&this->velocity[1], a[1]);
        atomicAdd((float*)&this->velocity[2], a[2]);
        #else
        std::atomic<float>{this->velocity[0]}.fetch_add(a[0]);
        std::atomic<float>{this->velocity[1]}.fetch_add(a[1]);
        std::atomic<float>{this->velocity[2]}.fetch_add(a[2]);
        #endif
        return this->velocity.xyz();
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

#define repulsionFactor(r, distance) r

#define FADEOFF(io,jo, ko) 1 //sqrt((io*io+jo*jo + 1.0))

static __host__ __device__ float maxx(float a, float b){
    return a > b ? a : b;
}

static __host__ __device__ float minn(float a, float b){
    return a < b ? a : b;
}


struct VelocitySpread:public HardamardOperation<VelocitySpread> {

    __host__ __device__ static inline
    void apply(SpaceField& velocityField, Particle& particle, int32x4& screenDims, float32x3 globalAddVelocity) {
        // Simple example: add spread to velocity
        if(particle.position[0] <=0 || particle.position[1] <=0 || particle.position[2] <=0){
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
        

        auto friction = 0.0;
        for (int i = -size; i <= size ; i++)
        {
            for (int j = -size; j <= size ; j++)
            {
                for (int k = -size; k <= size ; k++)
                {
                    friction += 1 / FADEOFF(i, j, k);
                }
            }
        }
        friction = 1. / friction;
        size_t width = screenDims[0];

        size_t height = screenDims[1];
        size_t depth = screenDims[2];
        

        float32x3 momentum(0.0, 0.0, 0.0);
        float allfilled = 0.0f;
        // float32x3 normals = float32x3(0.0f, 0.0f, 0.0f);
        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                for (int k = - SIZE; k <= SIZE; k++) {
                    auto& part = velocityField.get_field_relative(screenDims, int32x4(round(particle.position[0])+i, round(particle.position[1])+j, round(particle.position[2])+k, 0));
                    momentum += part.swap0();
                    if (part.occupant != nullptr) {
                        allfilled += part.occupant->color[3] > 200 ? 1.0f : 0.0f;
                    }
                    // if(part.occupant != nullptr){
                    //     normals += particle.position - part.occupant->position;
                    // }
                }
            }
        }

        // particle.normal = normals;


        if(allfilled < (SIZE*2+1) * (SIZE*2+1) * (SIZE*2+1)){
            // no velocity
            // particle.color = uint84(particle.color.xyz(), 0xff);
            particle.temperatureopacity[1] = 1.0f;
             
        }else{
            // particle.color = uint84(particle.color.xyz(), 0x00);
            particle.temperatureopacity[1] = 0.0f;
            momentum[1] -= 0.1f;
            momentum = momentum * 0.9f;
            momentum = momentum * friction;
            auto dis = sqrt(momentum.dot(momentum)+1.0);
            for (int i = - SIZE; i <= SIZE; i++) {
                for (int j = - SIZE; j <= SIZE; j++) {
                    for (int k = - SIZE; k <= SIZE; k++) {
                        auto& neighbor = velocityField.get_field_relative(screenDims, int32x4(round(particle.position[0])+ i , round(particle.position[1])+ j , round(particle.position[2]) + k, 0));
                        neighbor.atomic_plus_equals( momentum + (float32x3(i*dis, j*dis, k*dis) * particle.temperatureopacity[0]) / FADEOFF(i, j, k));
                    }
                }
            }

            return;
        }
              
        // momentum +=  ;
        particle.position += momentum - globalAddVelocity;
        // momentum += Particle(globalAddVelocity[0], globalAddVelocity[1],0.0)*0.1f;

        momentum[1] -= 0.1f;
        if ( particle.position[0] < 1 + size || particle.position[0] > width - 1 - size || particle.position[1] < 1 + size || particle.position[1] > height - 1 - size || particle.position[2] < 1 + size || particle.position[2] > depth - 1 -size)
        {

            // Particle beforebounce = particle;
            if (particle.position[1] < 1 + size || particle.position[1] > height - 1 - size)
            {
                particle.position[1] = maxx(2.0f + size, minn(float(height - (2.0 + size)), particle.position[1]));
                momentum[1] *= -0.25f;
            }
            if (particle.position[0] < 1 + size || particle.position[0] > width - 1 - size)
            {
                particle.position[0] = maxx(2.0f + size, minn(float(width - (2.0 + size)), particle.position[0]));
                momentum[0] *= -0.25f;
            }
            if (particle.position[2] < 1 + size || particle.position[2] > depth - 1 - size)
            {
                particle.position[2] = maxx(2.0f + size, minn(float(depth - (2.0 + size)), particle.position[2]));
                momentum[2] *= -0.25f;
            }
            
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
                //  newpos.atomic_plus_equals(momentum * -1.0f);
                // Particle assumed_oppos = {float(relative[0]) , float(relative[1]), float(relative[2]) };
                // auto norm = (particle - assumed_oppos);
                // // reflect momentum along normal
                // float dot = (momentum[0] * norm[0] + momentum[1] * norm[1] + momentum[2] * norm[2]) / (norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
                // momentum = momentum - norm * (2.0 * dot);
                momentum = momentum * 0.5; // bounce back
                newpos.atomic_plus_equals(momentum);
            
                newpos.addFilled(curra);
            }
            else{
                // particle.swapfilled(newpos);
                velocityField.get_field_relative(screenDims, oldrelative).addFilled(nullptr);

            }
        }

        
        // auto a = particle.swapfilled());
        // if(a)->swapfilled(particle);
        
        
\
        auto aa = momentum * friction;
        
        // auto& curr = *particle.filled;
        auto& mpos = velocityField.get_field_relative(screenDims, int32x4(round(particle.position[0]) , round(particle.position[1]) , round(particle.position[2]), 0));
       
        // mpos.atomic_plus_equals(aa);
        auto dis = sqrt(aa.dot(aa)+1.0);
        for (int i = - SIZE; i <= SIZE; i++) {
            for (int j = - SIZE; j <= SIZE; j++) {
                for (int k = - SIZE; k <= SIZE; k++) {
                    auto& neighbor = velocityField.get_field_relative(screenDims, int32x4(round(particle.position[0])+ i , round(particle.position[1])+ j , round(particle.position[2]) + k, 0));
                    neighbor.atomic_plus_equals( aa +  (float32x3(i*dis, j*dis, k*dis) * particle.temperatureopacity[0]) / FADEOFF(i, j, k) );
                }
            }
        }
    }
};



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

    int size = 128;
    int horizontal_size = 512;
    int depth_size = 512;


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
            patch[{j,k,i}] = Particle(i+5, j+5, k+5); // 10 by 10 block of particles
            patch[{j,k,i}].color = uint84(0xff,0xff,0xff, 0xff); // Green color
            float randomfactor = ((float)(rand() % 1000)) / 1000.0f * 0.1f;
            patch[{j,k,i}].temperatureopacity = float32x2(0.001, 1.0f); // random density and full opacity
            if (j >= 80){
                patch[{j,k,i}].color = uint84(0x00, 0x00, 0xff, 0x11); // Red color
                // repulsion at 0.01
                patch[{j,k,i}].temperatureopacity = float32x2(0.04f, 1.0f); // low density and half opacity

            }
            }

        }

    }

    auto patchcuda = patch.view(Shape<1>{-1}).to(window.device->default_memory_type);

    
    Tensor<SpaceField,3> Field = Tensor<SpaceField,3>({size*compression,horizontal_size*compression, depth_size*compression}, window.device->default_memory_type);
    
    Field[{{}}] = SpaceField();

    RenderStruct<float32x3, float32x2, uint84> particles_renderable(
        Shape<1>{100*100*100}
    );

    particles_renderable.count = 100*100*80;

    RenderStruct<float32x3, float32x2, uint84> particles_renderable_liquid(
       particles_renderable
    );

    particles_renderable_liquid.count = 100*100*20;
    particles_renderable_liquid.offset = 100*100*80;


    // sampler2D rock = load_texture("./image.png");
    // auto rockdevice = rock.to(window.device->default_memory_type, kOPENGLTEXTURE);
    Tensor<Particle, 1> particles = particles_renderable.view<Particle,1>({-1});
    // // particles[{{}}] = Particle(-10.0f, -10.0f, -10.0f); // Initialize all particles off-screen
    particles[{{}}] = patchcuda;
    particles_renderable.material = new Shader<ParticleShader>();
    // particles_renderable.material->textures_ids["texture1"] = (unsigned long long)rockdevice.storage_pointer - 0x10000;
    particles_renderable_liquid.material = particles_renderable.material;
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
    window.add_on_update([&](CurrentScreenInputInfo& info){

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
 

        VelocitySpread::run(
            Field[{0,{0,1},{0,1}}],
            particles,
            int32x4(horizontal_size*compression, size*compression, depth_size*compression, 0),
            info.relativeWindowMove.xxx() * right + info.relativeWindowMove.yyy() * forward.cross(right)
        );
        particles.device->synchronize_function();

        // // print aly gl errors
        
        // // set point size to 10
        particles_renderable.bind();
        camera.bind(particles_renderable.material->shader_program);
        // // // blend mode additive
        glEnable(GL_BLEND);
        // // // add blend mode, source + destination
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        mat4::identity().bind(particles_renderable.material->shader_program, "model");
        particles_renderable.draw();
 
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }

        // water, have it glow by having it be additive blended with itself
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(false); // disable depth writing for water to prevent z-fighting

        particles_renderable_liquid.bind();
        camera.bind(particles_renderable_liquid.material->shader_program);
        particles_renderable_liquid.draw();

        glDepthMask(true); // re-enable depth writing
        // 60fps
        // SDL_Delay(16);
    });
    // std::cout << Field << std::endl;
    window.displayLoop();

    return 0;
}