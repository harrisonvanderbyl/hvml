
#include "particle/chunk.hpp"



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


// uint8_t swappoint = 0b000; // 

static inline __host__ __device__ int32x4 tnf(int a, int b, int c, int d){ 
    return int32x4(a, b, c, 1 << d);
}






constexpr int SIZE = 1;
constexpr int size = SIZE;
constexpr int compression = 1;
constexpr int allowDensity = 1;
constexpr __device__ float surfaceTension = 1.1f; // how much to repel from empty space, negative values cause repulsion, positive values cause attraction
constexpr float airrepel = -0.2f; // how much to repel from empty space when not fully submerged, to prevent clumping and encourage spreading out
constexpr float waterviscosity = 1.0f;
constexpr float gravity = 0.1f;

#define FADEOFF(io,jo, ko) 1 //sqrt((io*io+jo*jo + 1.0))

static __host__ __device__ float maxx(float a, float b){
    return a > b ? a : b;
}

static __host__ __device__ float minn(float a, float b){
    return a < b ? a : b;
}


constexpr __device__ int awidth = 256;
constexpr __device__ int aheight = 128;
constexpr __device__ int adepth = 10240;

struct ParticleGetDistance :public HardamardOperation<ParticleGetDistance> {
    __host__ __device__ static inline
    void apply(Particle& particle, float32x3 cameraPos, float& adepthsortkeya, int& indeex) {
        float32x3 tocam = cameraPos - (&particle + indeex)->position.xyz();
        adepthsortkeya = sqrt((tocam).dot(tocam));
    };
};

struct VelocitySpread:public HardamardOperation<VelocitySpread> {

    __host__ __device__ static inline
    void apply(ParticleChunk<awidth,aheight,adepth>& velocityField, Particle& particlea, float32x3 globalAddVelocity, size_t framecount, int& index, int neighborhoodsize, int skippedframes){ 
        // Simple example: add spread to velocity
        Particle& particle = *(&particlea + index);

        bool firststep = false;
        if(globalAddVelocity.dot(globalAddVelocity) > 0.0f){
            particle.temperatureopacity[1] = 1.0f;
        }

        if(particle.temperatureopacity[1] < 1.0f){
            particle.temperatureopacity[1] += 1.0f/60.0f; // fade in over 1 second
            // return;
        }

        // if(particle.position[0] <=0 || particle.position[1] <=0 || particle.position[2] <=0){
        //     return;
        // }


        int neighborsize = neighborhoodsize;


        // if (adepthsortkey < 200.0f){
        //     neighborsize = 7;
        // }

        // int skipframe = (adepthsortkey / 100 + 1);
        // if (framecount % skipframe != 0){
        //     return;
        // }


        float32x3 oldparticle = particle.position;
        
        auto oldrelative = int32x3(round(oldparticle[0]), round(oldparticle[1]), round(oldparticle[2]));
        
        if(
            velocityField.get_chunk_relative(oldrelative).occupant == nullptr){
            // not initialized yet
            velocityField.get_chunk_relative(oldrelative).addFilled(&particle);
            firststep = true;
        }

        


        const int32x3 neighborIndexOffsets[15] = {
            int32x3(-1, 0, 0), // left
            int32x3(1, 0, 0), // right
            int32x3(0, -1, 0), // down
            int32x3(0, 1, 0), // up
            int32x3(0, 0, -1), // back
            int32x3(0, 0, 1), // front
            int32x3(0, 0, 0) // middle
            ,int32x3(1, -1, 1), // right-down
            int32x3(-1, -1, 1), // left-back
            int32x3(1, -1, -1), // right-front
            int32x3(-1, -1, -1), // left-front
            int32x3(1, 1, 1), // right-back
            int32x3(1, 1, -1),  // right-front
            int32x3(-1, 1, -1),  // left-front
            int32x3(-1, 1, 1)  // left-back
        };

        float attractionForce =  particle.temperatureopacity[0];


        int neighborallowance = neighborsize == 15 ? 2 : 0; // if we have diagonal neighbors, allow 2 empty neighbors for movement, otherwise require all neighbors to be filled for movement

        float friction = 1.0f/float(neighborsize);
        

        float32x3 momentum(0.0, 0.0, 0.0);

        int allfilled = 0;

        bool mesolid =  (particle.color[3] > 200);
        for (int i = 0; i < neighborsize; i++) {
            auto& part = velocityField.get_chunk_relative(neighborIndexOffsets[i] + int32x3(round(particle.position[0]), round(particle.position[1]), round(particle.position[2])));
            float32x4 neighborVelocity = part.swap0();
            momentum += neighborVelocity.xyz();

            if (part.occupant != nullptr) {
                bool partocsolid = (part.occupant->color[3] > 200);
                allfilled += (mesolid ^ partocsolid)? 0 : 1;
            }

        }

        // particle.normal = normals;
        float mix = 0.98f;
        
        if( allfilled < neighborsize - neighborallowance || !mesolid ){
            // no velocity
            // particle.color = uint84(particle.color.xyz(), 0xff);
            // particle.temperatureopacity[1] = 1.0f;
            particle.neighborsfilled = (float(allfilled*15)/(neighborsize)*(1.0f-mix) + particle.neighborsfilled*mix);

            auto nmomentum = (momentum ) ;

            // if((nmomentum.dot(nmomentum)) > 1.0f){
            //     nmomentum = nmomentum * (1.0f / sqrt(nmomentum.dot(nmomentum))); // normalize to prevent excessive speed
            // }
            particle.position += nmomentum- globalAddVelocity;
            
            if (firststep){
                attractionForce = 0.0f; // don't apply attraction force on first step to prevent clumping at spawn point
            }

            if(!mesolid){
                attractionForce *= 1.0f - pow(float(neighborsize - allfilled + 1 )/float(neighborsize),0.85f)*surfaceTension;
            }// particle.position += surfaceTensionVel;
            
            // attractionForce /= skippedframes; // divide by skipped frames to prevent excessive speed from attraction when skipping frames

        }else{
            // particle.color = uint84(particle.color.xyz(), 0x00);
            particle.temperatureopacity[1] = 0.0f;
            particle.neighborsfilled = 0.0f;
            // momentum = momentum * 0.95;
            
        }
              

        momentum[1] -= gravity;
      
        if (particle.position[1] < 1 + size || particle.position[1] > aheight - 1 - size)
        {
            particle.position[1] = maxx(2.0f + size, minn(float(aheight - (2.0 + size)), particle.position[1]));
            momentum[1] *= -0.0f;
        }

        if (particle.position[0] < 1 + size || particle.position[0] > awidth - 1 - size)
        {
            particle.position[0] = maxx(2.0f + size, minn(float(awidth - (2.0 + size)), particle.position[0]));
            momentum[0] *= -0.0f;
        }

        // if (particle.position[2] < 1 + size || particle.position[2] > adepth - 1 - size)
        // {
        //     particle.position[2] = maxx(2.0f + size, minn(float(adepth - (2.0 + size)), particle.position[2]));
        //     momentum[2] *= -0.0f;
        // }
        
        
        auto relative = int32x3(round(particle.position[0]), round(particle.position[1]), round(particle.position[2]));
        
        

        if(oldrelative[0] != relative[0] || oldrelative[1] != relative[1] || oldrelative[2] != relative[2]){

            // velocityField.get_field_relative(screenDims, oldrelative).filled = 0.0;   
            auto& newpos = velocityField.get_chunk_relative(relative);   
            // should be safe
            auto curra = newpos.addFilled(&particle, true);

            // // should be safe:
            
            if(curra != nullptr){ // collision
                particle.position = oldparticle; // revert position
                
                // momentum = momentum * 0.99; // lose momentum from collision
                newpos.atomic_plus_equals(float32x4(momentum*0.5, 0.0f)); // transfer momentum to other particle
                momentum *= 0.45; // lose momentum from collision
            }
            else{
                // particle.swapfilled(newpos);
                velocityField.get_chunk_relative(oldrelative).addFilled(nullptr);

            }
        }

        
        auto aa = momentum * friction;
        
        for (int i = 0; i < neighborsize; i++) {
            auto& neighbor = velocityField.get_chunk_relative(neighborIndexOffsets[i] + int32x3(round(particle.position[0]) , round(particle.position[1]) , round(particle.position[2])));
            neighbor.atomic_plus_equals( float32x4(aa + (float32x3(neighborIndexOffsets[i].xyz()) * attractionForce), 1.0f));
        }
    }
};

__weak float randomfloat(){
    return ((float)(rand() % 1000)) / 1000.0f;
}

struct ParticleStorageDisk{
    // a block of data on disk that can be loaded into memory in chunks, to allow for larger simulations than memory can hold, at the cost of speed
    bool are_particles_visible; // When particles are visible, but not simulated, these external particles can be used.

};



__weak int main(){

    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);


    float32x3 chunksizes = float32x3(awidth, aheight, adepth);
    float32x3 chunksperrotation = float32x3(20000,1,100);

    
    std::cout << "sizeof(size_t): " << sizeof(size_t) << std::endl;
    std::cout << "sizeof(unsigned long): " << sizeof(unsigned long) << std::endl;
    std::cout << "sizeof(size_t): " << sizeof(size_t) << std::endl;
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
    size_t num_particles = 100*100*1000;

   Tensor<Particle, 3> patch = Tensor<Particle, 3>({100,100,1000}, MemoryType::kDDR);
    for(int i = 0; i < 1000; i++){
        for(int j = 0; j < 100; j++){
            for(int k = 0; k < 100; k++){
            patch[{j,k,i}] = Particle(k+2, j+2, i+2);
            patch[{j,k,i}].color = uint84(0xff,0xff,0xff, 0xff); // Green color
            float randomfactor = ((float)(rand() % 1000)) / 1000.0f * 0.1f;
            patch[{j,k,i}].temperatureopacity = float32x2(0.00, 1.0f); // random density and full opacity
            if (j >= 50){
                patch[{j,k,i}].color = uint84(0x00, 0x00, 0x00, 0x11); // Red color
                // repulsion at 0.01
                patch[{j,k,i}].temperatureopacity = float32x2(waterviscosity, 1.0f); // low density and half opacity

            }
            }

        }

    }

    auto patchcuda = patch.view(Shape<1>{-1}).to(window.device->default_memory_type);

    size_t rco = 100*1000*50;
    ChunkGrid<awidth, aheight, adepth> Field({1,1,1}, window.device->default_memory_type);
    std::cout << "w,h,d: " << awidth << "," << aheight << "," << adepth << std::endl;
    // Field[{{}}] = SpaceField();
    std::cout << "Initialized field" << std::endl;
    std::cout << Field << std::endl;

    Tensor<int, 1> particleindex = Tensor<int, 1>({rco}, MemoryType::kDDR);
    for (int i = 0; i < rco; i++){
        particleindex[{i}] = i;
    }

    Tensor<int, 1> particleindexliquid = Tensor<int, 1>({num_particles - rco}, MemoryType::kDDR);
    for (int i = 0; i < num_particles - rco; i++){
        particleindexliquid[{i}] = rco + i;
    }

    int num_particles_simulate = 1'000'000;

    Tensor<int, 1> runnableIndex = Tensor<int, 1>({num_particles}, MemoryType::kDDR);
    for (int i = 0; i < num_particles; i++){
        runnableIndex[{i}] = i;
    }

    auto runnableIndexdevice = runnableIndex.to(window.device->default_memory_type);

    Tensor<float,1> adepthsortkey = Tensor<float,1>({num_particles}, window.device->default_memory_type);

    // auto particleindexcuda = particleindex.to(window.device->default_memory_type, kOPENGL);

    RenderStruct<float32x3, float32x2, float, uint84> particles_renderable(
        Shape<1>{num_particles},
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
    particles_renderable.material->textures_ids["texture1"] = (size_t)rockdevice.storage_pointer->data;
    particles_renderable_liquid.material = particles_renderable.material;
    particles_renderable_liquid.material->textures_ids["screentexture"] = (size_t)window.solidParticlesTexture.storage_pointer->data;

    window.setMouseGrab(true);

    glEnable(GL_PROGRAM_POINT_SIZE);
    Camera& camera = scene.getCamera();
    size_t last_frame_time = 0;
    size_t total_frames = 0;
    window.add_on_update([&](CurrentScreenInputInfo& info){
        size_t current_time = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
        size_t time_since_last_frame = current_time - last_frame_time;
        last_frame_time = current_time;
        float32x3 forward = camera.forward.normalized();
        float32x3 right = forward.cross(float32x3(0.0,1.0,0.0)).normalized();
        if(info.isKeyPressed(SDLK_A)){
            camera.position -= right * 1.0f;
        }
        if(info.isKeyPressed(SDLK_D)){
            camera.position += right * 1.0f;
        }
        if(info.isKeyPressed(SDLK_W)){
            camera.position += forward * 1.0f;
        }
        if(info.isKeyPressed(SDLK_S)){
            camera.position -= forward * 1.0f;
        }
        if (info.isKeyPressed(SDLK_SPACE)) {
            camera.position[1] += 1.0f;
        }
        if (info.isKeyPressed(SDLK_LCTRL)) {
            camera.position[1] -= 1.0f;
        }

        if (info.isMouseGrabbed()) {
            camera.forward = mat4::identity().rotated(-info.getMouseRel().first * 0.005f,float32x3(0.0f, 1.0f, 0.0f)) * (camera.forward );
            camera.forward = mat4::identity().rotated(-info.getMouseRel().second * 0.005f, right) * (camera.forward );
            camera.up = -(camera.forward).cross(right);
        }

        if(info.isKeyPressed(SDLK_TAB)){
            window.setMouseGrab(!info.isMouseGrabbed());
        }

        if(info.isKeyPressed(SDLK_O)) {
            // save particle positions to file
            particles.to(MemoryType::kDDR).to("particles.bin");
            particles.device->synchronize_function(); // ensure data is written to disk before continuing
        }

        if(info.isKeyPressed(SDLK_L)) {
            // load particle positions from file
            auto loadedparticles = Tensor<Particle, 1>(particles.shape,"particles.bin");
            particles.device->synchronize_function();
            auto particlesdevice = loadedparticles.to(MemoryType::kDDR).to(*particles.device);
            particles[{{}}] = particlesdevice;
            // reset Field to prevent collisions from old positions
            Field[{{}}] = 0;
            std::cout << "Loaded particles from file" << std::endl;
            std::cout << particlesdevice << std::endl;
            std::cout << particles << std::endl;
            particles.device->synchronize_function(); // ensure data is loaded before continuing
        }
 

       
        // // print aly gl errors
        
        // if still rendering, skip rendering this frame to avoid stuttering
        if(total_frames < 2){
            for (int i = 0; i < 10; i++){
                VelocitySpread::run(
                    Field[{{},{},{}}],
                    particles[{{0,1}}],
                    info.relativeWindowMove.xxx() * right + info.relativeWindowMove.yyy() * forward.cross(right),
                    total_frames,
                    runnableIndexdevice,
                    7,
                    1
            );
            }
        }else{
            // num_particles
            int lods = 8;
            for (int i = 0; i < lods; i++){
                // stagger each sublod so that they update on different frames to spread out the workload and prevent stuttering
                // skip 1,2,3,4,5,6,7,8,9 frames,
                // or
                // every 1 frame, 2 frames, 3 frames, 4 frames, 5 frames, 6 frames, 7 frames, 8 frames, 9 frames, 10 frames
                for (int j = 0; j < i+1; j++){
                    if(total_frames % (i+1) == j){
                        VelocitySpread::run(
                                Field[{{},{},{}}],
                                particles[{{0,1}}],
                                info.relativeWindowMove.xxx() * right + info.relativeWindowMove.yyy() * forward.cross(right),
                                total_frames,
                                runnableIndexdevice[{{i* (num_particles / lods) + j, (i+1)* (num_particles / lods), i+1}}],
                                7,
                                i+1
                                
                        );
                        
                    }
                }
            }
        }
        
        
        

        
        //
        if (total_frames % 60==6){
            particles_renderable.device->synchronize_function();

            ParticleGetDistance::run(
                particles[{{0,1}}],
                camera.position,
                adepthsortkey,
                runnableIndexdevice
            );


            particles_renderable.device->synchronize_function();

            call_sort(
                adepthsortkey.data,
                particles_renderable.indices.data,
                rco,
                kCUDA
            );
            particles_renderable.device->synchronize_function();

            ParticleGetDistance::run(
                particles[{{0,1}}],
                camera.position,
                adepthsortkey,
                runnableIndexdevice
            );


            particles_renderable.device->synchronize_function();

            call_sort(
                adepthsortkey.data,
                runnableIndexdevice.data,
                num_particles,
                kCUDA
            );

            particles_renderable.device->synchronize_function();
        }

        // VelocitySpread::run(
        //         Field[{{},{},{}}],
        //         particles[{{0,1}}],
        //         info.relativeWindowMove.xxx() * right + info.relativeWindowMove.yyy() * forward.cross(right),
        //         total_frames,
        //         runnableIndexdevice,
        //         7
        // );


        particles_renderable.bind();
        camera.bind(particles_renderable.material->shader_program);
        // // // blend mode additive
        // glDisable(GL_BLEND);
        // // // add blend mode, source + destination
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        mat4::identity().bind(particles_renderable.material->shader_program, "model");

        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        // set depthOnly to 1.0f
        GLuint depthOnlyLocation = glGetUniformLocation(particles_renderable.material->shader_program, "depthOnly");
        GLuint chunksizesLocation = glGetUniformLocation(particles_renderable.material->shader_program, "chunksizes");
        GLuint chunksperrotationLocation = glGetUniformLocation(particles_renderable.material->shader_program, "chunksperrotation");
        glUniform3fv(chunksizesLocation, 1, &chunksizes[0]);
        glUniform3fv(chunksperrotationLocation, 1, &chunksperrotation[0]);
        glUniform1f(depthOnlyLocation, 1.0f);
        
        particles_renderable.draw(); // cheap adepth-only

        // Main pass: full shading with early-z rejection
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        glUniform1f(depthOnlyLocation, 0.0f);
        particles_renderable.draw();
        // particles_renderable.draw();
 
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }

        // water, have it glow by having it be additive blended with itself
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        // glaDepthMask(false); // disable adepth writing for water to prevent z-fighting

        window.activateBackBuffer();

        particles_renderable_liquid.bind();
        glUniform3fv(chunksizesLocation, 1, &chunksizes[0]);
        glUniform3fv(chunksperrotationLocation, 1, &chunksperrotation[0]);
        camera.bind(particles_renderable_liquid.material->shader_program);
        particles_renderable_liquid.draw();

        // glaDepthMask(true); // re-enable adepth writing
        // 

        // model.skeletons[0][0] = mat4::identity().translated(particles[{{99*100*100,99*100*100+1}}].to(kDDR)[0].position);
        // model.skeletons[0][1] = mat4::identity().translated(particles[{{99*100*100+1,99*100*100+2}}].to(kDDR)[0].position);
        particles.device->synchronize_function(); // wait until physics is done before rendering? not really needed

        // 60fps
        // SDL_Delay(16);
        // print fps using current_time and last_frame_time
        
        size_t frame_time = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::milliseconds(1) - current_time;
        
        std::cout << "Frame time: " << frame_time << " ms, FPS: " << 1000.0f / frame_time << "\r" << std::flush;
        
        // limit to 60fps by sleeping for the remaining time
        // if (frame_time < 16) {
        //     SDL_Delay(16 - frame_time);
        // }

        total_frames++;
        
    });
    // std::cout << Field << std::endl;
    window.displayLoop();

    return 0;
}