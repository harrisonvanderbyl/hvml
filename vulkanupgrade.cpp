
#include "particle/chunk.hpp"
#include "display/materials/overlay.hpp"


__weak int main(){
    setenv("DRI_PRIME", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEO_DRIVER", "x11", 1);
    setenv("EGL_PLATFORM", "x11", 1);
    
    OpenGLDisplay window({1024,1024},  WP_ON_TOP);
    Scene scene(&window);


    global_device_manager.get_device(MemoryType::kDDR,0).default_compute_type = ComputeType::kCPU;

    scene.setCamera({0, 0, -5.0f}, {0, 0, 1.0f}, {0, 1, 0});

    VectorDisplay<float16x4> display({1024,1024}, kOPENGL);
    display[{{}}] = float16x4{0.0f,0.0f,0.0f,0.0f};
    display[{{0,100},{0,100}}] = float16x4{0.5f,0.5f,0.5f,1.0f};

    gltf model = gltf("examples/porygon/","scene.gltf");
    std::cout << model << std::endl;
    scene.loadGLTF(model);



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

        
 
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }

        SDL_Delay(16);
        // print fps using current_time and last_frame_time
        
        size_t frame_time = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::milliseconds(1) - current_time;
        
        std::cout << "Frame time: " << frame_time << " ms, FPS: " << 1000.0f / frame_time << "\r" << std::flush;
        
        // limit to 60fps by sleeping for the remaining time
        // if (frame_time < 16) {
        //     SDL_Delay(16 - frame_time);
        // }
        window.activateBackBuffer();
        display.present();

        total_frames++;
        
    });
    // std::cout << Field << std::endl;
    window.displayLoop();

    return 0;
}