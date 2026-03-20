
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#ifndef TENSOR_ENUMS_DEVICE_SUPPORT_OPENGL_DEVICE_HPP
#define TENSOR_ENUMS_DEVICE_SUPPORT_OPENGL_DEVICE_HPP

// OpenGL headers

// SDL headers
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

// Standard headers
#include <iostream>
#include <stdexcept>
#include <cstring>
#include "device/common.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unistd.h>
#include <map>
#include <set>
#include <functional>
#include <vector>
#include <chrono>
#include "bfloat16/bf16.hpp"




__weak SDL_Window* gl_window = nullptr;
__weak SDL_GLContext gl_context = nullptr;
__weak bool opengl_initialized = false;
__weak bool gl_functions_loaded = false;

__weak void loadGLFunctions() {
    if (gl_functions_loaded) {
        return; // Already loaded
    }
    
    // init glew
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to initialize GLEW");
    }

    gl_functions_loaded = true;
    std::cout << "OpenGL functions loaded successfully!" << std::endl;
}

#define GL_CHECK(call) \
    do { \
        call; \
        GLenum err = glGetError(); \
        if (err != GL_NO_ERROR) { \
            std::cerr << "OpenGL error at " << __FILE__ << ":" << __LINE__ \
                      << " - Error: 0x" << std::hex << err << std::dec << std::endl; \
        } \
    } while(0)

ComputeDeviceBase* create_opengl_compute_device(int device_id){
  
    std::cout << "Creating OpenGL compute device for device_id " << device_id << std::endl;
    if (device_id != 0) {
        std::cerr << "Invalid OpenGL device_id: " << device_id << " (only device 0 supported)" << std::endl;
        return nullptr;
    }

    ComputeDeviceBase* device = new ComputeDeviceBase();
    
    
    // Get OpenGL device info
    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    
    std::cout << "OpenGL Vendor: " << (vendor ? reinterpret_cast<const char*>(vendor) : "Unknown") << std::endl;
    std::cout << "OpenGL Renderer: " << (renderer ? reinterpret_cast<const char*>(renderer) : "Unknown") << std::endl;
    std::cout << "OpenGL Version: " << (version ? reinterpret_cast<const char*>(version) : "Unknown") << std::endl;
    
    // // Determine memory type based on vendor
    MemoryType mem = MemoryType::kUnknown_MEM;
    
    if (vendor) {
        const char* vendor_str = reinterpret_cast<const char*>(vendor);
        if (strstr(vendor_str, "NVIDIA") != nullptr) {
            // enable uint16_t support for NVIDIA GPUs

            // NVIDIA GPUs typically support uint16_t in shaders, but we should verify this with the renderer string

            mem = MemoryType::kCUDA_VRAM;
        } else if (strstr(vendor_str, "AMD") != nullptr || 
                   strstr(vendor_str, "ATI") != nullptr) {
            mem = MemoryType::kHIP_VRAM;
        } else if (strstr(vendor_str, "Intel") != nullptr) {
            mem = MemoryType::kDDR;
        } else {
            mem = MemoryType::kDDR;
        }
    }
    
    device->default_memory_type = mem;
    device->supports_memory_location[mem] = true;

    auto& mem_device = global_device_manager.get_device(mem, 0);
    mem_device.supports_compute_device[ComputeType::kOPENGL] = true;
    if(mem == kDDR){
         mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](AllocationMetadata metadata, void* existing_data) {
            
          
            GLuint buffer;
            glGenBuffers(1, &buffer);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glBufferStorageEXT(GL_ARRAY_BUFFER, metadata.byte_size, existing_data, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            return new BaseMemoryAllocation(metadata, reinterpret_cast<void*>(static_cast<uintptr_t>(buffer)));
        
        };

        mem_device.compute_type_converters[{ComputeType::kOPENGL, ComputeType::kCPU}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata){
            glBindBuffer(GL_ARRAY_BUFFER, (GLuint)(size_t)ptr);
            std::cout << ptr << " map_opengl\n" ;
            void* ptra = glMapBufferRange(GL_ARRAY_BUFFER, 0, metadata.byte_size, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT); 
            glBindBuffer(GL_ARRAY_BUFFER,0);
            return ptra;
        };

        mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [](void* ptr, BaseMemoryAllocation* original){

            glBindBuffer(GL_ARRAY_BUFFER, (GLuint)(size_t)original->data);
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        };

        
    }
    else{
        mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](AllocationMetadata meta, void* existing_data){
            GLuint buffer;
            glGenBuffers(1, &buffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, meta.byte_size, existing_data, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            return new BaseMemoryAllocation(meta, reinterpret_cast<void*>(static_cast<uintptr_t>(buffer)));
        };
    };

    mem_device.compute_type_converters[{ComputeType::kOPENGLTEXTURE, ComputeType::kCPU}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata){
            return (void*)0;
    };

    mem_device.compute_device_allocators[ComputeType::kOPENGLTEXTURE] = [](AllocationMetadata metadata, void* existing_data){
        GLuint texture;
        // metadata.shape.C == 3 for RGB, size.C == 4 for RGBA
        std::cout << "Metadata for OpenGL texture allocation: " << metadata.format << std::endl;

        if(metadata.format != 0){
            if(metadata.format == GL_DEPTH_COMPONENT24){

                glGenRenderbuffers(1, &texture);
                glBindRenderbuffer(GL_RENDERBUFFER, texture);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, metadata.shape.A, metadata.shape.B);
                glBindRenderbuffer(GL_RENDERBUFFER, 0);
            } else if (metadata.format == GL_DEPTH_COMPONENT32F){
                glGenRenderbuffers(1, &texture);
                glBindRenderbuffer(GL_RENDERBUFFER, texture);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, metadata.shape.A, metadata.shape.B);
                glBindRenderbuffer(GL_RENDERBUFFER, 0);
            }
            else{
                std::cerr << "Unsupported metadata format for OpenGL texture: " << metadata.format << std::endl;
                throw std::runtime_error("Unsupported metadata format for OpenGL texture");
            }
            

            GL_CHECK();
        } else
        { // guess format based on bitsize and channels

            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            if (metadata.type_size == 3) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, metadata.shape.A, metadata.shape.B, 0, GL_RGB, GL_UNSIGNED_BYTE, existing_data);
            } else if (metadata.type_size == 4) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, metadata.shape.A, metadata.shape.B, 0, GL_RGBA, GL_UNSIGNED_BYTE, existing_data);
            } else if (metadata.type_size == 6) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, metadata.shape.A, metadata.shape.B, 0, GL_RGB, GL_FLOAT, existing_data);
            } else if (metadata.type_size == 8) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, metadata.shape.A, metadata.shape.B, 0, GL_RGBA, GL_FLOAT, existing_data);
            } else {
                std::cerr << "Unsupported metadata.type_size for OpenGL texture: " << metadata.type_size << std::endl;
                throw std::runtime_error("Unsupported bitsize for OpenGL texture");
            }


            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

        }
        

        // prevent collisions, += 10k
        return new BaseMemoryAllocation(metadata, reinterpret_cast<void*>(static_cast<uintptr_t>(texture)));
        
    };

    mem_device.compute_device_deallocators[ComputeType::kOPENGL] = [](void* ptr) {
        GLuint buffer = static_cast<GLuint>(reinterpret_cast<uintptr_t>(ptr));
        glDeleteBuffers(1, &buffer);
    };

    mem_device.compute_device_deallocators[ComputeType::kOPENGLTEXTURE] = [](void* ptr) {
        GLuint texture = static_cast<GLuint>(reinterpret_cast<uintptr_t>(ptr));
        glDeleteTextures(1, &texture);
    };

    

    std::cout << "OpenGL device default memory type: " << mem << std::endl;
    
    global_device_manager.compute_device_counts[ComputeType::kOPENGL] = 1;
    global_device_manager.compute_devices[ComputeType::kOPENGL] = new ComputeDeviceBase*[1];
    global_device_manager.compute_devices[ComputeType::kOPENGL][0] = device;
    return device;
}

int count_opengl_devices(){
    if (opengl_initialized) {
        return 1;
    }
    
    
    
    return 1;
}

#endif // TENSOR_ENUMS_DEVICE_SUPPORT_OPENGL_DEVICE_HPP