#ifndef TENSOR_ENUMS_DEVICE_SUPPORT_OPENGL_DEVICE_HPP
#define TENSOR_ENUMS_DEVICE_SUPPORT_OPENGL_DEVICE_HPP

// OpenGL headers
#include <GL/gl.h>
#include <GL/glext.h>

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

// OpenGL function pointer typedefs
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
typedef void (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void (APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint *arrays);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
typedef void (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLVERTEXATTRIBIPOINTERPROC)(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORMMATRIX4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (APIENTRY *PFNGLUNIFORM3FVPROC)(GLint location, GLsizei count, const GLfloat *value);
typedef void (APIENTRY *PFNGLUNIFORM1FPROC)(GLint location, GLfloat value);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint value);
typedef void (APIENTRY *PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint *framebuffers);
typedef void (APIENTRY *PFNGLBINDFRAMEBUFFERPROC)(GLenum target, GLuint framebuffer);
typedef void (APIENTRY *PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint *framebuffers);
typedef void (APIENTRY *PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
typedef void (APIENTRY *PFNGLGENRENDERBUFFERSPROC)(GLsizei n, GLuint *renderbuffers);
typedef void (APIENTRY *PFNGLBINDRENDERBUFFERPROC)(GLenum target, GLuint renderbuffer);
typedef void (APIENTRY *PFNGLDELETERENDERBUFFERSPROC)(GLsizei n, const GLuint *renderbuffers);
typedef void (APIENTRY *PFNGLRENDERBUFFERSTORAGEPROC)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (APIENTRY *PFNGLFRAMEBUFFERRENDERBUFFERPROC)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
typedef GLenum (APIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum target);
typedef void (APIENTRY *PFNGLVIEWPORTPROC)(GLint x, GLint y, GLsizei width, GLsizei height);

typedef GLboolean (APIENTRY *PFNGLUNMAPBUFFERPROC)(GLenum target);

struct OpenGLFunctions {
    PFNGLCREATESHADERPROC glCreateShader = nullptr;
    PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
    PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
    PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
    PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
    PFNGLATTACHSHADERPROC glAttachShader = nullptr;
    PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
    PFNGLDELETESHADERPROC glDeleteShader = nullptr;
    PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
    PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
    PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
    PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
    PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
    PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
    PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
    PFNGLBUFFERDATAPROC glBufferData = nullptr;
    PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
    PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
    PFNGLVERTEXATTRIBIPOINTERPROC glVertexAttribIPointer = nullptr;
    PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
    PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = nullptr;
    PFNGLUNIFORM3FVPROC glUniform3fv = nullptr;
    PFNGLUNIFORM1FPROC glUniform1f = nullptr;
    PFNGLUNIFORM1IPROC glUniform1i = nullptr;
    PFNGLMAPBUFFERPROC glMapBuffer = nullptr;
    PFNGLUNMAPBUFFERPROC glUnmapBuffer = nullptr;
    PFNGLGETINTEGERI_VPROC glGetIntegeri_v = nullptr;
    PFNGLUNIFORM2IPROC glUniform2i = nullptr;
    PFNGLTEXBUFFERPROC glTexBuffer = nullptr; 
    PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer = nullptr;
    PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D = nullptr;
    PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers = nullptr;
    PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer = nullptr;
    PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers = nullptr;
    PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage = nullptr;
    PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus = nullptr;
    PFNGLVIEWPORTPROC glViewport = nullptr;
};

__weak OpenGLFunctions* GLFuncs = nullptr;
__weak SDL_Window* gl_window = nullptr;
__weak SDL_GLContext gl_context = nullptr;
__weak bool opengl_initialized = false;

__weak void loadGLFunctions() {
    if (GLFuncs != nullptr) {
        return; // Already loaded
    }
    
    GLFuncs = new OpenGLFunctions();
    OpenGLFunctions* funcs = GLFuncs;
    
    funcs->glCreateShader = (PFNGLCREATESHADERPROC)SDL_GL_GetProcAddress("glCreateShader");
    funcs->glShaderSource = (PFNGLSHADERSOURCEPROC)SDL_GL_GetProcAddress("glShaderSource");
    funcs->glCompileShader = (PFNGLCOMPILESHADERPROC)SDL_GL_GetProcAddress("glCompileShader");
    funcs->glGetShaderiv = (PFNGLGETSHADERIVPROC)SDL_GL_GetProcAddress("glGetShaderiv");
    funcs->glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
    funcs->glCreateProgram = (PFNGLCREATEPROGRAMPROC)SDL_GL_GetProcAddress("glCreateProgram");
    funcs->glAttachShader = (PFNGLATTACHSHADERPROC)SDL_GL_GetProcAddress("glAttachShader");
    funcs->glLinkProgram = (PFNGLLINKPROGRAMPROC)SDL_GL_GetProcAddress("glLinkProgram");
    funcs->glGetProgramiv = (PFNGLGETPROGRAMIVPROC)SDL_GL_GetProcAddress("glGetProgramiv");
    funcs->glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)SDL_GL_GetProcAddress("glGetProgramInfoLog");
    funcs->glDeleteShader = (PFNGLDELETESHADERPROC)SDL_GL_GetProcAddress("glDeleteShader");
    funcs->glDeleteProgram = (PFNGLDELETEPROGRAMPROC)SDL_GL_GetProcAddress("glDeleteProgram");
    funcs->glUseProgram = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    funcs->glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glGenVertexArrays");
    funcs->glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)SDL_GL_GetProcAddress("glBindVertexArray");
    funcs->glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glDeleteVertexArrays");
    funcs->glGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    funcs->glBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    funcs->glBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    funcs->glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");
    funcs->glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribPointer");
    funcs->glVertexAttribIPointer = (PFNGLVERTEXATTRIBIPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribIPointer");
    funcs->glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    funcs->glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)SDL_GL_GetProcAddress("glGetUniformLocation");
    funcs->glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)SDL_GL_GetProcAddress("glUniformMatrix4fv");
    funcs->glUniform3fv = (PFNGLUNIFORM3FVPROC)SDL_GL_GetProcAddress("glUniform3fv");
    funcs->glUniform1f = (PFNGLUNIFORM1FPROC)SDL_GL_GetProcAddress("glUniform1f");
    funcs->glUniform1i = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
    funcs->glMapBuffer = (PFNGLMAPBUFFERPROC)SDL_GL_GetProcAddress("glMapBuffer");
    funcs->glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)SDL_GL_GetProcAddress("glUnmapBuffer");
    funcs->glGetIntegeri_v = (PFNGLGETINTEGERI_VPROC)SDL_GL_GetProcAddress("glGetIntegeri_v");
    funcs->glUniform2i = (PFNGLUNIFORM2IPROC)SDL_GL_GetProcAddress("glUniform2i");
    funcs->glTexBuffer = (PFNGLTEXBUFFERPROC)SDL_GL_GetProcAddress("glTexBuffer");
    funcs->glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glGenFramebuffers");
    funcs->glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)SDL_GL_GetProcAddress("glBindFramebuffer");
    funcs->glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteFramebuffers");
    funcs->glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)SDL_GL_GetProcAddress("glFramebufferTexture2D");
    funcs->glGenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC)SDL_GL_GetProcAddress("glGenRenderbuffers");
    funcs->glBindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC)SDL_GL_GetProcAddress("glBindRenderbuffer");
    funcs->glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteRenderbuffers");
    funcs->glRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC)SDL_GL_GetProcAddress("glRenderbufferStorage");
    funcs->glFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC)SDL_GL_GetProcAddress("glFramebufferRenderbuffer");
    funcs->glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC)SDL_GL_GetProcAddress("glCheckFramebufferStatus");
    funcs->glViewport = (PFNGLVIEWPORTPROC)SDL_GL_GetProcAddress("glViewport");

    if (funcs->glCreateShader == nullptr || funcs->glShaderSource == nullptr ||
        funcs->glCompileShader == nullptr || funcs->glGetShaderiv == nullptr ||
        funcs->glGetShaderInfoLog == nullptr || funcs->glCreateProgram == nullptr ||
        funcs->glAttachShader == nullptr || funcs->glLinkProgram == nullptr ||
        funcs->glGetProgramiv == nullptr || funcs->glGetProgramInfoLog == nullptr ||
        funcs->glDeleteShader == nullptr || funcs->glDeleteProgram == nullptr ||
        funcs->glUseProgram == nullptr || funcs->glGenVertexArrays == nullptr ||
        funcs->glBindVertexArray == nullptr || funcs->glDeleteVertexArrays == nullptr ||
        funcs->glGenBuffers == nullptr || funcs->glBindBuffer == nullptr ||
        funcs->glBufferData == nullptr || funcs->glDeleteBuffers == nullptr ||
        funcs->glVertexAttribPointer == nullptr || funcs->glVertexAttribIPointer == nullptr ||
        funcs->glEnableVertexAttribArray == nullptr || funcs->glGetUniformLocation == nullptr ||
        funcs->glUniformMatrix4fv == nullptr || funcs->glUniform3fv == nullptr ||
        funcs->glUniform1f == nullptr || funcs->glUniform1i == nullptr ||
        funcs->glMapBuffer == nullptr || funcs->glUnmapBuffer == nullptr ||
        funcs->glGetIntegeri_v == nullptr || funcs->glUniform2i == nullptr || funcs->glTexBuffer == nullptr ||
        funcs->glGenFramebuffers == nullptr || funcs->glBindFramebuffer == nullptr ||
        funcs->glDeleteFramebuffers == nullptr || funcs->glFramebufferTexture2D == nullptr ||
        funcs->glGenRenderbuffers == nullptr || funcs->glBindRenderbuffer == nullptr ||
        funcs->glDeleteRenderbuffers == nullptr || funcs->glRenderbufferStorage == nullptr ||
        funcs->glFramebufferRenderbuffer == nullptr || funcs->glCheckFramebufferStatus == nullptr ||
        funcs->glViewport == nullptr) {
        throw std::runtime_error("Failed to load required OpenGL functions");
    }
    
    std::cout << "Successfully loaded OpenGL functions" << std::endl;
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
            mem = MemoryType::kUnknown_MEM;
        }
    }
    
    device->default_memory_type = mem;
    device->supports_memory_location[mem] = true;

    auto& mem_device = global_device_manager.get_device(mem, 0);
    mem_device.supports_compute_device[ComputeType::kOPENGL] = true;
    mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](Shape<-1> size, size_t bitsize, void* existing_data) {
        GLuint buffer;
        GLFuncs->glGenBuffers(1, &buffer);
        GLFuncs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        GLFuncs->glBufferData(GL_SHADER_STORAGE_BUFFER, size.total_size() * bitsize, existing_data, GL_DYNAMIC_DRAW);
        GLFuncs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return reinterpret_cast<void*>(static_cast<uintptr_t>(buffer));
    };
    mem_device.compute_device_allocators[ComputeType::kOPENGLTEXTURE] = [](Shape<-1> size, size_t bitsize, void* existing_data) {
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        // size.C == 3 for RGB, size.C == 4 for RGBA
        int channels;
        if(size.ndim() == 3){
            channels = size.C * bitsize;
        }else if(size.ndim() == 2){
            channels = bitsize;
        }else{
            std::cerr << "Unsupported number of dimensions for OpenGL texture: " << size.ndim() << std::endl;
            throw std::runtime_error("Unsupported number of dimensions for OpenGL texture");
        }


        if (bitsize == 3) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.A, size.B, 0, GL_RGB, GL_UNSIGNED_BYTE, existing_data);
        } else if (bitsize == 4) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.A, size.B, 0, GL_RGBA, GL_UNSIGNED_BYTE, existing_data);
        } else if (bitsize == 6) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.A, size.B, 0, GL_RGB, GL_FLOAT, existing_data);
        } else if (bitsize == 8) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.A, size.B, 0, GL_RGBA, GL_FLOAT, existing_data);
        } else {
            std::cerr << "Unsupported bitsize for OpenGL texture: " << bitsize << std::endl;
            throw std::runtime_error("Unsupported bitsize for OpenGL texture");
        }
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        // prevent collisions, += 10k
        texture += 0x10000;

        return reinterpret_cast<void*>(static_cast<uintptr_t>(texture));
    };

    mem_device.compute_device_deallocators[ComputeType::kOPENGL] = [](void* ptr) {
        GLuint buffer = static_cast<GLuint>(reinterpret_cast<uintptr_t>(ptr));
        GLFuncs->glDeleteBuffers(1, &buffer);
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