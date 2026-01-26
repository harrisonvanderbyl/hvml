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
#include "vector/float4.hpp"
#include "vector/int2.hpp"

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
        funcs->glGetIntegeri_v == nullptr) {
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
    mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](size_t size) {
        GLuint buffer;
        GLFuncs->glGenBuffers(1, &buffer);
        GLFuncs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        GLFuncs->glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
        GLFuncs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return reinterpret_cast<void*>(static_cast<uintptr_t>(buffer));
    };
   

    

    std::cout << "OpenGL device default memory type: " << mem << std::endl;
    
    // // Get compute capabilities (if available)
    // GLint max_compute_work_group_count[3];
    // GLFuncs->glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &max_compute_work_group_count[0]);
    // device->compute_units = max_compute_work_group_count[0];
    
    // // Register with memory device
    // 
    
    // // Setup OpenGL buffer allocator
    // mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](size_t size) {
    //     GLuint buffer;
    //     GLFuncs->glGenBuffers(1, &buffer);
    //     GLFuncs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    //     GLFuncs->glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
    //     GLFuncs->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    //     return reinterpret_cast<void*>(static_cast<uintptr_t>(buffer));
    // };
    
    // mem_device.compute_device_deallocators[ComputeType::kOPENGL] = [](void* ptr) {
    //     GLuint buffer = static_cast<GLuint>(reinterpret_cast<uintptr_t>(ptr));
    //     GLFuncs->glDeleteBuffers(1, &buffer);
    // };
    
    
    return device;
}

int count_opengl_devices(){
    if (opengl_initialized) {
        return 1;
    }
    
    
    
    return 1;
}

#endif // TENSOR_ENUMS_DEVICE_SUPPORT_OPENGL_DEVICE_HPP