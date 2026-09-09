// opengl_plugin.cpp — OpenGL backend plugin
//
// Built with:  g++-14 -std=c++20 -fPIC -shared \
//                  -I.. -I../../tensor \
//                  -o plugins/opengl/libopengl_plugin.so opengl_plugin.cpp \
//                  -lGL -lGLEW -lSDL3
//
// Priority 60 — loaded last; depends on a GL context existing.
//
// This plugin uses deferred init: plugin_register() does nothing except
// announce the plugin.  The actual GL context creation / device registration
// happens in plugin_init(), which is called by the display layer via
// dm->init_plugin("opengl") after a GL context is available.
//
// plugin_init() will reuse the currently-current GL context if one exists
// (i.e. BasicDisplay has already created its window + context).  If no
// context is current, it creates a hidden 1x1 SDL window as a fallback.

#include "plugin.hpp"

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

static SDL_Window* gl_window = nullptr;
static SDL_GLContext gl_context = nullptr;
static bool opengl_initialized = false;
static bool gl_functions_loaded = false;
static bool owns_gl_context = false;  // true if we created the context ourselves

static bool loadGLFunctions() {
    if (gl_functions_loaded) return true;

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "[opengl] Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return false;
    }
    gl_functions_loaded = true;
    std::cout << "[opengl] OpenGL functions loaded successfully" << std::endl;
    return true;
}

#define GL_CHECK(call)                                                        \
    do {                                                                      \
        call;                                                                 \
        GLenum err = glGetError();                                            \
        if (err != GL_NO_ERROR) {                                             \
            std::cerr << "[opengl] OpenGL error at " << __FILE__ << ":"       \
                      << __LINE__ << " - 0x" << std::hex << err               \
                      << std::dec << std::endl;                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
//  OpenGL ComputeDeviceBase
// ---------------------------------------------------------------------------

static ComputeDeviceBase* create_opengl_compute_device(int device_id) {
    std::cout << "[opengl] Creating OpenGL compute device " << device_id << std::endl;
    if (device_id != 0) {
        std::cerr << "[opengl] Invalid device_id: " << device_id << " (only 0 supported)" << std::endl;
        return nullptr;
    }

    ComputeDeviceBase* device = new ComputeDeviceBase();

    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);

    std::cout << "[opengl] Vendor: " << (vendor ? (const char*)vendor : "Unknown") << std::endl;
    std::cout << "[opengl] Renderer: " << (renderer ? (const char*)renderer : "Unknown") << std::endl;
    std::cout << "[opengl] Version: " << (version ? (const char*)version : "Unknown") << std::endl;

    MemoryType mem = MemoryType::kUnknown_MEM;
    if (vendor) {
        const char* v = (const char*)vendor;
        if (strstr(v, "NVIDIA") != nullptr) {
            mem = MemoryType::kCUDA_VRAM;
        } else if (strstr(v, "AMD") != nullptr || strstr(v, "ATI") != nullptr) {
            mem = MemoryType::kHIP_VRAM;
        } else if (strstr(v, "Intel") != nullptr) {
            mem = MemoryType::kDDR;
        } else {
            mem = MemoryType::kDDR;
        }
    }

    device->default_memory_type = mem;
    device->supports_memory_location[mem] = true;

    auto& mem_device = global_device_manager.get_device(mem, 0);
    mem_device.supports_compute_device[ComputeType::kOPENGL] = true;

    if (mem == kDDR) {
        mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](AllocationMetadata metadata, void* existing_data) {
            GLuint buffer;
            glGenBuffers(1, &buffer);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glBufferStorageEXT(GL_ARRAY_BUFFER, metadata.byte_size, existing_data,
                               GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            return new BaseMemoryAllocation(metadata, reinterpret_cast<void*>(static_cast<uintptr_t>(buffer)));
        };

        mem_device.compute_type_converters[{ComputeType::kOPENGL, ComputeType::kCPU}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
            glBindBuffer(GL_ARRAY_BUFFER, (GLuint)(size_t)ptr);
            void* ptra = glMapBufferRange(GL_ARRAY_BUFFER, 0, metadata.byte_size,
                                          GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            return ptra;
        };

        mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [](void* ptr, BaseMemoryAllocation* original) {
            glBindBuffer(GL_ARRAY_BUFFER, (GLuint)(size_t)original->data);
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        };
    } else {
        mem_device.compute_device_allocators[ComputeType::kOPENGL] = [](AllocationMetadata meta, void* existing_data) {
            GLuint buffer;
            glGenBuffers(1, &buffer);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, meta.byte_size, existing_data, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            return new BaseMemoryAllocation(meta, reinterpret_cast<void*>(static_cast<uintptr_t>(buffer)));
        };
    }

    mem_device.compute_type_converters[{ComputeType::kOPENGLTEXTURE, ComputeType::kCPU}] = [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
        return (void*)0;
    };

    // Texture allocator
    mem_device.compute_device_allocators[ComputeType::kOPENGLTEXTURE] = [](AllocationMetadata metadata, void* existing_data) {
        GLuint texture;
        if (metadata.format != 0) {
            if (metadata.format == GL_DEPTH_COMPONENT24) {
                glGenRenderbuffers(1, &texture);
                glBindRenderbuffer(GL_RENDERBUFFER, texture);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, metadata.shape.A, metadata.shape.B);
                glBindRenderbuffer(GL_RENDERBUFFER, 0);
            } else if (metadata.format == GL_DEPTH_COMPONENT32F) {
                glGenRenderbuffers(1, &texture);
                glBindRenderbuffer(GL_RENDERBUFFER, texture);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, metadata.shape.A, metadata.shape.B);
                glBindRenderbuffer(GL_RENDERBUFFER, 0);
            } else {
                throw std::runtime_error("Unsupported metadata format for OpenGL texture");
            }
        } else {
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
                throw std::runtime_error("Unsupported type_size for OpenGL texture");
            }
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
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

    std::cout << "[opengl] device default memory type: " << mem << std::endl;
    return device;
}

// ---------------------------------------------------------------------------
//  Plugin C ABI
// ---------------------------------------------------------------------------

extern "C" const char* plugin_name() {
    return "opengl";
}

extern "C" int plugin_priority() {
    return 60;
}

// Lightweight registration — no GL context yet.  The display layer will
// call plugin_init() once a GL context is available.
extern "C" void plugin_register(DeviceManager* dm) {
    std::cout << "[opengl] registered (deferred init — waiting for GL context)" << std::endl;
}

// Deferred init — called by dm->init_plugin("opengl") after the display
// layer has created a GL context.  If a context is already current we
// reuse it; otherwise we create a hidden fallback window.
extern "C" void plugin_init(DeviceManager* dm) {
    if (opengl_initialized) {
        // Already initialised — re-register the compute device in case dm
        // was reset (shouldn't happen, but be safe).
        dm->register_compute_device(ComputeType::kOPENGL, 0, create_opengl_compute_device(0));
        return;
    }

    // Check if a GL context is already current (e.g. BasicDisplay created one)
    if (SDL_GL_GetCurrentContext() != nullptr) {
        std::cout << "[opengl] Reusing existing GL context from display layer" << std::endl;
        gl_context = SDL_GL_GetCurrentContext();
        gl_window = SDL_GL_GetCurrentWindow();
        owns_gl_context = false;
    } else {
        // No context — create a hidden fallback window
        std::cout << "[opengl] No current GL context, creating hidden fallback window" << std::endl;
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "[opengl] SDL_Init failed: " << SDL_GetError() << std::endl;
            return;
        }

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

        gl_window = SDL_CreateWindow("deviceManager GL", 1, 1, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (!gl_window) {
            std::cerr << "[opengl] SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
            return;
        }

        gl_context = SDL_GL_CreateContext(gl_window);
        if (!gl_context) {
            std::cerr << "[opengl] SDL_GL_CreateContext failed: " << SDL_GetError() << std::endl;
            return;
        }

        SDL_GL_MakeCurrent(gl_window, gl_context);
        owns_gl_context = true;
    }

    if (!loadGLFunctions()) {
        std::cerr << "[opengl] GLEW init failed — OpenGL backend disabled" << std::endl;
        return;
    }
    opengl_initialized = true;

    ComputeDeviceBase* dev = create_opengl_compute_device(0);
    dm->register_compute_device(ComputeType::kOPENGL, 0, dev);
}
