

// X11 headers

// OpenGL headers
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>

// SDL headers
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

// CUDA headers
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Standard headers
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unistd.h>
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

// Global function pointers
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

void loadGLFunctions() {
    glCreateShader = (PFNGLCREATESHADERPROC)SDL_GL_GetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)SDL_GL_GetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)SDL_GL_GetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)SDL_GL_GetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)SDL_GL_GetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)SDL_GL_GetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)SDL_GL_GetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)SDL_GL_GetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)SDL_GL_GetProcAddress("glGetProgramInfoLog");
    glDeleteShader = (PFNGLDELETESHADERPROC)SDL_GL_GetProcAddress("glDeleteShader");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)SDL_GL_GetProcAddress("glDeleteProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)SDL_GL_GetProcAddress("glBindVertexArray");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glDeleteVertexArrays");
    glGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribPointer");
    glVertexAttribIPointer = (PFNGLVERTEXATTRIBIPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribIPointer");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)SDL_GL_GetProcAddress("glGetUniformLocation");
    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)SDL_GL_GetProcAddress("glUniformMatrix4fv");
    glUniform3fv = (PFNGLUNIFORM3FVPROC)SDL_GL_GetProcAddress("glUniform3fv");
    glUniform1f = (PFNGLUNIFORM1FPROC)SDL_GL_GetProcAddress("glUniform1f");
    glUniform1i = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
    glMapBuffer = (PFNGLMAPBUFFERPROC)SDL_GL_GetProcAddress("glMapBuffer");
    glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)SDL_GL_GetProcAddress("glUnmapBuffer");
}

#ifndef VECTOR_DISPLAY_HPP
#define VECTOR_DISPLAY_HPP



template <DeviceType device>
class GLBackend {
public:
    SDL_Window* window = nullptr;
    SDL_GLContext glctx = nullptr;

    GLuint tex = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint program = 0;

    int width = 0;
    int height = 0;

    cudaGraphicsResource* cudaTex = nullptr;
    cudaArray_t cudaArray = nullptr;
    void* cudaDevPtr = nullptr;
    bool resourcesMapped = false;

    void preinit() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error("Failed to initialize SDL: " + std::string(SDL_GetError()));
        }

        // Request OpenGL 3.3 Core profile
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        
        // Enable double buffering
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        
        // Ensure we get an accelerated context
        SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
    }

    void initialize(void* ptr, SDL_Window* win, int w, int h, Visual* visual = nullptr, int depth = 24, Colormap colormap = 0) {
        cudaDevPtr = ptr;
        width = w;
        height = h;
        window = win;

        if (!window) {
            throw std::runtime_error("Failed to create SDL window");
        }

        glctx = SDL_GL_CreateContext(window);
        if (!glctx) {
            std::string error = SDL_GetError();
            throw std::runtime_error("Failed to create OpenGL context: " + error);
        }
        
        // if (SDL_GL_MakeCurrent(window, glctx) != 0) {
        //     // std::string error = SDL_GetError();
        //     // throw std::runtime_error("Failed to make GL context current: " + error);
        // }
        SDL_GL_SetSwapInterval(0);

        

        loadGLFunctions();

        // Verify OpenGL functions loaded
        if (!glGenTextures || !glBindTexture || !glTexImage2D) {
            throw std::runtime_error("Failed to load required OpenGL functions");
        }

        // Create texture
        glGenTextures(1, &tex);
        if (tex == 0) {
            throw std::runtime_error("Failed to generate OpenGL texture");
        }

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        // Check for OpenGL errors
        GLenum glErr = glGetError();
        if (glErr != GL_NO_ERROR) {
            throw std::runtime_error("OpenGL error creating texture: " + std::to_string(glErr));
        }
        
        glBindTexture(GL_TEXTURE_2D, 0);

        // Ensure all OpenGL commands are complete before CUDA registration
        glFinish();

        // list all cudagraphics devices
        // Initialize CUDA device and set GL device
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found: " + std::string(cudaGetErrorString(err)));
        }

        // Find CUDA device that can interop with OpenGL
        int cudaDevice = -1;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (prop.major >= 3) { // Require compute capability 3.0+
                cudaDevice = i;
                break;
            }
        }

        if (cudaDevice == -1) {
            throw std::runtime_error("No suitable CUDA device found for GL interop");
        }

        // Set CUDA device for GL interop
        err = cudaSetDevice(cudaDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA GL device: " + std::string(cudaGetErrorString(err)));
        }

        // CUDA–GL interop registration
        err = cudaGraphicsGLRegisterImage(&cudaTex, tex, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if (err != cudaSuccess) {
            if(err == 999){
                std::cout << "CUDA–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
            }
            std::cout << "CUDA–GL interop registration failed with error code: " << err << std::endl;
            std::string errorMsg = "Failed to register CUDA-GL interop: " + 
                                  std::string(cudaGetErrorString(err));
            glDeleteTextures(1, &tex);
            throw std::runtime_error(errorMsg);
        }

        // Fullscreen quad
        float verts[] = {
            -1, -1, 0, 1,
             1, -1, 1, 1,
             1,  1, 1, 0,
            -1,  1, 0, 0
        };

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void*)(2 * sizeof(float)));

        glBindVertexArray(0);

        program = createProgram(vertexSrc, fragmentSrc);

       
    }

    void present() {
        
        // copy from cudaDevPtr to cudaTex
        //map cudaTex
        auto errMap = cudaGraphicsMapResources(1, &cudaTex);
        if (errMap != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA-GL resources: " + std::string(cudaGetErrorString(errMap)));
        }
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaTex, 0, 0);

        cudaError_t erra = cudaMemcpy2DToArray(
            cudaArray,
            0, 0,
            cudaDevPtr,
            width * sizeof(uint32_t),
            width * sizeof(uint32_t),
            height,
            cudaMemcpyDeviceToDevice
        );
        if (erra != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to CUDA array: " + std::string(cudaGetErrorString(erra)));
        }

        

        auto err = cudaGraphicsUnmapResources(1, &cudaTex);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to unmap CUDA-GL resources: " + std::string(cudaGetErrorString(err)));
        }
        resourcesMapped = false;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindVertexArray(vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(program, "tex"), 0);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        SDL_GL_SwapWindow(window);

    }

    void resize(int w, int h) {
        width = w;
        height = h;

        // // Unmap resources BEFORE unregistering
        // if (resourcesMapped && cudaTex) {
        //     cudaGraphicsUnmapResources(1, &cudaTex);
        //     resourcesMapped = false;
        // }

        // // Unregister old resource
        // if (cudaTex) {
        //     cudaGraphicsUnregisterResource(cudaTex);
        //     cudaTex = nullptr;
        // }

        // // Recreate texture
        // glBindTexture(GL_TEXTURE_2D, tex);
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
        //              GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        // glBindTexture(GL_TEXTURE_2D, 0);

        // // Re-register
        // cudaGraphicsGLRegisterImage(&cudaTex, tex, GL_TEXTURE_2D,
        //                             cudaGraphicsRegisterFlagsWriteDiscard);

        // // Re-map
        // cudaGraphicsMapResources(1, &cudaTex);
        // cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaTex, 0, 0);
        resourcesMapped = true;
    }

    ~GLBackend() {
        if (resourcesMapped && cudaTex) {
            cudaGraphicsUnmapResources(1, &cudaTex);
        }
        if (cudaTex) cudaGraphicsUnregisterResource(cudaTex);
        if (tex) glDeleteTextures(1, &tex);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (program) glDeleteProgram(program);
        if (glctx) SDL_GL_DestroyContext(glctx);
    }

private:
    static constexpr const char* vertexSrc = R"(
        #version 330 core
        layout(location = 0) in vec2 pos;
        layout(location = 1) in vec2 uv;
        out vec2 vUV;
        void main() {
            vUV = uv;
            gl_Position = vec4(pos, 0, 1);
        }
    )";

    static constexpr const char* fragmentSrc = R"(
        #version 330 core
        in vec2 vUV;
        out vec4 color;
        uniform sampler2D tex;
        void main() {
            color = texture(tex, vUV);
        }
    )";

    GLuint createProgram(const char* vs, const char* fs) {
        auto compile = [](GLenum type, const char* src) {
            GLuint s = glCreateShader(type);
            glShaderSource(s, 1, &src, nullptr);
            glCompileShader(s);
            
            GLint success;
            glGetShaderiv(s, GL_COMPILE_STATUS, &success);
            if (!success) {
                char infoLog[512];
                glGetShaderInfoLog(s, 512, nullptr, infoLog);
                std::cerr << "Shader compilation failed: " << infoLog << std::endl;
            }
            return s;
        };

        GLuint v = compile(GL_VERTEX_SHADER, vs);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        GLuint p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        
        GLint success;
        glGetProgramiv(p, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(p, 512, nullptr, infoLog);
            std::cerr << "Program linking failed: " << infoLog << std::endl;
        }
        
        glDeleteShader(v);
        glDeleteShader(f);
        return p;
    }

    DeviceType getDeviceType() const {
        return DeviceType::kCUDA;
    }
};





class CurrentScreenInputInfo {
private:
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    bool is_fullscreen = false;
    int mouse_x = 0;
    int mouse_y = 0;
    int mouse_move_x = 0;
    int mouse_move_y = 0;
    bool mouse_left_button = false;
    bool mouse_left_button_released = false;
    bool mouse_right_button = false;
    bool mouse_middle_button = false;
    bool mouse_wheel_up = false;
    bool mouse_wheel_down = false;
    bool mouse_wheel_left = false;
    bool mouse_wheel_right = false;
    float32x4 selectedarea = float32x4(0, 0, 0, 0);
    float32x2 lastClicked = float32x2(0, 0);
    
    std::map<int, bool> raw_key_states;
    int accumulated_mouse_x = 0;
    int accumulated_mouse_y = 0;
    
public:
    bool just_selected_area = false;
    
    void updateMousePositionAbsolute(int new_x, int new_y) {
        mouse_move_x = (new_x-x) - mouse_x;
        mouse_move_y = (new_y-y) - mouse_y;
        mouse_x = (new_x-x);
        mouse_y = (new_y-y);
        accumulated_mouse_x = (new_x-x);
        accumulated_mouse_y = (new_y-y);
    }

    float32x4 getSelectedArea() const {
        return selectedarea;
    }

    float32x2 getGlobalMousePosition() const {
        // sdl3 get global mouse position
        float gx, gy;
        SDL_GetGlobalMouseState(&gx, &gy);
        return float32x2(gx, gy);
    }
    
    void updateMouseButtonState(int button_code, bool pressed) {
        switch (button_code) {
            case SDL_BUTTON_LEFT:
                std::cout << "Mouse left button " << (pressed ? "pressed" : "released") << std::endl;
                mouse_left_button = pressed;
                if(pressed) {
                    lastClicked = getGlobalMousePosition();
                }else{
                    float32x2 mx = getGlobalMousePosition();
                    if (sqrt(pow(mx.x - lastClicked.x, 2) + pow(mx.y - lastClicked.y, 2)) > 5.0f) {
                        selectedarea = float32x4(lastClicked.x, lastClicked.y, mx.x - lastClicked.x, mx.y - lastClicked.y);
                    }
                    std::cout << "Selected area: " << selectedarea.x << ", " << selectedarea.y << ", " 
                              << selectedarea.z << ", " << selectedarea.w << std::endl;
                    just_selected_area = true;
                }
                break;
            case SDL_BUTTON_RIGHT:
                mouse_right_button = pressed;
                break;
            case SDL_BUTTON_MIDDLE:
                mouse_middle_button = pressed;
                break;
        }
    }
    
    void setScreenSize(int new_x, int new_y, int new_width, int new_height) {
        x = new_x;
        y = new_y;
        width = new_width;
        height = new_height;
    }
    
    void setFullscreen(bool fullscreen) {
        is_fullscreen = fullscreen;
    }
    
    int getX() const { return x; }
    int getY() const { return y; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    bool isFullscreen() const { return is_fullscreen; }
    int getMouseX() const { return mouse_x; }
    int getMouseY() const { return mouse_y; }
    float32x2 getMousePosition() const { return float32x2(mouse_x, mouse_y); }
    float32x2 getMouseMove() const { return float32x2(mouse_move_x, mouse_move_y); }
    float32x4 getScreenSize() const { return float32x4(x, y, width, height); }
    int getMouseMoveX() const { return mouse_move_x; }
    int getMouseMoveY() const { return mouse_move_y; }
    bool isMouseLeftButtonPressed() const { return mouse_left_button; }
    bool isMouseRightButtonPressed() const { return mouse_right_button; }
    bool isMouseMiddleButtonPressed() const { return mouse_middle_button; }
    bool isMouseWheelUp() const { return mouse_wheel_up; }
    bool isMouseWheelDown() const { return mouse_wheel_down; }
    bool isMouseWheelLeft() const { return mouse_wheel_left; }
    bool isMouseWheelRight() const { return mouse_wheel_right; }
   

    void clearWheelStates() {
        mouse_wheel_up = false;
        mouse_wheel_down = false;
        mouse_wheel_left = false;
        mouse_wheel_right = false;
    }

    // global coordinates for mouse position
    
    
    void clear_mouse_states() {
        just_selected_area = false;
    }
};

enum WindowProperties {
    WP_BORDERLESS = 1 << 0,
    WP_ALPHA_ENABLED = 1 << 1,
    WP_FULLSCREEN = 1 << 2,
    WP_CLICKTHROUGH = 1 << 3,
    WP_ON_TOP = 1 << 4
};

struct WindowPropertiesFlags {
    bool borderless = false;
    bool alpha_enabled = false;
    bool fullscreen = false;
    bool clickthrough = true;
    bool on_top = false;

    WindowPropertiesFlags(WindowProperties properties) {
        borderless = properties & WP_BORDERLESS;
        alpha_enabled = properties & WP_ALPHA_ENABLED;
        fullscreen = properties & WP_FULLSCREEN;
        clickthrough = properties & WP_CLICKTHROUGH;
        on_top = properties & WP_ON_TOP;
    }

    WindowPropertiesFlags(int flags) {
        borderless = flags & WP_BORDERLESS;
        alpha_enabled = flags & WP_ALPHA_ENABLED;
        fullscreen = flags & WP_FULLSCREEN;
        clickthrough = flags & WP_CLICKTHROUGH;
        on_top = flags & WP_ON_TOP;
    }

    operator WindowProperties() const {
        WindowProperties props = (WindowProperties)0;
        if (borderless) props = (WindowProperties)(props | WP_BORDERLESS);
        if (alpha_enabled) props = (WindowProperties)(props | WP_ALPHA_ENABLED);
        if (fullscreen) props = (WindowProperties)(props | WP_FULLSCREEN);
        if (clickthrough) props = (WindowProperties)(props | WP_CLICKTHROUGH);
        if (on_top) props = (WindowProperties)(props | WP_ON_TOP);
        return props;
    }
};

template <DeviceType Device = DeviceType::kCPU>
class VectorDisplay : public Tensor<uint84, 2> {
public:
    Display* display = nullptr;
    SDL_Window* window;
    Window* root_window = nullptr;
    // GC gc;
    // XImage* ximage = nullptr;
    Visual* visual = nullptr;
    Colormap colormap;
    int screen;
    int depth;
    bool borderless = false;
    bool alpha_enabled = false;
    bool is_fullscreen = false;
    bool clickthrough = true;
    bool on_top = false;
    CurrentScreenInputInfo current_screen_input_info;
    std::vector<std::function<void(CurrentScreenInputInfo&)>> display_loop_functions;
    
    GLBackend<Device> backend = GLBackend<Device>();
    
    
    VectorDisplay(Shape<2> shape = 0, WindowPropertiesFlags properties = (WindowProperties)0)
        : Tensor<uint84, 2>(shape, Device), borderless(properties.borderless), alpha_enabled(properties.alpha_enabled), is_fullscreen(properties.fullscreen),  clickthrough(properties.clickthrough), on_top(properties.on_top) {

        backend.preinit();

        std::cout << "finished preinit" << std::endl;
      
        window = SDL_CreateWindow(
            "CUDA → OpenGL",
            shape[1], shape[0],
            SDL_WINDOW_OPENGL | (is_fullscreen ? SDL_WINDOW_FULLSCREEN : 0) | (borderless ? SDL_WINDOW_BORDERLESS : 0) | (alpha_enabled ? SDL_WINDOW_TRANSPARENT : 0) | (on_top ? SDL_WINDOW_ALWAYS_ON_TOP : 0)
        );

        if (!window) {
            throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));
        }

        std::cout << "created window" << std::endl;

        display = (Display *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);

        screen = SDL_GetDisplayForWindow(window);
        
        // show window
        SDL_ShowWindow(window);

        backend.initialize(data, window, shape[1], shape[0], visual, depth, colormap);

       

        
        
    }
    
private:

    void updateMousePositionFromRoot() {
    }

    
    
public:
    void setWindowBorderless() {
        // Remove window decorations
        Atom motifHints = XInternAtom(display, "_MOTIF_WM_HINTS", False);
        if (motifHints != None) {
            struct {
                unsigned long flags;
                unsigned long functions;
                unsigned long decorations;
                long input_mode;
                unsigned long status;
            } hints = {0};
            
            hints.flags = 2; // MWM_HINTS_DECORATIONS
            hints.decorations = 0; // No decorations
            
            // XChangeProperty(display, window, motifHints, motifHints, 32,
            //                PropModeReplace, (unsigned char*)&hints, 5);
        }
    }
    
    void enableAlphaBlending() {
        // Set window opacity property for compositor
        Atom netWmWindowOpacity = XInternAtom(display, "_NET_WM_WINDOW_OPACITY", False);
        if (netWmWindowOpacity != None) {
            unsigned long opacity = 0xFFFFFFFF; // Fully opaque by default
            // XChangeProperty(display, window, netWmWindowOpacity, XA_CARDINAL, 32,
            //                PropModeReplace, (unsigned char*)&opacity, 1);
        }
        
        // Enable compositing for this window
        // XCompositeRedirectWindow(display, window, CompositeRedirectAutomatic);
    }
    
    void setWindowOpacity(float opacity) {
        if (!alpha_enabled) return;
        
        Atom netWmWindowOpacity = XInternAtom(display, "_NET_WM_WINDOW_OPACITY", False);
        if (netWmWindowOpacity != None) {
            unsigned long opacityValue = (unsigned long)(opacity * 0xFFFFFFFF);
            // XChangeProperty(display, window, netWmWindowOpacity, XA_CARDINAL, 32,
            //                PropModeReplace, (unsigned char*)&opacityValue, 1);
            // XFlush(display);
        }
    }
    
    // void createXImage() {
        
    // }
    
    void updateDisplay() {
    
        
        if (shape[0] > 0 && shape[1] > 0) {
            backend.present();
        }

        
    }

    void resizeDisplay() {
        
    }
    
    void displayLoop() {
        bool quit = false;

        while (!quit) {
            resizeDisplay();
            
            
            // Call display loop functions
            for (const auto& callback : display_loop_functions) {
                callback(current_screen_input_info);
            }

            // SDL event handling
            SDL_Event sdl_event;
            current_screen_input_info.clear_mouse_states();
            while (SDL_PollEvent(&sdl_event)) {
                if (sdl_event.type == SDL_EVENT_QUIT) {
                    quit = true;
                } 

                // mousedown
                else if (sdl_event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    current_screen_input_info.updateMouseButtonState(sdl_event.button.button, 1);
                }
                // mouseup
                else if (sdl_event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                    current_screen_input_info.updateMouseButtonState(sdl_event.button.button, 0);
                }
                // mousemotion
                else if (sdl_event.type == SDL_EVENT_MOUSE_MOTION) {
                    current_screen_input_info.updateMousePositionAbsolute(sdl_event.motion.x, sdl_event.motion.y);
                }
            }

            
            updateDisplay();
            updateMousePositionFromRoot();
        }
        
    }

    
    // Clear display with specified color
    void clear(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 0) {
        uint84 color;
        if (alpha_enabled && depth == 32) {
            color = (a << 24) | (r << 16) | (g << 8) | b;
        } else {
            color = (r << 16) | (g << 8) | b;
        }
        
        for (int y = 0; y < shape[0]; y++) {
            for (int x = 0; x < shape[1]; x++) {
                (*this)[y][x] = color;
            }
        }
    }
    
    ~VectorDisplay() {
      
        if (window) {
            // XDestroyWindow(display, window);
        }
        if (display) {
        }
    }
    
    void add_on_update(std::function<void(CurrentScreenInputInfo&)> func) {
        display_loop_functions.push_back(func);
    }
    
    
    // Utility functions for window management
    void moveWindow(int x, int y) {
        // XMoveWindow(display, window, x, y);
        // XFlush(display);
    }
    
    void resizeWindow(int width, int height) {
        // XResizeWindow(display, window, width, height);
        // XFlush(display);
    }
    
};

#endif
