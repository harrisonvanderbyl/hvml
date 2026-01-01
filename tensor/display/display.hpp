
#include "display/display_manager.hpp"
#include "display/selected_text.hpp"
// include opengl headers

#include <GL/gl.h>
#include <GL/glext.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_syswm.h>
#include <cuda_gl_interop.h>
#include <SDL2/SDL_opengl_glext.h>
// OpenGL function pointers for modern OpenGL functions

//glXSwapBuffers
#include <GL/glx.h>

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

// Framebuffer function pointers
typedef void (APIENTRY *PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint *framebuffers);
typedef void (APIENTRY *PFNGLBINDFRAMEBUFFERPROC)(GLenum target, GLuint framebuffer);
typedef void (APIENTRY *PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
typedef GLenum (APIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum target);
typedef void (APIENTRY *PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint *framebuffers);
typedef void (APIENTRY *PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
typedef void (APIENTRY *PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
typedef void (APIENTRY *PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
typedef void (APIENTRY *PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
typedef void (APIENTRY *PFNGLDELETETEXTURESPROC)(GLsizei n, const GLuint *textures);
typedef void (APIENTRY *PFNGLREADPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels);
typedef void (APIENTRY *PFNGLGENRENDERBUFFERSPROC)(GLsizei n, GLuint *renderbuffers);
typedef void (APIENTRY *PFNGLBINDRENDERBUFFERPROC)(GLenum target, GLuint renderbuffer);
typedef void (APIENTRY *PFNGLRENDERBUFFERSTORAGEPROC)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (APIENTRY *PFNGLFRAMEBUFFERRENDERBUFFERPROC)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
typedef void (APIENTRY *PFNGLDELETERENDERBUFFERSPROC)(GLsizei n, const GLuint *renderbuffers);
typedef void (APIENTRY *PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
typedef void (APIENTRY *PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
typedef void (APIENTRY *PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
typedef void (APIENTRY *PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
typedef void (APIENTRY *PFNGLDELETETEXTURESPROC)(GLsizei n, const GLuint *textures);
typedef void (APIENTRY *PFNGLREADPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint value);
typedef void (APIENTRY *PFNGLACTIVETEXTUREPROC)(GLenum texture);


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


#ifndef VECTOR_DISPLAY_HPP
#define VECTOR_DISPLAY_HPP

class PresentBackend {
public:
    virtual ~PresentBackend() = default;
    virtual void* initialize(Display* dpy, Window win, int width, int height, Visual* visual = nullptr, int depth = 24, Colormap colormap = 0) = 0;
    virtual DeviceType getDeviceType() const = 0;
    virtual void resize(int w, int h) = 0;
    virtual void present() = 0;
    virtual void preinit() = 0;
};

class XImageBackend : public PresentBackend {
public:
    XImage* ximage;
    GC gc;
    Display* dpy;
    Window win;
    void* pixels;
    // visual
    Visual* visual;
    int depth = 24;

    void preinit() override {
        // No pre-initialization needed for XImageBackend
    }

    void* initialize(Display* dpy, Window win, int width, int height, Visual* visual, int depth, Colormap colormap) override {
        this->dpy = dpy;
        this->win = win;
        gc = XCreateGC(dpy, win, 0, nullptr);
        this->visual = visual;
        this->depth = depth;
        int bitmap_pad = (depth > 16) ? 32 : (depth > 8) ? 16 : 8;


        pixels = DeviceAllocator<DeviceType::kCPU>::allocate(width * height * sizeof(uint32_t));

        // set all pixels to black
        // memset(pixels, 0, width * height * sizeof(uint32_t));
        
        ximage = XCreateImage(dpy, visual, depth, ZPixmap, 0,
                             (char*)pixels, width, height,
                             bitmap_pad, width * sizeof(uint84));
        
        
        if (!ximage) {
            throw std::runtime_error("Failed to create XImage");
        }
        
        
        ximage->data = (char*)pixels;
        ximage->byte_order = LSBFirst;
        ximage->bitmap_bit_order = LSBFirst;
        return pixels;
    }

    void present() override {
        XPutImage(dpy, win, gc, ximage, 0, 0, 0, 0,
                  ximage->width, ximage->height);
    }

    void resize(int w, int h) override {
        if (ximage) {
            ximage->data = nullptr;
            XDestroyImage(ximage);
        }
        ximage = XCreateImage(dpy, DefaultVisual(dpy, DefaultScreen(dpy)),
                              DefaultDepth(dpy, DefaultScreen(dpy)),
                              ZPixmap, 0, nullptr, w, h, 32, 0);;
        // pixels = new uint32_t[w * h];
        // ximage->data = //(char*)pixels;
        ximage->byte_order = LSBFirst;
        ximage->bitmap_bit_order = LSBFirst;
    }

    DeviceType getDeviceType() const override {
        return DeviceType::kCPU;
    }
};


class GLBackend : public PresentBackend {
public:
    Display* dpy = nullptr;
    Window win = 0;
    GLXContext ctx = nullptr;

    GLuint tex = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint program = 0;
    GLuint pbo = 0;  // Pixel Buffer Object for efficient transfers

    void* cudaDevPtr = nullptr;

    int width = 0;
    int height = 0;

    void preinit() override {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            printf("SDL_Init failed: %s\n", SDL_GetError());
        }

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

        if (SDL_GL_LoadLibrary(nullptr) < 0) {
            std::cerr << "SDL_GL_LoadLibrary failed: " << SDL_GetError() << std::endl;
        }
        
        // Load OpenGL function pointers using SDL
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
        
        // Check if all functions were loaded successfully
        if (!glCreateShader || !glShaderSource || !glCompileShader || !glGetShaderiv ||
            !glGetShaderInfoLog || !glCreateProgram || !glAttachShader || !glLinkProgram ||
            !glGetProgramiv || !glGetProgramInfoLog || !glDeleteShader || !glDeleteProgram ||
            !glUseProgram || !glGenVertexArrays || !glBindVertexArray || !glDeleteVertexArrays ||
            !glGenBuffers || !glBindBuffer || !glBufferData || !glDeleteBuffers ||
            !glVertexAttribPointer || !glEnableVertexAttribArray || !glGetUniformLocation ||
            !glUniformMatrix4fv || !glUniform3fv || !glUniform1f || !glVertexAttribIPointer ||
            !glUniform1i) {
            std::cerr << "Failed to load one or more OpenGL functions!" << std::endl;
        }
    }

    void* initialize(Display* dpy, Window win, int width, int height,
                     Visual* visual = nullptr, int depth = 24, Colormap colormap = 0) override {
        this->dpy = dpy;
        this->win = win;
        this->width = width;
        this->height = height;

        // Create GLX context
        static int attribs[] = {
            GLX_RGBA,
            GLX_DOUBLEBUFFER,
            GLX_DEPTH_SIZE, 24,
            None
        };

        XVisualInfo* vi = glXChooseVisual(dpy, DefaultScreen(dpy), attribs);
        if (!vi) {
            throw std::runtime_error("Failed to choose GLX visual");
        }

        ctx = glXCreateContext(dpy, vi, nullptr, GL_TRUE);
        if (!ctx) {
            XFree(vi);
            throw std::runtime_error("Failed to create GLX context");
        }

        glXMakeCurrent(dpy, win, ctx);
        XFree(vi);

        // Create texture
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindTexture(GL_TEXTURE_2D, 0);

        // Create PBO for efficient CPU/GPU transfers (optional but recommended)
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Allocate CUDA device memory
        cudaError_t err = cudaMalloc(&cudaDevPtr, width * height * 4);
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to allocate CUDA memory");
        }

        // Create fullscreen quad
        float verts[] = {
            // pos      // uv
            -1, -1,    0, 1,
             1, -1,    1, 1,
             1,  1,    1, 0,
            -1,  1,    0, 0
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

        // Simple shader
        const char* vs = R"(
            #version 330 core
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 uv;
            out vec2 vUV;
            void main() {
                vUV = uv;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
        )";

        const char* fs = R"(
            #version 330 core
            in vec2 vUV;
            out vec4 color;
            uniform sampler2D tex;
            void main() {
                color = texture(tex, vUV);
            }
        )";

        program = createProgram(vs, fs);

        return cudaDevPtr;
    }

    void present() override {
        glXMakeCurrent(dpy, win, ctx);

        // Copy CUDA data to texture
        // Option 1: Direct copy (slower)
        void* tempBuffer = malloc(width * height * 4);
        cudaMemcpy(tempBuffer, cudaDevPtr, width * height * 4, cudaMemcpyDeviceToHost);
        
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, tempBuffer);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        free(tempBuffer);

        // Option 2: Using PBO (faster, uncomment to use instead of Option 1)
        /*
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        void* pboMem = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        if (pboMem) {
            cudaMemcpy(pboMem, cudaDevPtr, width * height * 4, cudaMemcpyDeviceToHost);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
            
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        */

        // Render
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindVertexArray(vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(program, "tex"), 0);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        glBindVertexArray(0);
        glUseProgram(0);

        glXSwapBuffers(dpy, win);
    }

    void resize(int w, int h) override {
        width = w;
        height = h;

        // Reallocate CUDA memory
        if (cudaDevPtr) {
            cudaFree(cudaDevPtr);
        }
        cudaMalloc(&cudaDevPtr, w * h * 4);

        // Resize texture
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Resize PBO
        if (pbo) {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, nullptr, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }
    }

    DeviceType getDeviceType() const override {
        return DeviceType::kCUDA;
    }

    ~GLBackend() {
        if (cudaDevPtr) {
            cudaFree(cudaDevPtr);
        }
        if (pbo) glDeleteBuffers(1, &pbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (tex) glDeleteTextures(1, &tex);
        if (program) glDeleteProgram(program);
        if (ctx) {
            glXMakeCurrent(dpy, None, nullptr);
            glXDestroyContext(dpy, ctx);
        }
    }

private:
    GLuint compileShader(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        
        GLint success;
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "Shader compilation failed: " << log << std::endl;
        }
        
        return s;
    }

    GLuint createProgram(const char* vs, const char* fs) {
        GLuint v = compileShader(GL_VERTEX_SHADER, vs);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
        GLuint p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        
        GLint success;
        glGetProgramiv(p, GL_LINK_STATUS, &success);
        if (!success) {
            char log[512];
            glGetProgramInfoLog(p, 512, nullptr, log);
            std::cerr << "Program linking failed: " << log << std::endl;
        }
        
        glDeleteShader(v);
        glDeleteShader(f);
        return p;
    }
};

template <DeviceType T>
struct ToBackendSelector {
    using type = std::conditional_t<
        T == DeviceType::kCPU,
        XImageBackend,
        GLBackend
    >;
};


class CurrentScreenInputInfo {
private:
    MultiDisplayManager* display_manager = nullptr;
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
    bool mouse_right_button = false;
    bool mouse_middle_button = false;
    bool mouse_wheel_up = false;
    bool mouse_wheel_down = false;
    bool mouse_wheel_left = false;
    bool mouse_wheel_right = false;
    bool key_pressed[KEY_MAX] = {false};
    float32x4 selectedarea = float32x4(0, 0, 0, 0);
    float32x2 lastClicked = float32x2(0, 0);
    
    // Raw input tracking
    std::map<int, bool> raw_key_states;
    int accumulated_mouse_x = 0;
    int accumulated_mouse_y = 0;
    
public:
    SelectedTextReader* selected_text_reader = nullptr;
    
    
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
    
    void updateMouseButtonState(int button_code, bool pressed) {
        switch (button_code) {
            case BTN_LEFT:
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
                }
                break;
            case BTN_RIGHT:
                mouse_right_button = pressed;
                break;
            case BTN_MIDDLE:
                mouse_middle_button = pressed;
                break;
        }
    }
    
    void updateMouseWheel(int axis, int value) {
        if (axis == REL_WHEEL) {
            mouse_wheel_up = (value > 0);
            mouse_wheel_down = (value < 0);
        } else if (axis == REL_HWHEEL) {
            mouse_wheel_left = (value < 0);
            mouse_wheel_right = (value > 0);
        }
    }
    
    void updateKeyState(int keycode, bool pressed) {
        if (keycode >= 0 && keycode < KEY_MAX) {
            key_pressed[keycode] = pressed;
            raw_key_states[keycode] = pressed;
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
    
    // Getters
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
    bool isKeyPressed(int keycode) const {
        if (keycode >= 0 && keycode < KEY_MAX) {
            return key_pressed[keycode];
        }
        return false;
    }
    
    // Get current display info
    const DisplayInfo* getCurrentDisplayInfo() const {
        if (display_manager) {
            return display_manager->getCurrentDisplay();
        }
        return nullptr;
    }
    
    // Get all displays
    std::vector<DisplayInfo> getAllDisplays() const {
        if (display_manager) {
            return display_manager->getDisplays();
        }
        return {};
    }
    
    int getCurrentDisplayIndex() const {
        if (display_manager) {
            return display_manager->getCurrentDisplayIndex();
        }
        return 0;
    }

    void setDisplayManager(MultiDisplayManager* manager) {
        display_manager = manager;
    }
    MultiDisplayManager* getDisplayManager() const {
        return display_manager;
    }

    void clearWheelStates() {
        mouse_wheel_up = false;
        mouse_wheel_down = false;
        mouse_wheel_left = false;
        mouse_wheel_right = false;
    }

    // global coordinates for mouse position
    float32x2 getGlobalMousePosition() const {
        if (display_manager) {
            const DisplayInfo* display = display_manager->getCurrentDisplay();
            if (display) {
                auto local_pos = display->localToGlobal(mouse_x, mouse_y);
                return float32x2(local_pos.first, local_pos.second);
            }
        }
        return float32x2(mouse_x, mouse_y);
    }
    
};

enum WindowProperties {
    WP_BORDERLESS = 1 << 0,
    WP_ALPHA_ENABLED = 1 << 1,
    WP_FULLSCREEN = 1 << 2,
    WP_CLICKTHROUGH = 1 << 3
};

struct WindowPropertiesFlags {
    bool borderless = false;
    bool alpha_enabled = false;
    bool fullscreen = false;
    bool clickthrough = true;

    WindowPropertiesFlags(WindowProperties properties) {
        borderless = properties & WP_BORDERLESS;
        alpha_enabled = properties & WP_ALPHA_ENABLED;
        fullscreen = properties & WP_FULLSCREEN;
        clickthrough = properties & WP_CLICKTHROUGH;
    }

    operator WindowProperties() const {
        WindowProperties props = (WindowProperties)0;
        if (borderless) props = (WindowProperties)(props | WP_BORDERLESS);
        if (alpha_enabled) props = (WindowProperties)(props | WP_ALPHA_ENABLED);
        if (fullscreen) props = (WindowProperties)(props | WP_FULLSCREEN);
        if (clickthrough) props = (WindowProperties)(props | WP_CLICKTHROUGH);
        return props;
    }
};

template <DeviceType Device = DeviceType::kCPU, typename Backend = typename ToBackendSelector<Device>::type>
class VectorDisplay : public Tensor<uint84, 2> {
public:
    Display* display = nullptr;
    Window window;
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
    CurrentScreenInputInfo current_screen_input_info;
    std::vector<std::function<void(CurrentScreenInputInfo&)>> display_loop_functions;
    
    // Direct input reading
    DirectInputReader input_reader;

    
    // Multi-display support
    MultiDisplayManager display_manager;
    Backend backend = Backend();
    
    
    VectorDisplay(Shape<2> shape = 0, WindowPropertiesFlags properties = (WindowProperties)0)
        : Tensor<uint84, 2>(shape, nullptr, Device), borderless(properties.borderless), alpha_enabled(properties.alpha_enabled), is_fullscreen(properties.fullscreen), display_manager(nullptr), clickthrough(properties.clickthrough) {
        
        // Initialize the display with a black background
        // for (int y = 0; y < shape[0]; y++) {
        //     for (int x = 0; x < shape[1]; x++) {
        //         (*this)[y][x] = 0x00000000; // ARGB format: transparent black
        //     }
        // }
        // Open connection to X server
        backend.preinit();

        display = XOpenDisplay(nullptr);
        if (!display) {
            std::cerr << "Cannot open X display" << std::endl;
            return;
        }

        
        
        // Initialize display manager
        display_manager = MultiDisplayManager(display);
        current_screen_input_info.setDisplayManager(&display_manager);
        display_manager.detectDisplays();
        
        screen = DefaultScreen(display);
        
        // Check for composite extension if alpha is requested
        if (alpha_enabled) {
            int composite_event_base, composite_error_base;
            if (!XCompositeQueryExtension(display, &composite_event_base, &composite_error_base)) {
                std::cerr << "Warning: Composite extension not available, alpha blending may not work" << std::endl;
            }
        }
        
        // Find appropriate visual for alpha support
        if (alpha_enabled) {
            XVisualInfo vinfo;
            if (XMatchVisualInfo(display, screen, 32, TrueColor, &vinfo)) {
                visual = vinfo.visual;
                depth = vinfo.depth;
                colormap = XCreateColormap(display, RootWindow(display, screen), visual, AllocNone);
            } else {
                std::cerr << "Warning: 32-bit visual not found, falling back to default" << std::endl;
                visual = DefaultVisual(display, screen);
                depth = DefaultDepth(display, screen);
                colormap = DefaultColormap(display, screen);
            }
        } else {
            visual = DefaultVisual(display, screen);
            depth = DefaultDepth(display, screen);
            colormap = DefaultColormap(display, screen);
        }
        
        // Set window attributes
        // Replace the window creation and property setting section in your VectorDisplay constructor
// with this corrected version:

// Create window with proper input transparency
        XSetWindowAttributes attrs;
        attrs.colormap = colormap;
        if (borderless)
        attrs.border_pixel = 0;
        attrs.background_pixel = 0;
        unsigned long mask = CWColormap | CWBorderPixel | CWBackPixel | CWOverrideRedirect | CWDontPropagate;

    
        if (clickthrough)
        
        {
            attrs.event_mask = NoEventMask; // No events
        
        
            attrs.override_redirect = True; // Bypass window manager
            attrs.do_not_propagate_mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
            
        } else {
            attrs.event_mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
            attrs.override_redirect = False;
            attrs.do_not_propagate_mask = 0;
            mask |= CWEventMask;
        }
        
        // Create window
        window = XCreateWindow(display, RootWindow(display, screen),
                            0, 0, shape[1], shape[0], 0, depth, InputOutput,
                            visual, mask, &attrs);

        // CRITICAL: Set input region to empty to make window completely click-through
        if (clickthrough) {
                
            XserverRegion empty_region = XFixesCreateRegion(display, nullptr, 0);
            XFixesSetWindowShapeRegion(display, window, ShapeInput, 0, 0, empty_region);
            XFixesDestroyRegion(display, empty_region);

        }
        // Set window properties for overlay behavior
        XStoreName(display, window, "Vector Display");

        // Make window borderless
        if (borderless) {
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
                
                XChangeProperty(display, window, motifHints, motifHints, 32,
                            PropModeReplace, (unsigned char*)&hints, 5);
            }

            Atom netWmState = XInternAtom(display, "_NET_WM_STATE", False);
            Atom netWmStateAbove = XInternAtom(display, "_NET_WM_STATE_ABOVE", False);
            Atom netWmStateSkipTaskbar = XInternAtom(display, "_NET_WM_STATE_SKIP_TASKBAR", False);
            Atom netWmStateSkipPager = XInternAtom(display, "_NET_WM_STATE_SKIP_PAGER", False);

            Atom states[] = { netWmStateAbove, netWmStateSkipTaskbar, netWmStateSkipPager };
            XChangeProperty(display, window, netWmState, XA_ATOM, 32, 
                            PropModeReplace, (unsigned char*)states, 3);

            // Set window type to overlay/dock for proper stacking
            Atom windowType = XInternAtom(display, "_NET_WM_WINDOW_TYPE", False);
            Atom windowTypeDock = XInternAtom(display, "_NET_WM_WINDOW_TYPE_DOCK", False);
            XChangeProperty(display, window, windowType, XA_ATOM, 32,
                            PropModeReplace, (unsigned char*)&windowTypeDock, 1);
        }

        // Set window to stay on top
        

        // Ensure no input events are selected
        if (clickthrough)
        XSelectInput(display, window, NoEventMask);
                
        // Create graphics context

        data = (uint84*)backend.initialize(display, window, shape[1], shape[0], visual, depth, colormap);

        // (*this)[{{}}] = 0x00000000;
        // gc = XCreateGC(display, window, 0, nullptr);
        
        // Create XImage for pixel buffer
        // createXImage();
        
        // Map window
        XMapWindow(display, window);
        XFlush(display);

        // Set initial screen size based on current display
        if (is_fullscreen) {
        
            const DisplayInfo* current_display = display_manager.getCurrentDisplay();
            if (current_display) {
                current_screen_input_info.setScreenSize(current_display->x, current_display->y, 
                                                    current_display->width, current_display->height);
                
                // Position window on current display
                XMoveResizeWindow(display, window, current_display->x, current_display->y, 
                                current_display->width, current_display->height);
            } else {
                // Fallback for single display
                XWindowAttributes attr;
                XGetWindowAttributes(display, window, &attr);
                int x, y;
                unsigned int width, height, border_width, depth;
                Window root;
                XGetGeometry(display, window, &root, &x, &y, &width, &height, &border_width, &depth);
                current_screen_input_info.setScreenSize(x, y, width, height);
            }
        }

        current_screen_input_info.selected_text_reader = new SelectedTextReader(display, window);
        
        // Start direct input reading
        startDirectInputReading();
        
    }
    
private:
    void startDirectInputReading() {
        input_reader.start([this](const input_event& event, const std::string& device_name) {
            processInputEvent(event, device_name);
        });
    }

    void updateMousePositionFromRoot() {
        Window root = RootWindow(display, screen);
        Window child;
        int root_x, root_y, win_x, win_y;
        unsigned int mask;
        
        if (XQueryPointer(display, root, &root, &child, &root_x, &root_y, &win_x, &win_y, &mask)) {
            // root_x, root_y are the global screen coordinates
            current_screen_input_info.updateMousePositionAbsolute(root_x, root_y);
        }
    }

    
    void processInputEvent(const input_event& event, const std::string& device_name) {
        // Capture and process ALL events without filtering
        switch (event.type) {
            case EV_KEY:
                // Process ALL key events, including mouse buttons and keyboard keys
                if (event.code == BTN_LEFT || event.code == BTN_RIGHT || event.code == BTN_MIDDLE) {
                    current_screen_input_info.updateMouseButtonState(event.code, event.value);
                } else {
                    // For keyboard keys, update key state
                    current_screen_input_info.updateKeyState(event.code, event.value);
                }
                break;
                
            case EV_REL:
                // Process ALL relative events
                switch (event.code) {
                    case REL_X:
                        // current_screen_input_info.updateMousePositionRelative(event.value, 0);
                        break;
                    case REL_Y:
                        // current_screen_input_info.updateMousePositionRelative(0, event.value);
                        break;
                    case REL_WHEEL:
                    case REL_HWHEEL:
                        current_screen_input_info.updateMouseWheel(event.code, event.value);
                        break;
                }

                break;
                
            case EV_ABS:
                // Process ALL absolute positioning events
                switch (event.code) {
                    case ABS_X:
                        current_screen_input_info.updateMousePositionAbsolute(event.value, current_screen_input_info.getMouseY());
                        break;
                    case ABS_Y:
                        current_screen_input_info.updateMousePositionAbsolute(current_screen_input_info.getMouseX(), event.value);
                        break;
                }
                break;
                
            case EV_SYN:
                // Always process synchronization events
                if (event.code == SYN_REPORT) {
                    current_screen_input_info.clearWheelStates();
                }
                break;
        }

        
       
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
            
            XChangeProperty(display, window, motifHints, motifHints, 32,
                           PropModeReplace, (unsigned char*)&hints, 5);
        }
    }
    
    void enableAlphaBlending() {
        // Set window opacity property for compositor
        Atom netWmWindowOpacity = XInternAtom(display, "_NET_WM_WINDOW_OPACITY", False);
        if (netWmWindowOpacity != None) {
            unsigned long opacity = 0xFFFFFFFF; // Fully opaque by default
            XChangeProperty(display, window, netWmWindowOpacity, XA_CARDINAL, 32,
                           PropModeReplace, (unsigned char*)&opacity, 1);
        }
        
        // Enable compositing for this window
        XCompositeRedirectWindow(display, window, CompositeRedirectAutomatic);
    }
    
    void setWindowOpacity(float opacity) {
        if (!alpha_enabled) return;
        
        Atom netWmWindowOpacity = XInternAtom(display, "_NET_WM_WINDOW_OPACITY", False);
        if (netWmWindowOpacity != None) {
            unsigned long opacityValue = (unsigned long)(opacity * 0xFFFFFFFF);
            XChangeProperty(display, window, netWmWindowOpacity, XA_CARDINAL, 32,
                           PropModeReplace, (unsigned char*)&opacityValue, 1);
            XFlush(display);
        }
    }
    
    // void createXImage() {
        
    // }
    
    void updateDisplay() {
        // if (!backend. ximage) return;
        
        // // Update the image data pointer (in case tensor was reallocated)
        // ximage->data = (char*)this->data;
        // ximage->width = shape[1];
        // ximage->height = shape[0];
        // ximage->bytes_per_line = shape[1] * sizeof(uint84);
        
        if (shape[0] > 0 && shape[1] > 0) {
            backend.present();
            //     XPutImage(display, window, gc, ximage, 0, 0, 0, 0, shape[1], shape[0]);
        }
        // XFlush(display);
    }

    void resizeDisplay() {
        if (is_fullscreen) {
            XWindowAttributes attr;
            XGetWindowAttributes(display, window, &attr);
            if (attr.width != shape[1] || attr.height != shape[0]) {
                // if (ximage) {
                //     ximage->data = nullptr;
                //     XDestroyImage(ximage);
                // }

                // int bitmap_pad = (depth > 16) ? 32 : (depth > 8) ? 16 : 8;
                // this->data = (uint84*)realloc(data, attr.width * attr.height * sizeof(uint84));
                // shape[0] = attr.height;
                // shape[1] = attr.width;

                // ximage = XCreateImage(display, visual, depth, ZPixmap, 0,
                //                      (char*)this->data, shape[1], shape[0],
                //                      bitmap_pad, shape[1] * sizeof(uint84));
                // if (!ximage) {
                //     std::cerr << "Cannot create resized XImage" << std::endl;
                //     return;
                // }
                
                // ximage->byte_order = LSBFirst;
                // ximage->bitmap_bit_order = LSBFirst;
                // ximage->width = shape[1];
                // ximage->height = shape[0];
                // ximage->bytes_per_line = shape[1] * sizeof(uint84);
                
                // calculate_metadata();
                // XResizeWindow(display, window, ximage->width, ximage->height);
            } 
        } 
    }
    
    void displayLoop() {
        bool quit = false;
        XEvent event;
        
        while (!quit) {
            resizeDisplay();
            
            
            // Call display loop functions
            for (const auto& callback : display_loop_functions) {
                callback(current_screen_input_info);
            }

            // Handle minimal X11 events (mainly window management)
            while (XPending(display) > 0) {
                XNextEvent(display, &event);
                
                switch (event.type) {
                    case Expose:
                        updateDisplay();
                        break;
                    case ClientMessage:
                        if (event.xclient.data.l[0] == XInternAtom(display, "WM_DELETE_WINDOW", False)) {
                            quit = true;
                        }
                        break;
                    case ConfigureNotify:
                        // Window size changed
                        break;
                }
            }
            
            updateDisplay();
            // usleep(11111); // ~90 FPS
            updateMousePositionFromRoot();
        }
        
        // Stop input reading when display loop ends
        input_reader.stop();
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
        input_reader.stop();
        
        // if (ximage) {
        //     ximage->data = nullptr;
        //     XDestroyImage(ximage);
        // }
        // if (gc) {
        //     XFreeGC(display, gc);
        // }
        if (window) {
            XDestroyWindow(display, window);
        }
        if (display) {
            XCloseDisplay(display);
        }
    }
    
    void add_on_update(std::function<void(CurrentScreenInputInfo&)> func) {
        display_loop_functions.push_back(func);
    }
    
    
    // Utility functions for window management
    void moveWindow(int x, int y) {
        XMoveWindow(display, window, x, y);
        XFlush(display);
    }
    
    void resizeWindow(int width, int height) {
        XResizeWindow(display, window, width, height);
        XFlush(display);
    }
    
    // Multi-display functions
    void moveToDisplay(int display_index) {
        const DisplayInfo* target_display = display_manager.getDisplay(display_index);
        if (target_display) {
            display_manager.setCurrentDisplay(display_index);
            
            // Update window position and size to match display
            XMoveResizeWindow(display, window, target_display->x, target_display->y,
                            target_display->width, target_display->height);
            
            // Update screen info
            std::cout << "Moving to display: " << target_display->name << std::endl;
            std::cout << "Position: (" << target_display->x << ", " << target_display->y 
                      << "), Size: " << target_display->width << "x" << target_display->height << std::endl;
            current_screen_input_info.setScreenSize(target_display->x, target_display->y,
                                                   target_display->width, target_display->height);
            
            // Resize internal buffer if needed
            if (shape[0] != target_display->height || shape[1] != target_display->width) {
                // Resize tensor
                // if (ximage) {
                //     ximage->data = nullptr;
                //     XDestroyImage(ximage);
                // }
                
                // this->data = (uint84*)realloc(data, target_display->width * target_display->height * sizeof(uint84));
                // shape[0] = target_display->height;
                // shape[1] = target_display->width;
                
                // createXImage();
                // calculate_metadata();
            }
            
            XFlush(display);
            std::cout << "Moved to display " << display_index << ": " << target_display->name << std::endl;
        }
    }
    
    void moveToNextDisplay() {
        int next_display = (display_manager.getCurrentDisplayIndex() + 1) % display_manager.getDisplayCount();
        moveToDisplay(next_display);
    }
    
    void moveToPreviousDisplay() {
        int prev_display = (display_manager.getCurrentDisplayIndex() - 1 + display_manager.getDisplayCount()) % display_manager.getDisplayCount();
        moveToDisplay(prev_display);
    }
    
    void moveToPrimaryDisplay() {
        const auto& displays = display_manager.getDisplays();
        for (size_t i = 0; i < displays.size(); i++) {
            if (displays[i].is_primary) {
                moveToDisplay(i);
                return;
            }
        }
    }
    
    void moveToDisplayContainingMouse() {
        auto [global_x, global_y] = current_screen_input_info.getGlobalMousePosition();
        const DisplayInfo* containing_display = display_manager.getDisplayContaining(global_x, global_y);
        if (containing_display && containing_display->index != display_manager.getCurrentDisplayIndex()) {
            moveToDisplay(containing_display->index);
        }
    }
    
    // Get display information
    std::vector<DisplayInfo> getAllDisplays() const {
        return display_manager.getDisplays();
    }
    
    const DisplayInfo* getCurrentDisplay() const {
        return display_manager.getCurrentDisplay();
    }
    
    int getCurrentDisplayIndex() const {
        return display_manager.getCurrentDisplayIndex();
    }
    
    int getDisplayCount() const {
        return display_manager.getDisplayCount();
    }
    
    void refreshDisplays() {
        display_manager.detectDisplays();
    }
};

#endif
