#ifndef OPENGL_RENDERER_HPP
#define OPENGL_RENDERER_HPP

#include <GL/gl.h>
#include <GL/glext.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_syswm.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include "../tensor.hpp"
#include "../vector/vectors.hpp"
// #include "../vector/uint84.hpp"
#include "../file_loaders/gltf.hpp"
#include "./display.hpp"
#include <SDL2/SDL_opengl_glext.h>
// OpenGL function pointers for modern OpenGL functions
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



struct Camera {
    float32x3 position;
    float32x3 target;
    float32x3 up;
    float fov;
    float aspect;
    float near_plane;
    float far_plane;
    
    Camera() : position(0, 0, 5), target(0, 0, 0), up(0, 1, 0), 
               fov(45.0f), aspect(1.0f), near_plane(0.1f), far_plane(100.0f) {}

 

    
     mat4 getProjectionMatrix() {
        mat4 result;
        float tanHalfFov = tan(fov * 0.5f * M_PI / 180.0f);
        
        result.rows[0] = float32x4(1.0f / (aspect * tanHalfFov), 0, 0, 0);
        result.rows[1] = float32x4(0, 1.0f / tanHalfFov, 0, 0);
        result.rows[2] = float32x4(0, 0, -(far_plane + near_plane) / (far_plane - near_plane), -1);
        result.rows[3] = float32x4(0, 0, -(2.0f * far_plane * near_plane) / (far_plane - near_plane), 0);
        
        return result;
    }
    
    mat4 getViewMatrix() {
        // Simple lookAt implementation
        float32x3 f = target; // forward
        f.x -= position.x;
        f.y -= position.y;
        f.z -= position.z;
        
        // Normalize f
        float length = sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
        f.x /= length; f.y /= length; f.z /= length;
        
        // Right = f x up
        float32x3 r;
        r.x = f.y * up.z - f.z * up.y;
        r.y = f.z * up.x - f.x * up.z;
        r.z = f.x * up.y - f.y * up.x;
        
        // Normalize r
        length = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        r.x /= length; r.y /= length; r.z /= length;
        
        // Up = r x f
        float32x3 u;
        u.x = r.y * f.z - r.z * f.y;
        u.y = r.z * f.x - r.x * f.z;
        u.z = r.x * f.y - r.y * f.x;
        
        mat4 result;
        result.rows[0] = float32x4(r.x, u.x, -f.x, 0);
        result.rows[1] = float32x4(r.y, u.y, -f.y, 0);
        result.rows[2] = float32x4(r.z, u.z, -f.z, 0);
        result.rows[3] = float32x4(
            -(r.x * position.x + r.y * position.y + r.z * position.z),
            -(u.x * position.x + u.y * position.y + u.z * position.z),
            f.x * position.x + f.y * position.y + f.z * position.z,
            1
        );
        
        return result;
    }

    mat4 getWorldMatrix() {
        mat4 world;
        world.set_translate(-position);
        return world;
    }

    mat4 getViewProjectionMatrix() {
        return getProjectionMatrix() * getViewMatrix();
    }

    mat4 getViewProjectionWorldMatrix() {
        return getProjectionMatrix() * getViewMatrix() * getWorldMatrix();
    }

    mat4 getProjectionViewMatrix() {
        return getViewMatrix() * getProjectionMatrix();
    }
};

struct RenderTexture {
    GLuint texture;
    int width;
    int height;
    
    RenderTexture() : texture(0), width(0), height(0) {}
};

struct RenderMaterial {
    std::string name;
    GLuint texture;
    GLuint normal_map;
    GLuint metallic_map;
    bool double_sided;
    
    RenderMaterial() : texture(0), normal_map(0), metallic_map(0), double_sided(false) {}
};

struct RenderMesh {
    GLuint VAO;
    GLuint VBO_positions;
    GLuint VBO_normals;
    GLuint VBO_texcoords;
    GLuint EBO;
    GLuint BBO; // Bone Buffer Object for skeletal animation
    int index_count;
    PrimitiveType primitive_type;
    RenderMaterial material;
    
    Skeleton bone_matrices; // Pointer to bone matrices for skeletal animation
    
    RenderMesh(Skeleton bone_matrices) : VAO(0), VBO_positions(0), VBO_normals(0), VBO_texcoords(0), EBO(0), index_count(0), BBO(0), bone_matrices(bone_matrices),
                   primitive_type(PrimitiveType::TRIANGLES) {}
};



class OpenGLRenderer : public VectorDisplay{
private:
    SDL_Window* sdl_window;
    SDL_GLContext gl_context;
    GLuint shader_program;
    Camera camera;
    std::vector<RenderMesh> render_meshes;
    std::vector<RenderTexture> render_textures;
    
    // Framebuffer for off-screen rendering
    GLuint color_texture;
    GLuint depth_renderbuffer;
    int render_width, render_height;
    
    // Shader source code
    const char* vertex_shader_source = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;
        // boneID is used for skeletal animation
        layout (location = 3) in int boneID; // Assuming boneID is an integer attribute
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat4 bone_matrices[100]; // Array of bone matrices for skeletal animation
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        void main() {
            mat4 amodel = bone_matrices[boneID] * model ;
            FragPos = vec3(amodel * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(amodel))) * aNormal;
            TexCoord = aTexCoord;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
    )";
    
    const char* fragment_shader_source = R"(
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        uniform float objectAlpha;

        // texture, normal and metallic textures can be added here
        uniform sampler2D texture1;
        uniform sampler2D normalMap;
        uniform sampler2D metallicMap;

        
        void main() {

            vec3 color = texture(texture1, TexCoord).rgb * objectColor; // Apply texture color
            vec3 normal = texture(normalMap, TexCoord).rgb;
            normal = normalize(normal * 2.0 - 1.0); // Convert from [0,1] to [-1,1]
            float metallic = texture(metallicMap, TexCoord).r;
            vec3 objectColor = color ; // Simple metallic effect

            // Ambient
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse
            vec3 baseNormal = normalize(Normal);
            vec3 norm =baseNormal; // Use the normal from the vertex shader);
            
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            vec3 specularColor = vec3(1.0, 1.0, 1.0); // White specular highlight
            float metallicFactor = metallic; // Use metallic factor for specular intensity
            
            // Specular
            vec3 viewDir = normalize(-FragPos); // Assuming camera is at origin
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), metallicFactor * 128.0); // Adjust shininess based on metallic factor
            vec3 specular = spec * specularColor * lightColor;
            
            vec3 result = (ambient + diffuse) * objectColor + specular;
            
            // Output with proper alpha for transparency
            FragColor = vec4(result, objectAlpha);
            
            
        }
    )";
    
    bool loadOpenGLFunctions() {
        if (SDL_GL_LoadLibrary(nullptr) < 0) {
            std::cerr << "SDL_GL_LoadLibrary failed: " << SDL_GetError() << std::endl;
            return false;
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
        // array of mat4 for skeletons
        
       
        
        // Check if all functions were loaded successfully
        if (!glCreateShader || !glShaderSource || !glCompileShader || !glGetShaderiv ||
            !glGetShaderInfoLog || !glCreateProgram || !glAttachShader || !glLinkProgram ||
            !glGetProgramiv || !glGetProgramInfoLog || !glDeleteShader || !glDeleteProgram ||
            !glUseProgram || !glGenVertexArrays || !glBindVertexArray || !glDeleteVertexArrays ||
            !glGenBuffers || !glBindBuffer || !glBufferData || !glDeleteBuffers ||
            !glVertexAttribPointer || !glEnableVertexAttribArray || !glGetUniformLocation ||
            !glUniformMatrix4fv || !glUniform3fv || !glUniform1f || !glVertexAttribIPointer ||
            !glGenTextures ||
            !glBindTexture || !glTexImage2D || !glTexParameteri || !glDeleteTextures ||
            !glReadPixels ) {
            std::cerr << "Failed to load one or more OpenGL functions!" << std::endl;
            return false;
        }
        
        return true;
    }

    bool createShaderProgram() {
        GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
        glCompileShader(vertex_shader);
        
        GLint success;
        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(vertex_shader, 512, nullptr, infoLog);
            std::cerr << "Vertex Shader Compilation Failed: " << infoLog << std::endl;
            return false;
        }
        
        GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
        glCompileShader(fragment_shader);
        
        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
            std::cerr << "Fragment Shader Compilation Failed: " << infoLog << std::endl;
            return false;
        }
        
        shader_program = glCreateProgram();
        glAttachShader(shader_program, vertex_shader);
        glAttachShader(shader_program, fragment_shader);
        glLinkProgram(shader_program);
        
        glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(shader_program, 512, nullptr, infoLog);
            std::cerr << "Shader Program Linking Failed: " << infoLog << std::endl;
            return false;
        }
        
        // Clean up shaders
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        
        return true;
    }

    
    
   
    
    bool createFramebuffer(int width, int height) {
        render_width = width;
        render_height = height;
        
        // Generate framebuffer
        
        
        // Create color texture
        glGenTextures(1, &color_texture);
        glBindTexture(GL_TEXTURE_2D, color_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        
        
        std::cout << "Framebuffer created successfully: " << width << "x" << height << std::endl;
        return true;
    }
    
    void deleteFramebuffer() {
        if (color_texture) {
            glDeleteTextures(1, &color_texture);
            color_texture = 0;
        }
        
    }
    
public:
 //Platform-specific transparency setup
    float time = 0.0f;
    OpenGLRenderer(int width = 800, int height = 600) : VectorDisplay(Shape<2>{height, width}), 
                   sdl_window(nullptr), gl_context(nullptr), shader_program(0),
                   color_texture(0), depth_renderbuffer(0) {
        
        // Initialize SDL for OpenGL context creation
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }
        
        // Set OpenGL attributes
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
        // Set ARGB format for transparency

        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
        // set byte order to little-endian
        // SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
        
        // Create SDL window from existing X11 window
        SDL_SysWMinfo wmInfo;
        SDL_VERSION(&wmInfo.version);
        
        // Create a temporary SDL window to get OpenGL context
        sdl_window = SDL_CreateWindow("OpenGL Context", 
                                     SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                        width, height,
                                     SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        
        if (!sdl_window) {
            std::cerr << "SDL Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }
        
        // Create OpenGL context
        gl_context = SDL_GL_CreateContext(sdl_window);
        if (!gl_context) {
            std::cerr << "OpenGL context could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }
        
        // Make context current
        SDL_GL_MakeCurrent(sdl_window, gl_context);
        
        // Load OpenGL functions
        if (!loadOpenGLFunctions()) {
            std::cerr << "Failed to load OpenGL functions!" << std::endl;
            return;
        }
        
        // Create framebuffer for off-screen rendering
        float32x4 totalscreensallsize = current_screen_input_info.getDisplayManager()->getGlobalSize();
        if (!createFramebuffer(int(totalscreensallsize.z), int(totalscreensallsize.w))) {
            std::cerr << "Failed to create framebuffer!" << std::endl;
            return;
        }

        SDL_SetWindowSize(sdl_window, int(totalscreensallsize.z), int(totalscreensallsize.w));
        
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        
        // Enable blending for transparency
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // Create shader program
        if (!createShaderProgram()) {
            std::cerr << "Failed to create shader program!" << std::endl;
            return;
        }
        
        // Set camera aspect ratio
        camera.aspect = (float)width / (float)height;
        
        std::cout << "OpenGL Renderer initialized successfully!" << std::endl;

        add_on_update([this](CurrentScreenInputInfo& input_info) {
            // copy framebuffer content to the main window if using framebuffer
                
                
                
                // Update camera aspect ratio
                camera.aspect = (float)render_width / (float)render_height;

                glViewport(0, 0, render_width, render_height);
                

                render();
                glReadPixels(input_info.getX(), input_info.getY(), shape.B, shape.A, GL_BGRA, GL_UNSIGNED_BYTE, data);
                
            
        });
    }
    
   
    bool loadGLTF(const gltf& model) {
        std::cout << "Loading GLTF model with " << model.meshes.size() << " meshes" << std::endl;

        // load textures
        for (const auto& texture : model.textures) {
            RenderTexture render_texture;
            glGenTextures(1, &render_texture.texture);
            glBindTexture(GL_TEXTURE_2D, render_texture.texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture.shape.A, texture.shape.B, 0, GL_RGB, GL_UNSIGNED_BYTE, texture.data);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            render_texture.width = texture.shape.A;
            render_texture.height = texture.shape.B;
            glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture
            render_textures.push_back(render_texture);
            std::cout << "Loaded texture: " << texture.filename << " with ID: " << render_texture.texture << std::endl;

        }
        
        
        for (const auto& mesh : model.meshes) {
            std::cout << "Processing mesh: " << mesh.name << std::endl;
            
            for (const auto& primitive : mesh.primitives) {
                RenderMesh render_mesh(
                    model.skeletons[0]
                );
                
                // Generate VAO
                glGenVertexArrays(1, &render_mesh.VAO);
                glBindVertexArray(render_mesh.VAO);
                
                // Process positions
                auto pos_it = primitive.attributes.find("POSITION");
                if (pos_it != primitive.attributes.end()) {
                    Tensor<float32x3,1>  positions = pos_it->second;
                    glGenBuffers(1, &render_mesh.VBO_positions);
                    glBindBuffer(GL_ARRAY_BUFFER, render_mesh.VBO_positions);
                    glBufferData(GL_ARRAY_BUFFER, positions.shape[0] * sizeof(float32x3), positions.data, GL_STATIC_DRAW);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float32x3), (void*)0);
                    glEnableVertexAttribArray(0);
                }
                
                // Process normals
                auto norm_it = primitive.attributes.find("NORMAL");
                if (norm_it != primitive.attributes.end()) {
                    Tensor<float32x3,1> normals = norm_it->second;
                    glGenBuffers(1, &render_mesh.VBO_normals);
                    glBindBuffer(GL_ARRAY_BUFFER, render_mesh.VBO_normals);
                    glBufferData(GL_ARRAY_BUFFER, normals.shape[0] * sizeof(float32x3), normals.data, GL_STATIC_DRAW);
                    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float32x3), (void*)0);
                    glEnableVertexAttribArray(1);
                }
                
                // Process texture coordinates
                auto tex_it = primitive.attributes.find("TEXCOORD_0");
                if (tex_it != primitive.attributes.end()) {
                    Tensor<float32x32x2,1>  texcoords = tex_it->second;
                    glGenBuffers(1, &render_mesh.VBO_texcoords);
                    glBindBuffer(GL_ARRAY_BUFFER, render_mesh.VBO_texcoords);
                    glBufferData(GL_ARRAY_BUFFER, texcoords.shape[0] * sizeof(float32x32x2), texcoords.data, GL_STATIC_DRAW);
                    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float32x32x2), (void*)0);
                    glEnableVertexAttribArray(2);
                }

                auto joint_it = primitive.attributes.find("bone_ids");
                if (joint_it != primitive.attributes.end()) {
                    std::cout << "Processing joint IDs for mesh: " << mesh.name << std::endl;
                    Tensor<int,1> joint_ids = joint_it->second;
                    std::cout << "Joint IDs shape: " << joint_ids.shape.A << std::endl;
                    std::cout << joint_ids << std::endl;
                    glGenBuffers(1, &render_mesh.BBO);
                    glBindBuffer(GL_ARRAY_BUFFER, render_mesh.BBO);
                    glBufferData(GL_ARRAY_BUFFER, joint_ids.shape.A * sizeof(int), joint_ids.data, GL_STATIC_DRAW);
                    glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, sizeof(int), (void*)0);
                    glEnableVertexAttribArray(3);
                }
                
                // Process material
                if (primitive.materialIndex >= 0 && primitive.materialIndex < model.materials.size()) {
                    const auto& material = model.materials[primitive.materialIndex];
                    render_mesh.material.name = material.name;
                    render_mesh.material.texture = render_textures[material.baseColorTextureIndex].texture;
                    render_mesh.material.normal_map = render_textures[material.normalTextureIndex].texture;
                    render_mesh.material.metallic_map = render_textures[material.metallicRoughnessTextureIndex].texture;
                    render_mesh.material.double_sided = material.doubleSided;
                } else {
                    std::cerr << "Warning: Material index " << primitive.materialIndex << " out of range for mesh " << mesh.name << std::endl;
                }
                
                // Process indices
                glGenBuffers(1, &render_mesh.EBO);
                Tensor<int,1> primitive_indices = primitive.indices;
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, render_mesh.EBO);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, primitive_indices.shape[0] * sizeof(int), primitive.indices.data, GL_STATIC_DRAW);
                render_mesh.index_count = primitive.indices.shape.total_size();
                render_mesh.primitive_type = primitive.type;
                
                glBindVertexArray(0);

                
                render_meshes.push_back(render_mesh);
                std::cout << "Created render mesh with " << render_mesh.index_count << " indices" << std::endl;
            }


        }
        
        return true;
    }
    
    void setCamera(const float32x3& position, const float32x3& target, const float32x3& up = float32x3(0, 1, 0)) {
        camera.position = position;
        camera.target = target;
        camera.up = up;
    }
    
    // Add method to control transparency dynamically
    void setTransparency(float alpha) {
        // This will be used in the next render call
        // You can store this as a member variable if needed
    }

    Camera& getCamera() {
        return camera;
    }
    
    void render() {
        time += 0.01f; // Increment time for animation
        // Clear the screen
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        
        // Use shader program
        glUseProgram(shader_program);
        
        // Set up matrices
        mat4 model; // Identity matrix (all zeros except diagonal)
        model.rows[0].x = 1.0f;
        model.rows[1].y = 1.0f;
        model.rows[2].z = 1.0f;
        model.rows[3].w = 1.0f;
        
        mat4 view = camera.getViewMatrix();
        mat4 projection = camera.getProjectionMatrix();
        
        // Set uniforms
        GLint modelLoc = glGetUniformLocation(shader_program, "model");
        GLint viewLoc = glGetUniformLocation(shader_program, "view");
        GLint projLoc = glGetUniformLocation(shader_program, "projection");
        GLint lightPosLoc = glGetUniformLocation(shader_program, "lightPos");
        GLint lightColorLoc = glGetUniformLocation(shader_program, "lightColor");
        GLint objectColorLoc = glGetUniformLocation(shader_program, "objectColor");
        GLint objectAlphaLoc = glGetUniformLocation(shader_program, "objectAlpha");

        
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (float*)&model);
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, (float*)&view);
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, (float*)&projection);
       
        float lightPos[] = {2.0f, 2.0f, 2.0f};
        float lightColor[] = {1.0f, 1.0f, 1.0f};
        float objectColor[] = {1.0, 1.0f, 1.0f};
        float objectAlpha = 1.0f; // Semi-transparent for desktop pet
        
        glUniform3fv(lightPosLoc, 1, lightPos);
        glUniform3fv(lightColorLoc, 1, lightColor);
        glUniform3fv(objectColorLoc, 1, objectColor);
        glUniform1f(objectAlphaLoc, objectAlpha);
        
        
        // Render all meshes
        for (const auto& mesh : render_meshes) {

            // turn backface culling off
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        
            glBindVertexArray(mesh.VAO);
            // Bind vertex buffers
            glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO_positions);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float32x3), (void*)0);
            glEnableVertexAttribArray(0);
            if (mesh.VBO_normals) {
                glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO_normals);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float32x3), (void*)0);
                glEnableVertexAttribArray(1);
            }
            if (mesh.VBO_texcoords) {
                glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO_texcoords);
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float32x32x2), (void*)0);
                glEnableVertexAttribArray(2);
            }
            if (mesh.BBO) {
                glBindBuffer(GL_ARRAY_BUFFER, mesh.BBO);
                glVertexAttribIPointer(3, 1, GL_INT, GL_FALSE, (void*)0);
                glEnableVertexAttribArray(3);
            }

            // Bind buffers
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
            GLenum mode = GL_TRIANGLES; // Default to triangles
            switch (mesh.primitive_type) {
                case PrimitiveType::TRIANGLES:
                    mode = GL_TRIANGLES;
                    break;
                case PrimitiveType::LINES:
                    mode = GL_LINES;
                    break;
                case PrimitiveType::POINTS:
                    mode = GL_POINTS;
                    break;
                case PrimitiveType::LINE_LOOP:
                    mode = GL_LINE_LOOP;
                    break;
                case PrimitiveType::LINE_STRIP:
                    mode = GL_LINE_STRIP;
                    break;
                case PrimitiveType::POLYGON:
                    mode = GL_POLYGON;
                    break;
                case PrimitiveType::TRIANGLE_STRIP:
                    mode = GL_TRIANGLE_STRIP;
                    break;
                case PrimitiveType::TRIANGLE_FAN:
                    mode = GL_TRIANGLE_FAN;
                    break;
            }

            // apply materials
            GLint baseTexture = glGetUniformLocation(shader_program, "texture1");
            GLint normalMap = glGetUniformLocation(shader_program, "normalMap");
            GLint metallicMap = glGetUniformLocation(shader_program, "metallicMap");
            GLint bonesLoc = glGetUniformLocation(shader_program, "bone_matrices");
            
            glUniformMatrix4fv(bonesLoc, mesh.bone_matrices.shape.A, GL_TRUE, (float*)(void*)mesh.bone_matrices.data); // Assuming 100 bones for simplicity
            
            

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.material.texture);
            glUniform1i(baseTexture, 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, mesh.material.normal_map);
            glUniform1i(normalMap, 1);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, mesh.material.metallic_map);
            glUniform1i(metallicMap, 2);
            // Set double-sided rendering if needed
            if (mesh.material.double_sided) {
                glDisable(GL_CULL_FACE);
            } else {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }
            
            // indices 
            if (mesh.index_count > 0) {
                glDrawElements(mode, mesh.index_count, GL_UNSIGNED_INT, 0);
            } else {
                // If no indices, draw as points
                glDrawArrays(mode, 0, mesh.VBO_positions ? mesh.VBO_positions : 0);
            }
            glBindVertexArray(0);

        }
        
        glBindVertexArray(0);
        
        

        
    }
    
};


#endif // OPENGL_RENDERER_HPP
