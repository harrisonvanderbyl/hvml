#ifndef TENSOR_DISPLAY_SHADERBUILDER_HPP
#define TENSOR_DISPLAY_SHADERBUILDER_HPP

#include "tensor.hpp"
#include "ops/ops.hpp"
#include "vector/vectors.hpp"
#include "file_loaders/texture.hpp"
// function traits to extract argument types
#include <tuple>
#include <type_traits>

__weak std::map<std::string, GLuint> shader_program_cache;



struct Material
{
    virtual ~Material() {}
    virtual const char* getVertexShaderSource() {
        throw std::runtime_error("getVertexShaderSource not implemented for this material");
     };
    virtual const char* getFragmentShaderSource() {
        throw std::runtime_error("getVertexShaderSource not implemented for this material");
     };
    virtual const char* getGeometryShaderSource() {
        throw std::runtime_error("getVertexShaderSource not implemented for this material");
     };
    GLuint shader_program;
    std::string name;
    bool double_sided = false;
    std::map<std::string, GLuint> textures_ids;
    virtual bool createShaderProgram()
    {

        const char* vertex_shader_source = getVertexShaderSource();
        const char* fragment_shader_source = getFragmentShaderSource();

        std::string shader_key = std::string(vertex_shader_source) + std::string(fragment_shader_source);
        if (shader_program_cache.find(shader_key) == shader_program_cache.end())
        {


            GLuint vertex_shader = GLFuncs->glCreateShader(GL_VERTEX_SHADER);
            GLFuncs->glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
            GLFuncs->glCompileShader(vertex_shader);

            GLint success;
            GLFuncs->glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                char infoLog[512];
                GLFuncs->glGetShaderInfoLog(vertex_shader, 512, nullptr, infoLog);
                std::cerr << "Vertex Shader Compilation Failed: " << infoLog << std::endl;
                return false;
            }

            GLuint fragment_shader = GLFuncs->glCreateShader(GL_FRAGMENT_SHADER);
            GLFuncs->glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
            GLFuncs->glCompileShader(fragment_shader);

            GLFuncs->glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                char infoLog[512];
                GLFuncs->glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
                std::cerr << "Fragment Shader Compilation Failed: " << infoLog << std::endl;
                return false;
            }

            shader_program = GLFuncs->glCreateProgram();
            GLFuncs->glAttachShader(shader_program, vertex_shader);
            GLFuncs->glAttachShader(shader_program, fragment_shader);
            GLFuncs->glLinkProgram(shader_program);

            GLFuncs->glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
            if (!success)
            {
                char infoLog[512];
                GLFuncs->glGetProgramInfoLog(shader_program, 512, nullptr, infoLog);
                std::cerr << "Shader Program Linking Failed: " << infoLog << std::endl;
                return false;
            }

            // Clean up shaders
            GLFuncs->glDeleteShader(vertex_shader);
            GLFuncs->glDeleteShader(fragment_shader);
            shader_program_cache[shader_key] = shader_program;
        }else
        {
            shader_program = shader_program_cache[shader_key];
        }

        return true;
    }

    void bind()
    {
        if(shader_program == 0){
            std::cerr << "Shader program not created!, creating now..." << std::endl;
            this->createShaderProgram();
        }
        if (shader_program != 0)
        {
            GLFuncs->glUseProgram(shader_program);
        }
        else {
            std::cerr << "Failed to create shader program!" << std::endl;
            throw std::runtime_error("Failed to create shader program");
        }

        for (const auto& tex_pair : textures_ids)
        {
            GLuint texture_unit = tex_pair.second;
            glActiveTexture(GL_TEXTURE0 + texture_unit);
            glBindTexture(GL_TEXTURE_2D, tex_pair.second);
            GLint uniform_location = GLFuncs->glGetUniformLocation(shader_program, tex_pair.first.c_str());
            GLFuncs->glUniform1i(uniform_location, texture_unit);
        }
    }
};



template <typename T>
struct VertexAttribute
{
    static constexpr GLint size = 0;
    static constexpr GLenum type = 0;
    static constexpr GLboolean normalized = GL_FALSE;
};

template <>
struct VertexAttribute<float32x3>
{
    static constexpr GLint size = 3;
    static constexpr GLenum type = GL_FLOAT;
    static constexpr GLboolean normalized = GL_FALSE;
};
template <>
struct VertexAttribute<float32x2>
{
    static constexpr GLint size = 2;
    static constexpr GLenum type = GL_FLOAT;
    static constexpr GLboolean normalized = GL_FALSE;
};
template <>
struct VertexAttribute<float32x4>
{
    static constexpr GLint size = 4;
    static constexpr GLenum type = GL_FLOAT;
    static constexpr GLboolean normalized = GL_FALSE;
};
template <>
struct VertexAttribute<int>
{
    static constexpr GLint size = 1;
    static constexpr GLenum type = GL_INT;
    static constexpr GLboolean normalized = GL_FALSE;
};
template <>
struct VertexAttribute<uint84>
{
    static constexpr GLint size = 4;
    // this is actually 4 unsigned bytes packed into a uint32, but should be read as colors
    static constexpr GLenum type = GL_UNSIGNED_BYTE;
    static constexpr GLboolean normalized = GL_TRUE; // normalize means map 0-255 to 0.0-1.0
};
template <>
struct VertexAttribute<float>
{
    static constexpr GLint size = 1;
    static constexpr GLenum type = GL_FLOAT;
    static constexpr GLboolean normalized = GL_FALSE;
};

template <typename T>
struct VertexAttributeToGLType {
    static constexpr const char* get() {
        using A = VertexAttribute<T>;

        if constexpr (A::type == GL_FLOAT) {
            if constexpr (A::size == 1) return "float";
            if constexpr (A::size == 2) return "vec2";
            if constexpr (A::size == 3) return "vec3";
            if constexpr (A::size == 4) return "vec4";
        }

        if constexpr (A::type == GL_INT) {
            if constexpr (A::size == 1) return "int";
            if constexpr (A::size == 2) return "ivec2";
            if constexpr (A::size == 3) return "ivec3";
            if constexpr (A::size == 4) return "ivec4";
        }

        if constexpr (A::type == GL_UNSIGNED_BYTE) {
            if constexpr (A::normalized == GL_TRUE) {
                return "vec4";   // normalized color
            } else {
                return "uvec4";
            }
        }

        return "unknown";
    }
};







template <typename... Ts>
struct mytuple
{
    // tuple implementation with __device__ and __host__ constructors
    char data[(sizeof(Ts) + ... + 0)] = {};
    __host__ __device__ mytuple(Ts... args)
    {
        size_t offset = 0;
        ((*(Ts *)(data + offset) = args, offset += sizeof(Ts)), ...);
    }

    __host__ __device__ mytuple()
    {
        // default constructor
    }
};


template <typename... Args>
struct VertexLayout : public mytuple<Args...>
{
    static constexpr int num_attributes = sizeof...(Args);
    static constexpr std::array<GLint, num_attributes> sizes = {VertexAttribute<Args>::size...};
    static constexpr std::array<GLenum, num_attributes> types = {VertexAttribute<Args>::type...};
    static constexpr std::array<GLboolean, num_attributes> normalized = {VertexAttribute<Args>::normalized...};
    static constexpr std::array<size_t, num_attributes> attribute_sizes = {sizeof(Args)...};
};

template <typename... vertex_types>
struct CopyDataHelper : HardamardOperation<CopyDataHelper<vertex_types...>>
{

    __host__ __device__ static inline void apply(const vertex_types &...vals, mytuple<vertex_types...> &out)
    {
        out = VertexLayout<vertex_types...>({vals...});
    }
};

#define __shader  [[clang::annotate("shader")]]


template <typename shader_struct>
struct Shader: public Material
{
    const char* getVertexShaderSource() override
    {
        throw(std::runtime_error("getVertexShaderSource not implemented for this shader struct"));
    }

    const char* getFragmentShaderSource() override
    {
        throw(std::runtime_error("getFragmentShaderSource not implemented for this shader struct"));
    }

    const char* getGeometryShaderSource() override
    {
        return nullptr;
    }
};

struct ShaderProgram
{

   

    // get the input args of vertex function of ShaderSubProgram, ie
    // vertex ouputs
    float32x4 gl_Position;
    float gl_PointSize;
    
    // fragment builtins
    float32x2 gl_PointCoord;
    float gl_FragDepth;
    float32x4 gl_FragCoord;

    // fragment outputs
    uint84 FragColor;
    

    void discard(){};


    float32x3 reflect(const float32x3 &I, const float32x3 &N) {
        return I - N * 2.0f * dot(N, I);
    }

    mat4 inverse(const mat4 &m) {
        // Implement matrix inversion (omitted for brevity)
        return mat4::identity(); // Placeholder
    }

    mat4 transpose(const mat4 &m) {
        return mat4::identity(); // Placeholder
    }

    template <typename T>
    float length(const T &v) {
        return v.length();
    }

    template <typename T>
    T max(const T &v, const T &other) {
        return std::max(v, other);
    };

    template <typename T>
    T min(const T &v, const T &other) {
        return std::min(v, other);
    };

    template <typename T>
    float dot(const T &a, const T &b) {
        return a.dot(b);
    }

    template <typename T>
    T normalize(const T &v) {
        return v.normalized();
    }

    template <typename T>
    T floor(const T &v) {
        return v;
    }

    template <typename T>
    T mix(const T &a, const T &b, float t) {
        return a * (1.0f - t) + b * t;
    }
    
    uint84 texture2D(const sampler2D &sampler, const float32x2 &uv) {
        // Placeholder implementation
        return sampler[{int(uv[0]), int(uv[1])}];
    }

    // no template named function traits
  
};



#endif // TENSOR_DISPLAY_SHADERBUILDER_HPP