#ifndef SOLID_COLOR_SHADER_HPP
#define SOLID_COLOR_SHADER_HPP
#include "display/materials/shader.hpp"

// Minimal shader: solid color fill
template <int = 0>
struct __shader SolidColorShader : public ShaderProgram {
    void vertex(float32x2 position, float32x2 uv) {
        gl_Position = float32x4(position, 0, 1);
        fragment(uv);
    }
    void fragment(float32x2 uv) {
        FragColor = uint84(float32x4(1.0f, 0.0f, 0.0f, 1.0f)); // Red, full alpha
    }
};
template struct SolidColorShader<0>;

#endif
