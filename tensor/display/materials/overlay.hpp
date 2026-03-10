
#ifndef OVERLAY_SHADER_HPP
#define OVERLAY_SHADER_HPP

#include "display/materials/shader.hpp"

template <bool istexturebuffer>
struct __shader OverLayShader : public ShaderProgram
{
    using samplertype = typename std::conditional_t<istexturebuffer, samplerBuffer, sampler2D>;

    // === Vertex → Fragment interpolants ===
    samplertype bufferTex;
    int32x2 dimensions;

    void vertex(
        float32x2 position,
        float32x2 uv
    ){
        
        gl_Position = float32x4(position, 0, 1);
        fragment(
            uv
        );
    }

    template <bool B = istexturebuffer, std::enable_if_t<B, int> = 0>
    void fragment(float32x2 uv) {
        int32x2 pixelCoord = int32x2(uv * dimensions);
        int index = pixelCoord.y() * dimensions.x() + pixelCoord.x();
        FragColor = texelFetch(bufferTex, index);
    }

    template <bool B = istexturebuffer, std::enable_if_t<!B, int> = 0>
    void fragment(float32x2 uv) {
        FragColor = texture(bufferTex, float32x2(uv.x(), 1.0 - uv.y()));
    }
};


template struct OverLayShader<true>;
template struct OverLayShader<false>;

#endif