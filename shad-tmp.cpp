#include "display/materials/materials.hpp"
template <>
struct Shader<OverLayShader<true>> : public Material {
const char* getFragmentShaderSource() override {
    return R"(#version 330 core
out vec4 FragColor;
in vec2 uv;
uniform samplerBuffer bufferTex;
uniform ivec2 dimensions;

void main() {
{
    ivec2 pixelCoord = ivec2(uv * dimensions);
    int index = pixelCoord.y * dimensions.x + pixelCoord.x;
    FragColor = texelFetch(bufferTex, index);
}

}
)";
}

const char* getVertexShaderSource() override {
    return R"(#version 330 core
layout (location = 0) in vec2 aPosition;
layout (location = 1) in vec2 aUv;
uniform samplerBuffer bufferTex;
uniform ivec2 dimensions;
out vec2 uv;

void fragment(vec2 in_uv) {
    uv = in_uv;
}

void main() {
{
    gl_Position = vec4(aPosition, 0, 1);
    fragment(aUv);
}

}
)";
}

void init_uniforms(GLuint shader_program) override {
    uniform_setters["bufferTex"] = UniformSetter("bufferTex", shader_program);
    uniform_setters["dimensions"] = UniformSetter("dimensions", shader_program);
    };

};

template <>
struct Shader<OverLayShader<false>> : public Material {
const char* getFragmentShaderSource() override {
    return R"(#version 330 core
out vec4 FragColor;
in vec2 uv;
uniform sampler2D bufferTex;
uniform ivec2 dimensions;

void main() {
{
    FragColor = texture(bufferTex, vec2(uv.x, 1. - uv.y));
}

}
)";
}

const char* getVertexShaderSource() override {
    return R"(#version 330 core
layout (location = 0) in vec2 aPosition;
layout (location = 1) in vec2 aUv;
uniform sampler2D bufferTex;
uniform ivec2 dimensions;
out vec2 uv;

void fragment(vec2 in_uv) {
    uv = in_uv;
}

void main() {
{
    gl_Position = vec4(aPosition, 0, 1);
    fragment(aUv);
}

}
)";
}

void init_uniforms(GLuint shader_program) override {
    uniform_setters["bufferTex"] = UniformSetter("bufferTex", shader_program);
    uniform_setters["dimensions"] = UniformSetter("dimensions", shader_program);
    };

};

template <>
struct Shader<ParticleShader> : public Material {
const char* getFragmentShaderSource() override {
    return R"(#version 330 core
out vec4 FragColor;
in float vRadius;
in vec4 centerpos;
in vec4 camPos;
in vec4 vColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 projectionview;
uniform mat4 inv_view;
uniform mat4 inv_projection;
uniform float depthOnly;
uniform sampler2D texture1;
uniform sampler2D screentexture;
uniform vec3 chunksizes;
uniform vec3 chunksperrotation;

void main() {
{
    vec2 partuv = gl_PointCoord - 0.5F;
    float distSq = dot(partuv, partuv);
    if (distSq > 0.25F) {
        discard;
        return;
    }
    if (depthOnly > 0.5F) {
        FragColor = vec4(0, 0, 0, 255);
        gl_FragDepth = gl_FragCoord.z;
        return;
    }
    vec2 screenSize = vec2(1024.F, 1024.F);
    vec2 ndcxybn = (gl_FragCoord.xy / screenSize);
    vec2 ndc_xy = ndcxybn * 2.F - 1.F;
    float ndc_z = gl_FragCoord.z * 2.F - 1.F;
    vec4 clipPos = vec4(ndc_xy, ndc_z, 1.F);
    vec4 viewSpacePos = inv_projection * clipPos;
    viewSpacePos /= viewSpacePos.w;
    vec4 worldSpacePos = inv_view * viewSpacePos;
    vec3 raycastDir = normalize(worldSpacePos.xyz - camPos.xyz);
    vec3 oc = camPos.xyz - centerpos.xyz;
    float a = dot(raycastDir, raycastDir);
    float b = 2.F * dot(oc, raycastDir);
    float c = dot(oc, oc) - vRadius * vRadius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        discard;
        return;
    } else {
        float t = (-b - sqrt(discriminant)) / (2.F * a);
        vec3 intersectionPoint = camPos.xyz + vec3(t, t, t) * raycastDir;
        vec4 clipSpaceIntersection = projectionview * vec4(intersectionPoint, 1.F);
        gl_FragDepth = clipSpaceIntersection.z / clipSpaceIntersection.w;
        vec3 normal = normalize(intersectionPoint - centerpos.xyz);
        if (vColor.w < 0.5F) {
            float distortionStrength = 0.0199999996F;
            vec2 distortion = normal.xy * distortionStrength;
            vec2 distortedUV = clamp(ndcxybn + distortion, vec2(0.F), vec2(1.F));
            vec4 backgroundColor = texture2D(screentexture, distortedUV);
            vec3 viewDir = normalize(camPos.xyz - intersectionPoint);
            float fresnel = pow(1.F - abs(dot(viewDir, normal)), 3.F);
            vec3 waterTint = vec3(0.850000023F, 0.920000016F, 1.F);
            vec3 finalColor = backgroundColor.xyz * waterTint + vColor.xyz + vec3(fresnel * 0.143000007F);
            FragColor = vec4(finalColor, 1.F);
            return;
        }
        float lightIntensity = max(dot(normal, vec3(0, 1, 0)), 0.F);
        if (length(camPos.xyz - intersectionPoint) > 50.F) {
            vec2 uv = intersectionPoint.xz * 0.100000001F;
            uv = vec2(mod(uv.x, 1.F), mod(uv.y, 1.F));
            vec4 baseColor = texture2D(texture1, uv);
            vec4 finalColor = vec4(baseColor.xyz * lightIntensity, baseColor.w * vColor.w);
            FragColor = finalColor;
            return;
        }
        vec3 absNormal = abs(normal);
        vec2 uv;
        vec2 uv1;
        vec2 uv2;
        uv = intersectionPoint.yz * 0.100000001F;
        uv1 = intersectionPoint.xz * 0.100000001F;
        uv2 = intersectionPoint.xy * 0.100000001F;
        uv = vec2(mod(uv.x, 1.F), mod(uv.y, 1.F));
        uv1 = vec2(mod(uv1.x, 1.F), mod(uv1.y, 1.F));
        uv2 = vec2(mod(uv2.x, 1.F), mod(uv2.y, 1.F));
        vec4 base1 = texture2D(texture1, uv);
        vec4 base2 = texture2D(texture1, uv1);
        vec4 base3 = texture2D(texture1, uv2);
        float total = absNormal.x + absNormal.y + absNormal.z;
        vec3 weights = absNormal / total;
        vec4 baseColor = vColor * (base1 * weights.x + base2 * weights.y + base3 * weights.z);
        vec4 finalColor = vec4(baseColor.xyz * lightIntensity, baseColor.w);
        FragColor = finalColor;
    }
}

}
)";
}

const char* getVertexShaderSource() override {
    return R"(#version 330 core
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec2 aTemperatureopacity;
layout (location = 2) in float aNeighborsfilled;
layout (location = 3) in vec4 aColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 projectionview;
uniform mat4 inv_view;
uniform mat4 inv_projection;
uniform float depthOnly;
uniform sampler2D texture1;
uniform sampler2D screentexture;
uniform vec3 chunksizes;
uniform vec3 chunksperrotation;
out float vRadius;
out vec4 centerpos;
out vec4 camPos;
out vec4 vColor;

void fragment(float in_vRadius, vec4 in_centerpos, vec4 in_camPos, vec4 in_vColor) {
    vRadius = in_vRadius;
    centerpos = in_centerpos;
    camPos = in_camPos;
    vColor = in_vColor;
}

void main() {
{
    if (aNeighborsfilled < 0.5F) {
        gl_PointSize = 0.F;
        gl_Position = vec4(0.F);
        return;
    }
    vec3 tubesize = chunksizes.xyz * chunksperrotation.xyz;
    float R = tubesize.x / (2.F * 3.1415F);
    float r = (tubesize.z / (2.F * 3.1415F));
    float phi = (aPosition.x / tubesize.x + 0.5) * 2.F * 3.1415000000000002;
    float theta = (aPosition.z / tubesize.z - 0.25) * 2.F * 3.1415000000000002;
    float tubeR = r - aPosition.y;
    vec3 positiona = aPosition;
    vec4 worldposnature = model * vec4(positiona, 1.);
    vec4 mcol = aColor;
    float radiusgrow = 0.125F;
    float radius = -0.125F;
    if (aColor.w < 0.800000011F) {
        radius = 1.F;
        radiusgrow = 0.F;
        mcol.xyz = mix(vec3(aColor.xyz), vec3(0.5F), pow(1.F - float(aNeighborsfilled / 15.F), 3.F));
    }
    radius += float(aNeighborsfilled) * radiusgrow;
    radius = max(radius, 0.F);
    vec4 eyePos = view * worldposnature;
    gl_Position = projection * eyePos;
    float distance = length(eyePos.xyz);
    float focalLength = projection[1][1];
    float screenHeight = 1024.F;
    gl_PointSize = (radius * focalLength * screenHeight) / distance;
    vec4 camPos = vec4(inv_view[3][0], inv_view[3][1], inv_view[3][2], 1.);
    fragment(radius, worldposnature, camPos, mcol);
}

}
)";
}

void init_uniforms(GLuint shader_program) override {
    uniform_setters["model"] = UniformSetter("model", shader_program);
    uniform_setters["view"] = UniformSetter("view", shader_program);
    uniform_setters["projection"] = UniformSetter("projection", shader_program);
    uniform_setters["projectionview"] = UniformSetter("projectionview", shader_program);
    uniform_setters["inv_view"] = UniformSetter("inv_view", shader_program);
    uniform_setters["inv_projection"] = UniformSetter("inv_projection", shader_program);
    uniform_setters["depthOnly"] = UniformSetter("depthOnly", shader_program);
    uniform_setters["texture1"] = UniformSetter("texture1", shader_program);
    uniform_setters["screentexture"] = UniformSetter("screentexture", shader_program);
    uniform_setters["chunksizes"] = UniformSetter("chunksizes", shader_program);
    uniform_setters["chunksperrotation"] = UniformSetter("chunksperrotation", shader_program);
    };

};

template <>
struct Shader<BasicShader> : public Material {
const char* getFragmentShaderSource() override {
    return R"(#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 projectionview;
uniform mat4[100] bone_matrices;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform float objectAlpha;
uniform sampler2D texture1;
uniform sampler2D metallicMap;
uniform sampler2D normalMap;

void main() {
{
    vec4 diffuseColor = texture2D(texture1, TexCoord);
    vec4 specularColor = texture2D(metallicMap, TexCoord);
    vec3 ambient = diffuseColor.xyz * 0.10000000000000001;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.F);
    vec3 diffuse = diffuseColor.xyz * diff;
    vec3 viewDir = normalize(- FragPos);
    vec3 reflectDir = reflect(- lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.F), 32);
    vec3 specular = specularColor.xyz * spec;
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, objectAlpha);
}

}
)";
}

const char* getVertexShaderSource() override {
    return R"(#version 330 core
layout (location = 0) in vec3 aAPos;
layout (location = 1) in vec3 aANormal;
layout (location = 2) in vec2 aATexCoord;
layout (location = 3) in int aBoneID;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 projectionview;
uniform mat4[100] bone_matrices;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform float objectAlpha;
uniform sampler2D texture1;
uniform sampler2D metallicMap;
uniform sampler2D normalMap;
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void fragment(vec3 in_FragPos, vec3 in_Normal, vec2 in_TexCoord) {
    FragPos = in_FragPos;
    Normal = in_Normal;
    TexCoord = in_TexCoord;
}

void main() {
{
    mat4 amodel = bone_matrices[aBoneID] * model;
    vec3 FragPos = (amodel * vec4(aAPos, 1.F)).xyz;
    vec3 Normal = normalize(((inverse(transpose(amodel))) * vec4(aANormal, 1.)).xyz);
    vec2 TexCoord = aATexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.F);
    fragment(FragPos, Normal, TexCoord);
}

}
)";
}

void init_uniforms(GLuint shader_program) override {
    uniform_setters["model"] = UniformSetter("model", shader_program);
    uniform_setters["view"] = UniformSetter("view", shader_program);
    uniform_setters["projection"] = UniformSetter("projection", shader_program);
    uniform_setters["projectionview"] = UniformSetter("projectionview", shader_program);
    uniform_setters["bone_matrices"] = UniformSetter("bone_matrices", shader_program);
    uniform_setters["lightPos"] = UniformSetter("lightPos", shader_program);
    uniform_setters["lightColor"] = UniformSetter("lightColor", shader_program);
    uniform_setters["objectColor"] = UniformSetter("objectColor", shader_program);
    uniform_setters["objectAlpha"] = UniformSetter("objectAlpha", shader_program);
    uniform_setters["texture1"] = UniformSetter("texture1", shader_program);
    uniform_setters["metallicMap"] = UniformSetter("metallicMap", shader_program);
    uniform_setters["normalMap"] = UniformSetter("normalMap", shader_program);
    };

};

#include "./vulkanupgrade.cpp"

