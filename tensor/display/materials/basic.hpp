
#ifndef TENSOR_DISPLAY_MATERIALS_HPP
#define TENSOR_DISPLAY_MATERIALS_HPP
#include "display/materials/shader.hpp"

struct __shader BasicShader : public ShaderProgram
{
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 projectionview;
    mat4 bone_matrices[100]; // Array of bone matrices for skeletal animation
    float32x3 lightPos;
    float32x3 lightColor;
    float32x3 objectColor;
    float objectAlpha;
    sampler2D texture1;
    sampler2D metallicMap;
    sampler2D normalMap;
    // Shader source code

    void vertex(
        float32x3 aPos,
        float32x3 aNormal,
        float32x2 aTexCoord,
        int boneID // Assuming boneID is an integer attribute
    ){
        mat4 amodel = bone_matrices[boneID] * model ;
        float32x3 FragPos = (amodel * float32x4(aPos, 1.0f)).xyz();
        float32x3 Normal = normalize(((inverse(transpose(amodel))) * float32x4(aNormal,1.0)).xyz());
        float32x2 TexCoord = aTexCoord;

        gl_Position = projection * view * float32x4(FragPos, 1.0f);

        fragment(
            FragPos,
            Normal,
            TexCoord
        );
    }
    void fragment(
        float32x3 FragPos,
        float32x3 Normal,
        float32x2 TexCoord
    ){
        // use samplers to get texture colors
        uint84 diffuseColor = texture2D(texture1, TexCoord);
        uint84 specularColor = texture2D(metallicMap, TexCoord);
        // Ambient
        float32x3 ambient = diffuseColor.xyz() * 0.1;
        // Diffuse
        float32x3 norm = normalize(Normal);
        float32x3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0f);
        float32x3 diffuse = diffuseColor.xyz() * diff;
        // Specular
        float32x3 viewDir = normalize(-FragPos); // Assuming the camera is at the origin
        float32x3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 32);
        float32x3 specular = specularColor.xyz() * spec;
        float32x3 result = ambient + diffuse + specular;
        FragColor = uint84(result, objectAlpha);

    }    
};


#endif // TENSOR_DISPLAY_MATERIALS_HPP