#ifndef OPENGL_RENDERER_HPP
#define OPENGL_RENDERER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include "tensor.hpp"
#include "vector/vectors.hpp"
// #include "../vector/uint84.hpp"
#include "file_loaders/gltf.hpp"
#include "display/display.hpp"
#include "ops/ops.hpp"
#include "display/materials/materials.hpp"
#include "display/drawable/drawable.hpp"


class Scene
{
private:
    Shader<BasicShader> default_material = Shader<BasicShader>();
    std::vector<RenderStruct<float32x3, float32x3, float32x2, int>> render_meshes;
    std::vector<sampler2D> render_textures;
    std::vector<Shader<BasicShader>> render_materials;

    // Framebuffer for off-screen rendering
    int render_width, render_height;

    OpenGLDisplay* current_display;
public:
    Camera camera;
    // Platform-specific transparency setup
    float time = 0.0f;
    Scene(OpenGLDisplay* display) : current_display(display)
    {

        render_width = display->width;
        render_height = display->height;
        // Enable depth testing

        // Create default shader program for the scene
        if (!default_material.createShaderProgram())
        {
            std::cerr << "Failed to create default shader program!" << std::endl;
            return;
        }

        std::cout << "OpenGL Renderer initialized successfully!" << std::endl;

        current_display->add_on_update(
            [this](CurrentScreenInputInfo &info) {
                render();
            }
        );

    }

    

    bool loadGLTF(const gltf &model)
    {
        std::cout << "Loading GLTF model with " << model.meshes.size() << " meshes" << std::endl;

        // load textures
        for (const auto &texture : model.textures)
        {

            render_textures.push_back(texture.to(kCUDA_VRAM, kOPENGLTEXTURE));
            // std::cout << "Loaded texture: with ID: " << render_texture.texture << std::endl;
        }

        for (const auto &material : model.materials)
        {
            Shader<BasicShader> render_material = Shader<BasicShader>();
            render_material.name = material.name;
            render_material.double_sided = material.doubleSided;
            
            // Create shader program for each material
            if (!render_material.createShaderProgram())
            {
                std::cerr << "Failed to create shader program for material: " << material.name << std::endl;
                continue;
            }
            
            if (material.baseColorTextureIndex >= 0 && material.baseColorTextureIndex < render_textures.size())
            {
                auto glint = (GLuint)(unsigned long long)render_textures[material.baseColorTextureIndex].storage_pointer - 0x10000;
                render_material.textures_ids["texture1"] = glint;
            }
            if (material.normalTextureIndex >= 0 && material.normalTextureIndex < render_textures.size())
            {
                auto glint = (GLuint)(unsigned long long)render_textures[material.normalTextureIndex].storage_pointer - 0x10000;
                render_material.textures_ids["normalMap"] = glint;
            }
            if (material.metallicRoughnessTextureIndex >= 0 && material.metallicRoughnessTextureIndex < render_textures.size())
            {
                auto glint = (GLuint)(unsigned long long)render_textures[material.metallicRoughnessTextureIndex].storage_pointer - 0x10000;
                render_material.textures_ids["metallicMap"] = glint;
            }
            render_materials.push_back(render_material);
        }

        for (const auto &mesh : model.meshes)
        {
            std::cout << "Processing mesh: " << mesh.name << std::endl;

            for (const auto &primitive : mesh.primitives)
            {

                auto pos_it = primitive.attributes.find("POSITION");
                Tensor<float32x3, 1> positions = pos_it->second;
                auto norm_it = primitive.attributes.find("NORMAL");
                Tensor<float32x3, 1> normals = norm_it->second;
                auto tex_it = primitive.attributes.find("TEXCOORD_0");
                Tensor<float32x2, 1> texcoords = tex_it->second;
                auto joint_it = primitive.attributes.find("bone_ids");
                Tensor<int, 1> bone_ids = joint_it->second;
                std::cout << primitive.attributes.size() << std::endl;
                for (const auto &attr : primitive.attributes)
                {
                    std::cout << "Attribute: " << attr.first << " Shape: " << attr.second.shape << std::endl;
                }
                RenderStruct render_mesh(
                    model.skeletons[0],
                    primitive.indices,
                    positions.to(kCUDA_VRAM),
                    normals.to(kCUDA_VRAM),
                    texcoords.to(kCUDA_VRAM),
                    bone_ids.to(kCUDA_VRAM));

                // Process material
                if (primitive.materialIndex >= 0 && primitive.materialIndex < model.materials.size())
                {
                    render_mesh.material = &render_materials[primitive.materialIndex];
                }
                else
                {
                    std::cerr << "Warning: Material index " << primitive.materialIndex << " out of range for mesh " << mesh.name << std::endl;
                    render_mesh.material = &default_material; // Use default material
                }

                render_mesh.primitive_type = primitive.type; // Default to triangles

                GLFuncs->glBindVertexArray(0);

                render_meshes.push_back(render_mesh);
                std::cout << "Created render mesh with " << render_mesh.indices.shape << " indices" << std::endl;
            }
        }

        return true;
    }

    void setCamera(const float32x3 &position, const float32x3 &target, const float32x3 &up = float32x3(0, 1, 0))
    {
        camera.position = position;
        camera.forward = target;
        camera.up = up;
    }

    // Add method to control transparency dynamically
    void setTransparency(float alpha)
    {
        // This will be used in the next render call
        // You can store this as a member variable if needed
    }

    Camera &getCamera()
    {
        return camera;
    }

    void render()
    {

        camera.aspect = (float)render_width / (float)render_height;

        glViewport(0, 0, render_width, render_height);

        time += 0.01f; // Increment time for animation
        // Clear the screen

        float lightPos[] = {2.0f, 2.0f, 2.0f};
        float lightColor[] = {1.0f, 1.0f, 1.0f};
        float objectColor[] = {1.0, 1.0f, 1.0f};
        float objectAlpha = 1.0f; // Semi-transparent for desktop pet


        // Render all meshes
        for (auto &mesh : render_meshes)
        {
            // Bind material's shader
            mesh.bind();


            camera.bind(mesh.material->shader_program);
            
            GLint meshLightPosLoc = GLFuncs->glGetUniformLocation(mesh.material->shader_program, "lightPos");
            GLint meshLightColorLoc = GLFuncs->glGetUniformLocation(mesh.material->shader_program, "lightColor");
            GLint meshObjectColorLoc = GLFuncs->glGetUniformLocation(mesh.material->shader_program, "objectColor");
            GLint meshObjectAlphaLoc = GLFuncs->glGetUniformLocation(mesh.material->shader_program, "objectAlpha");

            GLFuncs->glUniform3fv(meshLightPosLoc, 1, lightPos);
            GLFuncs->glUniform3fv(meshLightColorLoc, 1, lightColor);
            GLFuncs->glUniform3fv(meshObjectColorLoc, 1, objectColor);
            GLFuncs->glUniform1f(meshObjectAlphaLoc, objectAlpha);

            // Draw mesh
            mesh.draw();
        }

        GLFuncs->glBindVertexArray(0);

        // Render cube points with default material
        
    }
};

#endif // OPENGL_RENDERER_HPP