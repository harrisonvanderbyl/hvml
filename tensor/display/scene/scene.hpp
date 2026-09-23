#ifndef VULKAN_RENDERER_HPP
#define VULKAN_RENDERER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include "tensor.hpp"
#include "vector/vectors.hpp"
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

    int render_width, render_height;

    VulkanDisplay* current_display;
public:
    float32x3 light_position = float32x3(2.0f, 2.0f, 2.0f);
    float32x3 light_color = float32x3(1.0f, 1.0f, 1.0f);
    float32x3 object_color = float32x3(1.0f, 1.0f, 1.0f);
    float object_alpha = 1.0f;

    Camera camera;
    float time = 0.0f;

    ~Scene() {
        // Prevent double-deallocate by nulling shared storage_pointers
        try {
            for (auto &m : render_materials) {
                for (auto it = m.textures_ids.begin(); it != m.textures_ids.end(); ++it) {
                    it->second.data = nullptr;
                    it->second.storage_pointer = nullptr;
                }
            }
            for (auto it = default_material.textures_ids.begin(); it != default_material.textures_ids.end(); ++it) {
                it->second.data = nullptr;
                it->second.storage_pointer = nullptr;
            }
        } catch (...) {}
    }

    Scene(VulkanDisplay* display) : current_display(display)
    {
        render_width = display->width;
        render_height = display->height;

        // Don't create the default material's pipeline here — it has no textures
        // or vertex layout yet. The pipeline will be created lazily in bind().
        default_material.double_sided = true;

        std::cout << "Vulkan Renderer initialized successfully!" << std::endl;

        current_display->add_on_update(
            [this](CurrentScreenInputInfo &info, VkCommandBuffer cmd) {
                render(cmd);
            }
        );
    }

    bool loadGLTF(const gltf &model)
    {
        std::cout << "Loading GLTF model with " << model.meshes.size() << " meshes" << std::endl;

        for (const auto &texture : model.textures)
        {
            // Allocate as kVULKANTEXTURE — the vulkan plugin creates a VkImage+VkImageView
            // on the rendering device, and storage_pointer->data holds the VkImageView.
            MemoryType memType = g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR;
            render_textures.push_back(texture.to(memType, kVULKANTEXTURE));
        }

        for (const auto &material : model.materials)
        {
            Shader<BasicShader> render_material = Shader<BasicShader>();
            render_material.name = material.name;
            render_material.double_sided = material.doubleSided;

            // Set textures BEFORE createShaderProgram() so the pipeline layout includes texture bindings
            if (material.baseColorTextureIndex >= 0 && material.baseColorTextureIndex < render_textures.size())
            {
                render_material.textures_ids["texture1"] = render_textures[material.baseColorTextureIndex];
            }
            if (material.normalTextureIndex >= 0 && material.normalTextureIndex < render_textures.size())
            {
                render_material.textures_ids["normalMap"] = render_textures[material.normalTextureIndex];
            }
            if (material.metallicRoughnessTextureIndex >= 0 && material.metallicRoughnessTextureIndex < render_textures.size())
            {
                render_material.textures_ids["metallicMap"] = render_textures[material.metallicRoughnessTextureIndex];
            }
            // Don't create the shader program here — vertex binding descriptions
            // are set later in RenderStruct::bind(). The pipeline will be created
            // lazily in bind() with both textures and vertex layout available.
            render_materials.push_back(render_material);
        }

        for (const auto &mesh : model.meshes)
        {
            std::cout << "Processing mesh: " << mesh.name << std::endl;

            for (auto &primitive : mesh.primitives)
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
                MemoryType memType = g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR;
                RenderStruct render_mesh(
                    model.skeletons[0],
                    primitive.indices,
                    positions.to(memType),
                    normals.to(memType),
                    texcoords.to(memType),
                    bone_ids.to(memType));

                if (primitive.materialIndex >= 0 && primitive.materialIndex < model.materials.size())
                {
                    render_mesh.material = &render_materials[primitive.materialIndex];
                }
                else
                {
                    std::cerr << "Warning: Material index " << primitive.materialIndex << " out of range for mesh " << mesh.name << std::endl;
                    render_mesh.material = &default_material;
                }

                render_mesh.primitive_type = primitive.type;

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

    void setTransparency(float alpha)
    {
    }

    Camera &getCamera()
    {
        return camera;
    }

    void render(VkCommandBuffer cmd)
    {
        camera.aspect = (float)render_width / (float)render_height;

        time += 0.01f;

        for (auto &mesh : render_meshes)
        {
            // Set vertex layout on material and create buffers first
            mesh.bind(cmd);

            // Set all uniforms BEFORE bind() flushes them to the GPU
            camera.bind(*mesh.material);

            // Set bone matrices — shader does bone_matrices[boneID] * model
            auto& boneSetter = mesh.material->uniform_setters["bone_matrices"];
            if (boneSetter.initialized && !mesh.bone_matrices.storage_pointer) {
                // bone_matrices not set, fill with identity
                static mat4 identity[100];
                for (int i = 0; i < 100; i++) identity[i] = mat4::identity();
                boneSetter.data.resize(sizeof(mat4) * 100);
                memcpy(boneSetter.data.data(), identity, sizeof(mat4) * 100);
                boneSetter.dirty = true;
            } else if (boneSetter.initialized && mesh.bone_matrices.storage_pointer) {
                size_t boneCount = mesh.bone_matrices.shape[0];
                boneSetter.data.resize(sizeof(mat4) * 100);
                memset(boneSetter.data.data(), 0, sizeof(mat4) * 100);
                memcpy(boneSetter.data.data(), mesh.bone_matrices.storage_pointer->data,
                       sizeof(mat4) * boneCount);
                boneSetter.dirty = true;
            }

            mesh.material->uniform_setters["lightPos"] = light_position;
            mesh.material->uniform_setters["lightColor"] = light_color;
            mesh.material->uniform_setters["objectColor"] = object_color;
            mesh.material->uniform_setters["objectAlpha"] = object_alpha;

            // bind() creates pipeline if needed, flushes uniforms, binds pipeline + descriptor set
            mesh.material->bind(cmd);

            mesh.draw(cmd);
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Scene &scene)
    {
        os << "Scene with " << scene.render_meshes.size() << " meshes, " << scene.render_textures.size() << " textures, and " << scene.render_materials.size() << " materials.";
        os << "\nMeshes:\n";
        for (const auto &mesh : scene.render_meshes)        {
            os << mesh << "\n";
        }
        return os;
    }
};

#endif // VULKAN_RENDERER_HPP