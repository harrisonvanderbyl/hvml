#ifndef DRAWABLE_HPP
#define DRAWABLE_HPP
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "file_loaders/gltf.hpp"
#include "ops/ops.hpp"
#include "display/materials/materials.hpp"


// Primitive topology values (shared with gltf.hpp)
// VK_TOPOLOGY_POINTS=0, VK_TOPOLOGY_LINES=1, ..., VK_TOPOLOGY_TRIANGLE_FAN=6

template <typename... vertex_types>
struct RenderStruct : Tensor<mytuple<vertex_types...>, 1>
{
    int primitive_type = VK_TOPOLOGY_POINTS;
    mat4 model_matrix = mat4::identity();
    Material* material = nullptr;
    Skeleton bone_matrices;

    // Vulkan resources
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    Tensor<int, 1> indices;
    int offset = 0;
    int count = -1;
    bool buffersCreated = false;

    RenderStruct() : Tensor<mytuple<vertex_types...>, 1>() {}

    VkPrimitiveTopology getTopology() const {
        switch (primitive_type) {
            case VK_TOPOLOGY_POINTS:         return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            case VK_TOPOLOGY_LINES:           return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            case VK_TOPOLOGY_LINE_STRIP:      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            case VK_TOPOLOGY_TRIANGLES:       return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            case VK_TOPOLOGY_TRIANGLE_STRIP:  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            case VK_TOPOLOGY_TRIANGLE_FAN:    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
            default:                          return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        }
    }

    void createVulkanBuffers()
    {
        if (buffersCreated) return;
        if (!g_vk_ctx) {
            std::cerr << "[vulkan] No context for buffer creation" << std::endl;
            return;
        }

        // When allocated with kVULKAN, storage_pointer->data is a VulkanBufferHandle*
        // (first field is VkBuffer) created by the vulkan plugin's kVULKAN allocator.
        if (this->storage_pointer && this->storage_pointer->data) {
            vertexBuffer = *(VkBuffer*)this->storage_pointer->data;
        }

        // Index buffer — also allocated with kVULKAN
        if (indices.storage_pointer && indices.storage_pointer->data) {
            indexBuffer = *(VkBuffer*)indices.storage_pointer->data;
        }

        buffersCreated = true;
    }

    RenderStruct(Tensor<vertex_types, 1>... tensors) : Tensor<mytuple<vertex_types...>, 1>(
        std::get<0>(std::make_tuple(tensors...)).shape,
        g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR,
        kVULKAN)
    {
        CopyDataHelper<vertex_types...>::run(tensors.to(*this->device)..., this->to_compute(this->device->default_compute_type));
    }

    RenderStruct(Shape<1> shape) : Tensor<mytuple<vertex_types...>, 1>(
            shape,
            g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR,
            kVULKAN)
    {
    }

    RenderStruct(Shape<1> shape, Tensor<int, 1> inindices) : Tensor<mytuple<vertex_types...>, 1>(
            shape,
            g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR,
            kVULKAN),
        indices(inindices.to(g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR, kVULKAN))
    {
    }

    RenderStruct(Skeleton bones, Tensor<int, 1> inindices, Tensor<vertex_types, 1>... inputs) : Tensor<mytuple<vertex_types...>, 1>(
        std::get<0>(std::make_tuple(inputs.shape...)), g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR, kVULKAN), bone_matrices(bones)
    {
        indices = inindices.to(g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR, kVULKAN);
        this->device->synchronize_function();

        CopyDataHelper<vertex_types...>::run(inputs.to(*this->device)..., this->to_compute(this->device->default_compute_type));
        this->primitive_type = VK_TOPOLOGY_TRIANGLES;
    }

    RenderStruct(const RenderStruct<vertex_types...>& other, Tensor<int, 1> inindices) : Tensor<mytuple<vertex_types...>, 1>(other), primitive_type(other.primitive_type), model_matrix(other.model_matrix), material(other.material), bone_matrices(other.bone_matrices), indices(inindices.to(g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR, kVULKAN))
    {
    }

    void draw(VkCommandBuffer cmd = VK_NULL_HANDLE) const
    {
        if (material == nullptr) {
            std::cerr << "No material assigned to RenderStruct, cannot draw!" << std::endl;
            return;
        }

        if (cmd == VK_NULL_HANDLE) return;

        if (!buffersCreated) {
            const_cast<RenderStruct*>(this)->createVulkanBuffers();
        }

        // Bind vertex buffer
        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

        // Bind index buffer and draw indexed, or draw arrays
        if (indexBuffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            if (count > 0) {
                vkCmdDrawIndexed(cmd, count, 1, offset, 0, 0);
            } else {
                vkCmdDrawIndexed(cmd, indices.shape[0], 1, 0, 0, 0);
            }
        } else {
            if (count > 0) {
                vkCmdDraw(cmd, count, 1, offset, 0);
            } else {
                vkCmdDraw(cmd, this->shape[0], 1, 0, 0);
            }
        }
    }

    void bind(VkCommandBuffer cmd = VK_NULL_HANDLE)
    {
        if (material == nullptr) {
            std::cerr << "No material assigned to RenderStruct, cannot bind!" << std::endl;
            return;
        }

        // Set vertex input layout on the material before pipeline creation
        if (material->vertexBindingDescs.empty()) {
            using VLayout = VertexLayout<vertex_types...>;
            VkVertexInputBindingDescription bindingDesc{};
            bindingDesc.binding = 0;
            bindingDesc.stride = sizeof(mytuple<vertex_types...>);
            bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            material->vertexBindingDescs.push_back(bindingDesc);

            size_t offset = 0;
            for (int i = 0; i < VLayout::num_attributes; i++) {
                VkVertexInputAttributeDescription attrDesc{};
                attrDesc.binding = 0;
                attrDesc.location = i;
                attrDesc.format = VLayout::formats[i];
                attrDesc.offset = offset;
                offset += VLayout::attribute_sizes[i];
                material->vertexAttrDescs.push_back(attrDesc);
            }
        }

        material->bind(cmd);
        material->uniform_setters["model"] = model_matrix;
    }
};

#endif