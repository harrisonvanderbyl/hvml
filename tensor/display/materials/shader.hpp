#ifndef TENSOR_DISPLAY_SHADERBUILDER_HPP
#define TENSOR_DISPLAY_SHADERBUILDER_HPP

#include "tensor.hpp"
#include "ops/ops.hpp"
#include "vector/vectors.hpp"
#include "file_loaders/texture.hpp"
#include "display/vulkan_context.hpp"
#include <vulkan/vulkan.h>
#include <tuple>
#include <type_traits>
#include <cstring>
#include <array>

// ---------------------------------------------------------------------------
//  Global Vulkan context pointer — set by VulkanDisplay before any Material
//  or RenderStruct is used.
// ---------------------------------------------------------------------------

__weak VulkanContext* g_vk_ctx = nullptr;

// Pipeline cache: keyed by shader source hash
__weak std::map<std::string, VkPipeline> g_pipeline_cache;
__weak std::map<std::string, VkPipelineLayout> g_pipeline_layout_cache;
__weak std::map<std::string, VkDescriptorSetLayout> g_desc_set_layout_cache;

// Override render pass for offscreen rendering.  When non-null, pipelines are
// created against this render pass instead of the swapchain render pass.
__weak VkRenderPass g_override_render_pass = VK_NULL_HANDLE;

// ---------------------------------------------------------------------------
//  UniformSetter — stores uniform values into a staging buffer that gets
//  copied to the GPU uniform buffer before draw time.
// ---------------------------------------------------------------------------

struct UniformSetter
{
    std::string name;
    bool initialized = false;
    std::vector<uint8_t> data;
    size_t offset = 0;
    bool dirty = true;

    UniformSetter() : name(""), initialized(false) {};

    UniformSetter(std::string aname)
    {
        this->name = aname;
        this->initialized = true;
        this->dirty = true;
    };

    void operator=(const UniformSetter& other) {
        name = other.name;
        initialized = other.initialized;
        data = other.data;
        offset = other.offset;
        dirty = true;
    }

    void operator=(Hvec<float, 1> value)
    {
        set(value);
    }

    template <typename T, int size>
    void operator=(const Hvec<T, size>& value)
    {
        set(value);
    }

    template <int size>
    void operator=(const Shape<size>& value)
    {
        if (!initialized) return;
        data.resize(sizeof(int) * size);
        for (int i = 0; i < size; i++) {
            int v = value[i];
            memcpy(data.data() + i * sizeof(int), &v, sizeof(int));
        }
        dirty = true;
    }

    template <typename T, int size>
    void set(const Hvec<T, size>& value)
    {
        if (!initialized) return;
        data.resize(sizeof(T) * size);
        memcpy(data.data(), value.data, sizeof(T) * size);
        dirty = true;
    }
};

struct UniformManager
{
    std::map<std::string, UniformSetter> uniform_setters;
    bool initialized = false;

    UniformManager()
    {
    }

    UniformManager(bool)
    {
        initialized = true;
    }

    UniformSetter& operator[](const std::string& uniform_name) {
        if (uniform_setters.find(uniform_name) != uniform_setters.end())
        {
            return uniform_setters[uniform_name];
        }
        else
        {
            UniformSetter setter(uniform_name);
            uniform_setters[uniform_name] = setter;
            return uniform_setters[uniform_name];
        };
    }
};

struct Material
{
    virtual ~Material() {}

    virtual const char* getVertexShaderSource() {
        throw std::runtime_error("getVertexShaderSource not implemented for this material");
    };

    virtual const char* getFragmentShaderSource() {
        throw std::runtime_error("getFragmentShaderSource not implemented for this material");
    };

    virtual const char* getGeometryShaderSource() {
        return nullptr;
    };

    // Returns uniform field names in GLSL UBO declaration order.
    // If empty, createUniformBuffer falls back to alphabetical (map) order.
    virtual std::vector<std::string> getUniformOrder() { return {}; }

    // Returns the size in bytes for each uniform field (in the same order as getUniformOrder()).
    // If empty, createUniformBuffer falls back to data.size() or 64 bytes.
    virtual std::vector<size_t> getUniformSizes() { return {}; }

    // Vulkan pipeline objects
    VkPipeline       pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet   descriptorSet = VK_NULL_HANDLE;
    VkDescriptorPool  descriptorPool = VK_NULL_HANDLE;

    // Uniform buffer
    VkBuffer       uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    void*          uniformBufferMapped = nullptr;
    size_t         uniformBufferSize = 0;
    bool           uniformBufferDirty = true;

    std::string name;
    bool double_sided = false;
    bool transparent = false;
    std::map<std::string, Tensor<void,-1>> textures_ids;
    std::map<std::string, uint32_t> texture_types;  // 0=2D, 1=buffer
    UniformManager uniform_setters;

    std::map<std::string, uint32_t> texture_binding_map;
    uint32_t next_texture_binding = 0;

    std::map<std::string, size_t> uniform_offsets;
    size_t next_uniform_offset = 0;

    // Parse sampler binding numbers from GLSL source so texture_binding_map
    // matches the shader's layout(set=0, binding=N) declarations, not the
    // alphabetical order of std::map.
    void buildTextureBindingMapFromShaders() {
        texture_binding_map.clear();
        auto parseSource = [&](const char* src) {
            if (!src) return;
            std::string s(src);
            // Match: layout(set = 0, binding = N) uniform sampler2D name;
            // or:   layout(set = 0, binding = N) uniform samplerBuffer name;
            size_t pos = 0;
            while (pos < s.size()) {
                size_t layoutPos = s.find("layout(", pos);
                if (layoutPos == std::string::npos) break;
                size_t semi = s.find(';', layoutPos);
                if (semi == std::string::npos) break;
                std::string stmt = s.substr(layoutPos, semi - layoutPos);
                pos = semi + 1;

                if (stmt.find("sampler2D") == std::string::npos &&
                    stmt.find("samplerBuffer") == std::string::npos) continue;
                if (stmt.find("binding") == std::string::npos) continue;

                // Extract binding number
                size_t bPos = stmt.find("binding");
                size_t eqPos = stmt.find('=', bPos);
                if (eqPos == std::string::npos) continue;
                // Parse the number after =
                uint32_t binding = 0;
                int consumed = 0;
                if (sscanf(stmt.c_str() + eqPos + 1, " %u%n", &binding, &consumed) < 1) continue;

                // Extract sampler name — last identifier before ';'
                size_t nameEnd = stmt.size();
                // Skip trailing whitespace
                while (nameEnd > 0 && isspace((unsigned char)stmt[nameEnd-1])) nameEnd--;
                size_t nameStart = nameEnd;
                while (nameStart > 0 && (isalnum((unsigned char)stmt[nameStart-1]) || stmt[nameStart-1] == '_')) nameStart--;
                if (nameStart >= nameEnd) continue;
                std::string name = stmt.substr(nameStart, nameEnd - nameStart);

                texture_binding_map[name] = binding;
            }
        };
        parseSource(getVertexShaderSource());
        parseSource(getFragmentShaderSource());
    }

    // Vertex input layout (set by RenderStruct before pipeline creation)
    std::vector<VkVertexInputBindingDescription> vertexBindingDescs;
    std::vector<VkVertexInputAttributeDescription> vertexAttrDescs;

    bool createShaderProgram()
    {
        if (pipeline != VK_NULL_HANDLE) return true;
        if (!g_vk_ctx) {
            std::cerr << "[vulkan] No Vulkan context set! Call VulkanDisplay first." << std::endl;
            return false;
        }

        const char* vertex_shader_source = getVertexShaderSource();
        const char* fragment_shader_source = getFragmentShaderSource();
        const char* geometry_shader_source = getGeometryShaderSource();

        if (!vertex_shader_source || !vertex_shader_source[0]) {
            std::cerr << "[vulkan] Vertex shader source is empty!" << std::endl;
            return false;
        }
        if (!fragment_shader_source || !fragment_shader_source[0]) {
            std::cerr << "[vulkan] Fragment shader source is empty!" << std::endl;
            return false;
        }

        std::string shader_key = std::string(vertex_shader_source) + std::string(fragment_shader_source);
        if (geometry_shader_source) shader_key += geometry_shader_source;
        // Include vertex layout in cache key so different vertex layouts get different pipelines
        for (auto &b : vertexBindingDescs) shader_key += ":" + std::to_string(b.stride);
        for (auto &a : vertexAttrDescs) shader_key += "," + std::to_string(a.location) + ":" + std::to_string(a.format) + ":" + std::to_string(a.offset);
        // Include texture bindings in cache key so different texture layouts get different pipelines
        for (auto& [name, tex] : textures_ids) shader_key += "#" + name;
        for (auto& [name, type] : texture_types) shader_key += "^" + name + std::to_string(type);
        // Include double_sided so cull mode changes get separate pipelines
        shader_key += "&ds=" + std::to_string(double_sided ? 1 : 0);
        // Include render pass in cache key so offscreen vs swapchain get different pipelines
        shader_key += "@rp=" + std::to_string((size_t)(g_override_render_pass ? g_override_render_pass : g_vk_ctx->renderPass));

        if (g_pipeline_cache.find(shader_key) != g_pipeline_cache.end()) {
            pipeline = g_pipeline_cache[shader_key];
            pipelineLayout = g_pipeline_layout_cache[shader_key];
            descSetLayout = g_desc_set_layout_cache[shader_key];
            // Rebuild texture_binding_map from shader source so createDescriptorSet()
            // binds textures to the correct bindings matching the GLSL declarations.
            buildTextureBindingMapFromShaders();
            uniform_setters = UniformManager(true);
            return true;
        }

        // Compile shaders to SPIR-V
        std::vector<uint8_t> vertSpv = g_vk_ctx->compileGLSL(vertex_shader_source, "vert");
        std::vector<uint8_t> fragSpv = g_vk_ctx->compileGLSL(fragment_shader_source, "frag");

        if (vertSpv.empty() || fragSpv.empty()) {
            std::cerr << "[vulkan] Failed to compile shaders" << std::endl;
            return false;
        }

        VkShaderModule vertModule = g_vk_ctx->createShaderModule(vertSpv);
        VkShaderModule fragModule = g_vk_ctx->createShaderModule(fragSpv);

        VkShaderModule geomModule = VK_NULL_HANDLE;
        if (geometry_shader_source) {
            std::vector<uint8_t> geomSpv = g_vk_ctx->compileGLSL(geometry_shader_source, "geom");
            if (!geomSpv.empty()) {
                geomModule = g_vk_ctx->createShaderModule(geomSpv);
            }
        }

        createDescriptorSetLayout();

        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount = 1;
        layoutCI.pSetLayouts = &descSetLayout;
        VK_CTX_CHECK(vkCreatePipelineLayout(g_vk_ctx->device, &layoutCI, nullptr, &pipelineLayout));

        std::vector<VkPipelineShaderStageCreateInfo> stages;
        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertModule;
        vertStage.pName = "main";
        stages.push_back(vertStage);

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragModule;
        fragStage.pName = "main";
        stages.push_back(fragStage);

        if (geomModule) {
            VkPipelineShaderStageCreateInfo geomStage{};
            geomStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            geomStage.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
            geomStage.module = geomModule;
            geomStage.pName = "main";
            stages.push_back(geomStage);
        }

        VkPipelineVertexInputStateCreateInfo vertexInputState{};
        vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputState.vertexBindingDescriptionCount = (uint32_t)vertexBindingDescs.size();
        vertexInputState.pVertexBindingDescriptions = vertexBindingDescs.data();
        vertexInputState.vertexAttributeDescriptionCount = (uint32_t)vertexAttrDescs.size();
        vertexInputState.pVertexAttributeDescriptions = vertexAttrDescs.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)g_vk_ctx->swapchainExtent.width;
        viewport.height = (float)g_vk_ctx->swapchainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = g_vk_ctx->swapchainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = double_sided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = g_vk_ctx->msaaSamples;
        multisampling.sampleShadingEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = transparent ? VK_TRUE : VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = (uint32_t)dynamicStates.size();
        dynamicState.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.stageCount = (uint32_t)stages.size();
        pipelineCI.pStages = stages.data();
        pipelineCI.pVertexInputState = &vertexInputState;
        pipelineCI.pInputAssemblyState = &inputAssembly;
        pipelineCI.pViewportState = &viewportState;
        pipelineCI.pRasterizationState = &rasterizer;
        pipelineCI.pMultisampleState = &multisampling;
        pipelineCI.pDepthStencilState = &depthStencil;
        pipelineCI.pColorBlendState = &colorBlending;
        pipelineCI.pDynamicState = &dynamicState;
        pipelineCI.layout = pipelineLayout;
        pipelineCI.renderPass = g_override_render_pass ? g_override_render_pass : g_vk_ctx->renderPass;
        pipelineCI.subpass = 0;

        VK_CTX_CHECK(vkCreateGraphicsPipelines(g_vk_ctx->device, VK_NULL_HANDLE,
            1, &pipelineCI, nullptr, &pipeline));

        vkDestroyShaderModule(g_vk_ctx->device, vertModule, nullptr);
        vkDestroyShaderModule(g_vk_ctx->device, fragModule, nullptr);
        if (geomModule) vkDestroyShaderModule(g_vk_ctx->device, geomModule, nullptr);

        g_pipeline_cache[shader_key] = pipeline;
        g_pipeline_layout_cache[shader_key] = pipelineLayout;
        g_desc_set_layout_cache[shader_key] = descSetLayout;

        uniform_setters = UniformManager(true);
        return true;
    }

    void createDescriptorSetLayout() {
        std::vector<VkDescriptorSetLayoutBinding> bindings;

        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding = 0;
        uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(uboBinding);

        // Parse sampler bindings from shader source so they match the GLSL declarations
        buildTextureBindingMapFromShaders();

        for (auto& [name, tex] : textures_ids) {
            auto bindIt = texture_binding_map.find(name);
            uint32_t binding = (bindIt != texture_binding_map.end()) ? bindIt->second : 1;

            VkDescriptorSetLayoutBinding texBindingDesc{};
            texBindingDesc.binding = binding;
            auto typeIt = texture_types.find(name);
            if (typeIt != texture_types.end() && typeIt->second == 1) {
                texBindingDesc.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            } else {
                texBindingDesc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            }
            texBindingDesc.descriptorCount = 1;
            texBindingDesc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings.push_back(texBindingDesc);
        }

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = (uint32_t)bindings.size();
        ci.pBindings = bindings.data();
        VK_CTX_CHECK(vkCreateDescriptorSetLayout(g_vk_ctx->device, &ci, nullptr, &descSetLayout));
    }

    void createUniformBuffer() {
        uniform_offsets.clear();
        next_uniform_offset = 0;

        // Use getUniformOrder() if available (matches GLSL UBO field order)
        auto order = getUniformOrder();
        auto sizes = getUniformSizes();
        if (order.empty()) {
            for (auto& [name, setter] : uniform_setters.uniform_setters) {
                uniform_offsets[name] = next_uniform_offset;
                size_t dataSize = setter.data.size();
                if (dataSize == 0) dataSize = 64;
                next_uniform_offset += ((dataSize + 15) / 16) * 16;
            }
        } else {
            for (size_t i = 0; i < order.size(); i++) {
                const auto& name = order[i];
                uniform_offsets[name] = next_uniform_offset;
                size_t dataSize = 0;
                if (i < sizes.size()) dataSize = sizes[i];
                if (dataSize == 0) {
                    auto it = uniform_setters.uniform_setters.find(name);
                    dataSize = (it != uniform_setters.uniform_setters.end()) ? it->second.data.size() : 0;
                }
                if (dataSize == 0) dataSize = 64;
                next_uniform_offset += ((dataSize + 15) / 16) * 16;
            }
        }
        uniformBufferSize = std::max((size_t)16, next_uniform_offset);

        if (uniformBuffer != VK_NULL_HANDLE) return;

        g_vk_ctx->createBuffer(uniformBufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBuffer, uniformBufferMemory);

        vkMapMemory(g_vk_ctx->device, uniformBufferMemory, 0, uniformBufferSize, 0, &uniformBufferMapped);
    }

    void flushUniforms() {
        if (uniformBuffer == VK_NULL_HANDLE) createUniformBuffer();

        for (auto& [name, setter] : uniform_setters.uniform_setters) {
            if (!setter.dirty || setter.data.empty()) continue;
            auto it = uniform_offsets.find(name);
            if (it == uniform_offsets.end()) continue;
            memcpy((uint8_t*)uniformBufferMapped + it->second, setter.data.data(), setter.data.size());
            setter.dirty = false;
        }
        uniformBufferDirty = false;
    }

    void createDescriptorSet() {
        if (descriptorSet != VK_NULL_HANDLE) return;

        // Recreate descriptor set layout if textures have been added since
        // createShaderProgram() created the initial layout
        if (descSetLayout != VK_NULL_HANDLE) {
            uint32_t expectedBindings = 1 + (uint32_t)textures_ids.size();
            // Check if the current layout has enough bindings
            // (simple heuristic: if texture_binding_map size != textures_ids size, rebuild)
            if (texture_binding_map.size() != textures_ids.size()) {
                if (descSetLayout) vkDestroyDescriptorSetLayout(g_vk_ctx->device, descSetLayout, nullptr);
                if (pipelineLayout) vkDestroyPipelineLayout(g_vk_ctx->device, pipelineLayout, nullptr);
                descSetLayout = VK_NULL_HANDLE;
                pipelineLayout = VK_NULL_HANDLE;
                texture_binding_map.clear();
                // Destroy old pipeline since it references the old layout
                if (pipeline) vkDestroyPipeline(g_vk_ctx->device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
        }

        if (descSetLayout == VK_NULL_HANDLE) {
            createDescriptorSetLayout();

            // Recreate pipeline layout
            VkPipelineLayoutCreateInfo layoutCI{};
            layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutCI.setLayoutCount = 1;
            layoutCI.pSetLayouts = &descSetLayout;
            VK_CTX_CHECK(vkCreatePipelineLayout(g_vk_ctx->device, &layoutCI, nullptr, &pipelineLayout));
        }

        if (descSetLayout == VK_NULL_HANDLE) return;
        if (descSetLayout == VK_NULL_HANDLE) return;

        // Count buffer vs image textures
        size_t numImageTextures = 0;
        size_t numBufferTextures = 0;
        for (auto& [name, tex] : textures_ids) {
            auto typeIt = texture_types.find(name);
            if (typeIt != texture_types.end() && typeIt->second == 1)
                numBufferTextures++;
            else
                numImageTextures++;
        }

        std::vector<VkDescriptorPoolSize> poolSizes;
        VkDescriptorPoolSize uboPool{};
        uboPool.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboPool.descriptorCount = 1;
        poolSizes.push_back(uboPool);
        if (numImageTextures > 0) {
            VkDescriptorPoolSize imgPool{};
            imgPool.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            imgPool.descriptorCount = numImageTextures;
            poolSizes.push_back(imgPool);
        }
        if (numBufferTextures > 0) {
            VkDescriptorPoolSize bufPool{};
            bufPool.type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            bufPool.descriptorCount = numBufferTextures;
            poolSizes.push_back(bufPool);
        }

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets = 1;
        poolCI.poolSizeCount = (uint32_t)poolSizes.size();
        poolCI.pPoolSizes = poolSizes.data();
        VK_CTX_CHECK(vkCreateDescriptorPool(g_vk_ctx->device, &poolCI, nullptr, &descriptorPool));

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &descSetLayout;
        VK_CTX_CHECK(vkAllocateDescriptorSets(g_vk_ctx->device, &ai, &descriptorSet));

        if (uniformBuffer == VK_NULL_HANDLE) createUniformBuffer();

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uniformBuffer;
        uboInfo.offset = 0;
        uboInfo.range = uniformBufferSize;

        VkWriteDescriptorSet uboWrite{};
        uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        uboWrite.dstSet = descriptorSet;
        uboWrite.dstBinding = 0;
        uboWrite.dstArrayElement = 0;
        uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboWrite.descriptorCount = 1;
        uboWrite.pBufferInfo = &uboInfo;

        std::vector<VkWriteDescriptorSet> writes = {uboWrite};
        std::vector<VkDescriptorImageInfo> imageInfos;
        std::vector<VkSampler> samplers;
        std::vector<VkBufferView> bufferViews;
        // Reserve to prevent reallocation — pImageInfo/pTexelBufferView pointers
        // in writes would dangle if the vectors grow.
        imageInfos.reserve(textures_ids.size());
        bufferViews.reserve(textures_ids.size());

        for (auto& [name, tex] : textures_ids) {
            auto bindIt = texture_binding_map.find(name);
            if (bindIt == texture_binding_map.end()) continue;

            auto typeIt = texture_types.find(name);
            bool isBuffer = (typeIt != texture_types.end() && typeIt->second == 1);

            if (isBuffer) {
                // Buffer texture (samplerBuffer) — use VkBufferView
                VkBufferView bufView = (VkBufferView)(size_t)tex.storage_pointer->data;
                if (bufView == VK_NULL_HANDLE) continue;
                bufferViews.push_back(bufView);

                VkWriteDescriptorSet texWrite{};
                texWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                texWrite.dstSet = descriptorSet;
                texWrite.dstBinding = bindIt->second;
                texWrite.dstArrayElement = 0;
                texWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                texWrite.descriptorCount = 1;
                texWrite.pTexelBufferView = &bufferViews.back();
                writes.push_back(texWrite);
            } else {
                // Image texture (sampler2D) — use VkImageView + sampler
                VkImageView imageView = (VkImageView)(size_t)tex.storage_pointer->data;
                if (imageView == VK_NULL_HANDLE) continue;

                VkSampler sampler = VK_NULL_HANDLE;
                VkSamplerCreateInfo sci{};
                sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                sci.magFilter = VK_FILTER_LINEAR;
                sci.minFilter = VK_FILTER_LINEAR;
                sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                sci.anisotropyEnable = VK_TRUE;
                sci.maxAnisotropy = 16;
                sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
                sci.unnormalizedCoordinates = VK_FALSE;
                sci.compareEnable = VK_FALSE;
                sci.compareOp = VK_COMPARE_OP_ALWAYS;
                sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                VK_CTX_CHECK(vkCreateSampler(g_vk_ctx->device, &sci, nullptr, &sampler));
                samplers.push_back(sampler);

                VkDescriptorImageInfo imgInfo{};
                imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imgInfo.imageView = imageView;
                imgInfo.sampler = sampler;
                imageInfos.push_back(imgInfo);

                VkWriteDescriptorSet texWrite{};
                texWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                texWrite.dstSet = descriptorSet;
                texWrite.dstBinding = bindIt->second;
                texWrite.dstArrayElement = 0;
                texWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                texWrite.descriptorCount = 1;
                texWrite.pImageInfo = &imageInfos.back();
                writes.push_back(texWrite);
            }
        }

        vkUpdateDescriptorSets(g_vk_ctx->device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    void bind(VkCommandBuffer cmd = VK_NULL_HANDLE)
    {
        if (pipeline == VK_NULL_HANDLE) {
            if (!createShaderProgram()) return;
        }

        if (uniformBuffer == VK_NULL_HANDLE) createUniformBuffer();
        if (descriptorSet == VK_NULL_HANDLE) createDescriptorSet();

        flushUniforms();

        if (cmd != VK_NULL_HANDLE) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        }
    }
};



template <typename T>
struct VertexAttribute
{
    static constexpr int size = 0;
    static constexpr VkFormat format = VK_FORMAT_UNDEFINED;
    static constexpr VkBool32 normalized = VK_FALSE;
};

template <>
struct VertexAttribute<float32x3>
{
    static constexpr int size = 3;
    static constexpr VkFormat format = VK_FORMAT_R32G32B32_SFLOAT;
    static constexpr VkBool32 normalized = VK_FALSE;
};
template <>
struct VertexAttribute<float32x2>
{
    static constexpr int size = 2;
    static constexpr VkFormat format = VK_FORMAT_R32G32_SFLOAT;
    static constexpr VkBool32 normalized = VK_FALSE;
};
template <>
struct VertexAttribute<float32x4>
{
    static constexpr int size = 4;
    static constexpr VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    static constexpr VkBool32 normalized = VK_FALSE;
};
template <>
struct VertexAttribute<int>
{
    static constexpr int size = 1;
    static constexpr VkFormat format = VK_FORMAT_R32_SINT;
    static constexpr VkBool32 normalized = VK_FALSE;
};
template <>
struct VertexAttribute<uint84>
{
    static constexpr int size = 4;
    static constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    static constexpr VkBool32 normalized = VK_TRUE;
};
template <>
struct VertexAttribute<float>
{
    static constexpr int size = 1;
    static constexpr VkFormat format = VK_FORMAT_R32_SFLOAT;
    static constexpr VkBool32 normalized = VK_FALSE;
};

template <>
struct VertexAttribute<uint16_t>
{
    static constexpr int size = 1;
    static constexpr VkFormat format = VK_FORMAT_R16_UINT;
    static constexpr VkBool32 normalized = VK_FALSE;
};

template <>
struct VertexAttribute<uint32_t>
{
    static constexpr int size = 1;
    static constexpr VkFormat format = VK_FORMAT_R32_UINT;
    static constexpr VkBool32 normalized = VK_FALSE;
};

template <typename T>
struct VertexAttributeToGLType {
    static constexpr const char* get() {
        using A = VertexAttribute<T>;

        if constexpr (A::format == VK_FORMAT_R32_SFLOAT) {
            return "float";
        }
        if constexpr (A::format == VK_FORMAT_R32G32_SFLOAT) {
            return "vec2";
        }
        if constexpr (A::format == VK_FORMAT_R32G32B32_SFLOAT) {
            return "vec3";
        }
        if constexpr (A::format == VK_FORMAT_R32G32B32A32_SFLOAT) {
            return "vec4";
        }
        if constexpr (A::format == VK_FORMAT_R32_SINT) {
            return "int";
        }
        if constexpr (A::format == VK_FORMAT_R8G8B8A8_UNORM) {
            return "vec4";
        }
        if constexpr (A::format == VK_FORMAT_R16_UINT) {
            return "uint16_t";
        }
        if constexpr (A::format == VK_FORMAT_R32_UINT) {
            return "uint";
        }
        return "unknown";
    }
};







template <typename... Ts>
struct mytuple
{
    // tuple implementation with __device__ and __host__ constructors
    uint8_t data[(sizeof(Ts) + ... + 0)] = {};
    __host__ __device__ mytuple(Ts... args)
    {
        size_t offset = 0;
        ((*(Ts *)(data + offset) = args, offset += sizeof(Ts)), ...);
    }

    __host__ __device__ mytuple()
    {
        // default constructor
    }

    friend std::ostream &operator<<(std::ostream &os, const mytuple &t)
    {
        os << "mytuple(";
        size_t offset = 0;
        ((os << (offset == 0 ? "" : ", ") << *reinterpret_cast<const Ts *>(t.data + offset), offset += sizeof(Ts)), ...);
        os << ")";
        return os;
    }
};


template <typename... Args>
struct VertexLayout : public mytuple<Args...>
{
    static constexpr int num_attributes = sizeof...(Args);
    static constexpr std::array<VkFormat, num_attributes> formats = {VertexAttribute<Args>::format...};
    static constexpr std::array<int, num_attributes> sizes = {VertexAttribute<Args>::size...};
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

struct IntToUnsignedLongConverter : public HardamardOperation<IntToUnsignedLongConverter>
{
    __host__ __device__ static inline void apply(const int &val, unsigned long &out)
    {
        out = static_cast<unsigned long>(val);
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


    float32x3 reflect(const float32x3 &I, const float32x3 &N); 

    mat4 inverse(const mat4 &m);

    mat4 transpose(const mat4 &m) {
        return mat4::identity(); // Placeholder
    }

    template <typename T>
    float length(const T &v);

    template <typename T>
    T max(const T &v, const T &other);

    template <typename T>
    T min(const T &v, const T &other);

    template <typename T>
    float dot(const T &a, const T &b);

    template <typename T>
    T normalize(const T &v);

    template <typename T>
    T abs(const T &v);

    template <typename T>
    T floor(const T &v);

    template <typename T>
    T mix(const T &a, const T &b, float t);
    
    uint84 texture2D(const sampler2D &sampler, const float32x2 &uv){
        return 0;
    };

    uint84 texture(const sampler2D &sampler, const float32x2 &uv){
        return 0;
    };

    uint84 texelFetch(const samplerBuffer &sampler, const int &uv){
        return 0;
    }

    template <typename T>
    T mod(const T &x, const T &y);

    template <typename T>
    T asin(const T &x);
    

    template <typename T>
    T atan(const T &y, const T &x);

    template <typename T>
    T acos(const T &x);

    template <typename T>
    T cos(const T &x);

    template <typename T>
    T sin(const T &x);
    // no template named function traits
    template <typename T, typename TT, typename TTT>
    T clamp(const T &x, const TT &minVal, const TTT &maxVal);
  
};



#endif // TENSOR_DISPLAY_SHADERBUILDER_HPP