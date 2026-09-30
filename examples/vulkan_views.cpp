// vulkan_views.cpp — render textures, buffer textures and in-place views.
//
// Build (from the repo root; plugins in tensor/device/plugins):
//   g++-14 -std=c++20 -I./tensor examples/vulkan_views.cpp -o vulkan_views \
//       -lSDL3 -lvulkan -ldl -rdynamic
//   DEVICE_PLUGIN_DIR=tensor/device/plugins ./vulkan_views            # window
//   DEVICE_PLUGIN_DIR=tensor/device/plugins ./vulkan_views --headless # checks only
//
// Every check below uses ONE allocation seen through different views —
// no copies between buffer, texel buffer, texture and render target.

#include "display/display.hpp"
#include <cstring>

// A material with hand-written GLSL: fills the target with a solid colour
// from a uniform (a full-screen triangle, no vertex buffer).
struct FillMaterial : public Material {
    FillMaterial() { depth_test = false; depth_write = false; double_sided = true; }
    const char* getVertexShaderSource() override {
        return R"(#version 450
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
})";
    }
    const char* getFragmentShaderSource() override {
        return R"(#version 450
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 0) uniform UBO { vec4 color; } ubo;
void main() { FragColor = ubo.color; })";
    }
    std::vector<std::string> getUniformOrder() override { return {"color"}; }
    std::vector<size_t> getUniformSizes() override { return {16}; }

    void fill(VkCommandBuffer cmd, float r, float g, float b, float a) {
        uniform_setters["color"] = float32x4(r, g, b, a);
        bind(cmd);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
};

static int failures = 0;

static void check(bool ok, const char* what) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << std::endl;
    if (!ok) failures++;
}

static bool pixel_is(const Tensor<uint84, 2>& img, size_t i, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    const uint8_t* p = (const uint8_t*)img.data.data + i * 4;
    auto near = [](int x, int y) { return std::abs(x - y) <= 1; };
    bool ok = near(p[0], r) && near(p[1], g) && near(p[2], b) && near(p[3], a);
    if (!ok) std::cout << "       pixel " << i << " = (" << (int)p[0] << "," << (int)p[1] << ","
                       << (int)p[2] << "," << (int)p[3] << ")" << std::endl;
    return ok;
}

static void run_checks() {
    VulkanContext& ctx = VulkanContext::get();
    const long W = 64, H = 32;
    FillMaterial fill;

    // 1. Render texture (optimal image, kTEXTURE | kSURFACE): render, read back.
    {
        DisplayTensor<uint84> rt({W, H});
        ctx.submit([&](VkCommandBuffer cmd) {
            rt.render(cmd, [&] { fill.fill(cmd, 1.0f, 0.0f, 0.0f, 1.0f); });
        });
        auto px = rt.read();
        check(pixel_is(px, 0, 255, 0, 0, 255) && pixel_is(px, W * H - 1, 255, 0, 0, 255),
              "render texture: rendered red, read back red");
    }

    // 2. Buffer texture (kVULKAN + kTEXELBUFFER): write from the CPU, blit it
    //    through its samplerBuffer view into a render texture.
    {
        DisplayTensor<uint84> buf({W, H}, kVULKAN);
        Tensor<uint84, 2> host({W, H}, MemoryType::kDDR);
        memset(host.data.data, 0, host.total_bytes);
        for (long i = 0; i < W * H; i++) {
            ((uint8_t*)host.data.data)[i * 4 + 1] = 200;
            ((uint8_t*)host.data.data)[i * 4 + 3] = 255;
        }
        buf.write(host);

        VulkanResource* tb = vk_resource(buf.texel_buffer());
        check(tb && tb->buffer_view, "buffer texture: texel_buffer() view has a VkBufferView");

        DisplayTensor<uint84> out({W, H});
        ctx.submit([&](VkCommandBuffer cmd) {
            out.render(cmd, [&] { buf.draw(cmd); });
        });
        auto px = out.read();
        check(pixel_is(px, 5, 0, 200, 0, 255), "buffer texture: blit via samplerBuffer reaches render texture");
    }

    // 3. The SAME buffer viewed in place as a sampler2D (linear image alias).
    {
        DisplayTensor<uint84> buf({W, H}, kVULKAN);
        Tensor<uint84, 2> host({W, H}, MemoryType::kDDR);
        for (long i = 0; i < W * H; i++) {
            uint8_t* p = (uint8_t*)host.data.data + i * 4;
            p[0] = 10; p[1] = 20; p[2] = 250; p[3] = 255;
        }
        buf.write(host);

        auto tex = buf.texture();                       // in place — no copy
        VulkanResource* base = buf.resource();
        VulkanResource* view = vk_resource(tex);
        check(view && view->image_view && view->memory == base->memory,
              "buffer → texture view shares the buffer's VkDeviceMemory");

        FillMaterial unused;
        struct SampleMaterial : public Material {
            SampleMaterial() { depth_test = false; depth_write = false; double_sided = true; sampler_nearest = true; }
            const char* getVertexShaderSource() override {
                return R"(#version 450
layout(location = 0) out vec2 uv;
void main() { vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2); uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); })";
            }
            const char* getFragmentShaderSource() override {
                return R"(#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 1) uniform sampler2D img;
void main() { FragColor = texture(img, uv); })";
            }
        } sample;
        sample.setTexture("img", buf);                  // plain kVULKAN tensor: converted to a sampler2D view

        DisplayTensor<uint84> out({W, H});
        ctx.submit([&](VkCommandBuffer cmd) {
            out.render(cmd, [&] { sample.bind(cmd); vkCmdDraw(cmd, 3, 1, 0, 0); });
        });
        check(pixel_is(out.read(), 7, 10, 20, 250, 255), "buffer sampled as sampler2D through its in-place image view");
    }

    // 4. kLINEAR render texture: render into it, then read the pixels through
    //    its buffer view (same memory).
    {
        DisplayTensor<uint84> lin({W, H}, kVULKANTEXTURE,
                                  AllocationFlags::kTEXTURE | AllocationFlags::kSURFACE | AllocationFlags::kLINEAR);
        ctx.submit([&](VkCommandBuffer cmd) {
            lin.render(cmd, [&] { fill.fill(cmd, 0.0f, 0.0f, 1.0f, 1.0f); });
        });
        auto asBuffer = lin.buffer();
        VulkanResource* br = vk_resource(asBuffer);
        check(br && br->buffer && br->memory == lin.resource()->memory,
              "linear texture → buffer view shares memory");

        // Read through the *buffer* view's handle
        Tensor<uint84, 2> host({W, H}, MemoryType::kDDR);
        auto download = (hvml_vk_download_fn)dlsym(RTLD_DEFAULT, "hvml_vk_download");
        download(br, host.data.data, host.total_bytes);
        check(pixel_is(host, 100, 0, 0, 255, 255), "pixels rendered into the texture are visible through the buffer view");

        // And as a texel buffer
        VulkanResource* tb = vk_resource(lin.texel_buffer());
        check(tb && tb->buffer_view, "linear texture → texel buffer view");

        // CPU pointer (host-visible memory on integrated/software GPUs)
        if (ctx.interopComputeType() == kCPU) {
            auto cpu = lin.compute();
            vkQueueWaitIdle(ctx.graphicsQueue);
            check(cpu.data.data != (uint84*)lin.resource() && pixel_is(cpu, 3, 0, 0, 255, 255),
                  "linear texture → CPU pointer to the same pixels");
        }
    }

    // 5. A buffer used directly as a render target (kVULKAN + kSURFACE → linear alias)
    {
        DisplayTensor<uint84> target({W, H}, kVULKAN,
                                     AllocationFlags::kTEXELBUFFER | AllocationFlags::kSURFACE);
        target.use_depth = false;
        ctx.submit([&](VkCommandBuffer cmd) {
            target.render(cmd, [&] { fill.fill(cmd, 0.0f, 1.0f, 1.0f, 1.0f); });
        });
        check(pixel_is(target.read(), 42, 0, 255, 255, 255), "buffer rendered into directly (kVULKAN | kSURFACE)");
    }

    // 6. OverLayShader blit keeps row 0 at the top for both paths
    {
        Tensor<uint84, 2> host({W, H}, MemoryType::kDDR);
        for (long i = 0; i < W * H; i++) {
            uint8_t* p = (uint8_t*)host.data.data + i * 4;
            bool firstRow = i < W;
            p[0] = firstRow ? 255 : 0; p[1] = 0; p[2] = firstRow ? 0 : 255; p[3] = 255;
        }
        DisplayTensor<uint84> asBuffer({W, H}, kVULKAN);
        DisplayTensor<uint84> asTexture({W, H});
        asBuffer.write(host);
        asTexture.write(host);
        for (auto* src : {&asBuffer, &asTexture}) {
            DisplayTensor<uint84> out({W, H});
            ctx.submit([&](VkCommandBuffer cmd) { out.render(cmd, [&] { src->draw(cmd); }); });
            auto px = out.read();
            check(pixel_is(px, 3, 255, 0, 0, 255) && pixel_is(px, W * H - 3, 0, 0, 255, 255),
                  src == &asBuffer ? "OverLayShader<true> (samplerBuffer) blit: row 0 at top"
                                   : "OverLayShader<false> (sampler2D) blit: row 0 at top");
        }
    }

    // 7. "Everything is a tensor": a 2-D tensor allocated only as a surface
    //    is rendered into, then used as a texture, a buffer and a CPU/HIP/CUDA
    //    pointer — all the same memory.
    {
        DisplayTensor<uint84> surf({W, H}, kVULKANTEXTURE, AllocationFlags::kSURFACE);
        ctx.submit([&](VkCommandBuffer cmd) {
            surf.render(cmd, [&] { fill.fill(cmd, 1.0f, 1.0f, 0.0f, 1.0f); });
        });

        auto tex = surf.to_compute(kVULKANTEXTURE, AllocationFlags::kTEXTURE);
        VulkanResource* tr = vk_resource(tex);
        check(tr && (tr->image_usage & VK_IMAGE_USAGE_SAMPLED_BIT), "surface-only allocation → texture view");

        DisplayTensor<uint84> out({W, H});
        ctx.submit([&](VkCommandBuffer cmd) { out.render(cmd, [&] { surf.draw(cmd); }); });
        check(pixel_is(out.read(), 9, 255, 255, 0, 255), "surface sampled as a texture in another pass");

        VulkanResource* br = vk_resource(surf.to_compute(kVULKAN));
        check(br && br->buffer && br->memory == surf.resource()->memory, "surface → buffer view (same memory)");

        if (ctx.interopComputeType() == kCPU) {
            auto cpu = surf.compute();
            check(pixel_is(cpu, 11, 255, 255, 0, 255), "surface → CPU pointer sees the rendered pixels");
        }

        // print() goes through the normal tensor path
        std::cout << "       " << surf.read() << std::endl;
    }

    // 8. kOPTIMAL opts out of buffer-backing; asking for a buffer view then
    //    fails with a message, but texture/surface views still work.
    {
        DisplayTensor<uint84> opt({W, H}, kVULKANTEXTURE, AllocationFlags::kSURFACE | AllocationFlags::kOPTIMAL);
        bool threw = false;
        try { auto b = opt.buffer(); } catch (const std::exception& e) {
            threw = true;
            std::cout << "       (expected) " << e.what() << std::endl;
        }
        check(threw, "kOPTIMAL texture refuses an in-place buffer view");
        check(vk_resource(opt.texture()) != nullptr, "kOPTIMAL surface still gives a texture view");
    }
}

int main(int argc, char** argv) {
    bool headless = argc > 1 && std::string(argv[1]) == "--headless";

    if (headless) {
        run_checks();
        std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
        return failures ? 1 : 0;
    }

    Window window({640, 480}, WP_RESIZABLE, "hvml — vulkan views");
    run_checks();

    // Window demo: a render texture and a buffer texture drawn every frame.
    DisplayTensor<uint84> canvas({320, 240});
    DisplayTensor<uint84> strip({320, 240}, kVULKAN);
    Tensor<uint84, 2> host({320, 240}, MemoryType::kDDR);
    for (long i = 0; i < 320 * 240; i++) {
        uint8_t* p = (uint8_t*)host.data.data + i * 4;
        p[0] = (uint8_t)(i % 320 * 255 / 320); p[1] = (uint8_t)(i / 320); p[2] = 128; p[3] = 255;
    }
    strip.write(host);

    FillMaterial fill;
    int frames = 0;
    int maxFrames = std::getenv("HVML_FRAMES") ? std::atoi(std::getenv("HVML_FRAMES")) : -1;
    window.add_on_update([&](CurrentScreenInputInfo&, VkCommandBuffer cmd) {
        float t = frames * 0.02f;
        canvas.render(cmd, [&] { fill.fill(cmd, 0.5f + 0.5f * std::sin(t), 0.2f, 0.6f, 1.0f); });
        // window.backbuffer() is the swapchain image as a DisplayTensor
        canvas.draw(cmd, 0, 0, window.backbuffer().width() / 2.0f, (float)window.backbuffer().height());
        strip.draw(cmd, window.width / 2.0f, 0, window.width / 2.0f, (float)window.height);
        if (maxFrames > 0 && frames == maxFrames / 2) window.resizeWindow(800, 600);   // exercise swapchain recreation
        if (maxFrames > 0 && ++frames >= maxFrames) {
            SDL_Event quit{};
            quit.type = SDL_EVENT_QUIT;
            SDL_PushEvent(&quit);
        }
    });
    window.displayLoop();

    std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
    return failures ? 1 : 0;
}
