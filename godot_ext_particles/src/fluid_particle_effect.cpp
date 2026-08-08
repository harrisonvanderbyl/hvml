#include "fluid_particle_effect.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

FluidParticleEffect::FluidParticleEffect() {
    set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_TRANSPARENT);
    set_enabled(true);
    set_access_resolved_color(true);
    set_access_resolved_depth(true);
}

FluidParticleEffect::~FluidParticleEffect() {}

void FluidParticleEffect::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_draw_shader_path", "path"), &FluidParticleEffect::set_draw_shader_path);
    ClassDB::bind_method(D_METHOD("get_draw_shader_path"),          &FluidParticleEffect::get_draw_shader_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "draw_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_draw_shader_path", "get_draw_shader_path");
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleEffect::set_particle_buffer(RID buf, int count,
                                               int gw, int gh, int gd)
{
    particle_buf  = buf;
    num_particles = count;
    grid_w = gw; grid_h = gh; grid_d = gd;
    pipeline_ready = false;  // Force rebuild when buffer dimensions change
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the graphics pipeline lazily on the render thread (required — RD
// graphics pipelines must be created with the correct framebuffer format, which
// is only known at render time).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleEffect::_build_pipeline(RenderingDevice *rd, int64_t fb_fmt) {
    // Load the combined vert+frag shader via Godot's resource system
    Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(draw_shader_path);
    if (!sf.is_valid()) {
        UtilityFunctions::printerr("FluidParticleEffect: cannot load ", draw_shader_path);
        return;
    }
    Ref<RDShaderSPIRV> spirv = sf->get_spirv();
    if (!spirv.is_valid()) {
        UtilityFunctions::printerr("FluidParticleEffect: no SPIR-V in particle_draw.glsl");
        return;
    }
    for (int s = 0; s < RenderingDevice::SHADER_STAGE_MAX; s++) {
        String err = spirv->get_stage_compile_error((RenderingDevice::ShaderStage)s);
        if (!err.is_empty())
            UtilityFunctions::printerr("FluidParticleEffect shader error (stage ", s, "): ", err);
    }
    if (!spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_VERTEX).is_empty() ||
        !spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_FRAGMENT).is_empty()) {
        return;
    }

    if (shader_rid.is_valid()) rd->free_rid(shader_rid);
    shader_rid = rd->shader_create_from_spirv(spirv);
    if (!shader_rid.is_valid()) return;

    // ── Rasterization: points, no culling ────────────────────────────────────
    Ref<RDPipelineRasterizationState> raster;
    raster.instantiate();
    raster->set_cull_mode(RenderingDevice::POLYGON_CULL_DISABLED);

    // ── Depth: test enabled, write enabled ───────────────────────────────────
    Ref<RDPipelineDepthStencilState> depth;
    depth.instantiate();
    depth->set_enable_depth_test(true);
    depth->set_enable_depth_write(true);
    depth->set_depth_compare_operator(RenderingDevice::COMPARE_OP_LESS);

    // ── Blending: src_alpha / one_minus_src_alpha ─────────────────────────────
    Ref<RDPipelineColorBlendStateAttachment> blend_att;
    blend_att.instantiate();
    blend_att->set_enable_blend(true);
    blend_att->set_src_color_blend_factor(RenderingDevice::BLEND_FACTOR_SRC_ALPHA);
    blend_att->set_dst_color_blend_factor(RenderingDevice::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
    blend_att->set_color_blend_op(RenderingDevice::BLEND_OP_ADD);
    blend_att->set_src_alpha_blend_factor(RenderingDevice::BLEND_FACTOR_ONE);
    blend_att->set_dst_alpha_blend_factor(RenderingDevice::BLEND_FACTOR_ZERO);
    blend_att->set_alpha_blend_op(RenderingDevice::BLEND_OP_ADD);

    Ref<RDPipelineColorBlendState> blend;
    blend.instantiate();
    blend->set_blend_constant(Color(0, 0, 0, 0));
    TypedArray<RDPipelineColorBlendStateAttachment> atts;
    atts.append(blend_att);
    blend->set_attachments(atts);

    Ref<RDPipelineMultisampleState> ms;
    ms.instantiate();

    if (pipeline_rid.is_valid()) rd->free_rid(pipeline_rid);
    pipeline_rid = rd->render_pipeline_create(
        shader_rid,
        fb_fmt,
        RenderingDevice::INVALID_ID,         // vertex format — none, we use gl_VertexIndex
        RenderingDevice::RENDER_PRIMITIVE_POINTS,
        raster,
        ms,
        depth,
        blend
    );

    pipeline_ready = pipeline_rid.is_valid();
    if (pipeline_ready)
        UtilityFunctions::print("FluidParticleEffect: graphics pipeline ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Push-constant layout (must match particle_draw.vert.glsl / .frag.glsl)
// ─────────────────────────────────────────────────────────────────────────────
struct DrawPushConstants {
    float proj_view[16];     // projection * view  (column-major)
    float view[16];          // view matrix
    float inv_view[16];      // inverse view
    float inv_proj[16];      // inverse projection
    float screen_size[2];
    float _pad[2];
    int   num_particles;
    float _pad2[3];
};

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleEffect::_render_callback(int32_t effect_callback_type,
                                            RenderData *render_data)
{
    if (!particle_buf.is_valid() || num_particles == 0) return;

    Ref<RenderSceneBuffersRD> scene_bufs =
        render_data->get_render_scene_buffers();
    if (!scene_bufs.is_valid()) return;

    RenderingDevice *rd = RenderingServer::get_singleton()
                              ->get_rendering_device();
    if (!rd) return;

    // ── Lazy pipeline build ───────────────────────────────────────────────────
    // We need the framebuffer format from the actual render target.
    RID color_img = scene_bufs->get_color_layer(0);
    RID depth_img = scene_bufs->get_depth_layer(0);
    if (!color_img.is_valid() || !depth_img.is_valid()) return;

    if (!pipeline_ready) {
        // Build a temporary framebuffer to get the format ID, then discard it.
        TypedArray<RID> fb_textures;
        fb_textures.append(color_img);
        fb_textures.append(depth_img);
        RID temp_fb = rd->framebuffer_create(fb_textures);
        int64_t fb_fmt = rd->framebuffer_get_format(temp_fb);
        rd->free_rid(temp_fb);
        _build_pipeline(rd, fb_fmt);
        if (!pipeline_ready) return;
    }

    // ── Framebuffer for this frame ────────────────────────────────────────────
    TypedArray<RID> fb_textures;
    fb_textures.append(color_img);
    fb_textures.append(depth_img);
    RID framebuffer = rd->framebuffer_create(fb_textures);

    // ── Uniform set: bind particle SSBO at set=0 binding=0 ───────────────────
    Ref<RDUniform> u;
    u.instantiate();
    u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    u->set_binding(0);
    u->add_id(particle_buf);

    TypedArray<RDUniform> uniforms;
    uniforms.append(u);
    RID uniform_set = rd->uniform_set_create(uniforms, shader_rid, 0);

    // ── Build push constants ──────────────────────────────────────────────────
    // Godot 4 render_data doesn't yet expose camera matrices directly via
    // GDExtension; we read them from the scene's CameraAttributes.
    // For now we pass identity — in a real project you'd pass them via a
    // shared UBO updated each frame by FluidParticleSystem::_process.
    DrawPushConstants pc{};
    pc.screen_size[0]  = (float)scene_bufs->get_internal_size().x;
    pc.screen_size[1]  = (float)scene_bufs->get_internal_size().y;
    pc.num_particles   = num_particles;
    // proj_view / view / inv_view / inv_proj — filled below if available
    // (identity fallback keeps the code compiling; wire up camera matrices
    //  via a shared UBO in FluidParticleSystem::_process for real use)
    for (int i = 0; i < 4; i++) pc.proj_view[i*4+i] = pc.view[i*4+i]
                                = pc.inv_view[i*4+i] = pc.inv_proj[i*4+i] = 1.0f;

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(DrawPushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(DrawPushConstants));

    // ── Draw ──────────────────────────────────────────────────────────────────
    // No vertex buffer — vertex shader uses gl_VertexIndex to index the SSBO.
    // draw_list_draw(list, use_indices=false, instances=1, procedural_vertices=N)
    // draw_list_begin(framebuffer, draw_flags, clear_colors, clear_depth, ...)
    // Godot 4.3 API: no separate initial/final action enums — use DrawFlags.
    // DRAW_DEFAULT (0) = load existing contents, store result.
    int64_t draw_list = rd->draw_list_begin(
        framebuffer,
        (BitField<RenderingDevice::DrawFlags>)0,  // DRAW_DEFAULT
        PackedColorArray()                         // no colour clear
    );

    rd->draw_list_bind_render_pipeline(draw_list, pipeline_rid);
    rd->draw_list_bind_uniform_set(draw_list, uniform_set, 0);
    rd->draw_list_set_push_constant(draw_list, pc_bytes, sizeof(DrawPushConstants));
    rd->draw_list_draw(draw_list, false, 1, num_particles);
    rd->draw_list_end();

    // Framebuffer and uniform set are per-frame temporaries
    rd->free_rid(uniform_set);
    rd->free_rid(framebuffer);
}
