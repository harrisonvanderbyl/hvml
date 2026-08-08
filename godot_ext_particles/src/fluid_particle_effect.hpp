#pragma once
#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rd_framebuffer_pass.hpp>
#include <godot_cpp/classes/rd_pipeline_rasterization_state.hpp>
#include <godot_cpp/classes/rd_pipeline_depth_stencil_state.hpp>
#include <godot_cpp/classes/rd_pipeline_color_blend_state.hpp>
#include <godot_cpp/classes/rd_pipeline_color_blend_state_attachment.hpp>
#include <godot_cpp/classes/rd_pipeline_multisample_state.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/render_data.hpp>

namespace godot {

// ─────────────────────────────────────────────────────────────────────────────
// FluidParticleEffect — CompositorEffect that renders particles directly on the
// render thread using the same storage buffer written by the compute shaders.
// Zero CPU-GPU copies: gl_VertexIndex in the vertex shader indexes into the
// particle SSBO bound as a storage buffer uniform.
// ─────────────────────────────────────────────────────────────────────────────
class FluidParticleEffect : public CompositorEffect {
    GDCLASS(FluidParticleEffect, CompositorEffect)

public:
    FluidParticleEffect();
    ~FluidParticleEffect();

    // Shader path (Inspector-editable)
    void   set_draw_shader_path(const String &v) { draw_shader_path = v; pipeline_ready = false; }
    String get_draw_shader_path() const { return draw_shader_path; }

    // Called by FluidParticleSystem after its GPU resources are ready.
    // particle_storage_buf is the RID of the compute-written particle buffer.
    void set_particle_buffer(RID particle_storage_buf, int count,
                             int grid_w, int grid_h, int grid_d);

    void _render_callback(int32_t effect_callback_type,
                          RenderData *render_data) override;

protected:
    static void _bind_methods();

private:
    // Shader path
    String draw_shader_path = "res://addons/fluid_particles/shaders/particle_draw.glsl";

    // Particle buffer RID shared with compute (owned by FluidParticleSystem)
    RID  particle_buf;
    int  num_particles = 0;
    int  grid_w = 128, grid_h = 64, grid_d = 128;

    // Graphics pipeline (built once on first _render_callback)
    RID  shader_rid;
    RID  pipeline_rid;
    bool pipeline_ready = false;

    void _build_pipeline(RenderingDevice *rd, int64_t framebuffer_format);
};

}  // namespace godot
