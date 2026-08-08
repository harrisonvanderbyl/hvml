#pragma once
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/immediate_mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>

namespace godot {

// ──────────────────────────────────────────────────────────────────────────────
// Mirrors the GPU-side Particle struct (std430, 32 bytes)
// ──────────────────────────────────────────────────────────────────────────────
struct GPUParticle {
    float  position[3];      // x, y, z
    float  _pad0;
    uint32_t color_packed;   // r|g|b|a packed as uint8 × 4
    float  attraction_force; // temperatureopacity[0]
    float  opacity_fade;     // temperatureopacity[1]
    float  neighbors_filled;
    float  _pad1;
    // 36 bytes → pad to 48 for cleaner alignment
    float  _pad2[3];
};  // 48 bytes

// ──────────────────────────────────────────────────────────────────────────────
// Mirrors the GPU-side ChunkCell struct (std430, 32 bytes)
// ──────────────────────────────────────────────────────────────────────────────
struct GPUChunkCell {
    int32_t  occupant;
    uint32_t vel_x_bits, vel_y_bits, vel_z_bits, vel_w_bits;
    uint32_t _pad[3];
};  // 32 bytes

// ──────────────────────────────────────────────────────────────────────────────
// FluidParticleSystem — main node
// ──────────────────────────────────────────────────────────────────────────────
class FluidParticleSystem : public Node3D {
    GDCLASS(FluidParticleSystem, Node3D)

public:
    enum DebugMode {
        DEBUG_NONE         = 0,  // normal simulation + CompositorEffect render
        DEBUG_BOUNDING_BOX = 1,  // wire-frame grid AABB overlay
        DEBUG_SIMPLE_POINTS= 2,  // point-only render (no ray-sphere), simulation runs
        DEBUG_STATIC       = 3,  // no compute, particles at spawn positions only
    };

    FluidParticleSystem();
    ~FluidParticleSystem();

    // Godot lifecycle
    void _enter_tree() override;
    void _process(double delta) override;
    void _exit_tree() override;

    // Scripting interface (Inspector-visible properties)
    void set_grid_width(int v);     int get_grid_width()     const { return grid_width; }
    void set_grid_height(int v);    int get_grid_height()    const { return grid_height; }
    void set_grid_depth(int v);     int get_grid_depth()     const { return grid_depth; }
    void set_num_particles(int v);  int get_num_particles()  const { return num_particles; }
    void set_gravity(float v)       { gravity = v; }
    float get_gravity()             const { return gravity; }
    void set_surface_tension(float v) { surface_tension = v; }
    float get_surface_tension()     const { return surface_tension; }
    void set_water_viscosity(float v) { water_viscosity = v; }
    float get_water_viscosity()     const { return water_viscosity; }
    void set_attraction_force(float v) { attraction_force = v; }
    float get_attraction_force()    const { return attraction_force; }
    void set_neighbor_mode(int v)   { neighbor_mode = (v >= 15 ? 15 : 6); }
    int  get_neighbor_mode()        const { return neighbor_mode; }

    // Simulation toggle
    void set_simulation_active(bool v) { simulation_active = v; }
    bool get_simulation_active()   const { return simulation_active; }

    // Initial chunk fill
    void set_use_initial_chunk(bool v) { use_initial_chunk = v; }
    bool get_use_initial_chunk()  const { return use_initial_chunk; }
    void set_initial_chunk_origin(Vector3 v) { initial_chunk_origin = v; }
    Vector3 get_initial_chunk_origin() const { return initial_chunk_origin; }
    void set_initial_chunk_size(Vector3i v) { initial_chunk_size = v; }
    Vector3i get_initial_chunk_size() const { return initial_chunk_size; }
    void set_initial_chunk_color(Color v) { initial_chunk_color = v; }
    Color get_initial_chunk_color() const { return initial_chunk_color; }
    void set_initial_chunk_attraction(float v) { initial_chunk_attraction = v; }
    float get_initial_chunk_attraction() const { return initial_chunk_attraction; }

    // Shader paths (Inspector-editable, default to addon)
    void   set_clear_shader_path(const String &v)  { clear_shader_path = v; }
    String get_clear_shader_path()  const { return clear_shader_path; }
    void   set_physics_shader_path(const String &v) { physics_shader_path = v; }
    String get_physics_shader_path() const { return physics_shader_path; }
    void   set_sortkey_shader_path(const String &v) { sortkey_shader_path = v; }
    String get_sortkey_shader_path() const { return sortkey_shader_path; }
    void   set_debug_shader_path(const String &v)  { debug_shader_path = v; }
    String get_debug_shader_path()  const { return debug_shader_path; }

    // Spawn helpers callable from GDScript
    void spawn_block(Vector3 origin, int w, int h, int d, Color color, float attraction);
    void add_velocity_impulse(Vector3 impulse);
    void reset_grid();  // explicitly clear chunk grid (occupants + velocities)

    // Debug
    void  set_debug_mode(int v);
    int   get_debug_mode() const { return debug_mode; }
    AABB  get_grid_aabb()  const {
        return AABB(Vector3(0,0,0), Vector3((float)grid_width,(float)grid_height,(float)grid_depth));
    }

    // ── Access for child source/sink nodes ────────────────────────────────
    // These expose the GPU particle buffer + RenderingDevice so that
    // FluidSource / FluidSink can dispatch their own compute shaders against
    // the same buffer without any CPU round-trip.
    RenderingDevice *get_rd()           const { return rd; }
    RID              get_particle_buf() const { return particle_buf; }
    RID              get_chunk_buf()    const { return chunk_buf; }
    int              get_grid_w()       const { return grid_width; }
    int              get_grid_h()       const { return grid_height; }
    int              get_grid_d()       const { return grid_depth; }

protected:
    static void _bind_methods();

private:
    // ── shader paths (Inspector-editable) ────────────────────────────────
    String clear_shader_path   = "res://addons/fluid_particles/shaders/clear_grid.glsl";
    String physics_shader_path = "res://addons/fluid_particles/shaders/velocity_spread.glsl";
    String sortkey_shader_path = "res://addons/fluid_particles/shaders/depth_sort_key.glsl";
    String debug_shader_path   = "res://addons/fluid_particles/shaders/particle_debug.gdshader";

    // ── simulation parameters (matching particles.cpp reference) ──────────
    int   grid_width      = 256;    // awidth
    int   grid_height     = 256;    // aheight
    int   grid_depth      = 256;    // adepth
    int   num_particles   = 10000000; // 100*100*1000
    float gravity         = 0.1f;   // gravity
    float surface_tension = 1.1f;   // surfaceTension
    float water_viscosity = 1.0f;   // waterviscosity
    float attraction_force = 1.0f;  // global multiplier on per-particle attraction/repulsion
    int   neighbor_mode   = 7;      // 6+center (reference uses 7)

    // ── simulation toggle ───────────────────────────────────────────────────
    bool  simulation_active = true;

    // ── initial chunk fill (matching particles.cpp reference spawn) ────────
    bool      use_initial_chunk       = true;
    Vector3   initial_chunk_origin    = Vector3(2, 2, 2);
    Vector3i  initial_chunk_size      = Vector3i(32, 32, 32);
    Color     initial_chunk_color     = Color(1.0f, 1.0f, 1.0f, 1.0f);  // solid white
    float     initial_chunk_attraction = 0.0f;  // solid particles have 0 attraction

    // ── rendering ─────────────────────────────────────────────────────────
    // Rendering is handled by FluidParticleEffect (CompositorEffect), which
    // reads particle_buf directly on the render thread — no CPU copy.
    // The system just holds a pointer to notify the effect when the buffer
    // RID changes (e.g. after a resize).
    class FluidParticleEffect *render_effect = nullptr;

    // ── RenderingDevice compute pipeline ──────────────────────────────────
    RenderingDevice *rd = nullptr;

    // GPU buffers (RIDs)
    RID particle_buf;
    RID chunk_buf;
    RID runnable_buf;
    RID sort_key_buf;
    RID uniform_buf;      // push-constant / UBO for per-dispatch params

    // Pipelines
    RID clear_pipeline;
    RID physics_pipeline;
    RID sortkey_pipeline;
    // Shader RIDs (needed for uniform_set_create)
    RID clear_shader;
    RID physics_shader;
    RID sortkey_shader;

    // Uniform sets (one per pipeline)
    RID clear_uniform_set;
    RID physics_uniform_set;
    RID sortkey_uniform_set;

    // ── runtime state ──────────────────────────────────────────────────────
    uint64_t  frame_count  = 0;
    Vector3   pending_impulse;
    bool      impulse_pending = false;
    bool      gpu_ready    = false;

    // ── debug ──────────────────────────────────────────────────────────────
    int              debug_mode      = DEBUG_NONE;
    MeshInstance3D  *debug_bb_node   = nullptr;   // bounding-box wire frame
    Ref<ImmediateMesh> debug_bb_mesh;
    MeshInstance3D  *debug_pts_node  = nullptr;   // simple-points fallback
    Ref<ArrayMesh>   debug_pts_mesh;
    Ref<ShaderMaterial> debug_material;

    void _rebuild_debug_bb();
    void _update_debug_points();
    void _ensure_debug_nodes();
    void _build_gpu_resources();
    void _destroy_gpu_resources();
    void _dispatch_clear_grid();
    void _dispatch_physics(Vector3 global_add_velocity);
    void _dispatch_sortkey();

    RID  _load_shader(const String &glsl_path, const String &entry);
    RID  _make_uniform_set(RID pipeline, TypedArray<RDUniform> uniforms, uint32_t set_index);
};

}  // namespace godot

VARIANT_ENUM_CAST(godot::FluidParticleSystem::DebugMode);
