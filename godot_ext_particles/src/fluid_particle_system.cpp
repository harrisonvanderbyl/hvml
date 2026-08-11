#include "fluid_particle_system.hpp"
#include "fluid_source_sink.hpp"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/immediate_mesh.hpp>
#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace godot;

// ─────────────────────────────────────────────────────────────────────────────
// Push-constant layout sent to every compute dispatch (std430 / 16-byte align)
// ─────────────────────────────────────────────────────────────────────────────
struct PushConstants {
    int32_t grid_w, grid_h, grid_d;
    int32_t num_particles;
    float   surface_tension;
    float   water_viscosity;
    float   attraction_force;
    float   _pad0;
    float   gravity[3];      // vec3 in grid space — local or world depending on gravity_local
    int32_t frame_count;
    float   global_vel[3];
    int32_t neighbor_mode;
    int32_t num_runnable;
    int32_t _pad1[3];
};


// ─────────────────────────────────────────────────────────────────────────────
FluidParticleSystem::FluidParticleSystem() {}
FluidParticleSystem::~FluidParticleSystem() {}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_grid_width","v"),    &FluidParticleSystem::set_grid_width);
    ClassDB::bind_method(D_METHOD("get_grid_width"),        &FluidParticleSystem::get_grid_width);
    ClassDB::bind_method(D_METHOD("set_grid_height","v"),   &FluidParticleSystem::set_grid_height);
    ClassDB::bind_method(D_METHOD("get_grid_height"),       &FluidParticleSystem::get_grid_height);
    ClassDB::bind_method(D_METHOD("set_grid_depth","v"),    &FluidParticleSystem::set_grid_depth);
    ClassDB::bind_method(D_METHOD("get_grid_depth"),        &FluidParticleSystem::get_grid_depth);
    ClassDB::bind_method(D_METHOD("set_num_particles","v"), &FluidParticleSystem::set_num_particles);
    ClassDB::bind_method(D_METHOD("get_num_particles"),     &FluidParticleSystem::get_num_particles);
    ClassDB::bind_method(D_METHOD("set_gravity","v"),       &FluidParticleSystem::set_gravity);
    ClassDB::bind_method(D_METHOD("get_gravity"),           &FluidParticleSystem::get_gravity);
    ClassDB::bind_method(D_METHOD("set_gravity_local","v"), &FluidParticleSystem::set_gravity_local);
    ClassDB::bind_method(D_METHOD("get_gravity_local"),     &FluidParticleSystem::get_gravity_local);
    ClassDB::bind_method(D_METHOD("set_surface_tension","v"),  &FluidParticleSystem::set_surface_tension);
    ClassDB::bind_method(D_METHOD("get_surface_tension"),      &FluidParticleSystem::get_surface_tension);
    ClassDB::bind_method(D_METHOD("set_water_viscosity","v"),  &FluidParticleSystem::set_water_viscosity);
    ClassDB::bind_method(D_METHOD("get_water_viscosity"),      &FluidParticleSystem::get_water_viscosity);
    ClassDB::bind_method(D_METHOD("set_attraction_force","v"), &FluidParticleSystem::set_attraction_force);
    ClassDB::bind_method(D_METHOD("get_attraction_force"),     &FluidParticleSystem::get_attraction_force);
    ClassDB::bind_method(D_METHOD("set_neighbor_mode","v"), &FluidParticleSystem::set_neighbor_mode);
    ClassDB::bind_method(D_METHOD("get_neighbor_mode"),     &FluidParticleSystem::get_neighbor_mode);
    ClassDB::bind_method(D_METHOD("set_simulation_active","v"), &FluidParticleSystem::set_simulation_active);
    ClassDB::bind_method(D_METHOD("get_simulation_active"),     &FluidParticleSystem::get_simulation_active);
    ClassDB::bind_method(D_METHOD("set_use_initial_chunk","v"), &FluidParticleSystem::set_use_initial_chunk);
    ClassDB::bind_method(D_METHOD("get_use_initial_chunk"),     &FluidParticleSystem::get_use_initial_chunk);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_origin","v"), &FluidParticleSystem::set_initial_chunk_origin);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_origin"),     &FluidParticleSystem::get_initial_chunk_origin);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_size","v"), &FluidParticleSystem::set_initial_chunk_size);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_size"),     &FluidParticleSystem::get_initial_chunk_size);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_color","v"), &FluidParticleSystem::set_initial_chunk_color);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_color"),     &FluidParticleSystem::get_initial_chunk_color);
    ClassDB::bind_method(D_METHOD("set_initial_chunk_attraction","v"), &FluidParticleSystem::set_initial_chunk_attraction);
    ClassDB::bind_method(D_METHOD("get_initial_chunk_attraction"),     &FluidParticleSystem::get_initial_chunk_attraction);
    ClassDB::bind_method(D_METHOD("spawn_block","origin","w","h","d","color","attraction"),
                         &FluidParticleSystem::spawn_block);
    ClassDB::bind_method(D_METHOD("add_velocity_impulse","impulse"),
                         &FluidParticleSystem::add_velocity_impulse);
    ClassDB::bind_method(D_METHOD("reset_grid"),
                         &FluidParticleSystem::reset_grid);

    ClassDB::bind_method(D_METHOD("get_grid_aabb"),      &FluidParticleSystem::get_grid_aabb);

    // Shader paths
    ClassDB::bind_method(D_METHOD("set_clear_shader_path","v"),   &FluidParticleSystem::set_clear_shader_path);
    ClassDB::bind_method(D_METHOD("get_clear_shader_path"),        &FluidParticleSystem::get_clear_shader_path);
    ClassDB::bind_method(D_METHOD("set_physics_shader_path","v"), &FluidParticleSystem::set_physics_shader_path);
    ClassDB::bind_method(D_METHOD("get_physics_shader_path"),      &FluidParticleSystem::get_physics_shader_path);
    ClassDB::bind_method(D_METHOD("set_sortkey_shader_path","v"), &FluidParticleSystem::set_sortkey_shader_path);
    ClassDB::bind_method(D_METHOD("get_sortkey_shader_path"),      &FluidParticleSystem::get_sortkey_shader_path);
    ClassDB::bind_method(D_METHOD("set_render_material","v"),  &FluidParticleSystem::set_render_material);
    ClassDB::bind_method(D_METHOD("get_render_material"),       &FluidParticleSystem::get_render_material);

    // Shader paths group
    ADD_GROUP("Shaders", "");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "clear_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_clear_shader_path",   "get_clear_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "physics_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_physics_shader_path", "get_physics_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "sortkey_shader_path",
        PROPERTY_HINT_FILE, "*.glsl"), "set_sortkey_shader_path", "get_sortkey_shader_path");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "render_material",
        PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_render_material", "get_render_material");
    ADD_GROUP("", "");

    ADD_PROPERTY(PropertyInfo(Variant::INT,   "grid_width"),      "set_grid_width",      "get_grid_width");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "grid_height"),     "set_grid_height",     "get_grid_height");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "grid_depth"),      "set_grid_depth",      "get_grid_depth");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "num_particles"),   "set_num_particles",   "get_num_particles");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"),         "set_gravity",         "get_gravity");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,    "gravity_local"),   "set_gravity_local",   "get_gravity_local");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_tension"), "set_surface_tension", "get_surface_tension");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "water_viscosity"), "set_water_viscosity", "get_water_viscosity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attraction_force", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_attraction_force", "get_attraction_force");
    ADD_PROPERTY(PropertyInfo(Variant::INT,   "neighbor_mode"),   "set_neighbor_mode",   "get_neighbor_mode");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,  "simulation_active"), "set_simulation_active", "get_simulation_active");

    // Initial chunk fill group
    ADD_GROUP("Initial Chunk", "initial_chunk_");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,  "use_initial_chunk"),  "set_use_initial_chunk",  "get_use_initial_chunk");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,  "origin"),  "set_initial_chunk_origin",  "get_initial_chunk_origin");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "size"),   "set_initial_chunk_size",   "get_initial_chunk_size");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR,  "color"),    "set_initial_chunk_color",   "get_initial_chunk_color");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attraction", PROPERTY_HINT_RANGE, "-2,2,0.01"), "set_initial_chunk_attraction", "get_initial_chunk_attraction");
    ADD_GROUP("", "");
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::set_grid_width(int v)   { grid_width  = v; }
void FluidParticleSystem::set_grid_height(int v)  { grid_height = v; }
void FluidParticleSystem::set_grid_depth(int v)   { grid_depth  = v; }
void FluidParticleSystem::set_num_particles(int v){ num_particles = v; }

// ─────────────────────────────────────────────────────────────────────────────
// Default render shader path. Used to pre-fill the render_shader resource when
// it has not been set (or has been cleared) in the inspector.
// ─────────────────────────────────────────────────────────────────────────────
static const char *DEFAULT_RENDER_SHADER_PATH =
    "res://addons/fluid_particles/shaders/particle_render.gdshader";

// Loads the default particle render shader code into a new Shader resource.
static Ref<Shader> _load_default_render_shader() {
    Ref<Shader> shader;
    shader.instantiate();
    Ref<FileAccess> sf = FileAccess::open(DEFAULT_RENDER_SHADER_PATH, FileAccess::READ);
    if (sf.is_valid()) {
        shader->set_code(sf->get_as_text());
    } else {
        UtilityFunctions::printerr("FluidParticleSystem: cannot open default render shader: ",
                                    DEFAULT_RENDER_SHADER_PATH);
    }
    return shader;
}

void FluidParticleSystem::set_render_material(const Ref<ShaderMaterial> &v) {
    render_material = v;

    if (render_material.is_valid()) {
        // If the material has no shader or empty code, pre-fill with the default
        // particle render shader so the user can edit code + uniforms immediately.
        Ref<Shader> shader = render_material->get_shader();
        if (shader.is_null() || shader->get_code().is_empty()) {
            render_material->set_shader(_load_default_render_shader());
        }
    }

    // If the render node already exists, swap the material override live.
    // Using the user's ShaderMaterial directly means shader/uniform edits
    // are reflected instantly by Godot's own resource signaling.
    if (render_node) {
        if (render_material.is_valid()) {
            render_node->set_material_override(render_material);
        } else if (internal_material.is_valid()) {
            render_node->set_material_override(internal_material);
        }
    }
}

Ref<ShaderMaterial> FluidParticleSystem::get_render_material() const {
    return render_material;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ensure the render node exists (lazy-created): a MeshInstance3D whose
// ArrayMesh is rebuilt every frame from a CPU readback of particle_buf,
// shaded by the render_shader resource (defaults to particle_render.gdshader).
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_ensure_render_node() {
    if (render_node) return;

    // If the user hasn't set a render_material, create an internal default
    // material from the addon's particle_render.gdshader. This is NOT exposed
    // as the property — the property stays null so the user knows they're
    // using the built-in default.
    if (render_material.is_null() && internal_material.is_null()) {
        internal_material.instantiate();
        internal_material->set_shader(_load_default_render_shader());
    }

    // Use the user's material if set, otherwise the internal default.
    Ref<ShaderMaterial> mat = render_material.is_valid() ? render_material : internal_material;

    render_mesh.instantiate();
    render_node = memnew(MeshInstance3D);
    render_node->set_mesh(render_mesh);
    render_node->set_material_override(mat);
    add_child(render_node);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU readback → ArrayMesh(PRIMITIVE_POINTS) with CUSTOM0 (attraction_force,
// opacity_fade, neighbors_filled), rebuilt every frame from particle_buf.
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_update_render_mesh() {
    if (!render_mesh.is_valid() || !rd || !particle_buf.is_valid()) return;

    PackedByteArray raw = rd->buffer_get_data(particle_buf);
    if (raw.is_empty()) return;
    const GPUParticle *parts = reinterpret_cast<const GPUParticle*>(raw.ptr());

    PackedVector3Array  positions;
    PackedColorArray    colors;
    PackedFloat32Array  custom0;  // (attraction_force, opacity_fade, neighbors_filled, 0) per vertex
    Vector3 *pp = nullptr;
    Color   *cp = nullptr;
    float   *c0 = nullptr;

    // First pass: count active (non-NaN) particles
    int active_count = 0;
    for (int i = 0; i < num_particles; i++) {
        float x = parts[i].position[0];
        if (x == x) active_count++;  // NaN check: x != x means inactive
    }

    if (active_count == 0) {
        if (render_mesh->get_surface_count() > 0) render_mesh->clear_surfaces();
        return;
    }

    positions.resize(active_count);
    colors.resize(active_count);
    custom0.resize(active_count * 4);
    pp = positions.ptrw();
    cp = colors.ptrw();
    c0 = custom0.ptrw();

    int out = 0;
    for (int i = 0; i < num_particles; i++) {
        float x = parts[i].position[0];
        if (x != x) continue;  // skip inactive (NaN sentinel)
        pp[out] = Vector3(parts[i].position[0], parts[i].position[1], parts[i].position[2]);
        uint32_t c = parts[i].color_packed;
        cp[out] = Color(((c>>0)&0xff)/255.f, ((c>>8)&0xff)/255.f,
                        ((c>>16)&0xff)/255.f, ((c>>24)&0xff)/255.f);
        c0[out*4 + 0] = parts[i].attraction_force;
        c0[out*4 + 1] = parts[i].opacity_fade;
        c0[out*4 + 2] = parts[i].neighbors_filled;
        c0[out*4 + 3] = 0.0f;
        out++;
    }

    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX]  = positions;
    arrays[Mesh::ARRAY_COLOR]   = colors;
    arrays[Mesh::ARRAY_CUSTOM0] = custom0;
    if (render_mesh->get_surface_count() > 0) render_mesh->clear_surfaces();
    render_mesh->add_surface_from_arrays(
        Mesh::PRIMITIVE_POINTS, arrays, TypedArray<Array>(), Dictionary(),
        (BitField<Mesh::ArrayFormat>)((int64_t)Mesh::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT));
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_enter_tree() {
    _build_gpu_resources();
    _ensure_render_node();
}

void FluidParticleSystem::_exit_tree() {
    _destroy_gpu_resources();
    // render_node is a child and freed automatically
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_build_gpu_resources() {
    // Use a LOCAL RenderingDevice: rendering goes through a normal
    // MeshInstance3D + CPU-readback ArrayMesh (see _update_render_mesh), not
    // a shared render-thread device. A local device gives deterministic
    // submit()/sync() semantics needed for correct, non-stale readbacks.
    rd = RenderingServer::get_singleton()->create_local_rendering_device();
    if (!rd) {
        UtilityFunctions::printerr("FluidParticleSystem: Could not create RenderingDevice.");
        return;
    }

    // ── Particle buffer ──────────────────────────────────────────────────────
    {
        int64_t buf_size = (int64_t)num_particles * sizeof(GPUParticle);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);

        // Default initialise: all particles start INACTIVE (NaN position)
        // so FluidSource nodes can spawn them on demand. If use_initial_chunk
        // is checked, activate a block of particles at initial_chunk_origin.
        GPUParticle *p = reinterpret_cast<GPUParticle*>(data.ptrw());
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_val;
        std::memcpy(&nan_val, &nan_bits, sizeof(float));

        // Pack initial chunk color
        uint32_t init_col = ((uint32_t)(initial_chunk_color.r * 255) & 0xff)
                          | (((uint32_t)(initial_chunk_color.g * 255) & 0xff) << 8)
                          | (((uint32_t)(initial_chunk_color.b * 255) & 0xff) << 16)
                          | (((uint32_t)(initial_chunk_color.a * 255) & 0xff) << 24);

        for (int i = 0; i < num_particles; i++) {
            p[i].position[0] = nan_val;  // inactive sentinel
            p[i].position[1] = 0.0f;
            p[i].position[2] = 0.0f;
            p[i].color_packed     = init_col;
            p[i].attraction_force = initial_chunk_attraction;
            p[i].opacity_fade     = 1.0f;
            p[i].neighbors_filled = 0.0f;
        }

        // If initial chunk is enabled, activate particles in a block.
        // All liquid so they repulse each other and flow.
        if (use_initial_chunk) {
            int idx = 0;
            for (int iz = 0; iz < initial_chunk_size.z && idx < num_particles; iz++) {
                for (int iy = 0; iy < initial_chunk_size.y && idx < num_particles; iy++) {
                    for (int ix = 0; ix < initial_chunk_size.x && idx < num_particles; ix++, idx++) {
                        p[idx].position[0] = initial_chunk_origin.x + ix;
                        p[idx].position[1] = initial_chunk_origin.y + iy;
                        p[idx].position[2] = initial_chunk_origin.z + iz;
                        // Liquid: attraction = water_viscosity (enables repulsion)
                        p[idx].color_packed     = 0x00u | (0x00u << 8) | (0x00u << 16) | (0x11u << 24);
                        p[idx].attraction_force = water_viscosity;
                        p[idx].opacity_fade     = 1.0f;
                        p[idx].neighbors_filled = 0.0f;
                    }
                }
            }
        }
        particle_buf = rd->storage_buffer_create(buf_size, data);
    }

    // ── Chunk grid buffer ────────────────────────────────────────────────────
    {
        int64_t cell_count = (int64_t)grid_width * grid_height * grid_depth;
        int64_t buf_size   = cell_count * sizeof(GPUChunkCell);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);
        // Set all occupants to -1
        GPUChunkCell *cells = reinterpret_cast<GPUChunkCell*>(data.ptrw());
        for (int64_t i = 0; i < cell_count; i++) {
            cells[i].occupant = -1;
        }
        chunk_buf = rd->storage_buffer_create(buf_size, data);
    }

    // ── Runnable index buffer (identity permutation) ─────────────────────────
    {
        int64_t buf_size = (int64_t)num_particles * sizeof(int32_t);
        PackedByteArray data;
        data.resize(buf_size);
        int32_t *idx = reinterpret_cast<int32_t*>(data.ptrw());
        for (int i = 0; i < num_particles; i++) idx[i] = i;
        runnable_buf = rd->storage_buffer_create(buf_size, data);
    }

    // ── Sort-key buffer ───────────────────────────────────────────────────────
    {
        int64_t buf_size = (int64_t)num_particles * sizeof(float);
        PackedByteArray data;
        data.resize(buf_size);
        data.fill(0);
        sort_key_buf = rd->storage_buffer_create(buf_size, data);
    }

    // ── Load and compile compute shaders ─────────────────────────────────────
    // Loaded via Godot's resource system (imports .glsl -> RDShaderFile).

    // Load compute shaders via Godot's resource system (imports .glsl -> RDShaderFile)
    auto compile_shader = [&](const String &res_path) -> RID {
        Ref<RDShaderFile> sf = ResourceLoader::get_singleton()->load(res_path);
        if (!sf.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: cannot load shader resource: ", res_path);
            return RID();
        }
        Ref<RDShaderSPIRV> spirv = sf->get_spirv();
        if (!spirv.is_valid()) {
            UtilityFunctions::printerr("FluidParticleSystem: no SPIR-V in: ", res_path);
            return RID();
        }
        String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
        if (!err.is_empty()) {
            UtilityFunctions::printerr("FluidParticleSystem: shader error (", res_path, "):\n", err);
            return RID();
        }
        return rd->shader_create_from_spirv(spirv);
    };

    clear_shader   = compile_shader(clear_shader_path);
    physics_shader = compile_shader(physics_shader_path);
    sortkey_shader = compile_shader(sortkey_shader_path);

    if (clear_shader.is_valid())   clear_pipeline   = rd->compute_pipeline_create(clear_shader);
    if (physics_shader.is_valid()) physics_pipeline = rd->compute_pipeline_create(physics_shader);
    if (sortkey_shader.is_valid()) sortkey_pipeline = rd->compute_pipeline_create(sortkey_shader);

    // ── Build uniform sets ────────────────────────────────────────────────────
    // All three pipelines share the same buffer bindings:
    //   binding 0 → particles
    //   binding 1 → chunk_grid
    //   binding 2 → runnable_indices
    //   binding 3 → sort_keys

    auto make_storage_uniform = [](RID buf, uint32_t binding) -> Ref<RDUniform> {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(buf);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.append(make_storage_uniform(particle_buf,  0));
    uniforms.append(make_storage_uniform(chunk_buf,     1));
    uniforms.append(make_storage_uniform(runnable_buf,  2));
    uniforms.append(make_storage_uniform(sort_key_buf,  3));

    if (clear_shader.is_valid()) {
        clear_uniform_set   = rd->uniform_set_create(uniforms, clear_shader,   0);
    }
    if (physics_shader.is_valid()) {
        physics_uniform_set = rd->uniform_set_create(uniforms, physics_shader, 0);
    }
    if (sortkey_shader.is_valid()) {
        sortkey_uniform_set = rd->uniform_set_create(uniforms, sortkey_shader, 0);
    }

    // ── Rendering ──────────────────────────────────────────────────────────
    // Rendering goes through a normal MeshInstance3D (render_node) whose
    // ArrayMesh is rebuilt each frame from a CPU readback of particle_buf
    // (see _update_render_mesh). This uses Godot's ordinary render pipeline
    // (automatic camera matrices, sorting, shadows) via particle_render.gdshader.
    gpu_ready = true;
    UtilityFunctions::print("FluidParticleSystem: GPU resources ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_destroy_gpu_resources() {
    if (!rd) return;
    gpu_ready = false;

    if (clear_uniform_set.is_valid())   rd->free_rid(clear_uniform_set);
    if (physics_uniform_set.is_valid()) rd->free_rid(physics_uniform_set);
    if (sortkey_uniform_set.is_valid()) rd->free_rid(sortkey_uniform_set);

    if (clear_pipeline.is_valid())   rd->free_rid(clear_pipeline);
    if (physics_pipeline.is_valid()) rd->free_rid(physics_pipeline);
    if (sortkey_pipeline.is_valid()) rd->free_rid(sortkey_pipeline);
    if (clear_shader.is_valid())     rd->free_rid(clear_shader);
    if (physics_shader.is_valid())   rd->free_rid(physics_shader);
    if (sortkey_shader.is_valid())   rd->free_rid(sortkey_shader);

    if (particle_buf.is_valid())  rd->free_rid(particle_buf);
    if (chunk_buf.is_valid())     rd->free_rid(chunk_buf);
    if (runnable_buf.is_valid())  rd->free_rid(runnable_buf);
    if (sort_key_buf.is_valid())  rd->free_rid(sort_key_buf);

    // rd is a local device we own — free it.
    memdelete(rd);
    rd = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_process(double delta) {
    if (!gpu_ready) return;

    // Simulation can be paused from the editor
    if (!simulation_active) {
        _update_render_mesh();  // keep the render mesh in sync with frozen state
        return;
    }

    Vector3 impulse = Vector3();
    if (impulse_pending) {
        impulse = pending_impulse;
        impulse_pending = false;
    }

    // NOTE: No per-frame clear — the chunk grid stores persistent velocity state
    // that is read-consumed by swap0() and re-written by atomic_plus_equals().
    // Clearing every frame would zero all velocities and force firststep=true on
    // every particle, breaking the simulation. The grid is initialized once at
    // _build_gpu_resources() and only cleared explicitly via reset_grid().

    // 1. Physics step (velocity_spread.glsl)
    _dispatch_physics(impulse);

    // 2. Source/sink children dispatch into the same RD command buffer
    for (int i = 0; i < get_child_count(); i++) {
        FluidSourceBase *ss = Object::cast_to<FluidSourceBase>(get_child(i));
        if (ss) ss->dispatch_into_parent();
    }

    // 3. Sort keys (every 60 frames)
    if (frame_count % 60 == 0) {
        _dispatch_sortkey();
    }

    // rd is a local RenderingDevice we own, so submit + sync deterministically
    // before reading the buffer back on the CPU below.
    rd->submit();
    rd->sync();

    // Rebuild the ArrayMesh from the compute results every frame, shaded by
    // particle_render.gdshader (real ray-sphere/liquid rendering) through a
    // normal MeshInstance3D — automatic camera matrices/sorting/shadows come
    // from Godot's own render pipeline.
    _update_render_mesh();

    frame_count++;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build and submit a compute list for one pipeline
// ─────────────────────────────────────────────────────────────────────────────
static void dispatch_compute(RenderingDevice *rd,
                             RID pipeline, RID uniform_set,
                             const PushConstants &pc,
                             uint32_t x_groups, uint32_t y_groups = 1, uint32_t z_groups = 1)
{
    if (!pipeline.is_valid() || !uniform_set.is_valid()) return;

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    int64_t cl = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(cl, pipeline);
    rd->compute_list_bind_uniform_set(cl, uniform_set, 0);
    rd->compute_list_set_push_constant(cl, pc_bytes, sizeof(PushConstants));
    rd->compute_list_dispatch(cl, x_groups, y_groups, z_groups);
    rd->compute_list_end();
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_clear_grid() {
    PushConstants pc{};
    pc.grid_w         = grid_width;
    pc.grid_h         = grid_height;
    pc.grid_d         = grid_depth;
    pc.num_particles  = num_particles;

    int64_t cell_count = (int64_t)grid_width * grid_height * grid_depth;
    uint32_t groups = (uint32_t)((cell_count + 63) / 64);
    dispatch_compute(rd, clear_pipeline, clear_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_physics(Vector3 global_add_velocity) {
    // Resolve gravity into grid (local) space.
    // The grid lives in the node's local space, so world-space gravity must be
    // transformed by the inverse of the node's global basis.
    Vector3 grav = gravity_vec;
    if (!gravity_local) {
        // World-space gravity → transform into node local space
        Basis inv_basis = get_global_transform().basis.inverse();
        grav = inv_basis.xform(grav);
    }

    PushConstants pc{};
    pc.grid_w           = grid_width;
    pc.grid_h           = grid_height;
    pc.grid_d           = grid_depth;
    pc.num_particles    = num_particles;
    pc.surface_tension  = surface_tension;
    pc.water_viscosity  = water_viscosity;
    pc.attraction_force = attraction_force;
    pc.gravity[0]       = grav.x;
    pc.gravity[1]       = grav.y;
    pc.gravity[2]       = grav.z;
    pc.global_vel[0]    = global_add_velocity.x;
    pc.global_vel[1]    = global_add_velocity.y;
    pc.global_vel[2]    = global_add_velocity.z;
    pc.frame_count      = (int32_t)(frame_count & 0x7fffffff);
    pc.neighbor_mode    = neighbor_mode;
    pc.num_runnable     = num_particles;

    uint32_t groups = (uint32_t)((num_particles + 63) / 64);
    dispatch_compute(rd, physics_pipeline, physics_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::_dispatch_sortkey() {
    PushConstants pc{};
    pc.grid_w        = grid_width;
    pc.grid_h        = grid_height;
    pc.grid_d        = grid_depth;
    pc.num_particles = num_particles;
    // Camera position is baked into the sort key shader via global_vel[0..2]
    // We pass (0,0,0) here; in _process you could pass the camera world position.
    uint32_t groups = (uint32_t)((num_particles + 63) / 64);
    dispatch_compute(rd, sortkey_pipeline, sortkey_uniform_set, pc, groups);
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::spawn_block(Vector3 origin, int w, int h, int d,
                                      Color color, float attraction)
{
    if (!rd) return;

    PackedByteArray raw = rd->buffer_get_data(particle_buf);
    if (raw.is_empty()) return;
    GPUParticle *particles = reinterpret_cast<GPUParticle*>(raw.ptrw());

    uint32_t col_packed = ((uint32_t)(color.r * 255) & 0xff)
                        | (((uint32_t)(color.g * 255) & 0xff) << 8)
                        | (((uint32_t)(color.b * 255) & 0xff) << 16)
                        | (((uint32_t)(color.a * 255) & 0xff) << 24);

    int idx = 0;
    for (int iz = 0; iz < d && idx < num_particles; iz++) {
        for (int iy = 0; iy < h && idx < num_particles; iy++) {
            for (int ix = 0; ix < w && idx < num_particles; ix++, idx++) {
                particles[idx].position[0] = origin.x + ix;
                particles[idx].position[1] = origin.y + iy;
                particles[idx].position[2] = origin.z + iz;
                particles[idx].color_packed        = col_packed;
                particles[idx].attraction_force    = attraction;
                particles[idx].opacity_fade        = 1.0f;
                particles[idx].neighbors_filled    = 0.0f;
            }
        }
    }

    rd->buffer_update(particle_buf, 0, raw.size(), raw);
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::add_velocity_impulse(Vector3 impulse) {
    pending_impulse  = impulse;
    impulse_pending  = true;
}

// ─────────────────────────────────────────────────────────────────────────────
void FluidParticleSystem::reset_grid() {
    if (!gpu_ready) return;
    _dispatch_clear_grid();
    rd->submit();
    rd->sync();
}
