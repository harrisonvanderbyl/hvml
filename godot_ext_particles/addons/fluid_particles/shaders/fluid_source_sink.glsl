#[compute]
#version 450

// ─── Fluid Source / Sink compute shader ──────────────────────────────────────
// One shader handles both operations:
//   mode = 0  → SOURCE: find inactive particles (position.x == NaN) and
//                activate them within radius of the source position, setting
//                their attributes to the source's values.
//   mode = 1  → SINK:   find active particles within radius of the sink
//                position and deactivate them (set position.x = NaN).
//
// Inactive sentinel: position.x = NaN (0x7FC00000 when stored as uint bits).
// This costs zero extra memory — NaN is already representable in the float
// position field.
//
// Dispatch: 1 group of 64 threads. Each thread scans a strided subset of
// particles. Atomic operations ensure only the right number of particles are
// claimed/released.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant, std430) uniform PushConstants {
    vec3  world_pos;     // source/sink position in grid space
    float radius;        // effect radius
    vec4  color;         // color to assign (source mode only)
    float attraction;    // attraction_force to assign (source mode only)
    float opacity_fade;  // opacity_fade to assign (source mode only)
    int   mode;          // 0 = source, 1 = sink
    int   max_count;     // max particles to activate/deactivate this dispatch
    int   num_particles; // total particles in buffer
    int   grid_w, grid_h, grid_d;
    int   _pad;
} pc;

struct Particle {
    vec3  position;
    float _pad0;
    uint  color_packed;
    float attraction_force;
    float opacity_fade;
    float neighbors_filled;
    float _pad1[4];
};

layout(set = 0, binding = 0, std430) buffer ParticleBuffer { Particle particles[]; };

// Shared atomic counter — how many particles we've claimed/released this dispatch
shared int s_claimed;

bool is_inactive(int idx) {
    // NaN check: a float is NaN if it != itself
    return !(particles[idx].position.x == particles[idx].position.x);
}

void set_inactive(int idx) {
    // 0x7FC00000 = quiet NaN
    particles[idx].position = vec3(uintBitsToFloat(0x7FC00000u), 0.0, 0.0);
}

void main() {
    if (gl_LocalInvocationIndex == 0u) {
        s_claimed = 0;
    }
    barrier();

    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= pc.num_particles) return;

    float r2 = pc.radius * pc.radius;

    if (pc.mode == 0) {
        // ── SOURCE: activate inactive particles near world_pos ──────────────
        if (!is_inactive(int(gid))) return;

        vec3 d = particles[gid].position - pc.world_pos;
        // Inactive particles have NaN position, so the distance check above
        // would fail. Instead, we assign position first, then check.
        // Actually: inactive particles have NaN position, so we can't distance-check.
        // Strategy: each thread claims an inactive particle via atomicAdd on s_claimed,
        // then assigns it a position within radius of world_pos.
        // This is a "spawn" model — we grab any inactive particle and place it
        // near the source, rather than finding inactive particles that are already near.

        int slot = atomicAdd(s_claimed, 1);
        if (slot >= pc.max_count) return;

        // Place at a random-ish offset within radius using gid as seed
        uint seed = gid * 2654435761u + uint(pc.grid_w);
        float ang1 = float(seed % 1000u) / 1000.0 * 6.2831853;
        float ang2 = float((seed / 1000u) % 1000u) / 1000.0 * 3.14159265;
        float r = pc.radius * float((seed / 1000000u) % 1000u) / 1000.0;

        particles[gid].position = pc.world_pos + vec3(
            r * sin(ang2) * cos(ang1),
            r * sin(ang2) * sin(ang1),
            r * cos(ang2)
        );

        // Clamp to grid bounds
        particles[gid].position.x = clamp(particles[gid].position.x, 2.0, float(pc.grid_w) - 2.0);
        particles[gid].position.y = clamp(particles[gid].position.y, 2.0, float(pc.grid_h) - 2.0);
        particles[gid].position.z = clamp(particles[gid].position.z, 2.0, float(pc.grid_d) - 2.0);

        // Pack color
        uint cr = uint(clamp(pc.color.r * 255.0, 0.0, 255.0));
        uint cg = uint(clamp(pc.color.g * 255.0, 0.0, 255.0));
        uint cb = uint(clamp(pc.color.b * 255.0, 0.0, 255.0));
        uint ca = uint(clamp(pc.color.a * 255.0, 0.0, 255.0));
        particles[gid].color_packed     = cr | (cg << 8) | (cb << 16) | (ca << 24);
        particles[gid].attraction_force = pc.attraction;
        particles[gid].opacity_fade     = pc.opacity_fade;
        particles[gid].neighbors_filled = 0.0;

    } else {
        // ── SINK: deactivate active particles near world_pos ─────────────────
        if (is_inactive(int(gid))) return;

        vec3 d = particles[gid].position - pc.world_pos;
        if (dot(d, d) > r2) return;

        int slot = atomicAdd(s_claimed, 1);
        if (slot >= pc.max_count) return;

        set_inactive(int(gid));
    }
}
