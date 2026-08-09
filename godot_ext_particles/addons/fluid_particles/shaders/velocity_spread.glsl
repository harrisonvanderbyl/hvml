#[compute]
#version 450

// No float atomic extension needed: velocities stored as uint bit-patterns.
// atomicCompSwap on buffer members (not inout params) is standard GLSL.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant, std430) uniform PushConstants {
    int   grid_w, grid_h, grid_d;
    int   num_particles;
    float surface_tension;
    float water_viscosity;
    float attraction_force;
    float _pad0;         // padding before vec3s
    vec3  gravity;       // gravity vector in grid space
    int   frame_count;
    vec3  global_vel;
    int   neighbor_mode;
    int   num_runnable;
    int   _pad1[3];
} pc;

struct Particle {
    vec3  position;
    float _pad0;
    uint  color_packed;
    float attraction_force;
    float opacity_fade;
    float neighbors_filled;
    float _pad1[4];  // 16 bytes trailing pad → total 48 bytes, matches GPUParticle
};

struct ChunkCell {
    int  occupant;
    uint vel_x_bits;
    uint vel_y_bits;
    uint vel_z_bits;
    uint vel_w_bits;
    uint _pad[3];
};

layout(set = 0, binding = 0, std430) buffer ParticleBuffer { Particle  particles[]; };
layout(set = 0, binding = 1, std430) buffer ChunkBuffer    { ChunkCell cells[]; };
layout(set = 0, binding = 2, std430) buffer RunnableBuffer { int runnable_indices[]; };
layout(set = 0, binding = 3, std430) buffer SortKeyBuffer  { float sort_keys[]; };

int cell_index(ivec3 p) {
    p = clamp(p, ivec3(0), ivec3(pc.grid_w-1, pc.grid_h-1, pc.grid_d-1));
    return p.x + p.y * pc.grid_w + p.z * pc.grid_w * pc.grid_h;
}

bool in_bounds(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(pc.grid_w, pc.grid_h, pc.grid_d)));
}

// Swap velocity to 0, return old value. atomicExchange on uint is always valid.
vec3 cell_swap0(int cidx) {
    return vec3(
        uintBitsToFloat(atomicExchange(cells[cidx].vel_x_bits, 0u)),
        uintBitsToFloat(atomicExchange(cells[cidx].vel_y_bits, 0u)),
        uintBitsToFloat(atomicExchange(cells[cidx].vel_z_bits, 0u))
    );
}

// Atomic float-add via CAS loop directly on buffer members (not inout params).
void cell_atomic_add(int cidx, vec3 v) {
    uint prev, next;
    do { prev = cells[cidx].vel_x_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.x); } while (atomicCompSwap(cells[cidx].vel_x_bits, prev, next) != prev);
    do { prev = cells[cidx].vel_y_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.y); } while (atomicCompSwap(cells[cidx].vel_y_bits, prev, next) != prev);
    do { prev = cells[cidx].vel_z_bits; next = floatBitsToUint(uintBitsToFloat(prev) + v.z); } while (atomicCompSwap(cells[cidx].vel_z_bits, prev, next) != prev);
}

bool cell_try_place(int cidx, int pidx) {
    return atomicCompSwap(cells[cidx].occupant, -1, pidx) == -1;
}

void cell_clear_occupant(int cidx) {
    atomicExchange(cells[cidx].occupant, -1);
}

const ivec3 NEIGHBOR_OFFSETS[15] = ivec3[15](
    ivec3(-1,0,0), ivec3(1,0,0), ivec3(0,-1,0), ivec3(0,1,0),
    ivec3(0,0,-1), ivec3(0,0,1), ivec3(0,0,0),
    ivec3(1,-1,1), ivec3(-1,-1,1), ivec3(1,-1,-1), ivec3(-1,-1,-1),
    ivec3(1,1,1),  ivec3(1,1,-1), ivec3(-1,1,-1), ivec3(-1,1,1)
);

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= pc.num_runnable) return;
    int particle_idx = runnable_indices[gid];
    if (particle_idx < 0 || particle_idx >= pc.num_particles) return;

    Particle p = particles[particle_idx];

    // Skip inactive particles (position.x == NaN sentinel from source/sink)
    if (!(p.position.x == p.position.x)) return;

    if (length(pc.global_vel) > 0.0) p.opacity_fade = 1.0;
    if (p.opacity_fade < 1.0) p.opacity_fade += 1.0 / 60.0;

    bool  firststep    = false;
    ivec3 old_cell_pos = ivec3(round(p.position));
    int   old_cidx     = cell_index(old_cell_pos);

    if (cells[old_cidx].occupant == -1) {
        cell_try_place(old_cidx, particle_idx);
        firststep = true;
    }

    int   n_size      = pc.neighbor_mode;
    int   n_allowance = (n_size == 15) ? 2 : 0;
    float friction    = 1.0 / float(n_size);
    vec3  momentum    = vec3(0.0);
    int   allfilled   = 0;
    bool  me_solid    = ((p.color_packed >> 24) & 0xffu) > 200u;

    for (int i = 0; i < n_size; i++) {
        ivec3 npos = old_cell_pos + NEIGHBOR_OFFSETS[i];
        if (!in_bounds(npos)) continue;
        int ncidx = cell_index(npos);
        momentum += cell_swap0(ncidx);
        int occ = cells[ncidx].occupant;
        if (occ >= 0 && occ < pc.num_particles) {
            bool part_solid = ((particles[occ].color_packed >> 24) & 0xffu) > 200u;
            allfilled += (me_solid ^^ part_solid) ? 0 : 1;
        }
    }

    float attract = p.attraction_force * pc.attraction_force;
    float mix_f   = 0.98;

    if (allfilled < n_size - n_allowance || !me_solid) {
        p.neighbors_filled = float(allfilled * 15) / float(n_size) * (1.0 - mix_f) + p.neighbors_filled * mix_f;
        p.position += momentum - pc.global_vel;
        if (firststep) attract = 0.0;
        if (!me_solid) {
            float coverage = float(n_size - allfilled + 1) / float(n_size);
            attract *= 1.0 - pow(coverage, 0.85) * pc.surface_tension;
        }
    } else {
        p.opacity_fade     = 0.0;
        p.neighbors_filled = 0.0;
    }

    momentum -= pc.gravity;

    float sz = 1.0;
    if (p.position.y < 1.0 + sz || p.position.y > float(pc.grid_h) - 1.0 - sz) {
        p.position.y = clamp(p.position.y, 2.0 + sz, float(pc.grid_h) - 2.0 - sz);
        momentum.y = 0.0;
    }
    if (p.position.x < 1.0 + sz || p.position.x > float(pc.grid_w) - 1.0 - sz) {
        p.position.x = clamp(p.position.x, 2.0 + sz, float(pc.grid_w) - 2.0 - sz);
        momentum.x = 0.0;
    }
    if (p.position.z < 1.0 + sz || p.position.z > float(pc.grid_d) - 1.0 - sz) {
        p.position.z = clamp(p.position.z, 2.0 + sz, float(pc.grid_d) - 2.0 - sz);
        momentum.z = 0.0;
    }

    ivec3 new_cell_pos = ivec3(round(p.position));
    int   new_cidx     = cell_index(new_cell_pos);

    if (new_cell_pos != old_cell_pos) {
        if (!cell_try_place(new_cidx, particle_idx)) {
            p.position   = vec3(old_cell_pos);
            cell_atomic_add(new_cidx, momentum * 0.5);
            momentum    *= 0.45;
            new_cell_pos = old_cell_pos;
        } else {
            cell_clear_occupant(old_cidx);
        }
    }

    vec3 spread = momentum * friction;
    for (int i = 0; i < n_size; i++) {
        ivec3 npos = new_cell_pos + NEIGHBOR_OFFSETS[i];
        if (!in_bounds(npos)) continue;
        cell_atomic_add(cell_index(npos), spread + vec3(NEIGHBOR_OFFSETS[i]) * attract);
    }

    // NOTE: attraction_force is NOT written back — it's a per-particle constant
    // set at spawn. Writing the modified local 'attract' back would cause it to
    // decay to zero each frame via the surface tension multiplier.
    particles[particle_idx] = p;
}
