#[vertex]
#version 450

// ── Particle struct (std430, matches compute shaders) ─────────────────────────
struct Particle {
    vec3  position;
    float _pad0;
    uint  color_packed;
    float attraction_force;
    float opacity_fade;
    float neighbors_filled;
    float _pad1[4];
};

layout(set = 0, binding = 0, std430) readonly buffer ParticleBuffer {
    Particle particles[];
};

layout(push_constant, std430) uniform PushConstants {
    mat4  proj_view;
    mat4  view;
    mat4  inv_view;
    mat4  inv_proj;
    vec2  screen_size;
    vec2  _pad;
    int   num_particles;
    vec3  _pad2;
} pc;

layout(location = 0) out vec4  v_world_pos;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_radius;

void main() {
    int idx = gl_VertexIndex;
    if (idx >= pc.num_particles) {
        gl_Position  = vec4(0.0);
        gl_PointSize = 0.0;
        return;
    }

    Particle p = particles[idx];

    // Skip inactive particles (NaN sentinel from source/sink)
    if (!(p.position.x == p.position.x)) {
        gl_Position  = vec4(0.0);
        gl_PointSize = 0.0;
        return;
    }

    if (p.neighbors_filled < 0.5) {
        gl_Position  = vec4(0.0);
        gl_PointSize = 0.0;
        return;
    }

    bool is_solid = ((p.color_packed >> 24) & 0xffu) > 200u;

    float radius_grow = is_solid ? 0.125 : 0.0;
    float radius      = is_solid ? -0.125 : 1.0;
    radius += p.neighbors_filled * radius_grow;
    radius  = max(radius, 0.0);

    v_radius    = radius;
    v_world_pos = vec4(p.position, 1.0);

    vec4 col = vec4(
        float((p.color_packed >>  0) & 0xffu) / 255.0,
        float((p.color_packed >>  8) & 0xffu) / 255.0,
        float((p.color_packed >> 16) & 0xffu) / 255.0,
        float((p.color_packed >> 24) & 0xffu) / 255.0
    );
    if (!is_solid) {
        float blend = pow(1.0 - p.neighbors_filled / 15.0, 3.0);
        col.rgb = mix(col.rgb, vec3(0.5), blend);
    }
    v_color = col;

    vec4 eye_pos = pc.view * v_world_pos;
    gl_Position  = pc.proj_view * v_world_pos;

    float dist      = length(eye_pos.xyz);
    float focal_len = pc.proj_view[1][1];
    gl_PointSize = max((radius * focal_len * pc.screen_size.y) / max(dist, 0.001), 0.0);
}

#[fragment]
#version 450

layout(push_constant, std430) uniform PushConstants {
    mat4  proj_view;
    mat4  view;
    mat4  inv_view;
    mat4  inv_proj;
    vec2  screen_size;
    vec2  _pad;
    int   num_particles;
    vec3  _pad2;
} pc;

layout(set = 1, binding = 0) uniform sampler2D texture1;
layout(set = 1, binding = 1) uniform sampler2D screen_texture;

layout(location = 0) in vec4  v_world_pos;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_radius;

layout(location = 0) out vec4 frag_color;

void main() {
    vec2  puv = gl_PointCoord - vec2(0.5);
    if (dot(puv, puv) > 0.25) discard;

    bool is_solid = (v_color.a > (200.0 / 255.0));

    vec2  ndc_xy  = (gl_FragCoord.xy / pc.screen_size) * 2.0 - 1.0;
    float ndc_z   = gl_FragCoord.z * 2.0 - 1.0;
    vec4  clip    = vec4(ndc_xy, ndc_z, 1.0);
    vec4  view_p  = pc.inv_proj * clip;
    view_p       /= view_p.w;
    vec4  world_p = pc.inv_view * view_p;

    vec3 cam_pos = vec3(pc.inv_view[3][0], pc.inv_view[3][1], pc.inv_view[3][2]);
    vec3 ray_dir = normalize(world_p.xyz - cam_pos);

    vec3  oc   = cam_pos - v_world_pos.xyz;
    float a    = dot(ray_dir, ray_dir);
    float b    = 2.0 * dot(oc, ray_dir);
    float c    = dot(oc, oc) - v_radius * v_radius;
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0) discard;

    float t      = (-b - sqrt(disc)) / (2.0 * a);
    vec3  hit    = cam_pos + t * ray_dir;
    vec3  normal = normalize(hit - v_world_pos.xyz);

    vec4 clip_hit = pc.proj_view * vec4(hit, 1.0);
    gl_FragDepth  = (clip_hit.z / clip_hit.w) * 0.5 + 0.5;

    if (!is_solid) {
        vec2  screen_uv    = gl_FragCoord.xy / pc.screen_size;
        vec2  distorted_uv = clamp(screen_uv + normal.xy * 0.02, vec2(0.0), vec2(1.0));
        vec3  bg           = texture(screen_texture, distorted_uv).rgb;
        vec3  view_dir     = normalize(cam_pos - hit);
        float fresnel      = pow(1.0 - abs(dot(view_dir, normal)), 3.0);
        frag_color = vec4(bg * vec3(0.85, 0.92, 1.0) + v_color.rgb + vec3(fresnel * 0.143), 1.0);
        return;
    }

    float light = max(dot(normal, vec3(0.0, 1.0, 0.0)), 0.0);
    vec3  abs_n = abs(normal);
    vec3  w     = abs_n / (abs_n.x + abs_n.y + abs_n.z + 0.0001);
    float dist_to_cam = length(cam_pos - hit);
    vec3  base;

    if (dist_to_cam > 50.0) {
        base = texture(texture1, mod(hit.xz * 0.1, 1.0)).rgb * v_color.rgb;
    } else {
        vec3 t1 = texture(texture1, mod(hit.yz * 0.1, 1.0)).rgb;
        vec3 t2 = texture(texture1, mod(hit.xz * 0.1, 1.0)).rgb;
        vec3 t3 = texture(texture1, mod(hit.xy * 0.1, 1.0)).rgb;
        base = (t1 * w.x + t2 * w.y + t3 * w.z) * v_color.rgb;
    }

    frag_color = vec4(base * light, v_color.a);
}
