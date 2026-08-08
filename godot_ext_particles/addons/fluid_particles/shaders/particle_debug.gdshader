// Simple debug point shader — vertex color only, no ray-sphere intersection.
// Used by DEBUG_SIMPLE_POINTS and DEBUG_STATIC modes.
// Particles appear as flat colored discs sized by distance to camera.

shader_type spatial;
render_mode unshaded, depth_draw_always, cull_disabled, skip_vertex_transform;

varying flat vec4 v_color;

void vertex() {
    v_color = COLOR;

    vec4 eye_pos = VIEW_MATRIX * MODEL_MATRIX * vec4(VERTEX, 1.0);
    POSITION  = PROJECTION_MATRIX * eye_pos;

    // Fixed world-space radius of 0.5 units projected to screen pixels
    float dist      = max(length(eye_pos.xyz), 0.001);
    float focal_len = PROJECTION_MATRIX[1][1];
    POINT_SIZE = max((0.5 * focal_len * VIEWPORT_SIZE.y) / dist, 1.5);
}

void fragment() {
    // Round point sprite
    vec2 p = POINT_COORD - vec2(0.5);
    if (dot(p, p) > 0.25) discard;

    ALBEDO = v_color.rgb;
    ALPHA  = max(v_color.a, 0.5);  // boost alpha so invisible particles are still faintly visible
}
