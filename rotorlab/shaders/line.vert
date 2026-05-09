#version 450

// Push constants (mirrors LinePushConstants in Rust).
layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    vec2 viewport;
    vec2 _pad;
} pc;

// Instance attributes (binding 0, instance rate).
layout(location = 0) in vec4  in_a;
layout(location = 1) in vec4  in_b;
layout(location = 2) in vec4  in_color;
layout(location = 3) in float in_thickness;

// Outputs to the fragment shader.
layout(location = 0) out vec4  v_color;
layout(location = 1) out vec2  v_quad_uv;
layout(location = 2) out float v_aa_width;

void main() {
    // Project endpoints to clip space.
    vec4 ca = pc.view_proj * in_a;
    vec4 cb = pc.view_proj * in_b;

    // Convert to screen space (NDC times half-viewport).
    vec2 sa = (ca.xy / ca.w) * 0.5 * pc.viewport;
    vec2 sb = (cb.xy / cb.w) * 0.5 * pc.viewport;

    // Screen-space line direction and unit perpendicular normal.
    vec2 dir = sb - sa;
    float len = length(dir);
    vec2 n = (len > 1e-6) ? vec2(-dir.y, dir.x) / len : vec2(0.0, 1.0);

    // TRIANGLE_STRIP corner mapping (matches the strip used by point.vert):
    //   0: (a, -t/2)   1: (b, -t/2)   2: (a, +t/2)   3: (b, +t/2)
    bool at_b   = (gl_VertexIndex & 1) == 1;
    float side  = ((gl_VertexIndex & 2) == 0) ? -1.0 : 1.0;

    vec4 base = at_b ? cb : ca;

    // Half-thickness offset:
    //   screen_offset_pixels = side * (thickness / 2) * n
    //   clip_offset = screen_offset_pixels * 2 / viewport, scaled by w to
    //   survive the perspective divide.
    vec2 screen_offset = side * (in_thickness * 0.5) * n;
    vec2 clip_offset = (screen_offset * 2.0) / pc.viewport;

    base.xy += clip_offset * base.w;

    gl_Position = base;
    v_color = in_color;
    // Across-line UV is in [-1, +1] so the fragment can fade |y| -> 1 the
    // same way the point disk fades |uv| -> 1.
    v_quad_uv = vec2(at_b ? 1.0 : 0.0, side);
    // Reciprocal half-thickness, plumbed for future fragment effects (dash
    // patterns, arrowheads). Unused by line.frag in Plan 3.
    v_aa_width = (in_thickness > 0.0) ? (2.0 / in_thickness) : 0.0;
}
