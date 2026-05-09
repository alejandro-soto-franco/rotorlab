#version 450

// Push constants (mirrors PointPushConstants in Rust).
layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    float time;
    vec2 viewport;
    float _pad;
} pc;

// Instance attributes (binding 0, instance rate).
layout(location = 0) in vec4 in_world_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in float in_radius;

// Outputs to the fragment shader.
layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_quad_uv;

void main() {
    // Quad corners in [-1, 1] from gl_VertexIndex, TRIANGLE_STRIP order.
    // 0: (-1, -1), 1: (1, -1), 2: (-1, 1), 3: (1, 1).
    vec2 corner = vec2(
        ((gl_VertexIndex & 1) == 0) ? -1.0 : 1.0,
        ((gl_VertexIndex & 2) == 0) ? -1.0 : 1.0
    );

    // Transform world position to clip space.
    vec4 clip = pc.view_proj * in_world_pos;

    // Convert screen-pixel radius into clip-space offset.
    // radius_pixels / (viewport_pixels / 2) = radius_clip per axis.
    vec2 radius_clip = (in_radius * 2.0) / pc.viewport;

    // Apply perspective divide compensation so the disk stays pixel-sized
    // regardless of depth (offset is added before the implicit divide by w).
    clip.xy += corner * radius_clip * clip.w;

    gl_Position = clip;
    v_color = in_color;
    v_quad_uv = corner;
}
