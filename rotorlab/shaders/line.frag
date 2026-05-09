#version 450

layout(location = 0) in vec4  v_color;
layout(location = 1) in vec2  v_quad_uv;
layout(location = 2) in float v_aa_width;

layout(location = 0) out vec4 out_color;

void main() {
    // Across-axis coordinate in [0, 1]: 0 at the line centre, 1 at the edge.
    float across = abs(v_quad_uv.y);

    // Edge AA via screen-space derivative: blend over one pixel of width.
    float aa = fwidth(across);
    float alpha = 1.0 - smoothstep(1.0 - aa, 1.0, across);

    out_color = vec4(v_color.rgb, v_color.a * alpha);
}
