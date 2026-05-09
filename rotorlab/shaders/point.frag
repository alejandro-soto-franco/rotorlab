#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_quad_uv;

layout(location = 0) out vec4 out_color;

void main() {
    // Filled disk: discard pixels outside the unit disk |uv| > 1.
    float r = length(v_quad_uv);
    if (r > 1.0) {
        discard;
    }

    // Edge antialiasing: blend over the width of one screen-space pixel via
    // fwidth (always available in standard graphics pipelines, no extension
    // required).
    float aa = fwidth(r);
    float alpha = 1.0 - smoothstep(1.0 - aa, 1.0, r);

    out_color = vec4(v_color.rgb, v_color.a * alpha);
}
