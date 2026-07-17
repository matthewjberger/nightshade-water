// The pool box: floor and walls, drawn by tracing the tile texture with the
// projected caustics and underwater tint. Ported from the original cubeShader.

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = position;
    out.clip = u.view_proj * vec4<f32>(position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = get_wall_color(in.position);
    if in.position.y < water_height(in.position.xz * 0.5 + 0.5) {
        color *= UNDERWATER_COLOR * 1.2;
    }
    return vec4<f32>(color, 1.0);
}
