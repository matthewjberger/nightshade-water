// Caustics: a floor-aligned quad that adds focused light to the pool floor,
// driven by the GPU height field. Where the surface is concave it focuses light
// (positive Laplacian of the height), which is what a caustic is. Additively
// blended onto the floor, then seen through the water.

struct RenderUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    shallow: vec4<f32>,
    deep: vec4<f32>,
    params: vec4<f32>,  // x = extent, y = water level, z = height scale, w = floor y
}

@group(0) @binding(0) var<uniform> u: RenderUniform;
@group(0) @binding(1) var state_tex: texture_2d<f32>;
@group(0) @binding(2) var state_samp: sampler;

const TEXEL: f32 = 1.0 / 256.0;

struct VertexInput {
    @location(0) plane: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let extent = u.params.x;
    let floor_y = u.params.w;
    let world = vec3<f32>(in.plane.x * extent, floor_y + 0.05, in.plane.y * extent);
    var out: VertexOutput;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.uv = in.plane * 0.5 + 0.5;
    return out;
}

fn height(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(state_tex, state_samp, uv, 0.0).r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center = height(in.uv);
    let left = height(in.uv + vec2<f32>(-TEXEL, 0.0));
    let right = height(in.uv + vec2<f32>(TEXEL, 0.0));
    let down = height(in.uv + vec2<f32>(0.0, -TEXEL));
    let up = height(in.uv + vec2<f32>(0.0, TEXEL));

    let laplacian = left + right + down + up - 4.0 * center;
    let caustic = clamp(laplacian * 900.0, 0.0, 1.6);
    let color = vec3<f32>(0.45, 0.72, 1.0) * caustic;
    return vec4<f32>(color, 1.0);
}
