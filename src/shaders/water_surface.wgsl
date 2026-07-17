// Renders the water volume: the animated top surface plus skirts down to the
// floor. Vertices flagged 0 are the surface (displaced by the simulated height);
// flagged 1 are the skirt bottom at the floor. Alpha blended over the scene and
// tinted by depth so the pool reads as filled with water.

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

struct VertexInput {
    @location(0) data: vec3<f32>,  // xy = plane in [-1,1], z = flag (0 surface, 1 skirt bottom)
}

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let plane = in.data.xy;
    let uv = plane * 0.5 + 0.5;
    let info = textureSampleLevel(state_tex, state_samp, uv, 0.0);
    let extent = u.params.x;
    let level = u.params.y;
    let height_scale = u.params.z;
    let floor_y = u.params.w;

    var world_y = floor_y;
    if in.data.z < 0.5 {
        world_y = level + info.r * height_scale;
    }
    let world = vec3<f32>(plane.x * extent, world_y, plane.y * extent);

    var out: VertexOutput;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let info = textureSampleLevel(state_tex, state_samp, in.uv, 0.0);
    let normal_y = sqrt(max(0.0, 1.0 - info.b * info.b - info.a * info.a));
    let normal = normalize(vec3<f32>(info.b, normal_y, info.a));

    let view = normalize(u.camera_pos.xyz - in.world);
    let n_dot_v = max(dot(normal, view), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - n_dot_v, 5.0);

    let depth = clamp((u.params.y - in.world.y) / 3.0, 0.0, 1.0);
    let surface = 1.0 - depth;
    let water_tint = mix(u.shallow.rgb, u.deep.rgb, depth);
    let sky = vec3<f32>(0.72, 0.83, 0.95);
    var color = mix(water_tint, sky, fresnel * surface);

    let sun = normalize(u.sun_dir.xyz);
    let half_vec = normalize(view + sun);
    let specular = pow(max(dot(normal, half_vec), 0.0), 220.0) * surface;
    color += u.sun_color.rgb * specular;

    let alpha = clamp(0.4 + depth * 0.4 + fresnel * surface * 0.3 + specular, 0.0, 0.94);
    return vec4<f32>(color, alpha);
}
