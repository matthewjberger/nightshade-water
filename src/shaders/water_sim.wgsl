// GPU water height-field simulation, ported from Evan Wallace's WebGL Water.
// State texel packs (height, velocity, normal.x, normal.z). Each stage reads A
// into B, then `normals` reads B into A, so A always holds the current state.
// `inject` (drops and the sphere volume) runs once per frame while `update` (the
// wave step) runs a frame-rate-corrected number of times, so interaction is not
// multiplied by the number of steps.

struct SimUniform {
    drop: vec4<f32>,          // xy = center in [0,1], z = radius, w = strength
    flags: vec4<f32>,         // x = drop active, y = texel size (1/N)
    sphere_old: vec4<f32>,    // xyz = old center in [-1,1] pool space, w = radius
    sphere_new: vec4<f32>,    // xyz = new center in [-1,1] pool space
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> sim: SimUniform;

const PI: f32 = 3.141592653589793;

fn load_clamped(coord: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    let clamped = clamp(coord, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
    return textureLoad(src_tex, clamped, 0);
}

fn volume_in_sphere(plane_x: f32, plane_z: f32, center: vec3<f32>, radius: f32) -> f32 {
    let to_center = vec3<f32>(plane_x - center.x, -center.y, plane_z - center.z);
    let t = length(to_center) / radius;
    let thickness = exp(-pow(t * 1.5, 6.0));
    let y_min = min(0.0, center.y - thickness);
    let y_max = min(max(0.0, center.y + thickness), y_min + 2.0 * thickness);
    return (y_max - y_min) * 0.1;
}

@compute @workgroup_size(8, 8, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(src_tex));
    if i32(id.x) >= size.x || i32(id.y) >= size.y {
        return;
    }
    textureStore(dst_tex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(src_tex));
    if i32(id.x) >= size.x || i32(id.y) >= size.y {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    var info = textureLoad(src_tex, coord, 0);

    let left = load_clamped(coord + vec2<i32>(-1, 0), size).r;
    let right = load_clamped(coord + vec2<i32>(1, 0), size).r;
    let down = load_clamped(coord + vec2<i32>(0, -1), size).r;
    let up = load_clamped(coord + vec2<i32>(0, 1), size).r;
    let average = (left + right + down + up) * 0.25;

    info.g += (average - info.r) * 2.0;
    info.g *= 0.995;
    info.r += info.g;

    textureStore(dst_tex, coord, info);
}

@compute @workgroup_size(8, 8, 1)
fn inject(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(src_tex));
    if i32(id.x) >= size.x || i32(id.y) >= size.y {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    var info = textureLoad(src_tex, coord, 0);

    if sim.flags.x > 0.5 {
        let uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + 0.5) / vec2<f32>(size);
        let distance = length(sim.drop.xy - uv);
        let falloff = max(0.0, 1.0 - distance / sim.drop.z);
        info.r += (0.5 - cos(falloff * PI) * 0.5) * sim.drop.w;
    }

    let plane_x = (f32(coord.x) + 0.5) / f32(size.x) * 2.0 - 1.0;
    let plane_z = (f32(coord.y) + 0.5) / f32(size.y) * 2.0 - 1.0;
    let added = volume_in_sphere(plane_x, plane_z, sim.sphere_old.xyz, sim.sphere_old.w);
    let removed = volume_in_sphere(plane_x, plane_z, sim.sphere_new.xyz, sim.sphere_new.w);
    info.r += added - removed;

    textureStore(dst_tex, coord, info);
}

@compute @workgroup_size(8, 8, 1)
fn normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(src_tex));
    if i32(id.x) >= size.x || i32(id.y) >= size.y {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    var info = textureLoad(src_tex, coord, 0);

    let delta = sim.flags.y;
    let right = load_clamped(coord + vec2<i32>(1, 0), size).r;
    let up = load_clamped(coord + vec2<i32>(0, 1), size).r;
    let dx = vec3<f32>(delta, right - info.r, 0.0);
    let dy = vec3<f32>(0.0, up - info.r, delta);
    let normal = normalize(cross(dy, dx));
    info.b = normal.x;
    info.a = normal.z;

    textureStore(dst_tex, coord, info);
}
