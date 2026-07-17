// Shared helpers for the traced pool, ported from Evan Wallace's WebGL Water.
// Prepended to every render shader (the way the original concatenates
// helperFunctions). All rendering happens in the normalized pool space: the
// pool interior is x and z in [-1, 1], the floor is at y = -1, the water rests
// at y = 0, and the walls rise to the rim at y = 2/12.

const IOR_AIR: f32 = 1.0;
const IOR_WATER: f32 = 1.333;
const ABOVEWATER_COLOR: vec3<f32> = vec3<f32>(0.25, 1.0, 1.25);
const UNDERWATER_COLOR: vec3<f32> = vec3<f32>(0.4, 0.9, 1.0);
const POOL_HEIGHT: f32 = 1.0;

struct RenderUniform {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    sphere: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: RenderUniform;
@group(0) @binding(1) var water_tex: texture_2d<f32>;
@group(0) @binding(2) var tiles_tex: texture_2d<f32>;
@group(0) @binding(3) var caustic_tex: texture_2d<f32>;
@group(0) @binding(4) var sky_tex: texture_cube<f32>;
@group(0) @binding(5) var samp_clamp: sampler;
@group(0) @binding(6) var samp_repeat: sampler;
@group(0) @binding(7) var samp_cube: sampler;

fn light_dir() -> vec3<f32> {
    return u.light.xyz;
}

fn sphere_center() -> vec3<f32> {
    return u.sphere.xyz;
}

fn sphere_radius() -> f32 {
    return u.sphere.w;
}

fn water_height(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(water_tex, samp_clamp, uv, 0.0).r;
}

fn intersect_cube(origin: vec3<f32>, ray: vec3<f32>, cube_min: vec3<f32>, cube_max: vec3<f32>) -> vec2<f32> {
    let t_min = (cube_min - origin) / ray;
    let t_max = (cube_max - origin) / ray;
    let t1 = min(t_min, t_max);
    let t2 = max(t_min, t_max);
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);
    return vec2<f32>(t_near, t_far);
}

fn intersect_sphere(origin: vec3<f32>, ray: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let to_sphere = origin - center;
    let a = dot(ray, ray);
    let b = 2.0 * dot(to_sphere, ray);
    let c = dot(to_sphere, to_sphere) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant > 0.0 {
        let t = (-b - sqrt(discriminant)) / (2.0 * a);
        if t > 0.0 {
            return t;
        }
    }
    return 1.0e6;
}

fn get_sphere_color(point: vec3<f32>) -> vec3<f32> {
    let radius = sphere_radius();
    let center = sphere_center();
    var color = vec3<f32>(0.5);

    color *= 1.0 - 0.9 / pow((1.0 + radius - abs(point.x)) / radius, 3.0);
    color *= 1.0 - 0.9 / pow((1.0 + radius - abs(point.z)) / radius, 3.0);
    color *= 1.0 - 0.9 / pow((point.y + 1.0 + radius) / radius, 3.0);

    let sphere_normal = (point - center) / radius;
    let refracted_light = refract(-light_dir(), vec3<f32>(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(-refracted_light, sphere_normal)) * 0.5;
    if point.y < water_height(point.xz * 0.5 + 0.5) {
        let caustic_uv = 0.75 * (point.xz - point.y * refracted_light.xz / refracted_light.y) * 0.5 + 0.5;
        let caustic = textureSampleLevel(caustic_tex, samp_clamp, caustic_uv, 0.0);
        diffuse *= caustic.r * 4.0;
    }
    color += diffuse;
    return color;
}

fn get_wall_color(point: vec3<f32>) -> vec3<f32> {
    var scale = 0.5;

    var wall_color: vec3<f32>;
    var normal: vec3<f32>;
    if abs(point.x) > 0.999 {
        wall_color = textureSampleLevel(tiles_tex, samp_repeat, point.yz * 0.5 + vec2<f32>(1.0, 0.5), 0.0).rgb;
        normal = vec3<f32>(-point.x, 0.0, 0.0);
    } else if abs(point.z) > 0.999 {
        wall_color = textureSampleLevel(tiles_tex, samp_repeat, point.yx * 0.5 + vec2<f32>(1.0, 0.5), 0.0).rgb;
        normal = vec3<f32>(0.0, 0.0, -point.z);
    } else {
        wall_color = textureSampleLevel(tiles_tex, samp_repeat, point.xz * 0.5 + 0.5, 0.0).rgb;
        normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    scale /= length(point);
    scale *= 1.0 - 0.9 / pow(length(point - sphere_center()) / sphere_radius(), 4.0);

    let refracted_light = -refract(-light_dir(), vec3<f32>(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(refracted_light, normal));
    if point.y < water_height(point.xz * 0.5 + 0.5) {
        let caustic_uv = 0.75 * (point.xz - point.y * refracted_light.xz / refracted_light.y) * 0.5 + 0.5;
        let caustic = textureSampleLevel(caustic_tex, samp_clamp, caustic_uv, 0.0);
        scale += diffuse * caustic.r * 2.0 * caustic.g;
    } else {
        let t = intersect_cube(point, refracted_light, vec3<f32>(-1.0, -POOL_HEIGHT, -1.0), vec3<f32>(1.0, 2.0, 1.0));
        diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refracted_light.y * t.y - 2.0 / 12.0)));
        scale += diffuse * 0.5;
    }

    return wall_color * scale;
}
