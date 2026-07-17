// Projected caustics, ported from the original causticsShader. Each water plane
// vertex is traced along the refracted sunlight onto the floor. The fragment
// stage measures how much the triangle shrank or grew with screen-space
// derivatives: light that focuses onto a smaller area is brighter. The result
// bakes into a texture that lights the floor, walls, and ball, carrying the
// ball's blob shadow in the green channel and the pool-rim shadow in red.

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) old_pos: vec3<f32>,
    @location(1) new_pos: vec3<f32>,
    @location(2) ray: vec3<f32>,
}

fn project(origin: vec3<f32>, ray: vec3<f32>, refracted_light: vec3<f32>) -> vec3<f32> {
    let t_cube = intersect_cube(origin, ray, vec3<f32>(-1.0, -POOL_HEIGHT, -1.0), vec3<f32>(1.0, 2.0, 1.0));
    let hit = origin + ray * t_cube.y;
    let t_plane = (-hit.y - 1.0) / refracted_light.y;
    return hit + refracted_light * t_plane;
}

@vertex
fn vs_main(@location(0) vertex: vec2<f32>) -> VertexOutput {
    var info = textureSampleLevel(water_tex, samp_clamp, vertex * 0.5 + 0.5, 0.0);
    let softened = info.ba * 0.5;
    let normal = vec3<f32>(softened.x, sqrt(max(0.0, 1.0 - dot(softened, softened))), softened.y);

    let refracted_light = refract(-light_dir(), vec3<f32>(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    let ray = refract(-light_dir(), normal, IOR_AIR / IOR_WATER);
    let base = vec3<f32>(vertex.x, 0.0, vertex.y);

    var out: VertexOutput;
    out.old_pos = project(base, refracted_light, refracted_light);
    out.new_pos = project(base + vec3<f32>(0.0, info.r, 0.0), ray, refracted_light);
    out.ray = ray;
    out.clip = vec4<f32>(0.75 * (out.new_pos.xz + refracted_light.xz / refracted_light.y), 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let old_area = length(dpdx(in.old_pos)) * length(dpdy(in.old_pos));
    let new_area = length(dpdx(in.new_pos)) * length(dpdy(in.new_pos));
    var color = vec4<f32>(old_area / new_area * 0.2, 1.0, 0.0, 0.0);

    let refracted_light = refract(-light_dir(), vec3<f32>(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);

    let dir = (sphere_center() - in.new_pos) / sphere_radius();
    let area = cross(dir, refracted_light);
    var shadow = dot(area, area);
    let dist = dot(dir, -refracted_light);
    shadow = 1.0 + (shadow - 1.0) / (0.05 + dist * 0.025);
    shadow = clamp(1.0 / (1.0 + exp(-shadow)), 0.0, 1.0);
    shadow = mix(1.0, shadow, clamp(dist * 2.0, 0.0, 1.0));
    color.g = shadow;

    let t = intersect_cube(in.new_pos, -refracted_light, vec3<f32>(-1.0, -POOL_HEIGHT, -1.0), vec3<f32>(1.0, 2.0, 1.0));
    color.r *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (in.new_pos.y - refracted_light.y * t.y - 2.0 / 12.0)));

    return color;
}
