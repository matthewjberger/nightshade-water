// The water surface: for each fragment, trace a reflected ray (into the sky
// cubemap with a sun glint) and a refracted ray (into the traced pool and
// sphere), and blend them by fresnel. Ported from the original waterShaders.
// vs_main displaces the plane grid by the simulated height; fs_above and
// fs_under are the two culling passes for the above and underwater views.

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_main(@location(0) vertex: vec2<f32>) -> VertexOutput {
    let info = textureSampleLevel(water_tex, samp_clamp, vertex * 0.5 + 0.5, 0.0);
    let position = vec3<f32>(vertex.x, info.r, vertex.y);
    var out: VertexOutput;
    out.position = position;
    out.clip = u.view_proj * vec4<f32>(position, 1.0);
    return out;
}

fn get_surface_ray_color(origin: vec3<f32>, ray: vec3<f32>, water_color: vec3<f32>) -> vec3<f32> {
    var color: vec3<f32>;
    let q = intersect_sphere(origin, ray, sphere_center(), sphere_radius());
    if q < 1.0e6 {
        color = get_sphere_color(origin + ray * q);
    } else if ray.y < 0.0 {
        let t = intersect_cube(origin, ray, vec3<f32>(-1.0, -POOL_HEIGHT, -1.0), vec3<f32>(1.0, 2.0, 1.0));
        color = get_wall_color(origin + ray * t.y);
    } else {
        let t = intersect_cube(origin, ray, vec3<f32>(-1.0, -POOL_HEIGHT, -1.0), vec3<f32>(1.0, 2.0, 1.0));
        let hit = origin + ray * t.y;
        if hit.y < 2.0 / 12.0 {
            color = get_wall_color(hit);
        } else {
            color = textureSampleLevel(sky_tex, samp_cube, ray, 0.0).rgb;
            color += vec3<f32>(pow(max(0.0, dot(light_dir(), ray)), 5000.0)) * vec3<f32>(10.0, 8.0, 6.0);
        }
    }
    if ray.y < 0.0 {
        color *= water_color;
    }
    return color;
}

fn surface_normal(position: vec3<f32>) -> vec3<f32> {
    var coord = position.xz * 0.5 + 0.5;
    var info = textureSampleLevel(water_tex, samp_clamp, coord, 0.0);
    for (var i = 0; i < 5; i = i + 1) {
        coord += info.ba * 0.005;
        info = textureSampleLevel(water_tex, samp_clamp, coord, 0.0);
    }
    return vec3<f32>(info.b, sqrt(max(0.0, 1.0 - dot(info.ba, info.ba))), info.a);
}

@fragment
fn fs_above(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = surface_normal(in.position);
    let incoming_ray = normalize(in.position - u.eye.xyz);

    let reflected_ray = reflect(incoming_ray, normal);
    let refracted_ray = refract(incoming_ray, normal, IOR_AIR / IOR_WATER);
    let fresnel = mix(0.25, 1.0, pow(1.0 - dot(normal, -incoming_ray), 3.0));

    let reflected_color = get_surface_ray_color(in.position, reflected_ray, ABOVEWATER_COLOR);
    let refracted_color = get_surface_ray_color(in.position, refracted_ray, ABOVEWATER_COLOR);

    return vec4<f32>(mix(refracted_color, reflected_color, fresnel), 1.0);
}

@fragment
fn fs_under(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = -surface_normal(in.position);
    let incoming_ray = normalize(in.position - u.eye.xyz);

    let reflected_ray = reflect(incoming_ray, normal);
    let refracted_ray = refract(incoming_ray, normal, IOR_WATER / IOR_AIR);
    let fresnel = mix(0.5, 1.0, pow(1.0 - dot(normal, -incoming_ray), 3.0));

    let reflected_color = get_surface_ray_color(in.position, reflected_ray, UNDERWATER_COLOR);
    let refracted_color = get_surface_ray_color(in.position, refracted_ray, vec3<f32>(1.0)) * vec3<f32>(0.8, 1.0, 1.1);

    return vec4<f32>(mix(reflected_color, refracted_color, (1.0 - fresnel) * length(refracted_ray)), 1.0);
}
