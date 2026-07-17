//! The water demo plugin: builds the scene, registers the GPU water pass, and
//! feeds interaction (ripple drops, the draggable ball) into it.

use crate::scene::{self, POOL_HALF, SPHERE_RADIUS, SPHERE_SIM_RADIUS, WORLD_SCALE};
use crate::water_pass::{WaterGpuPass, WaterParams};
use nightshade::ecs::camera::components::PanOrbitCamera;
use nightshade::prelude::*;
use std::sync::{Arc, Mutex};

const DROP_RADIUS: f32 = 0.03;
const DROP_STRENGTH: f32 = 0.02;
const BALL_START: [f32; 3] = [-0.4, -0.5, 0.2];

#[derive(Clone, Copy, PartialEq)]
enum DragMode {
    None,
    Ball,
    Drops,
    Orbit,
}

/// App-wide state for the demo.
pub struct WaterState {
    params: Arc<Mutex<WaterParams>>,
    ball_entity: Entity,
    camera_entity: Entity,
    ball_center: Vec3,
    ball_previous_center: Vec3,
    ball_velocity: Vec3,
    ball_rotation: Quat,
    drag_mode: DragMode,
    drag_plane_normal: Vec3,
    drag_prev_hit: Vec3,
    pending_drop: Option<Vec2>,
    paused: bool,
}

/// The demo plugin.
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.world.res_mut::<Window>().title = "Nightshade Water".to_string();
        let params = Arc::new(Mutex::new(WaterParams::default()));

        app.insert_resource(WaterState {
            params: params.clone(),
            ball_entity: Entity::default(),
            camera_entity: Entity::default(),
            ball_center: Vec3::new(BALL_START[0], BALL_START[1], BALL_START[2]),
            ball_previous_center: Vec3::new(BALL_START[0], BALL_START[1], BALL_START[2]),
            ball_velocity: Vec3::zeros(),
            ball_rotation: Quat::identity(),
            drag_mode: DragMode::None,
            drag_plane_normal: Vec3::new(0.0, 0.0, 1.0),
            drag_prev_hit: Vec3::zeros(),
            pending_drop: None,
            paused: false,
        });

        let graph_params = params;
        app.add_render_graph_config(move |graph, device, _format, resources| {
            let pass = WaterGpuPass::new(device, graph_params.clone());
            render_graph_add_pass(
                graph,
                Box::new(pass),
                &[("color", resources.scene_color), ("depth", resources.depth)],
            )
            .unwrap();
        });

        app.add_system(Stage::Startup, initialize);
        app.add_systems(Stage::Update, (handle_input, simulate, update_title));
    }
}

fn initialize(
    mut state: ResMut<WaterState>,
    mut render_settings: ResMut<RenderSettings>,
    mut debug_draw: ResMut<DebugDraw>,
    mut active_camera: ResMut<ActiveCamera>,
    world: &mut World,
) {
    render_settings.atmosphere = Atmosphere::Hdr;
    render_settings.show_sky = true;
    render_settings.bloom_enabled = true;
    render_settings.bloom_intensity = 0.3;
    render_settings.bloom_threshold = 1.3;
    debug_draw.show_grid = false;

    scene::load_environment(world);
    scene::load_textures(world);
    scene::spawn_pool(world);
    state.ball_entity = scene::spawn_ball(world, ball_world_position(state.ball_center));
    let camera = scene::spawn_camera(world);
    active_camera.0 = Some(camera);
    state.camera_entity = camera;
    scene::spawn_help(world);
}

fn handle_input(mut state: ResMut<WaterState>, input: Res<Input>, world: &mut World) {
    let mouse = input.mouse;
    let cursor = mouse.position;
    let left_pressed = mouse.state.contains(MouseState::LEFT_JUST_PRESSED);
    let left_held = mouse.state.contains(MouseState::LEFT_CLICKED);
    let cursor_moved = mouse.position_delta.norm() > 1.5;
    let toggle_pause = input.keyboard.just_pressed(KeyCode::Space);
    let quit = input.keyboard.just_pressed(KeyCode::KeyQ);

    if quit {
        world.res_mut::<Window>().should_exit = true;
    }
    if toggle_pause {
        state.paused = !state.paused;
    }

    if left_pressed {
        state.drag_mode = classify_press(&state, world, cursor);
        if state.drag_mode == DragMode::Ball {
            match grab_plane(&state, world, cursor) {
                Some((normal, hit)) => {
                    state.drag_plane_normal = normal;
                    state.drag_prev_hit = hit;
                }
                None => state.drag_mode = DragMode::Orbit,
            }
        }
    }
    if !left_held {
        state.drag_mode = DragMode::None;
    }

    let camera_enabled = !matches!(state.drag_mode, DragMode::Ball | DragMode::Drops);
    if let Some(camera) = world.get_mut::<PanOrbitCamera>(state.camera_entity) {
        camera.enabled = camera_enabled;
    }

    state.pending_drop = None;
    match state.drag_mode {
        DragMode::Ball => {
            if let Some(ray) = PickingRay::from_screen_position(world, cursor)
                && let Some(hit) = intersect_ray_plane(
                    ray.origin,
                    ray.direction,
                    state.drag_prev_hit,
                    state.drag_plane_normal,
                )
            {
                let delta = hit - state.drag_prev_hit;
                let mut ball_world = ball_world_position(state.ball_center) + delta;
                let xz_limit = POOL_HALF * (1.0 - SPHERE_SIM_RADIUS);
                ball_world.x = ball_world.x.clamp(-xz_limit, xz_limit);
                ball_world.z = ball_world.z.clamp(-xz_limit, xz_limit);
                ball_world.y = ball_world
                    .y
                    .clamp((SPHERE_SIM_RADIUS - 1.0) * WORLD_SCALE, WORLD_SCALE);
                state.ball_center = Vec3::new(
                    ball_world.x / POOL_HALF,
                    ball_world.y / WORLD_SCALE,
                    ball_world.z / POOL_HALF,
                );
                state.ball_velocity = Vec3::zeros();
                state.drag_prev_hit = hit;
            }
        }
        DragMode::Drops => {
            if (left_pressed || cursor_moved)
                && let Some(point) = get_ground_position_from_screen(world, cursor, scene::WATER_Y)
                && point.x.abs() <= POOL_HALF
                && point.z.abs() <= POOL_HALF
            {
                state.pending_drop = Some(Vec2::new(
                    (point.x / POOL_HALF) * 0.5 + 0.5,
                    (point.z / POOL_HALF) * 0.5 + 0.5,
                ));
            }
        }
        DragMode::None | DragMode::Orbit => {}
    }
}

fn simulate(mut state: ResMut<WaterState>, time: Res<Time>, world: &mut World) {
    let delta = time.delta_time.min(0.05);
    if !state.paused {
        if state.drag_mode == DragMode::Ball {
            state.ball_velocity = Vec3::zeros();
        } else {
            step_ball_physics(&mut state, delta);
        }
        apply_ball_rotation(&mut state, delta);
    }

    if let Some(transform) = world.get_mut::<LocalTransform>(state.ball_entity) {
        transform.translation = ball_world_position(state.ball_center);
        transform.rotation = state.ball_rotation;
    }

    let params_handle = state.params.clone();
    let mut params = params_handle.lock().unwrap();
    params.sphere_old = [
        state.ball_previous_center.x,
        state.ball_previous_center.y,
        state.ball_previous_center.z,
    ];
    params.sphere_new = [
        state.ball_center.x,
        state.ball_center.y,
        state.ball_center.z,
    ];
    params.sphere_radius = SPHERE_SIM_RADIUS;
    match state.pending_drop.take() {
        Some(uv) if !state.paused => {
            params.drop_active = true;
            params.drop_center = [uv.x, uv.y];
            params.drop_radius = DROP_RADIUS;
            params.drop_strength = DROP_STRENGTH;
        }
        _ => params.drop_active = false,
    }
    drop(params);

    state.ball_previous_center = state.ball_center;
}

fn update_title(state: Res<WaterState>, time: Res<Time>, mut window: ResMut<Window>) {
    let fps = time.frames_per_second;
    let status = if state.paused { "paused" } else { "running" };
    window.title = format!("Nightshade Water | {fps:.0} FPS | {status}");
}

fn classify_press(state: &WaterState, world: &World, cursor: Vec2) -> DragMode {
    if let Some(ray) = PickingRay::from_screen_position(world, cursor) {
        let ball = ball_world_position(state.ball_center);
        if ray_hits_sphere(ray.origin, ray.direction, ball, SPHERE_RADIUS) {
            return DragMode::Ball;
        }
    }
    if let Some(point) = get_ground_position_from_screen(world, cursor, scene::WATER_Y)
        && point.x.abs() <= POOL_HALF
        && point.z.abs() <= POOL_HALF
    {
        return DragMode::Drops;
    }
    DragMode::Orbit
}

fn step_ball_physics(state: &mut WaterState, delta: f32) {
    let radius = SPHERE_SIM_RADIUS;
    let submerged = ((radius - state.ball_center.y) / (2.0 * radius)).clamp(0.0, 1.0);
    let gravity = Vec3::new(0.0, -4.0, 0.0);
    state.ball_velocity += gravity * (delta - 1.1 * delta * submerged);

    let speed_squared = state.ball_velocity.dot(&state.ball_velocity);
    if speed_squared > 1.0e-6 {
        let drag = state.ball_velocity.normalize() * (submerged * delta * speed_squared);
        state.ball_velocity -= drag;
    }

    state.ball_center += state.ball_velocity * delta;
    if state.ball_center.y < radius - 1.0 {
        state.ball_center.y = radius - 1.0;
        state.ball_velocity.y = state.ball_velocity.y.abs() * 0.7;
    }

    let limit = 1.0 - radius;
    state.ball_center.x = state.ball_center.x.clamp(-limit, limit);
    state.ball_center.z = state.ball_center.z.clamp(-limit, limit);
}

fn apply_ball_rotation(state: &mut WaterState, delta: f32) {
    let ambient_axis = nalgebra_glm::normalize(&Vec3::new(0.3, 1.0, 0.2));
    state.ball_rotation =
        nalgebra_glm::quat_angle_axis(0.4 * delta, &ambient_axis) * state.ball_rotation;

    let previous = ball_world_position(state.ball_previous_center);
    let current = ball_world_position(state.ball_center);
    let horizontal = Vec3::new(current.x - previous.x, 0.0, current.z - previous.z);
    let distance = horizontal.norm();
    if distance > 1.0e-5 {
        let motion = horizontal / distance;
        let axis = nalgebra_glm::cross(&Vec3::new(0.0, 1.0, 0.0), &motion);
        let axis_length = axis.norm();
        if axis_length > 1.0e-5 {
            let angle = distance / SPHERE_RADIUS;
            let roll = nalgebra_glm::quat_angle_axis(angle, &(axis / axis_length));
            state.ball_rotation = roll * state.ball_rotation;
        }
    }
    state.ball_rotation = nalgebra_glm::quat_normalize(&state.ball_rotation);
}

fn grab_plane(state: &WaterState, world: &World, cursor: Vec2) -> Option<(Vec3, Vec3)> {
    let ray = PickingRay::from_screen_position(world, cursor)?;
    let normal = -ray.direction.normalize();
    let ball = ball_world_position(state.ball_center);
    let hit = intersect_ray_plane(ray.origin, ray.direction, ball, normal)?;
    Some((normal, hit))
}

fn intersect_ray_plane(
    origin: Vec3,
    direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = plane_normal.dot(&direction);
    if denominator.abs() < 1.0e-6 {
        return None;
    }
    let distance = plane_normal.dot(&(plane_point - origin)) / denominator;
    if distance < 0.0 {
        return None;
    }
    Some(origin + direction * distance)
}

fn ray_hits_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> bool {
    let to_center = origin - center;
    let a = direction.dot(&direction);
    let b = 2.0 * to_center.dot(&direction);
    let c = to_center.dot(&to_center) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }
    (-b - discriminant.sqrt()) / (2.0 * a) > 0.0
}

fn ball_world_position(center: Vec3) -> Vec3 {
    Vec3::new(
        center.x * POOL_HALF,
        center.y * WORLD_SCALE,
        center.z * POOL_HALF,
    )
}
