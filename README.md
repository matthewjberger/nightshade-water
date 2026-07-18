# Nightshade Water

<img width="2560" height="1392" alt="Screenshot 2026-07-18 135346" src="https://github.com/user-attachments/assets/084b5b6d-0b26-4710-ac06-debb8089c03e" />

I ported [Evan Wallace's WebGL Water](https://madebyevan.com/webgl-water/) to the
[Nightshade](https://github.com/matthewjberger/nightshade) engine, GPU and all.
The pool, the water, and the ball are drawn by ray tracing the way the original
does, added at the app layer through the render graph API.

The ripple height field runs on the GPU. Compute shaders step the field across a
pair of ping-pong textures every frame. A custom render pass then traces the
scene in the normalized pool space where the pool spans [-1, 1] in x and z, the
floor sits at y = -1, and the water rests at y = 0. The pool reflects an HDR
environment, and the ball is a beach ball that floats, bobs, and rolls.

## Controls

| Input | Action |
|-------|--------|
| Click or drag the pool | Add ripples |
| Drag the ball | Move it (it pushes water and rolls) |
| Drag outside the pool | Orbit the camera |
| Scroll | Zoom |
| `Space` | Pause |
| `Q` or `Esc` | Quit |

There is a small panel in the top left:

- **Rain** scatters drops across the surface and turns on falling rain particles.
- **Ball** shows or hides the beach ball.
- **Reset** flattens the pool.

## Run

```bash
just run        # native
just run-wasm   # web (WebGPU)
```

WebGPU works in Chromium browsers (Chrome, Brave, Vivaldi, Edge) and Firefox 141+.

## How it works

The whole scene is a custom `PassNode` registered with
`App::add_render_graph_config`. It draws the pool, the water, and the ball
itself, so nothing but the camera, the HDR skybox, the rain particles, and the
UI lives in the ECS scene.

`src/water_pass.rs` holds the `WaterGpuPass`. It owns the ping-pong state
textures, the compute pipelines, a caustics texture, and the render pipelines for
the pool, the water, and the ball. Each frame it steps the simulation on a fixed
clock so ripple speed does not depend on frame rate, projects the caustics
texture, then traces the pool, the water surface, and the ball into the scene.

`src/shaders/water_sim.wgsl` is the simulation, ported straight from the
original: `v += (avg - h) * 2`, `v *= 0.995`, `h += v` with a reflective
boundary, raised-cosine drops, and the moving-sphere volume displacement that
pushes water around the ball.

`src/shaders/helpers.wgsl` carries the shared tracing code: the cube and sphere
intersection, the tiled walls with their ambient occlusion and projected
caustics, and the ball shading. It is prepended to the render shaders.

`src/shaders/water_surface.wgsl` traces the surface. For each fragment it casts a
reflected ray into the HDR sky with a sun glint and a refracted ray into the
tiled pool and the ball, then blends them by fresnel. It draws in two culling
passes so the underwater view is correct when the camera dips below the surface.

`src/shaders/pool.wgsl` and `src/shaders/sphere.wgsl` draw the walls and the ball
with the same shared shading, tinted below the waterline. The near walls are
removed by backface culling as the camera orbits.

`src/shaders/caustics.wgsl` projects each water vertex along the refracted
sunlight onto the floor and measures the triangle area change with screen-space
derivatives. Light that focuses onto a smaller area is brighter. The result
lights the floor, the walls, and the ball and carries the ball's blob shadow.

`src/plugin.rs` is `WaterPlugin`. It feeds interaction into the pass through a
shared `WaterParams` handle: ripple drops, the draggable ball with buoyancy and
rolling, rain, and reset.

`src/scene.rs` sets up the camera, the HDR skybox, the rain emitter, the control
panel, and the help text.

## Notes

Reflection, refraction, and the projected caustics are all here, traced the way
the original does them. The ball is a procedural beach ball on the traced sphere
rather than a model, so it can carry the same caustics, ambient occlusion, and
underwater tint everywhere it appears, including in the water's reflection and
refraction.

## License

Apache-2.0 or MIT, your choice. The original WebGL Water is MIT, Copyright 2011
Evan Wallace.
