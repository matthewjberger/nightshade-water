# Nightshade Water

I ported [Evan Wallace's WebGL Water](https://madebyevan.com/webgl-water/) to the
[Nightshade](https://github.com/matthewjberger/nightshade) engine, GPU and all.
It runs in the app layer through the render graph API, so the engine itself is
untouched.

The ripple height field runs on the GPU. Compute shaders step the field across a
pair of ping-pong textures every frame, and a custom render pass draws the water
by sampling that state on the GPU. The pool is lit from an HDR environment, the
tiles use a PBR material, and the beach ball is a glTF model that floats, bobs,
and rolls in the water.

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
- **Reset** flattens the pool.
- **Walls** shows or hides the two pool walls.
- **Ball** shows or hides the beach ball.

## Run

```bash
just run        # native
just run-wasm   # web (WebGPU)
```

WebGPU works in Chromium browsers (Chrome, Brave, Vivaldi, Edge) and Firefox 141+.

## How it works

The water is a custom `PassNode` registered with `App::add_render_graph_config`.
None of it touches the engine.

`src/water_pass.rs` holds the `WaterGpuPass`. It owns two `Rgba16Float` ping-pong
textures that pack height, velocity, and the surface normal, the compute
pipelines, and the render pipeline. Each frame it steps the simulation on a fixed
clock so ripple speed does not depend on frame rate, runs the `update` and
`normals` compute passes separately so the writes are ordered, then draws the
caustics quad and the water grid.

`src/shaders/water_sim.wgsl` is the simulation, ported straight from the
original: `v += (avg - h) * 2`, `v *= 0.995`, `h += v` with a reflective
boundary, raised-cosine drops, and the moving-sphere volume displacement that
pushes water around the ball.

`src/shaders/water_surface.wgsl` draws the surface. The vertex stage samples the
height texture to displace the grid and drops skirts down to the floor so the
pool looks filled. The fragment stage uses the simulated normals for fresnel and
specular and tints by depth.

`src/shaders/caustics.wgsl` adds light to the floor from the height field. Where
the surface is concave it focuses light, so the pass brightens the floor by the
positive Laplacian of the height.

`src/plugin.rs` is `WaterPlugin`. It builds the scene, registers the pass, and
feeds interaction into it through a shared `WaterParams` handle: ripple drops,
the draggable ball with buoyancy and rolling, rain, and reset. Systems take `Res`
and `ResMut` params for engine resources.

`src/scene.rs` builds the tiled pool, the glTF ball, the HDR environment, the
camera, the control panel, and the help text. The tile normal map is generated
from the albedo at startup.

## What is and isn't here

On the GPU: the ripple simulation, the water render, drop injection, the sphere
displacement, the caustics, and buoyancy.

Left out: the original's analytic ray-traced reflection and refraction. The
surface reflects the HDR environment and reads the scene behind it through alpha
blending instead.

## License

Apache-2.0 or MIT, your choice. The original WebGL Water is MIT, Copyright 2011
Evan Wallace.
