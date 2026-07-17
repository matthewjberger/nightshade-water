# Nightshade Water

A GPU-driven port of [Evan Wallace's WebGL Water](https://madebyevan.com/webgl-water/)
to the [Nightshade](https://github.com/matthewjberger/nightshade) engine, built
on the app-layer render-graph API.

Like the original, the ripple height field is simulated on the GPU: compute
shaders advance the field over ping-pong textures every frame, and a custom
render pass draws the water volume by sampling that state directly on the GPU.
The pool is HDR-lit with image-based lighting, the tiles are a PBR material, and
the beach ball is a glTF model that floats and rolls in the water.

## Controls

| Input | Action |
|-------|--------|
| Click / drag the pool | Add ripples |
| Drag the ball | Move it (it displaces water and rolls) |
| Drag outside the pool | Orbit the camera |
| Scroll | Zoom |
| `Space` | Pause |
| `Q` / `Esc` | Quit |

## Run

```bash
# native
just run

# web (WebGPU)
just run-wasm
```

> WebGPU runs in Chromium-based browsers (Chrome, Brave, Vivaldi, Edge) and in
> Firefox 141+.

## How it works

The water is entirely GPU-driven, added at the app layer as a custom
`PassNode` through `App::add_render_graph_config` — no engine changes.

- **`src/water_pass.rs`** — the `WaterGpuPass`. It owns two ping-pong
  `Rgba16Float` storage textures (packing height, velocity, and normal), the
  compute pipelines, and the water render pipeline. Each frame `execute` clears
  the state on the first frame, runs the `update` compute pass (the wave
  equation and the ball/drop displacement) and the `normals` compute pass in
  separate passes so the writes are barriered, then draws the displaced grid
  volume, alpha blended into `scene_color`.
- **`src/shaders/water_sim.wgsl`** — the compute simulation, ported from the
  original: `v += (avg − h) · 2; v ·= 0.995; h += v` with a reflective boundary,
  plus raised-cosine drops and the moving-sphere volume displacement.
- **`src/shaders/water_surface.wgsl`** — the surface: the vertex stage samples
  the height texture to displace the grid (top surface and skirts to the floor),
  the fragment stage uses the simulated normals for fresnel/specular and tints
  by depth so the pool reads as filled with water.
- **`src/plugin.rs`** — `WaterPlugin`: builds the scene, registers the pass, and
  feeds interaction (ripple drops, the draggable/rolling ball with buoyancy)
  into the pass through a shared `WaterParams` handle. Systems take `Res`/
  `ResMut` params for engine resources.
- **`src/scene.rs`** — the tiled pool, the glTF ball, the HDR environment, the
  camera, and the help overlay. The PBR tile normal map is generated from the
  tile albedo at startup.

## What is and isn't ported

Ported and GPU-driven: the ripple simulation (compute ping-pong), the water
render, drop injection, the moving sphere's displacement, buoyancy, and the
tiled pool.

Not ported: the original's screen-space caustics and its analytic ray-traced
reflection/refraction. The surface here reflects the HDR environment and reads
the scene through alpha blending.

## License

Licensed under either of Apache-2.0 or MIT, at your option. The original WebGL
Water is MIT, Copyright 2011 Evan Wallace.
