//! WebGL Water, ported to the Nightshade engine.
//!
//! A GPU-driven port of Evan Wallace's WebGL Water pool demo, built on
//! nightshade's app-layer render-graph API. The ripple height field is
//! simulated on the GPU with compute shaders over ping-pong textures, and the
//! water grid is drawn by a custom render pass that samples that state, the
//! same technique as the original.
//!
//! ## Layout
//!
//! - `src/plugin.rs` — the [`WaterPlugin`]: owns the scene, registers the GPU
//!   water pass, and feeds interaction into it.
//! - `src/water_pass.rs` — the custom `PassNode`: the compute simulation and
//!   the water render pipeline.
//! - `src/shaders/` — the WGSL for the simulation and the surface.
//! - `src/scene.rs` — the pool, ball, sky, camera, and help overlay.

mod plugin;
mod scene;
mod water_pass;

pub use plugin::WaterPlugin;
