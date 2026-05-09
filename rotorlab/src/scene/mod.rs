//! Top-level scene orchestration: configuration, frame loop, output sink.
//!
//! Plan 3 Task 2 lays down the public surface: [`SceneConfig`],
//! [`Output`], [`H264Preset`], and [`Scene`] with `new`, `render`, and
//! `Drop`. Plan 3 Tasks 3 through 11 grow the scene with drawables,
//! a real timeline, and stock animations.

pub mod config;
pub mod descriptor_pool;
pub mod frame_context;
#[allow(clippy::module_inception)]
pub mod scene;

pub use config::{H264Preset, Output, SceneConfig};
pub use descriptor_pool::DescriptorPool;
pub use frame_context::FrameContext;
pub use scene::Scene;
