//! Per-frame plumbing handed to each [`Drawable`](crate::object::Drawable) during recording.
//!
//! [`FrameContext`] is a borrowed bundle assembled inside
//! [`Scene::render_with`](crate::scene::Scene::render_with) and passed to every
//! [`Drawable::record`](crate::object::Drawable::record). It carries
//! the active command buffer, the scene's pipeline cache, the scene's
//! descriptor pool, and a pair of mutable instance accumulators that
//! drawables push into.
//!
//! Plan 3 Task 12. Tasks 5 / 7 / 8 left `record` as a no-op shell that
//! only ran the geometry-to-instance conversion for its side effect;
//! Task 12 wires real recording by giving every drawable somewhere to
//! deposit its [`PointInstance`] / [`LineInstance`] payloads. The
//! [`Scene::render_with`](crate::scene::Scene::render_with) frame loop drains the accumulators each
//! frame, uploads them to the per-scene
//! [`PointInstanceBuffer`](crate::render::pipelines::PointInstanceBuffer)
//! / [`LineInstanceBuffer`](crate::render::pipelines::LineInstanceBuffer),
//! and issues the bind + draw + readback.

use ash::vk;

use crate::render::PipelineCache;
use crate::render::pipelines::{LineInstance, PointInstance};
use crate::scene::descriptor_pool::DescriptorPool;

/// Per-frame recording context borrowed by every drawable.
///
/// The lifetime parameter `'a` ties the context to the per-frame
/// borrow of the scene's pipeline cache, descriptor pool, and instance
/// accumulators; it cannot outlive the frame's recording window. The
/// `cmd_buf` is a raw Vulkan handle (it does not own anything), so it
/// is copied by value.
pub struct FrameContext<'a> {
    /// The recording command buffer for the current frame.
    pub cmd_buf: vk::CommandBuffer,
    /// The scene-owned pipeline cache; drawables fetch their
    /// pipeline through `pipelines.get_or_create(...)`.
    pub pipelines: &'a PipelineCache,
    /// The scene-owned descriptor pool; unused in Plan 3 but plumbed
    /// for future drawables that need uniform-buffer or
    /// sampled-image sets.
    pub descriptor_pool: &'a DescriptorPool,
    /// Per-frame point-instance accumulator. `Point::record` pushes
    /// one [`PointInstance`] per drawable; `Plane::record` does not
    /// push points (only edges).
    pub point_instances: &'a mut Vec<PointInstance>,
    /// Per-frame line-instance accumulator. `Line::record` pushes
    /// one [`LineInstance`] per drawable; `Plane::record` pushes
    /// four (one per wireframe edge).
    pub line_instances: &'a mut Vec<LineInstance>,
}
