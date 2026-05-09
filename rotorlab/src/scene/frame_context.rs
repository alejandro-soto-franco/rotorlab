//! Per-frame plumbing handed to each [`Drawable`] during recording.
//!
//! [`FrameContext`] is a borrowed bundle assembled inside
//! [`Scene::render`](crate::scene::Scene) and passed to every
//! [`Drawable::record`](crate::object::Drawable::record). It carries
//! the active command buffer, the scene's pipeline cache, and the
//! scene's descriptor pool. Plan 3 Task 3 only defines the type; Tasks
//! 5 onward will construct it inside the frame loop.

use ash::vk;

use crate::render::PipelineCache;
use crate::scene::descriptor_pool::DescriptorPool;

/// Per-frame recording context borrowed by every drawable.
///
/// The lifetime parameter `'a` ties the context to the per-frame
/// borrow of the scene's pipeline cache and descriptor pool; it
/// cannot outlive the frame's recording window. The `cmd_buf` is a
/// raw Vulkan handle (it does not own anything), so it is copied by
/// value.
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
}
