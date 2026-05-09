//! Concrete graphics pipelines and the [`Pipeline`] trait that lets
//! [`PipelineCache`](crate::render::PipelineCache) store them
//! generically.

use ash::vk;

pub mod point;
pub mod triangle;

pub use point::{PointInstance, PointInstanceBuffer, PointPipeline, PointPushConstants};
pub use triangle::TrianglePipeline;

/// Object-safe interface for a compiled graphics pipeline.
///
/// Implemented by every concrete pipeline (Plan 3 ships
/// [`TrianglePipeline`], Tasks 4 and 6 add the point and line
/// pipelines). Stored by
/// [`PipelineCache`](crate::render::PipelineCache) as
/// `Arc<dyn Pipeline>` so drawables can fetch a pipeline by
/// [`PipelineId`](crate::render::pipeline_cache::PipelineId) without
/// the cache caring about construction signatures or vertex formats.
///
/// `Send + Sync` because the cache is itself shared across the
/// recording surface; the underlying Vulkan handles are immutable
/// after pipeline creation, so concurrent reads are safe.
pub trait Pipeline: Send + Sync {
    /// The raw `VkPipeline` handle, suitable for
    /// `cmd_bind_pipeline`.
    fn raw(&self) -> vk::Pipeline;
    /// The `VkPipelineLayout` paired with this pipeline, suitable for
    /// `cmd_push_constants` and `cmd_bind_descriptor_sets`.
    fn layout(&self) -> vk::PipelineLayout;
}

impl Pipeline for TrianglePipeline {
    fn raw(&self) -> vk::Pipeline {
        self.raw
    }
    fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Pipeline for PointPipeline {
    fn raw(&self) -> vk::Pipeline {
        self.raw
    }
    fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}
