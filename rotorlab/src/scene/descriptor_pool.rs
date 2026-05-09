//! Per-scene Vulkan descriptor pool.
//!
//! Plan 3 pipelines (point, line, plane) communicate per-draw state
//! exclusively through push constants, so no descriptor sets are
//! allocated from this pool during the Plan 3 frame loop. The pool is
//! created anyway so that [`FrameContext`](crate::scene::FrameContext)
//! has a stable, per-frame slot to point at, and so future pipelines
//! (textured glyph, shaded mesh, anything that needs a uniform buffer)
//! can allocate sets from it without restructuring the scene.
//!
//! The pool is sized for one descriptor set with a single
//! uniform-buffer binding. Some drivers reject `max_sets = 0`, so we
//! size it small but non-zero rather than attempting to create an
//! empty pool.

use std::sync::Arc;

use ash::vk;

use crate::error::RenderError;
use crate::render::Device;

/// A small `VkDescriptorPool` owned by the scene.
///
/// In Plan 3 nothing allocates from this pool; it exists so future
/// drawables that need uniform-buffer descriptor sets can plumb
/// through [`FrameContext`](crate::scene::FrameContext) without
/// changing the scene surface.
pub struct DescriptorPool {
    /// Owning device. Held as an `Arc` so the pool's [`Drop`] can
    /// destroy the handle through a still-live `VkDevice`.
    pub device: Arc<Device>,
    /// The raw `VkDescriptorPool` handle.
    pub raw: vk::DescriptorPool,
}

impl DescriptorPool {
    /// Allocate a new descriptor pool sized for a small number of
    /// uniform-buffer sets.
    ///
    /// The pool currently advertises capacity for one
    /// `UNIFORM_BUFFER` set; this is wasted space in Plan 3 (no
    /// drawable allocates from it) but matches the eventual usage and
    /// avoids the `max_sets = 0` driver trap.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Vulkan`] if the underlying
    /// `vkCreateDescriptorPool` call fails.
    pub fn new(device: Arc<Device>) -> Result<Self, RenderError> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        // Safety: `device.raw` is a valid VkDevice; `info` is fully
        // populated and references `pool_sizes`, which lives for the
        // duration of this call. No allocation callbacks are used.
        // The returned handle becomes the sole owner of the pool and
        // is destroyed in `Drop`.
        let raw = unsafe { device.raw.create_descriptor_pool(&info, None) }?;

        Ok(Self { device, raw })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        // Safety: caller ensures no in-flight Vulkan work uses sets
        // allocated from this pool (Plan 3 never allocates any). The
        // device is kept alive via `self.device` (`Arc<Device>`).
        unsafe {
            self.device.raw.destroy_descriptor_pool(self.raw, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{Device, Instance};

    #[test]
    fn create_descriptor_pool_smoke() {
        let Ok(instance) = Instance::new() else {
            eprintln!("skip: no vulkan");
            return;
        };
        let Ok(device) = Device::new(instance) else {
            eprintln!("skip: no device");
            return;
        };
        let pool = DescriptorPool::new(device).expect("create pool");
        drop(pool);
    }
}
