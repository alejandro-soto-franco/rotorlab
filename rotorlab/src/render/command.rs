//! Command pool + command buffer helpers.

use crate::error::RenderError;
use crate::render::device::Device;
use ash::vk;
use std::sync::Arc;

/// Owns a transient command pool tied to the device's graphics queue family.
pub struct CommandPool {
    /// Owning device.
    pub device: Arc<Device>,
    /// The wrapped ash command pool.
    pub raw: vk::CommandPool,
}

impl CommandPool {
    /// Create a transient + reset-able command pool on the device's graphics
    /// queue family.
    pub fn new(device: Arc<Device>) -> Result<Self, RenderError> {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(device.graphics_family_index);

        // Safety: `device.raw` is a valid VkDevice. `info` is fully initialised
        // above and valid for the duration of this call. The returned
        // VkCommandPool handle is owned solely by `Self` and destroyed in Drop.
        let raw =
            unsafe { device.raw.create_command_pool(&info, None) }.map_err(RenderError::Vulkan)?;

        Ok(Self { device, raw })
    }

    /// Allocate one primary command buffer from this pool.
    pub fn allocate_primary(&self) -> Result<vk::CommandBuffer, RenderError> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // Safety: `self.device.raw` is a valid VkDevice. `self.raw` is a valid
        // VkCommandPool owned by `self`. `info` is fully initialised. The
        // returned command buffer lifetimes are tied to this pool.
        let mut buffers = unsafe { self.device.raw.allocate_command_buffers(&info) }
            .map_err(RenderError::Vulkan)?;

        // allocate_command_buffers returns exactly command_buffer_count entries;
        // we requested 1, so pop is always Some.
        Ok(buffers.remove(0))
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // Safety: destroying the pool also frees all command buffers allocated
        // from it; caller ensures no in-flight GPU work uses them.
        unsafe {
            self.device.raw.destroy_command_pool(self.raw, None);
        }
    }
}
