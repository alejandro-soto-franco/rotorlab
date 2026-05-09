//! A headless render target: color image + memory + readback staging buffer.

use crate::error::RenderError;
use crate::render::device::Device;
use ash::vk;
use std::sync::Arc;

/// One image we can render into and copy back to host memory.
///
/// Resources: VkImage (color attachment), backing VkDeviceMemory,
/// VkImageView (for the framebuffer), VkBuffer (staging, host-visible) for
/// readback, backing VkDeviceMemory for the staging buffer.
///
/// Format: VK_FORMAT_B8G8R8A8_UNORM (matches FFmpeg `bgra` raw input).
/// Image usage: COLOR_ATTACHMENT | TRANSFER_SRC.
/// Buffer usage: TRANSFER_DST.
pub struct HeadlessRenderTarget {
    /// The device this target lives on.
    pub device: Arc<Device>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The color image.
    pub image: vk::Image,
    /// Memory backing the image.
    pub image_memory: vk::DeviceMemory,
    /// Image view for the framebuffer attachment.
    pub image_view: vk::ImageView,
    /// Host-visible staging buffer for readback (size = width * height * 4 bytes).
    pub staging_buffer: vk::Buffer,
    /// Memory backing the staging buffer.
    pub staging_memory: vk::DeviceMemory,
}

impl HeadlessRenderTarget {
    /// Create a new headless render target of the given size.
    pub fn new(device: Arc<Device>, width: u32, height: u32) -> Result<Self, RenderError> {
        // === Image ===

        // Step 1: build VkImageCreateInfo.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // Step 2: create the image.
        // Safety: `device.raw` is a valid VkDevice. `image_info` is fully
        // initialised above and valid for the duration of this call.
        let image =
            unsafe { device.raw.create_image(&image_info, None) }.map_err(RenderError::Vulkan)?;

        // === Image memory ===

        // Step 3: query memory requirements for the image.
        // Safety: `image` was just successfully created by `device.raw` and
        // has not been destroyed. The call only reads driver-internal state.
        let image_requirements = unsafe { device.raw.get_image_memory_requirements(image) };

        // Step 4: find a DEVICE_LOCAL memory type satisfying the image's
        // requirements bitmask.
        // Safety: `device.physical` is a valid VkPhysicalDevice belonging to
        // `device.instance.raw`. The call only reads physical device properties.
        let mem_props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical)
        };
        let image_mem_type = find_memory_type(
            &mem_props,
            image_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| {
            RenderError::OutOfMemory("no DEVICE_LOCAL memory type for image".to_owned())
        })?;

        // Step 5-6: allocate and bind image memory.
        let image_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(image_requirements.size)
            .memory_type_index(image_mem_type);

        // Safety: `device.raw` is valid. `image_alloc_info` is fully initialised.
        // The returned `VkDeviceMemory` becomes exclusively owned by this struct.
        let image_memory = unsafe { device.raw.allocate_memory(&image_alloc_info, None) }
            .map_err(RenderError::Vulkan)?;

        // Step 7: bind image memory.
        // Safety: `image` and `image_memory` are valid handles belonging to
        // `device.raw`. Offset 0 satisfies alignment because the allocation
        // was sized and typed by `get_image_memory_requirements`.
        unsafe { device.raw.bind_image_memory(image, image_memory, 0) }
            .map_err(RenderError::Vulkan)?;

        // === Image view ===

        // Step 8-9: create the image view.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .components(vk::ComponentMapping::default())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // Safety: `device.raw` is valid. `view_info` references `image` which
        // is alive and bound. No allocation callbacks used.
        let image_view = unsafe { device.raw.create_image_view(&view_info, None) }
            .map_err(RenderError::Vulkan)?;

        // === Staging buffer ===

        // Step 10: compute byte size (BGRA = 4 bytes per pixel).
        let size_bytes = width as u64 * height as u64 * 4;

        // Step 11: build VkBufferCreateInfo.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // Step 12: create the staging buffer.
        // Safety: `device.raw` is valid. `buffer_info` is fully initialised.
        let staging_buffer =
            unsafe { device.raw.create_buffer(&buffer_info, None) }.map_err(RenderError::Vulkan)?;

        // Step 13: query buffer memory requirements.
        // Safety: `staging_buffer` was just successfully created and has not
        // been destroyed. The call only reads driver-internal state.
        let buf_requirements = unsafe { device.raw.get_buffer_memory_requirements(staging_buffer) };

        // Step 14: find a HOST_VISIBLE | HOST_COHERENT memory type.
        let staging_mem_type = find_memory_type(
            &mem_props,
            buf_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| {
            RenderError::OutOfMemory(
                "no HOST_VISIBLE+COHERENT memory type for staging buffer".to_owned(),
            )
        })?;

        // Step 15: allocate and bind staging buffer memory.
        let staging_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_requirements.size)
            .memory_type_index(staging_mem_type);

        // Safety: `device.raw` is valid. `staging_alloc_info` is fully
        // initialised. The returned `VkDeviceMemory` becomes exclusively owned
        // by this struct.
        let staging_memory = unsafe { device.raw.allocate_memory(&staging_alloc_info, None) }
            .map_err(RenderError::Vulkan)?;

        // Safety: `staging_buffer` and `staging_memory` are valid handles
        // belonging to `device.raw`. Offset 0 satisfies alignment requirements
        // returned by `get_buffer_memory_requirements`.
        unsafe {
            device
                .raw
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
        }
        .map_err(RenderError::Vulkan)?;

        // Step 16: return the fully-initialised target.
        Ok(Self {
            device,
            width,
            height,
            image,
            image_memory,
            image_view,
            staging_buffer,
            staging_memory,
        })
    }

    /// Read back the image contents into a host buffer of length `width * height * 4`
    /// (BGRA bytes). Caller must ensure the image-to-buffer copy command has already
    /// been submitted and completed (Task 10's FrameRecorder handles this).
    pub fn read_to_host(&self, out: &mut [u8]) -> Result<(), RenderError> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        assert_eq!(
            out.len(),
            expected,
            "out slice length must match width*height*4"
        );

        // Step 1: map the staging memory into host address space.
        // Safety: `self.staging_memory` is a valid HOST_VISIBLE | HOST_COHERENT
        // allocation belonging to `self.device.raw`. Offset 0, WHOLE_SIZE covers
        // the entire allocation. No other thread holds a concurrent map.
        let mapped = unsafe {
            self.device
                .raw
                .map_memory(
                    self.staging_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| RenderError::Mapping(format!("map_memory failed: {e:?}")))?
        };

        // Step 2: copy from mapped pointer into the caller's buffer.
        // Safety: `mapped` points to at least `expected` bytes of valid,
        // readable host memory (HOST_COHERENT; no explicit cache flush needed).
        // `out` is a valid mutable slice of length `expected`. The regions do
        // not overlap (one is device-allocated, the other is caller-owned heap).
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), expected);
        }

        // Step 3: unmap; invalidation is implicit with HOST_COHERENT memory.
        // Safety: `self.staging_memory` is currently mapped (Step 1 succeeded).
        unsafe {
            self.device.raw.unmap_memory(self.staging_memory);
        }

        Ok(())
    }
}

impl Drop for HeadlessRenderTarget {
    fn drop(&mut self) {
        // Safety: caller ensures no in-flight GPU work uses these resources.
        // Destruction order: child views before parents, buffers before backing memory.
        unsafe {
            self.device.raw.destroy_buffer(self.staging_buffer, None);
            self.device.raw.free_memory(self.staging_memory, None);
            self.device.raw.destroy_image_view(self.image_view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.image_memory, None);
        }
    }
}

/// Find the first memory type index satisfying both `memory_type_bits` (a
/// requirements bitmask) and `required_flags` (the property flags we want).
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..properties.memory_type_count {
        let bit = 1u32 << i;
        if (memory_type_bits & bit) != 0
            && properties.memory_types[i as usize]
                .property_flags
                .contains(required_flags)
        {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Instance;

    #[test]
    fn create_render_target_smoke() {
        let Ok(instance) = Instance::new() else {
            eprintln!("skip: no vulkan");
            return;
        };
        let Ok(device) = Device::new(instance) else {
            eprintln!("skip: no device");
            return;
        };
        let target = HeadlessRenderTarget::new(device, 1920, 1080).expect("create target");
        drop(target);
    }
}
