//! Physical and logical device selection + graphics queue.

use crate::error::RenderError;
use crate::render::instance::Instance;
use ash::{Device as AshDevice, vk};
use std::sync::Arc;

/// A logical Vulkan device + the graphics queue we'll use for headless rendering.
pub struct Device {
    /// The instance this device was created from.
    pub instance: Arc<Instance>,
    /// The physical device chosen.
    pub physical: vk::PhysicalDevice,
    /// The wrapped ash device.
    pub raw: AshDevice,
    /// The graphics-capable queue (we use a single queue family in v0.0.x).
    pub graphics_queue: vk::Queue,
    /// The index of the graphics queue family.
    pub graphics_family_index: u32,
}

impl Device {
    /// Create a logical device by selecting the first physical device whose
    /// queue families include a graphics-capable queue. v0.0.x of `rotorlab`
    /// targets a single device with no surface (headless).
    pub fn new(instance: Arc<Instance>) -> Result<Arc<Self>, RenderError> {
        // Step 1: enumerate physical devices.
        // Safety: `instance.raw` is a valid VkInstance; the returned handles
        // are device handles owned by the loader and remain valid until the
        // instance is destroyed. We only read them here to select a device.
        let physical_devices =
            unsafe { instance.raw.enumerate_physical_devices() }.map_err(RenderError::Vulkan)?;

        if physical_devices.is_empty() {
            return Err(RenderError::NoPhysicalDevice);
        }

        // Step 2: find the first device that exposes a graphics-capable queue family.
        let (physical, graphics_family_index) = physical_devices
            .into_iter()
            .find_map(|dev| {
                // Safety: `dev` is a valid VkPhysicalDevice handle belonging to
                // `instance.raw`. This call only reads device properties; no
                // device-side state is modified.
                let queue_props = unsafe {
                    instance
                        .raw
                        .get_physical_device_queue_family_properties(dev)
                };
                queue_props
                    .into_iter()
                    .enumerate()
                    .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .map(|(idx, _)| (dev, idx as u32))
            })
            .ok_or(RenderError::NoPhysicalDevice)?;

        // Step 3: build the single VkDeviceQueueCreateInfo.
        // The priorities slice must outlive `queue_create_info`, so bind it here.
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family_index)
            .queue_priorities(&queue_priorities);

        // Step 4: build VkDeviceCreateInfo — no extensions, no layers.
        let queue_create_infos = [queue_create_info];
        let create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

        // Step 5: create the logical device.
        // Safety: `physical` is a valid VkPhysicalDevice from the instance.
        // `create_info` and all referenced slices are valid for the duration of
        // this call. No allocation callbacks are used (None). The returned
        // `AshDevice` becomes the sole owner of the VkDevice handle.
        let raw = unsafe { instance.raw.create_device(physical, &create_info, None) }
            .map_err(RenderError::Vulkan)?;

        // Step 6: fetch the queue handle (queue index 0 within the family).
        // Safety: `raw` is a valid VkDevice, `graphics_family_index` is the
        // family we requested in DeviceQueueCreateInfo, and queue index 0 is
        // always present (we requested one queue in the priority array above).
        let graphics_queue = unsafe { raw.get_device_queue(graphics_family_index, 0) };

        // Step 7: return the fully-initialised Device.
        Ok(Arc::new(Self {
            instance,
            physical,
            raw,
            graphics_queue,
            graphics_family_index,
        }))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Safety: this is the only owner of `self.raw`; no in-flight Vulkan
        // operations remain because all child resources hold an Arc<Device>
        // and are destroyed before the device. The parent instance is kept
        // alive via `self.instance` (Arc<Instance>) and will be dropped after.
        unsafe {
            self.raw.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_device_smoke() {
        let instance = match Instance::new() {
            Ok(i) => i,
            Err(_) => {
                eprintln!("skipping: no Vulkan");
                return;
            }
        };
        let device = match Device::new(instance) {
            Ok(d) => d,
            Err(RenderError::NoPhysicalDevice) => {
                eprintln!("skipping: no physical device");
                return;
            }
            Err(e) => panic!("unexpected: {e}"),
        };
        drop(device);
    }
}
