//! VkInstance creation and lifecycle.

use crate::error::RenderError;
use ash::{Entry, Instance as AshInstance, vk};
use std::sync::Arc;

/// A Vulkan instance, the entry-point handle for everything else.
///
/// Wraps `ash::Instance` plus the dynamically-loaded `ash::Entry` it was
/// created from. Drops the instance on destruction.
pub struct Instance {
    /// The dynamically-loaded Vulkan loader.
    pub entry: Entry,
    /// The wrapped ash instance.
    pub raw: AshInstance,
}

impl Instance {
    /// Create a new Vulkan instance, validating headless requirements.
    ///
    /// Loads the Vulkan loader dynamically (`Entry::load()`), constructs an
    /// application info struct with `engineName = "rotorlab"`, no enabled
    /// extensions in v0.0.x of `rotorlab`, and creates the VkInstance.
    pub fn new() -> Result<Arc<Self>, RenderError> {
        // Safety: dynamically loads libvulkan.so from the system library
        // search path. No Vulkan calls are made here; the returned Entry
        // holds only function pointers. Safe to call from a single thread at
        // construction time.
        let entry = unsafe { Entry::load() }
            .map_err(|_| RenderError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"rotorlab")
            .application_version(0)
            .engine_name(c"rotorlab")
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // No enabled extensions and no enabled layers in v0.0.x (headless, no validation).

        let raw =
            // Safety: `entry` is a valid, fully-loaded Vulkan loader. `create_info`
            // and `app_info` are valid for the duration of this call. No allocation
            // callbacks are used (None). The returned `AshInstance` becomes the
            // sole owner of the VkInstance handle.
            unsafe { entry.create_instance(&create_info, None) }.map_err(RenderError::Vulkan)?;

        Ok(Arc::new(Self { entry, raw }))
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // Safety: this is the only owner of `self.raw`; no in-flight Vulkan
        // operations remain because all child resources (Device, etc.) hold
        // an Arc<Instance> and are destroyed before the instance.
        unsafe {
            self.raw.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: creating an instance succeeds on a system with a working
    /// Vulkan stack. Skipped on systems where the loader has no usable driver
    /// (`ERROR_INITIALIZATION_FAILED`, `ERROR_INCOMPATIBLE_DRIVER`) or any
    /// other early-init Vulkan error. Mirrors the skip policy of
    /// `device::tests::create_device_smoke`.
    #[test]
    fn create_instance_smoke() {
        match Instance::new() {
            Ok(instance) => {
                drop(instance);
            }
            Err(e) => {
                eprintln!("skipping: Vulkan instance creation failed ({e})");
            }
        }
    }
}
