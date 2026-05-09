//! VkRenderPass for color-only headless rendering.

use crate::error::RenderError;
use crate::render::device::Device;
use ash::vk;
use std::sync::Arc;

/// A render pass with one color attachment.
///
/// Format: B8G8R8A8_UNORM; load op CLEAR; store op STORE; final layout TRANSFER_SRC_OPTIMAL
/// (so the next command-buffer step can copy out of it without a layout transition barrier).
pub struct RenderPass {
    /// The device this pass lives on.
    pub device: Arc<Device>,
    /// The wrapped ash render pass.
    pub raw: vk::RenderPass,
}

impl RenderPass {
    /// Create a render pass matching the headless target's format + usage.
    pub fn new(device: Arc<Device>) -> Result<Self, RenderError> {
        // Step 1: one color attachment — BGRA8_UNORM, CLEAR on load, STORE on
        // store, transitions from UNDEFINED to TRANSFER_SRC_OPTIMAL so the
        // caller can issue vkCmdCopyImageToBuffer immediately after the pass
        // without inserting an explicit pipeline barrier.
        let attachment_desc = vk::AttachmentDescription::default()
            .format(vk::Format::B8G8R8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        // Step 2: reference the single attachment as a color attachment during
        // the subpass. COLOR_ATTACHMENT_OPTIMAL is the required layout while
        // the subpass is executing fragment writes.
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        // Bind the slice here so it outlives the SubpassDescription and the
        // subsequent RenderPassCreateInfo call.
        let color_refs = [color_ref];

        // Step 3: a single graphics subpass that writes to the color attachment.
        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);

        // Step 4: subpass dependency from the implicit external subpass.
        // Ensures that color writes are visible before the pass begins outputting
        // to the attachment (covers multi-frame or multi-pass cases).
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        // Bind all arrays before building the create-info so lifetime rules
        // are satisfied (the builder stores raw pointers into these slices).
        let attachments = [attachment_desc];
        let subpasses = [subpass_desc];
        let dependencies = [dependency];

        // Step 5: assemble the RenderPassCreateInfo.
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        // Step 6: create the render pass.
        // Safety: `device.raw` is a valid VkDevice. `info` and all referenced
        // slices are alive for the duration of this call. No allocation
        // callbacks are used (None). The returned VkRenderPass handle is owned
        // solely by `Self` and is destroyed in the Drop impl.
        let raw =
            unsafe { device.raw.create_render_pass(&info, None) }.map_err(RenderError::Vulkan)?;

        // Step 7: return the initialised RenderPass.
        Ok(Self { device, raw })
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        // Safety: this is the sole owner of `self.raw`. No in-flight GPU work
        // references this render pass because all downstream resources (frame-
        // buffers, command buffers) must be destroyed before the RenderPass is
        // dropped. The device is kept alive via `self.device` (Arc<Device>).
        unsafe {
            self.device.raw.destroy_render_pass(self.raw, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Instance;

    #[test]
    fn create_render_pass_smoke() {
        let Ok(instance) = Instance::new() else {
            eprintln!("skip: no vulkan");
            return;
        };
        let Ok(device) = Device::new(instance) else {
            eprintln!("skip: no device");
            return;
        };
        let pass = RenderPass::new(device).expect("render pass");
        drop(pass);
    }
}
