//! Single-frame recording + submission + readback orchestration.

use crate::error::RenderError;
use crate::render::command::CommandPool;
use crate::render::device::Device;
use crate::render::pipelines::triangle::{TrianglePipeline, TriangleVertexBuffer};
use crate::render::render_pass::RenderPass;
use crate::render::render_target::HeadlessRenderTarget;
use ash::vk;
use std::sync::Arc;

/// Records and submits one frame of triangle rendering.
///
/// Owns the command pool, framebuffer, command buffer, and fence;
/// gets handed references to the pipeline, vertex buffer, and target on each
/// render call.
pub struct FrameRecorder {
    /// Owning device.
    pub device: Arc<Device>,
    /// Command pool that owns the command buffer.
    pub pool: CommandPool,
    /// Framebuffer over the headless target's image view.
    pub framebuffer: vk::Framebuffer,
    /// The single primary command buffer used per frame.
    pub command_buffer: vk::CommandBuffer,
    /// Fence to wait on after submission.
    pub fence: vk::Fence,
    /// The width the framebuffer was sized for.
    pub width: u32,
    /// The height the framebuffer was sized for.
    pub height: u32,
}

impl FrameRecorder {
    /// Allocate a framebuffer + command buffer + fence ready to record one frame.
    pub fn new(
        device: Arc<Device>,
        render_pass: &RenderPass,
        target: &HeadlessRenderTarget,
    ) -> Result<Self, RenderError> {
        // Step 1: build the framebuffer over the target's image view.
        let attachments = [target.image_view];
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass.raw)
            .attachments(&attachments)
            .width(target.width)
            .height(target.height)
            .layers(1);

        // Safety: `device.raw` is a valid VkDevice. `render_pass.raw` and
        // `target.image_view` are valid handles belonging to the same device.
        // `fb_info` is fully initialised and all slices are alive. The returned
        // VkFramebuffer is owned solely by `Self` and destroyed in Drop.
        let framebuffer = unsafe { device.raw.create_framebuffer(&fb_info, None) }
            .map_err(RenderError::Vulkan)?;

        // Step 2: create a command pool and allocate one primary command buffer.
        let pool = CommandPool::new(device.clone())?;
        let command_buffer = pool.allocate_primary()?;

        // Step 3: create an unsignaled fence.
        let fence_info = vk::FenceCreateInfo::default(); // no SIGNALED flag = unsignaled

        let fence =
            // Safety: `device.raw` is a valid VkDevice. `fence_info` is fully
            // initialised. The returned VkFence is owned solely by `Self` and
            // destroyed in Drop.
            unsafe { device.raw.create_fence(&fence_info, None) }.map_err(RenderError::Vulkan)?;

        Ok(Self {
            device,
            pool,
            framebuffer,
            command_buffer,
            fence,
            width: target.width,
            height: target.height,
        })
    }

    /// Record + submit + wait + copy-to-staging the frame.
    ///
    /// On return, `target.staging_buffer` contains the rendered BGRA bytes,
    /// ready to be mapped via `HeadlessRenderTarget::read_to_host`.
    pub fn render_triangle(
        &self,
        render_pass: &RenderPass,
        pipeline: &TrianglePipeline,
        vertex_buffer: &TriangleVertexBuffer,
        target: &HeadlessRenderTarget,
        clear_color: [f32; 4],
    ) -> Result<(), RenderError> {
        // Step 1: reset the command buffer to record fresh commands.
        // Safety: `self.command_buffer` is a valid primary command buffer
        // allocated from `self.pool`, and no GPU work referencing it is
        // in-flight (we wait on `self.fence` after every submission).
        unsafe {
            self.device
                .raw
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(RenderError::Vulkan)?;
        }

        // Step 2: begin recording with ONE_TIME_SUBMIT semantics.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // Safety: `self.command_buffer` is in the initial state after the reset
        // above. `begin_info` is fully initialised.
        unsafe {
            self.device
                .raw
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(RenderError::Vulkan)?;
        }

        // Step 3: begin the render pass.
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear_color,
            },
        }];
        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: self.width,
                height: self.height,
            });
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.raw)
            .framebuffer(self.framebuffer)
            .render_area(render_area)
            .clear_values(&clear_values);

        // Safety: `self.command_buffer` is recording. All handles (`render_pass`,
        // `self.framebuffer`) are valid and belong to the same device.
        unsafe {
            self.device.raw.cmd_begin_render_pass(
                self.command_buffer,
                &rp_begin,
                vk::SubpassContents::INLINE,
            );
        }

        // Step 4: bind the graphics pipeline.
        // Safety: `self.command_buffer` is inside the render pass subpass that
        // the pipeline was created for. `pipeline.raw` is a valid VkPipeline.
        unsafe {
            self.device.raw.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw,
            );
        }

        // Step 5: bind the vertex buffer at binding 0, offset 0.
        // Safety: `vertex_buffer.buffer` is a valid VERTEX_BUFFER-capable
        // VkBuffer containing `TRIANGLE_VERTICES` at offset 0.
        unsafe {
            self.device.raw.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                &[vertex_buffer.buffer],
                &[0],
            );
        }

        // Step 6: draw 3 vertices, 1 instance, first vertex 0, first instance 0.
        // Safety: the vertex buffer contains exactly 3 vertices.
        unsafe {
            self.device.raw.cmd_draw(self.command_buffer, 3, 1, 0, 0);
        }

        // Step 7: end the render pass. The render pass's final_layout for the
        // color attachment is TRANSFER_SRC_OPTIMAL, so the driver transitions
        // the image automatically at vkCmdEndRenderPass.
        // Safety: we are inside a render pass subpass; this ends it correctly.
        unsafe {
            self.device.raw.cmd_end_render_pass(self.command_buffer);
        }

        // Step 8: copy the image into the staging buffer.
        // No explicit pipeline barrier is needed: the render pass's final_layout
        // transition to TRANSFER_SRC_OPTIMAL guarantees the image is in the
        // correct layout and all writes are visible.
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            });

        // Safety: `target.image` is in TRANSFER_SRC_OPTIMAL layout (set by the
        // render pass final_layout above). `target.staging_buffer` is a valid
        // TRANSFER_DST buffer large enough to hold `width * height * 4` bytes.
        unsafe {
            self.device.raw.cmd_copy_image_to_buffer(
                self.command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target.staging_buffer,
                &[region],
            );
        }

        // Step 9: end recording.
        // Safety: all commands have been recorded; no render pass is active.
        unsafe {
            self.device
                .raw
                .end_command_buffer(self.command_buffer)
                .map_err(RenderError::Vulkan)?;
        }

        // Step 10: submit the command buffer to the graphics queue.
        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);

        // Safety: `self.device.graphics_queue` is the device's graphics queue.
        // `self.fence` is unsignaled (we reset it after every wait below).
        // The command buffer is fully recorded and references only valid handles.
        unsafe {
            self.device
                .raw
                .queue_submit(self.device.graphics_queue, &[submit], self.fence)
                .map_err(RenderError::Vulkan)?;
        }

        // Step 11: wait for the GPU to finish.
        // Safety: `self.fence` was submitted to the queue in Step 10. `u64::MAX`
        // is effectively an infinite timeout; on a functioning GPU this will
        // return SUCCESS within microseconds for a single-triangle frame.
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(RenderError::Vulkan)?;
        }

        // Step 12: reset the fence to unsignaled for the next render call.
        // Safety: `self.fence` is currently in the signaled state (Step 11
        // returned SUCCESS). Resetting it here makes it safe to reuse.
        unsafe {
            self.device
                .raw
                .reset_fences(&[self.fence])
                .map_err(RenderError::Vulkan)?;
        }

        Ok(())
    }
}

impl Drop for FrameRecorder {
    fn drop(&mut self) {
        // Safety: caller is responsible for ensuring no in-flight GPU work
        // references `self.fence` or `self.framebuffer`. The command pool (and
        // therefore `self.command_buffer`) is dropped automatically after these
        // explicit destroys because `self.pool` is a field of `Self` and Rust
        // drops fields in declaration order after the explicit ones here finish.
        unsafe {
            self.device.raw.destroy_fence(self.fence, None);
            self.device.raw.destroy_framebuffer(self.framebuffer, None);
        }
    }
}
