//! PointPipeline: an instanced billboard-quad pipeline that renders each
//! point as a screen-space-sized antialiased filled disk.
//!
//! Each instance is one [`PointInstance`] (world-space position, RGBA color,
//! screen-pixel radius). The pipeline emits a `TRIANGLE_STRIP` of four
//! vertices per instance using `gl_VertexIndex` (no vertex buffer for the
//! corners is bound; the only buffer is the instance buffer at binding 0).
//!
//! Push constants carry the camera `view_proj` matrix, a per-frame `time`,
//! and the framebuffer `viewport` size in pixels (the vertex shader needs
//! the viewport size to convert pixel radii into clip-space offsets).
//!
//! Plan 3 Task 4. The PNG hash-compare baseline test the spec mentions is
//! deferred to Plan 3 Task 13 (visual regression baselines); the smoke
//! tests in this file prove pipeline + instance buffer + push-constants
//! compile and bind without Vulkan validation errors.

use crate::error::RenderError;
use crate::render::device::Device;
use crate::render::pipeline::create_shader_module;
use crate::render::render_pass::RenderPass;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::mem;
use std::sync::Arc;

/// One instance of a point: homogeneous world-space position, linear RGBA
/// color, and a radius in screen-space pixels.
///
/// Radius is in **screen-space pixels**: a point with `radius = 8.0` always
/// covers an 8-pixel-radius disk on screen, regardless of how far the point
/// is from the camera or how aggressive the projection is. This matches the
/// default `Point` / `Dot` semantics in ManimGL and is the natural choice
/// for explainer videos where reading positions matters more than reading
/// distances.
///
/// `_pad` brings the struct to 48 bytes, which is friendly to std140-style
/// uniform layouts and a multiple of 16 bytes (the typical instance-buffer
/// alignment requirement on desktop GPUs). `world_pos.w` is reserved at
/// `1.0` (a homogeneous PGA3 point); `0.0` would denote a direction and is
/// not currently used.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct PointInstance {
    /// Homogeneous world-space position. `w = 1.0` for finite points.
    pub world_pos: [f32; 4],
    /// Linear-space RGBA color. Alpha is multiplied by the disk's edge
    /// antialiasing alpha in the fragment shader.
    pub color: [f32; 4],
    /// Disk radius in **screen-space pixels** (constant size on screen).
    pub radius: f32,
    /// Padding to bring the struct up to 48 bytes (16-byte aligned).
    pub _pad: [f32; 3],
}

/// Push-constant block for the point pipeline.
///
/// Total size is `64 + 4 + 8 + 4 = 80` bytes, well under Vulkan's
/// minimum-required push-constant size of 128 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct PointPushConstants {
    /// Camera view-projection matrix (column-major, GLSL `mat4`).
    pub view_proj: [[f32; 4]; 4],
    /// Per-frame time in seconds (free for the shader to consume).
    pub time: f32,
    /// Viewport size in pixels: `[width, height]`. Used to convert the
    /// screen-pixel radius into clip-space offsets.
    pub viewport: [f32; 2],
    /// Trailing pad so the block is a multiple of 4 bytes (and explicit).
    pub _pad: f32,
}

const POINT_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/point.vert.spv"));
const POINT_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/point.frag.spv"));

/// Instanced billboard-disk pipeline.
pub struct PointPipeline {
    /// Owning device.
    pub device: Arc<Device>,
    /// Pipeline layout (one push-constant range, no descriptor sets).
    pub layout: vk::PipelineLayout,
    /// The pipeline itself.
    pub raw: vk::Pipeline,
}

impl PointPipeline {
    /// Build the pipeline against a render pass.
    ///
    /// Viewport and scissor are dynamic state: the caller binds them per
    /// command buffer via `cmd_set_viewport` / `cmd_set_scissor`. This lets
    /// the same pipeline handle resolution changes in Plan 5+ without a
    /// rebuild.
    pub fn new(device: Arc<Device>, render_pass: &RenderPass) -> Result<Self, RenderError> {
        // Step 1: build shader modules from pre-compiled SPIR-V.
        let vert_module = create_shader_module(&device, POINT_VERT_SPV)?;
        let frag_module = create_shader_module(&device, POINT_FRAG_SPV)?;

        // Step 2: shader stage create infos. Entry-point name is c"main"
        // (a 'static CStr) so its lifetime trivially outlives the structs.
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"main"),
        ];

        // Step 3: vertex input state.
        // Binding 0: one PointInstance per instance (stride = 48 bytes).
        // No per-vertex buffer: the quad corners are emitted from
        // gl_VertexIndex inside the shader.
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<PointInstance>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)];
        // Attribute 0: world_pos (R32G32B32A32_SFLOAT, offset 0).
        // Attribute 1: color     (R32G32B32A32_SFLOAT, offset 16).
        // Attribute 2: radius    (R32_SFLOAT,          offset 32).
        let attrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(16),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32_SFLOAT)
                .offset(32),
        ];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attrs);

        // Step 4: input assembly. TRIANGLE_STRIP with no primitive restart;
        // the shader emits four corners per instance via gl_VertexIndex.
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .primitive_restart_enable(false);

        // Step 5: viewport state. Counts must match the dynamic-state
        // declaration; the actual values come from cmd_set_viewport /
        // cmd_set_scissor at record time.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Step 6: rasterization. Fill, no culling (the quad is always
        // front-facing in screen space), CCW front face, no depth bias.
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        // Step 7: multisample. One sample, no sample shading.
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        // Step 8: color blend attachment. Standard alpha blend (matches
        // TrianglePipeline) so the disk's antialiased edge composites
        // smoothly against the framebuffer.
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );

        // Step 9: color blend state. No logic op, one attachment.
        let blend_attachments = [blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&blend_attachments);

        // Step 10: dynamic state (viewport + scissor).
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Step 11: pipeline layout. One push-constant range covering the
        // full PointPushConstants block, visible to vertex + fragment.
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(mem::size_of::<PointPushConstants>() as u32);
        let push_ranges = [push_range];
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
        // Safety: `device.raw` is a valid VkDevice. `layout_info` is fully
        // populated. The returned layout is owned by `Self` and destroyed
        // in Drop.
        let layout = unsafe { device.raw.create_pipeline_layout(&layout_info, None) }?;

        // Step 12: assemble GraphicsPipelineCreateInfo.
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass.raw)
            .subpass(0);

        // Step 13: create the pipeline. Same error-shape unwrap as
        // TrianglePipeline (Result<Vec<P>, (Vec<P>, vk::Result)>).
        // Safety: `device.raw` is valid. `pipeline_info` and every struct
        // it references are valid for the duration of this call.
        let raw = unsafe {
            device
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, r)| RenderError::Vulkan(r))?
        .into_iter()
        .next()
        .ok_or(RenderError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        // Step 14: destroy the shader modules. They are no longer needed
        // once the pipeline is fully created; leaving them alive would
        // leak device memory.
        // Safety: both modules are valid VkShaderModule handles, were not
        // used elsewhere since creation, and the pipeline that referenced
        // them has been fully created.
        unsafe {
            device.raw.destroy_shader_module(vert_module, None);
            device.raw.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            device,
            layout,
            raw,
        })
    }
}

impl Drop for PointPipeline {
    fn drop(&mut self) {
        // Safety: caller ensures no in-flight GPU work uses this pipeline.
        // `self.device` (Arc<Device>) keeps the device alive.
        unsafe {
            self.device.raw.destroy_pipeline(self.raw, None);
            self.device.raw.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// A host-visible instance buffer for point instances.
///
/// Allocated once for `capacity` instances. `upload` writes a slice into
/// the buffer and updates `len` (which the caller uses as the
/// `instance_count` argument to `vkCmdDraw`). Resizing is a Plan 5+
/// optimisation; uploading more instances than `capacity` returns
/// [`RenderError::OutOfMemory`].
pub struct PointInstanceBuffer {
    /// Owning device.
    pub device: Arc<Device>,
    /// The buffer handle.
    pub buffer: vk::Buffer,
    /// Memory backing the buffer.
    pub memory: vk::DeviceMemory,
    /// Number of instances the buffer can hold.
    pub capacity: u32,
    /// Number of instances actually written by the most recent `upload`.
    pub len: u32,
}

impl PointInstanceBuffer {
    /// Allocate a host-visible instance buffer sized for `capacity`
    /// [`PointInstance`] entries.
    pub fn new(device: Arc<Device>, capacity: u32) -> Result<Self, RenderError> {
        let size_bytes = (mem::size_of::<PointInstance>() as u64) * capacity as u64;

        // Step 1: build the buffer create info. We will bind this as a
        // vertex buffer (instance-rate binding 0 in the pipeline).
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes.max(1))
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // Step 2: create the buffer.
        // Safety: `device.raw` is valid. `buffer_info` is fully populated.
        // The returned handle is owned by `Self` and destroyed in Drop.
        let buffer = unsafe { device.raw.create_buffer(&buffer_info, None) }?;

        // Step 3: query memory requirements.
        // Safety: `buffer` is a valid VkBuffer just created above.
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };

        // Step 4: query physical-device memory properties.
        // Safety: `device.physical` is a valid VkPhysicalDevice belonging
        // to `device.instance.raw`; the call only reads driver state.
        let mem_props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical)
        };

        // Step 5: find a HOST_VISIBLE | HOST_COHERENT memory type.
        let required_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type_index =
            find_memory_type(&mem_props, requirements.memory_type_bits, required_flags)
                .ok_or_else(|| {
                    RenderError::OutOfMemory(
                        "no HOST_VISIBLE | HOST_COHERENT memory type for point instance buffer"
                            .into(),
                    )
                })?;

        // Step 6: allocate device memory.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        // Safety: `device.raw` is valid; `alloc_info` is fully populated.
        let memory = unsafe { device.raw.allocate_memory(&alloc_info, None) }?;

        // Step 7: bind the buffer to memory at offset 0 (offset 0 always
        // satisfies any required alignment).
        // Safety: `buffer` and `memory` are valid handles.
        unsafe { device.raw.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self {
            device,
            buffer,
            memory,
            capacity,
            len: 0,
        })
    }

    /// Write `instances` into the buffer. Updates `self.len` on success.
    /// Returns [`RenderError::OutOfMemory`] if `instances.len() > capacity`.
    pub fn upload(&mut self, instances: &[PointInstance]) -> Result<(), RenderError> {
        if instances.len() as u64 > self.capacity as u64 {
            return Err(RenderError::OutOfMemory(format!(
                "point instance upload of {} exceeds capacity {}",
                instances.len(),
                self.capacity
            )));
        }

        // Empty upload is a no-op aside from resetting len; skip the map
        // entirely (mapping a zero-byte range is undefined).
        if instances.is_empty() {
            self.len = 0;
            return Ok(());
        }

        let bytes = std::mem::size_of_val(instances);

        // Step 1: map the memory.
        // Safety: `self.memory` is a valid HOST_VISIBLE | HOST_COHERENT
        // allocation; offset 0 + WHOLE_SIZE covers the full allocation;
        // no other thread holds a concurrent mapping.
        let mapped = unsafe {
            self.device
                .raw
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }?;

        // Step 2: copy the instance slice into the mapped region.
        // Safety: `mapped` points to at least `bytes` bytes of host-visible
        // device memory (we sized the allocation for `capacity` instances
        // and verified `instances.len() <= capacity`); `instances` is a
        // valid slice of the same byte length; HOST_COHERENT memory needs
        // no explicit flush.
        unsafe {
            std::ptr::copy_nonoverlapping(
                instances.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                bytes,
            );
        }

        // Step 3: unmap.
        // Safety: `self.memory` is currently mapped (Step 1 succeeded).
        unsafe { self.device.raw.unmap_memory(self.memory) };

        self.len = instances.len() as u32;
        Ok(())
    }
}

impl Drop for PointInstanceBuffer {
    fn drop(&mut self) {
        // Safety: caller ensures no in-flight GPU work uses this buffer.
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

/// File-private memory-type finder (duplicates the helpers in
/// `triangle.rs` and `render_target.rs`; dedupe is a future refactor).
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
    // The PNG hash-compare baseline test the spec mentions is deferred to
    // Plan 3 Task 13 ("Visual regression baselines"). The smoke tests here
    // prove the pipeline + instance buffer + push constants compile and
    // bind without Vulkan validation errors, which is most of the work;
    // Task 13 will add pixel-level validation.

    use super::*;
    use crate::render::{Device, Instance, RenderPass};

    #[test]
    fn point_push_constants_size_is_eighty() {
        // Acceptance criterion 7.
        assert_eq!(std::mem::size_of::<PointPushConstants>(), 80);
    }

    #[test]
    fn point_instance_size_is_forty_eight() {
        // 16 (world_pos) + 16 (color) + 4 (radius) + 12 (_pad) = 48.
        assert_eq!(std::mem::size_of::<PointInstance>(), 48);
    }

    #[test]
    fn create_point_pipeline_smoke() {
        let Ok(i) = Instance::new() else { return };
        let Ok(d) = Device::new(i) else { return };
        let pass = RenderPass::new(d.clone()).expect("render pass");
        let p = PointPipeline::new(d, &pass).expect("point pipeline");
        drop(p);
    }

    #[test]
    fn create_point_instance_buffer_smoke() {
        let Ok(i) = Instance::new() else { return };
        let Ok(d) = Device::new(i) else { return };
        let buf = PointInstanceBuffer::new(d, 16).expect("instance buffer");
        drop(buf);
    }

    #[test]
    fn point_instance_buffer_upload_within_capacity() {
        let Ok(i) = Instance::new() else { return };
        let Ok(d) = Device::new(i) else { return };
        let mut buf = PointInstanceBuffer::new(d, 4).expect("buffer");
        let instances = [
            PointInstance {
                world_pos: [0.0, 0.0, 0.0, 1.0],
                color: [1.0, 0.0, 0.0, 1.0],
                radius: 8.0,
                _pad: [0.0; 3],
            },
            PointInstance {
                world_pos: [1.0, 0.0, 0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
                radius: 8.0,
                _pad: [0.0; 3],
            },
        ];
        buf.upload(&instances).expect("upload");
        assert_eq!(buf.len, 2);
    }

    #[test]
    fn point_instance_buffer_upload_overflow_errors() {
        let Ok(i) = Instance::new() else { return };
        let Ok(d) = Device::new(i) else { return };
        let mut buf = PointInstanceBuffer::new(d, 1).expect("buffer");
        let instances = [
            PointInstance {
                world_pos: [0.0; 4],
                color: [0.0; 4],
                radius: 1.0,
                _pad: [0.0; 3],
            },
            PointInstance {
                world_pos: [0.0; 4],
                color: [0.0; 4],
                radius: 1.0,
                _pad: [0.0; 3],
            },
        ];
        assert!(buf.upload(&instances).is_err());
    }
}
