//! TrianglePipeline: a hello-world graphics pipeline that renders one tricolor
//! triangle to the headless render target's color attachment.

use crate::error::RenderError;
use crate::render::device::Device;
use crate::render::pipeline::create_shader_module;
use crate::render::render_pass::RenderPass;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::mem;
use std::sync::Arc;

/// One vertex of the triangle: 2D position + RGB color.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TriangleVertex {
    /// 2D clip-space position.
    pub pos: [f32; 2],
    /// Linear RGB color.
    pub color: [f32; 3],
}

/// The pre-baked tricolor triangle (corners at top, bottom-left, bottom-right).
pub const TRIANGLE_VERTICES: [TriangleVertex; 3] = [
    TriangleVertex {
        pos: [0.0, -0.6],
        color: [1.0, 0.0, 0.0],
    },
    TriangleVertex {
        pos: [-0.6, 0.6],
        color: [0.0, 1.0, 0.0],
    },
    TriangleVertex {
        pos: [0.6, 0.6],
        color: [0.0, 0.0, 1.0],
    },
];

const TRIANGLE_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const TRIANGLE_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

/// The triangle pipeline.
pub struct TrianglePipeline {
    /// Owning device.
    pub device: Arc<Device>,
    /// Pipeline layout (no descriptors, no push constants in v0.0.x).
    pub layout: vk::PipelineLayout,
    /// The pipeline itself.
    pub raw: vk::Pipeline,
}

impl TrianglePipeline {
    /// Build the pipeline against a render pass + viewport size.
    pub fn new(
        device: Arc<Device>,
        render_pass: &RenderPass,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        // Step 1: build shader modules from pre-compiled SPIR-V.
        let vert_module = create_shader_module(&device, TRIANGLE_VERT_SPV)?;
        let frag_module = create_shader_module(&device, TRIANGLE_FRAG_SPV)?;

        // Step 2: shader stage create infos.
        // The entry-point name must outlive the stage info structs; c"main" is
        // a static CStr literal so its lifetime is 'static.
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
        // Binding 0: one TriangleVertex per vertex (stride = 20 bytes).
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<TriangleVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        // Attribute 0: pos  — R32G32_SFLOAT,    offset 0
        // Attribute 1: color — R32G32B32_SFLOAT, offset 8 (2 * 4 bytes)
        let attrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(8),
        ];
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attrs);

        // Step 4: input assembly — triangle list, no primitive restart.
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Step 5: viewport state — one full-frame viewport + matching scissor.
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        // Step 6: rasterization — fill, no culling, CCW front face, no depth bias.
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        // Step 7: multisample — 1 sample, no sample shading.
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        // Step 8: color blend attachment — standard alpha blend, write all channels.
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

        // Step 9: color blend state — no logic op, one attachment.
        let blend_attachments = [blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&blend_attachments);

        // Step 10: pipeline layout — no descriptor set layouts, no push constants.
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // Safety: `device.raw` is a valid VkDevice; `layout_info` is fully
        // populated with empty descriptor / push-constant arrays. The returned
        // layout handle is owned solely by `Self` and destroyed in Drop.
        let layout = unsafe { device.raw.create_pipeline_layout(&layout_info, None) }?;

        // Step 11: assemble the VkGraphicsPipelineCreateInfo.
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(layout)
            .render_pass(render_pass.raw)
            .subpass(0);

        // Step 12: create the pipeline. The unusual return type on error is
        // Result<Vec<Pipeline>, (Vec<Pipeline>, vk::Result)>; we extract the
        // vk::Result from the error tuple.
        // Safety: `device.raw` is a valid VkDevice; `pipeline_info` and all
        // referenced structs are valid for the duration of this call.
        let raw = unsafe {
            device
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, r)| RenderError::Vulkan(r))?
        .into_iter()
        .next()
        .ok_or(RenderError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        // Step 13: destroy the shader modules — no longer needed after pipeline
        // creation; leaving them alive would leak device memory.
        // Safety: `vert_module` and `frag_module` are valid VkShaderModule
        // handles that have not been used for any other purpose since creation,
        // and the pipeline that references them has already been fully created.
        unsafe {
            device.raw.destroy_shader_module(vert_module, None);
            device.raw.destroy_shader_module(frag_module, None);
        }

        // Step 14: return the completed pipeline.
        Ok(Self {
            device,
            layout,
            raw,
        })
    }
}

impl Drop for TrianglePipeline {
    fn drop(&mut self) {
        // Safety: caller ensures no in-flight work uses this pipeline.
        // The device is kept alive via `self.device` (Arc<Device>).
        unsafe {
            self.device.raw.destroy_pipeline(self.raw, None);
            self.device.raw.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// A device-allocated vertex buffer containing `TRIANGLE_VERTICES`.
///
/// Host-visible memory in v0.0.x for simplicity; if frame rate matters we'll
/// add a staging-buffer-then-device-local upload path later.
pub struct TriangleVertexBuffer {
    /// Owning device.
    pub device: Arc<Device>,
    /// The buffer handle.
    pub buffer: vk::Buffer,
    /// Memory backing the buffer.
    pub memory: vk::DeviceMemory,
}

impl TriangleVertexBuffer {
    /// Allocate a host-visible vertex buffer and copy `TRIANGLE_VERTICES` into it.
    pub fn new(device: Arc<Device>) -> Result<Self, RenderError> {
        let size_bytes = (std::mem::size_of::<TriangleVertex>() * 3) as u64;

        // Step 2: build the buffer create info.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // Step 3: create the buffer.
        // Safety: `device.raw` is a valid VkDevice; `buffer_info` is fully
        // populated. The returned handle is owned by `Self` and destroyed in Drop.
        let buffer = unsafe { device.raw.create_buffer(&buffer_info, None) }?;

        // Step 4: query memory requirements.
        // Safety: `buffer` is a valid VkBuffer just created above.
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };

        // Step 5: get physical device memory properties.
        // Safety: `device.physical` is a valid VkPhysicalDevice selected during
        // device creation and remains valid for the lifetime of the instance.
        let mem_props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical)
        };

        // Step 6: find a HOST_VISIBLE | HOST_COHERENT memory type.
        let required_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type_index =
            find_memory_type(&mem_props, requirements.memory_type_bits, required_flags)
                .ok_or_else(|| {
                    RenderError::OutOfMemory(
                        "no HOST_VISIBLE | HOST_COHERENT memory type for vertex buffer".into(),
                    )
                })?;

        // Step 7: allocate device memory.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        // Safety: `device.raw` is a valid VkDevice; `alloc_info` is fully
        // populated with a valid memory type index.
        let memory = unsafe { device.raw.allocate_memory(&alloc_info, None) }?;

        // Step 8: bind the buffer to the allocated memory.
        // Safety: `buffer` and `memory` are valid handles; offset 0 satisfies
        // the buffer's alignment requirement because it is always a multiple of
        // any required alignment.
        unsafe { device.raw.bind_buffer_memory(buffer, memory, 0) }?;

        // Step 9: map the memory.
        // Safety: `memory` is valid and unbound to any other host mapping;
        // vk::WHOLE_SIZE maps the full allocation.
        let mapped = unsafe {
            device
                .raw
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }?;

        // Step 10: copy vertex data into the mapped region.
        // Safety: `mapped` points to at least `size_bytes` bytes of host-visible
        // device memory; `TRIANGLE_VERTICES` is a valid array of the same size;
        // the memory is HOST_COHERENT so no explicit flush is needed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                TRIANGLE_VERTICES.as_ptr().cast::<u8>(),
                mapped.cast::<u8>(),
                size_bytes as usize,
            );
        }

        // Step 11: unmap memory.
        // Safety: `memory` is currently mapped (step 9 succeeded); we will not
        // access `mapped` after this point.
        unsafe { device.raw.unmap_memory(memory) };

        Ok(Self {
            device,
            buffer,
            memory,
        })
    }
}

impl Drop for TriangleVertexBuffer {
    fn drop(&mut self) {
        // Safety: caller ensures no in-flight GPU work uses this buffer.
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

/// File-private memory-type finder (duplicates the one in render_target.rs;
/// dedupe is a future refactor).
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
    use crate::render::{Device, Instance, RenderPass};

    #[test]
    fn create_pipeline_smoke() {
        let Ok(i) = Instance::new() else { return };
        let Ok(d) = Device::new(i) else { return };
        let pass = RenderPass::new(d.clone()).expect("render pass");
        let p = TrianglePipeline::new(d, &pass, 1920, 1080).expect("pipeline");
        drop(p);
    }

    #[test]
    fn create_vertex_buffer_smoke() {
        let Ok(i) = Instance::new() else { return };
        let Ok(d) = Device::new(i) else { return };
        let vbo = TriangleVertexBuffer::new(d).expect("vbo");
        drop(vbo);
    }
}
