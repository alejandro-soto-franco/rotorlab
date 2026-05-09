//! Shared helpers for graphics pipeline construction.

use crate::error::RenderError;
use crate::render::device::Device;
use ash::vk;

/// Compile a SPIR-V byte slice into a `VkShaderModule` on the given device.
///
/// The `spv` slice must be a whole number of 32-bit words (a SPIR-V
/// invariant). The bytes are copied into an aligned `Vec<u32>` because
/// `include_bytes!` provides only 1-byte alignment on the embedded data.
pub fn create_shader_module(device: &Device, spv: &[u8]) -> Result<vk::ShaderModule, RenderError> {
    assert_eq!(spv.len() % 4, 0, "SPIR-V byte length must be 4-aligned");
    // Copy into an owned, 4-byte-aligned buffer. `u32::from_le_bytes` reads
    // each 4-byte chunk in little-endian order, which matches the SPIR-V
    // on-disk format (SPIR-V magic: 0x07230203, little-endian).
    let code: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    // Safety: `info` is fully populated with a valid SPIR-V word slice;
    // device handle is valid for the duration of this call.
    let module = unsafe { device.raw.create_shader_module(&info, None) }?;
    Ok(module)
}
