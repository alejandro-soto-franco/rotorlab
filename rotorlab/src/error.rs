//! Error types for the rotorlab engine.

use thiserror::Error;

/// Top-level error from any rotorlab operation.
#[derive(Debug, Error)]
pub enum RotorlabError {
    /// Render-side failure (Vulkan, shader compilation, framebuffer readback).
    #[error("render error: {0}")]
    Render(#[from] RenderError),
    /// Encode-side failure (FFmpeg child process, IO).
    #[error("encode error: {0}")]
    Encode(#[from] EncodeError),
}

/// Errors from the render module.
#[derive(Debug, Error)]
pub enum RenderError {
    /// A Vulkan API call returned a non-success result.
    #[error("Vulkan call failed: {0:?}")]
    Vulkan(ash::vk::Result),
    /// Shader compilation or linking failed.
    #[error("shader compilation failed: {0}")]
    Shader(String),
    /// Memory allocation failed.
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    /// No suitable physical device was found.
    #[error("no suitable Vulkan physical device found")]
    NoPhysicalDevice,
    /// Image / buffer mapping failed.
    #[error("memory mapping failed: {0}")]
    Mapping(String),
}

impl From<ash::vk::Result> for RenderError {
    fn from(r: ash::vk::Result) -> Self {
        RenderError::Vulkan(r)
    }
}

/// Errors from the encode module.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// IO error talking to the FFmpeg child process.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// FFmpeg exited non-zero.
    #[error("ffmpeg exited with status {0}: {1}")]
    FfmpegExit(i32, String),
    /// FFmpeg binary not found on PATH.
    #[error("ffmpeg binary not found on PATH")]
    FfmpegNotFound,
    /// Image encoding failure (PNG save, format conversion).
    #[error("image encoding error: {0}")]
    Image(#[from] image::ImageError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_error_displays() {
        let e = RenderError::NoPhysicalDevice;
        assert!(format!("{e}").contains("physical device"));
    }

    #[test]
    fn render_into_rotorlab() {
        let r: RotorlabError = RenderError::NoPhysicalDevice.into();
        assert!(matches!(r, RotorlabError::Render(_)));
    }

    #[test]
    fn vk_result_into_render() {
        let r: RenderError = ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY.into();
        assert!(matches!(r, RenderError::Vulkan(_)));
    }
}
