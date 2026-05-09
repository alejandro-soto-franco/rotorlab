//! FFmpeg child-process wrapper.
//!
//! Spawns `ffmpeg` with raw BGRA stdin pipe; writes frames via `submit_frame`;
//! `finish()` closes stdin and waits for the child to exit cleanly.

use crate::error::EncodeError;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

/// One-shot FFmpeg encoder for a fixed-size, fixed-fps BGRA stream.
pub struct FfmpegEncoder {
    child: Child,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    fps: u32,
    frames_written: u64,
}

impl FfmpegEncoder {
    /// Spawn ffmpeg writing to `output_path`. Resolution and fps are baked in.
    /// Quality preset: x264 CRF 18 (visually lossless), preset slow.
    pub fn new(output_path: &Path, width: u32, height: u32, fps: u32) -> Result<Self, EncodeError> {
        let child = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgra",
                "-s",
                &format!("{width}x{height}"),
                "-r",
                &fps.to_string(),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                output_path.to_str().expect("utf8 path"),
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    EncodeError::FfmpegNotFound
                } else {
                    EncodeError::Io(e)
                }
            })?;
        if child.stdin.is_none() {
            return Err(EncodeError::Io(std::io::Error::other(
                "ffmpeg stdin not piped",
            )));
        }
        Ok(Self {
            child,
            width,
            height,
            fps,
            frames_written: 0,
        })
    }

    /// Pipe one BGRA frame to ffmpeg.
    pub fn submit_frame(&mut self, bgra: &[u8]) -> Result<(), EncodeError> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if bgra.len() != expected {
            return Err(EncodeError::Io(std::io::Error::other(format!(
                "frame size mismatch: got {} bytes, expected {}",
                bgra.len(),
                expected
            ))));
        }
        let stdin = self.child.stdin.as_mut().expect("piped");
        stdin.write_all(bgra)?;
        self.frames_written += 1;
        Ok(())
    }

    /// Close stdin and wait for ffmpeg to exit.
    /// Returns the number of frames submitted on success.
    pub fn finish(mut self) -> Result<u64, EncodeError> {
        // Closing stdin signals EOF to ffmpeg.
        drop(self.child.stdin.take());
        let status = self.child.wait()?;
        if !status.success() {
            // Read stderr for diagnostics.
            let mut stderr_text = String::new();
            if let Some(mut stderr) = self.child.stderr.take() {
                use std::io::Read;
                let _ = stderr.read_to_string(&mut stderr_text);
            }
            return Err(EncodeError::FfmpegExit(
                status.code().unwrap_or(-1),
                stderr_text
                    .lines()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ));
        }
        Ok(self.frames_written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn encode_blank_frames() {
        let dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = dir.path().join("blank.mp4");
        let (w, h, fps) = (320u32, 180, 30);
        let mut enc = match FfmpegEncoder::new(&path, w, h, fps) {
            Ok(e) => e,
            Err(EncodeError::FfmpegNotFound) => {
                eprintln!("skip: ffmpeg not on PATH");
                return;
            }
            Err(e) => panic!("ffmpeg spawn failed: {e}"),
        };
        let frame = vec![0u8; (w * h * 4) as usize];
        for _ in 0..30 {
            enc.submit_frame(&frame).unwrap();
        }
        let n = enc.finish().expect("finish ok");
        assert_eq!(n, 30);
        assert!(path.exists());
        let bytes = std::fs::metadata(&path).unwrap().len();
        assert!(bytes > 1000, "mp4 should be at least 1 KB, got {bytes}");
    }
}
