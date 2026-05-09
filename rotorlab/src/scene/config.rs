//! Scene configuration: framerate, resolution, clear color, output sink.

use std::path::PathBuf;

/// x264 `-preset` mapping for the [`Output::Mp4`] encoder.
///
/// The preset trades encode time against compression efficiency; see the
/// `x264 --help` output for the full picture. `Veryslow` is the highest
/// quality at a given CRF; `Ultrafast` is the cheapest. The default used
/// by the engine elsewhere (the `triangle_demo` baseline) is [`Slow`],
/// which matches `FfmpegEncoder::new`'s hardcoded preset.
///
/// [`Slow`]: H264Preset::Slow
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum H264Preset {
    /// Fastest encode, largest output. Useful for smoke tests.
    Ultrafast,
    /// Roughly 8x faster than `Medium`, ~30% larger output.
    Veryfast,
    /// Roughly 2x faster than `Medium`.
    Fast,
    /// x264's default preset.
    Medium,
    /// Slower than `Medium`, smaller output. Engine default.
    Slow,
    /// Substantially slower; marginal gains over `Slow` in most cases.
    Veryslow,
}

impl H264Preset {
    /// Return the preset name as the literal string x264 expects on its
    /// `-preset` argument. Stable lifetime; safe to pass straight to
    /// `Command::arg`.
    pub fn as_str(&self) -> &'static str {
        match self {
            H264Preset::Ultrafast => "ultrafast",
            H264Preset::Veryfast => "veryfast",
            H264Preset::Fast => "fast",
            H264Preset::Medium => "medium",
            H264Preset::Slow => "slow",
            H264Preset::Veryslow => "veryslow",
        }
    }
}

/// Where rendered frames go.
#[derive(Clone, Debug)]
pub enum Output {
    /// Single `.mp4` file produced by piping BGRA frames into a child
    /// `ffmpeg` process. `crf` controls quality (lower is better;
    /// 18 is visually lossless, 23 is ffmpeg's default). `preset` is the
    /// x264 speed/efficiency tradeoff.
    Mp4 {
        /// Path to the output `.mp4` file. Will be overwritten if it
        /// exists.
        path: PathBuf,
        /// x264 constant rate factor. Lower = higher quality.
        crf: u32,
        /// x264 speed/efficiency preset.
        preset: H264Preset,
    },
    /// Directory of zero-padded `frame_NNNNNN.png` files, one per frame.
    /// Useful for visual regression baselines and for downstream tools
    /// that prefer image sequences.
    PngSequence {
        /// Directory to write into. Created by the caller; the engine
        /// only writes files inside it.
        dir: PathBuf,
    },
}

/// Top-level configuration for a [`Scene`](crate::scene::Scene).
///
/// Fields use `Default` values appropriate for a 1080p60 explainer video
/// with a near-black background. Override fields with struct-update
/// syntax: `SceneConfig { fps: 30, ..Default::default() }`.
///
/// `Default` does not provide a sensible [`Output`]; callers must
/// supply one explicitly. The default value is a [`PngSequence`] under
/// the current working directory, which is rarely what the caller wants
/// in production but is harmless in tests.
///
/// [`PngSequence`]: Output::PngSequence
#[derive(Clone, Debug)]
pub struct SceneConfig {
    /// Frames per second. Default: `60`.
    pub fps: u32,
    /// Output resolution as `(width, height)` in pixels. Default:
    /// `(1920, 1080)`.
    pub resolution: (u32, u32),
    /// Clear color in linear RGBA, range `[0.0, 1.0]`. Default:
    /// `[0.02, 0.02, 0.02, 1.0]` (near-black).
    pub background: [f32; 4],
    /// Output sink: mp4 file or PNG sequence.
    pub output: Output,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            fps: 60,
            resolution: (1920, 1080),
            background: [0.02, 0.02, 0.02, 1.0],
            output: Output::PngSequence {
                dir: PathBuf::from("."),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_1080p60_near_black() {
        let cfg = SceneConfig::default();
        assert_eq!(cfg.fps, 60);
        assert_eq!(cfg.resolution, (1920, 1080));
        assert_eq!(cfg.background[3], 1.0);
        for c in &cfg.background[..3] {
            assert!(*c < 0.1, "background should be near-black, got {c}");
        }
    }

    #[test]
    fn h264_preset_strings_match_x264() {
        assert_eq!(H264Preset::Ultrafast.as_str(), "ultrafast");
        assert_eq!(H264Preset::Veryfast.as_str(), "veryfast");
        assert_eq!(H264Preset::Fast.as_str(), "fast");
        assert_eq!(H264Preset::Medium.as_str(), "medium");
        assert_eq!(H264Preset::Slow.as_str(), "slow");
        assert_eq!(H264Preset::Veryslow.as_str(), "veryslow");
    }

    #[test]
    fn struct_update_syntax_works() {
        let cfg = SceneConfig {
            fps: 30,
            ..Default::default()
        };
        assert_eq!(cfg.fps, 30);
        assert_eq!(cfg.resolution, (1920, 1080));
    }
}
