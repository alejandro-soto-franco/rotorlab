//! Frame loop scaffolding: opens GPU resources + an output sink, runs
//! one render call per frame across the timeline, finalises the sink.
//!
//! The `Scene` is the public entry point for end users:
//!
//! ```no_run
//! use rotorlab::camera::Camera;
//! use rotorlab::scene::{Output, Scene, SceneConfig};
//! use rotorlab_ga::pga3::point;
//! use std::path::PathBuf;
//!
//! let cfg = SceneConfig {
//!     output: Output::Mp4 {
//!         path: PathBuf::from("out.mp4"),
//!         crf: 18,
//!         preset: rotorlab::scene::H264Preset::Slow,
//!     },
//!     ..Default::default()
//! };
//! let camera = Camera::look_at(
//!     point(0.0, 0.0, 5.0),
//!     point(0.0, 0.0, 0.0),
//!     [0.0, 1.0, 0.0],
//! );
//! let scene = Scene::new(cfg, camera).expect("scene init");
//! let _frames = scene.render().expect("render");
//! ```

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::animation::{Animation, Timeline};
use crate::camera::Camera;
use crate::encode::FfmpegEncoder;
use crate::error::{EncodeError, RotorlabError};
use crate::render::{
    Device, FrameRecorder, HeadlessRenderTarget, Instance, PipelineCache, RenderPass,
};
use crate::scene::config::{Output, SceneConfig};
use crate::scene::descriptor_pool::DescriptorPool;

/// Output sink resolved at construction time.
///
/// `Output::Mp4` becomes a child `ffmpeg` process; `Output::PngSequence`
/// becomes a directory plus a frame counter. Held inside an `Option` on
/// the [`Scene`] so [`Scene::render`] can take ownership without
/// forcing every move on the struct.
enum OutputSink {
    /// FFmpeg child receiving raw BGRA frames over stdin.
    Mp4(FfmpegEncoder),
    /// Directory of `frame_NNNNNN.png` files.
    PngSequence {
        /// Output directory.
        dir: PathBuf,
        /// Number of PNG frames written so far.
        frames_written: u64,
    },
}

/// A scene: GPU resources + camera + timeline + pipeline cache + output
/// sink, ready to render.
///
/// Construct with [`Scene::new`], drive with [`Scene::render`]. Dropping
/// without rendering is safe; any in-flight FFmpeg child is closed
/// cleanly in [`Drop`].
///
/// The `'anim` lifetime parameter is the borrow lifetime of any data
/// the scheduled animations may capture (typically a drawable owned by
/// the caller). Callers that schedule only owning animations can use
/// the unconstrained form `Scene<'static>`, which is what type
/// inference produces when there is nothing to bind against.
pub struct Scene<'anim> {
    /// Frames-per-second for the output stream.
    fps: u32,
    /// Output resolution as `(width, height)`.
    resolution: (u32, u32),
    /// Linear-RGBA clear color used between drawables.
    background: [f32; 4],
    /// Held so the device's parent stays alive at least as long as the
    /// device. The instance is read by no other field directly but the
    /// Vulkan loader requires it to outlive every device.
    #[allow(dead_code)]
    instance: Arc<Instance>,
    /// Vulkan device shared by every render-side resource.
    #[allow(dead_code)]
    device: Arc<Device>,
    /// Single VkRenderPass used by the clear-only frame loop.
    render_pass: RenderPass,
    /// Headless target the frame loop renders into and reads back from.
    target: HeadlessRenderTarget,
    /// Per-frame recorder; reused across the loop.
    recorder: FrameRecorder,
    /// Camera that frames the scene. Currently held but not yet
    /// uploaded to the GPU; Plan 3 Task 3 wires `view_proj` into the
    /// per-frame uniform.
    #[allow(dead_code)]
    camera: Camera,
    /// Timeline of scheduled animations. Owns each `Box<dyn Animation
    /// + 'anim>` and is dispatched once per frame by
    /// [`Scene::render`].
    timeline: Timeline<'anim>,
    /// Cache of GPU pipelines keyed by drawable kind. Populated
    /// lazily by drawables during recording.
    #[allow(dead_code)]
    pipeline_cache: PipelineCache,
    /// Per-scene descriptor pool. Plumbed through
    /// [`crate::scene::FrameContext`] for future drawables; not
    /// allocated from in Plan 3.
    ///
    /// Holds its own `Arc<Device>` clone, so the underlying
    /// `VkDescriptorPool` is destroyed against a live device whatever
    /// the field-drop order ends up being.
    #[allow(dead_code)]
    descriptor_pool: DescriptorPool,
    /// Output sink. `None` once `render` has consumed it. `Some` while
    /// the scene is alive and unrendered, so `Drop` can close it
    /// gracefully.
    sink: Option<OutputSink>,
}

impl<'anim> Scene<'anim> {
    /// Construct a scene from configuration and a camera.
    ///
    /// On success, the Vulkan instance, device, render pass, headless
    /// render target, frame recorder, and output sink are all live.
    /// `Output::Mp4` spawns a child `ffmpeg` process; `Output::PngSequence`
    /// creates the directory if it does not already exist.
    ///
    /// # Errors
    ///
    /// Returns [`RotorlabError::Render`] if any Vulkan resource fails to
    /// initialise (no GPU, missing driver, out-of-memory). Returns
    /// [`RotorlabError::Encode`] if the FFmpeg child fails to spawn or
    /// the PNG output directory cannot be created.
    pub fn new(config: SceneConfig, camera: Camera) -> Result<Self, RotorlabError> {
        let (width, height) = config.resolution;

        let instance = Instance::new()?;
        let device = Device::new(instance.clone())?;
        let render_pass = RenderPass::new(device.clone())?;
        let target = HeadlessRenderTarget::new(device.clone(), width, height)?;
        let recorder = FrameRecorder::new(device.clone(), &render_pass, &target)?;
        let descriptor_pool = DescriptorPool::new(device.clone())?;

        let sink = match &config.output {
            Output::Mp4 { path, crf, preset } => {
                let enc = FfmpegEncoder::with_quality(
                    path,
                    width,
                    height,
                    config.fps,
                    *crf,
                    preset.as_str(),
                )?;
                OutputSink::Mp4(enc)
            }
            Output::PngSequence { dir } => {
                fs::create_dir_all(dir).map_err(EncodeError::Io)?;
                OutputSink::PngSequence {
                    dir: dir.clone(),
                    frames_written: 0,
                }
            }
        };

        Ok(Self {
            fps: config.fps,
            resolution: config.resolution,
            background: config.background,
            instance,
            device,
            render_pass,
            target,
            recorder,
            camera,
            timeline: Timeline::new(),
            pipeline_cache: PipelineCache::new(),
            descriptor_pool,
            sink: Some(sink),
        })
    }

    /// Schedule an iterable of animations to start at the timeline's
    /// current cursor.
    ///
    /// Thin wrapper over [`Timeline::play`]: the cursor advances by the
    /// schedule's full footprint (max `start_offset + run_time` across
    /// the iterable), the input box-iterable is consumed, and the
    /// timeline owns the boxed animations from then on.
    pub fn play<I>(&mut self, anims: I)
    where
        I: IntoIterator<Item = Box<dyn Animation + 'anim>>,
    {
        self.timeline.play(anims);
    }

    /// Advance the timeline cursor by `seconds` without scheduling any
    /// animation.
    ///
    /// Thin wrapper over [`Timeline::wait`]: at the configured FPS this
    /// produces `ceil(seconds * fps)` extra clear-only frames in the
    /// output stream.
    pub fn wait(&mut self, seconds: f32) {
        self.timeline.wait(seconds);
    }

    /// Run the frame loop and finalise the output sink.
    ///
    /// Frame count is `ceil(timeline.total_run_time() * fps)`. Each
    /// frame dispatches every still-active timeline entry to its
    /// `now = frame_index / fps`, then clears + reads back at
    /// `config.background`. Real drawable rendering lands in Plan 3
    /// Tasks 4 through 8. Returns the number of frames written to the
    /// sink.
    ///
    /// Consumes `self`; the [`Drop`] impl is a no-op after this call
    /// because the sink has already been finalised.
    ///
    /// # Errors
    ///
    /// Returns [`RotorlabError::Render`] on any Vulkan failure during
    /// the loop. Returns [`RotorlabError::Encode`] if writing a frame
    /// to the sink fails or if FFmpeg exits non-zero.
    pub fn render(mut self) -> Result<u64, RotorlabError> {
        let total_run_time = self.timeline.total_run_time();
        let total_frames = (total_run_time * self.fps as f32).ceil() as u32;

        let (width, height) = self.resolution;
        let mut host = vec![0u8; (width as usize) * (height as usize) * 4];

        // Invariant: sink is set in `Scene::new` and only taken here.
        // `render` consumes self, so it cannot be called twice.
        let mut sink = self.sink.take().unwrap();

        for f in 0..total_frames {
            let now = f as f32 / self.fps as f32;
            self.timeline.dispatch_frame(now);
            self.recorder
                .clear_only(&self.render_pass, &self.target, self.background)
                .map_err(RotorlabError::Render)?;
            self.target
                .read_to_host(&mut host)
                .map_err(RotorlabError::Render)?;
            match &mut sink {
                OutputSink::Mp4(enc) => enc.submit_frame(&host).map_err(RotorlabError::Encode)?,
                OutputSink::PngSequence {
                    dir,
                    frames_written,
                } => {
                    write_bgra_png(dir, *frames_written, width, height, &host)
                        .map_err(RotorlabError::Encode)?;
                    *frames_written += 1;
                }
            }
        }

        match sink {
            OutputSink::Mp4(enc) => enc.finish().map_err(RotorlabError::Encode),
            OutputSink::PngSequence { frames_written, .. } => Ok(frames_written),
        }
    }
}

impl Drop for Scene<'_> {
    /// Close any still-open output sink cleanly.
    ///
    /// If [`Scene::render`] was called the sink is already gone and
    /// this is a no-op. If the scene was dropped without rendering, an
    /// open FFmpeg child is told EOF on stdin and reaped; an
    /// in-progress PNG directory is left as-is (the caller can inspect
    /// what was written).
    fn drop(&mut self) {
        if let Some(sink) = self.sink.take() {
            match sink {
                OutputSink::Mp4(enc) => {
                    // Best-effort: we ignore both the frame count and
                    // the exit status because there is no caller to
                    // surface them to. FFmpeg may produce an empty or
                    // truncated mp4; this is the documented behaviour
                    // when a scene is dropped without rendering.
                    let _ = enc.finish();
                }
                OutputSink::PngSequence { .. } => {
                    // Nothing to finalise for a PNG sequence; the
                    // already-written frames are valid PNGs on disk.
                }
            }
        }
    }
}

/// Write one BGRA frame as `dir/frame_NNNNNN.png`.
///
/// The image crate works in RGBA, so we convert in place into a fresh
/// buffer. Padded to 6 zero-filled digits, matching ffmpeg's
/// `image2 -start_number 0` convention.
fn write_bgra_png(
    dir: &std::path::Path,
    index: u64,
    width: u32,
    height: u32,
    bgra: &[u8],
) -> Result<(), EncodeError> {
    let mut rgba = vec![0u8; bgra.len()];
    for (src, dst) in bgra.chunks_exact(4).zip(rgba.chunks_exact_mut(4)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = src[3];
    }
    // Buffer-shape mismatch is a caller-side bug (wrong width/height vs.
    // bgra length), not an `image` crate failure, so it stays an `Io`
    // variant. The save error below comes from inside the `image` crate
    // and gets the structured `Image` variant via `?`.
    let img = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| EncodeError::Io(std::io::Error::other("rgba buffer size mismatch")))?;
    let path = dir.join(format!("frame_{index:06}.png"));
    img.save(&path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::config::H264Preset;
    use rotorlab_ga::pga3::point;
    use tempfile::tempdir;

    fn default_camera() -> Camera {
        Camera::look_at(point(0.0, 0.0, 5.0), point(0.0, 0.0, 0.0), [0.0, 1.0, 0.0])
    }

    /// Make a small PngSequence config so tests do not allocate
    /// 1080p-sized framebuffers (and do not need ffmpeg on PATH).
    fn small_png_config(dir: PathBuf) -> SceneConfig {
        SceneConfig {
            fps: 30,
            resolution: (64, 36),
            output: Output::PngSequence { dir },
            ..Default::default()
        }
    }

    #[test]
    fn scene_constructs_with_default_config() {
        let dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = small_png_config(dir.path().to_path_buf());
        let Ok(scene) = Scene::new(cfg, default_camera()) else {
            eprintln!("skip: vulkan unavailable");
            return;
        };
        drop(scene);
    }

    #[test]
    fn scene_drop_closes_encoder_cleanly() {
        let dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        // Try Mp4 first (exercises FFmpeg child); fall back to PNG if
        // ffmpeg is not installed. Either path proves Drop does not
        // panic.
        let mp4_path = dir.path().join("scene_drop.mp4");
        let mp4_cfg = SceneConfig {
            fps: 30,
            resolution: (64, 36),
            output: Output::Mp4 {
                path: mp4_path,
                crf: 23,
                preset: H264Preset::Ultrafast,
            },
            ..Default::default()
        };
        match Scene::new(mp4_cfg, default_camera()) {
            Ok(scene) => {
                drop(scene);
                // Best-effort wait for the ffmpeg child to reap; on
                // most systems this is microseconds. We do not assert
                // exit status because waitpid timing is finicky and
                // the test is about "no panic on drop".
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(RotorlabError::Encode(EncodeError::FfmpegNotFound)) => {
                eprintln!("skip: ffmpeg not on PATH; falling back to PngSequence drop check");
                let cfg = small_png_config(dir.path().to_path_buf());
                let Ok(scene) = Scene::new(cfg, default_camera()) else {
                    eprintln!("skip: vulkan unavailable");
                    return;
                };
                drop(scene);
            }
            Err(_) => {
                eprintln!("skip: vulkan unavailable");
            }
        }
    }

    #[test]
    fn scene_with_zero_length_timeline_produces_zero_frames() {
        let dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = small_png_config(dir.path().to_path_buf());
        let Ok(scene) = Scene::new(cfg, default_camera()) else {
            eprintln!("skip: vulkan unavailable");
            return;
        };
        let n = scene.render().expect("render zero-length timeline");
        assert_eq!(n, 0);
    }

    /// Test stub used by the Scene-level dispatch tests below. Records
    /// every alpha it observes so the test can assert the trace shape
    /// produced by the frame loop. Mirrors the in-module
    /// `AlphaTrace` from [`crate::animation::timeline`] tests but
    /// re-declared here so the visibility surface stays clean.
    struct SceneAlphaTrace {
        calls: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
        run_time: f32,
    }

    impl crate::animation::Animation for SceneAlphaTrace {
        fn run_time(&self) -> f32 {
            self.run_time
        }
        fn rate_func(&self) -> crate::animation::RateFunc {
            crate::animation::RateFunc::Linear
        }
        fn interpolate(&mut self, alpha: f32) {
            self.calls.lock().unwrap().push(alpha);
        }
    }

    #[test]
    fn scene_wait_one_second_produces_60_frames_at_60_fps() {
        let dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = SceneConfig {
            fps: 60,
            resolution: (64, 36),
            output: Output::PngSequence {
                dir: dir.path().to_path_buf(),
            },
            ..Default::default()
        };
        let Ok(mut scene) = Scene::new(cfg, default_camera()) else {
            eprintln!("skip: vulkan unavailable");
            return;
        };
        scene.wait(1.0);
        let n = scene.render().expect("render wait(1.0) at 60 fps");
        assert_eq!(n, 60);
    }

    #[test]
    fn scene_play_dispatches_animations_through_render_loop() {
        let dir = match tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = small_png_config(dir.path().to_path_buf());
        let Ok(mut scene) = Scene::new(cfg, default_camera()) else {
            eprintln!("skip: vulkan unavailable");
            return;
        };
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let trace = SceneAlphaTrace {
            calls: calls.clone(),
            run_time: 1.0,
        };
        scene.play([Box::new(trace) as Box<dyn crate::animation::Animation>]);
        let n = scene.render().expect("render play(trace)");
        // small_png_config picks 30 fps, run_time=1.0 -> 30 frames.
        assert_eq!(n, 30);
        let observed = calls.lock().unwrap().clone();
        assert_eq!(observed.len(), 30);
        assert!((observed[0] - 0.0).abs() < 1e-6);
        // Last frame's alpha is 29/30 (linear ease, frame index f goes
        // 0..30, now = f/30, raw = (f/30) / 1.0 = f/30; the f == 30
        // tick is excluded because total_frames == 30).
        let last = *observed.last().unwrap();
        assert!(
            (last - (29.0_f32 / 30.0)).abs() < 1e-6,
            "expected last alpha ~= 29/30, got {last}",
        );
    }
}
