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
use crate::object::Drawable;
use crate::render::pipelines::line::{LineInstanceBuffer, LinePipeline, LinePushConstants};
use crate::render::pipelines::point::{PointInstanceBuffer, PointPipeline, PointPushConstants};
use crate::render::{
    Device, FrameRecorder, HeadlessRenderTarget, Instance, PipelineCache, RenderPass,
};
use crate::scene::config::{Output, SceneConfig};
use crate::scene::descriptor_pool::DescriptorPool;
use crate::scene::frame_context::FrameContext;

/// Per-scene instance-buffer capacity for [`Point`](crate::object::Point)
/// drawables. 1024 is comfortable for every Plan 3 demo and is well
/// under the 16 MiB budget desktop GPUs offer for a single
/// host-visible buffer.
const POINT_INSTANCE_CAPACITY: u32 = 1024;

/// Per-scene instance-buffer capacity for [`Line`](crate::object::Line)
/// drawables. Same rationale as
/// [`POINT_INSTANCE_CAPACITY`](POINT_INSTANCE_CAPACITY).
const LINE_INSTANCE_CAPACITY: u32 = 1024;

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
/// Construct with [`Scene::new`], drive with [`Scene::render`] (clear
/// only) or [`Scene::render_with`] (drawables). Dropping without
/// rendering is safe; any in-flight FFmpeg child is closed cleanly in
/// [`Drop`].
///
/// The `'anim` lifetime parameter is the borrow lifetime of any data
/// the scheduled animations may capture (typically a drawable owned by
/// the caller). Callers that schedule only owning animations can use
/// the unconstrained form `Scene<'static>`, which is what type
/// inference produces when there is nothing to bind against.
///
/// Field-drop order matters: Vulkan handles must be torn down before
/// the [`Device`] that created them. Rust drops fields in declaration
/// order, so the per-frame resources are listed before
/// `device` and `instance`.
pub struct Scene<'anim> {
    /// Frames-per-second for the output stream.
    fps: u32,
    /// Output resolution as `(width, height)`.
    resolution: (u32, u32),
    /// Linear-RGBA clear color used between drawables.
    background: [f32; 4],
    /// Camera that frames the scene. Read-only once the scene is
    /// constructed.
    camera: Camera,
    /// Timeline of scheduled animations. Owns each `Box<dyn Animation
    /// + 'anim>` and is dispatched once per frame by
    /// [`Scene::render_with`].
    timeline: Timeline<'anim>,
    /// Output sink. `None` once `render` has consumed it. `Some` while
    /// the scene is alive and unrendered, so `Drop` can close it
    /// gracefully.
    sink: Option<OutputSink>,
    /// Headless target the frame loop renders into and reads back from.
    target: HeadlessRenderTarget,
    /// Per-frame recorder; reused across the loop.
    recorder: FrameRecorder,
    /// Per-scene point instance buffer. Drained and re-uploaded each
    /// frame from the per-frame [`FrameContext::point_instances`]
    /// accumulator.
    point_buffer: PointInstanceBuffer,
    /// Per-scene line instance buffer. Drained and re-uploaded each
    /// frame from the per-frame [`FrameContext::line_instances`]
    /// accumulator.
    line_buffer: LineInstanceBuffer,
    /// GPU pipeline for [`Point`](crate::object::Point) drawables.
    point_pipeline: PointPipeline,
    /// GPU pipeline for [`Line`](crate::object::Line) and the wireframe
    /// edges of [`Plane`](crate::object::Plane) drawables.
    line_pipeline: LinePipeline,
    /// Single VkRenderPass used by the frame loop.
    render_pass: RenderPass,
    /// Per-scene descriptor pool. Plumbed through
    /// [`crate::scene::FrameContext`] for future drawables; not
    /// allocated from in Plan 3.
    descriptor_pool: DescriptorPool,
    /// Cache of GPU pipelines keyed by drawable kind. Populated
    /// lazily by drawables during recording.
    pipeline_cache: PipelineCache,
    /// Vulkan device shared by every render-side resource. Held last
    /// among the GPU fields so it outlives every child.
    #[allow(dead_code)]
    device: Arc<Device>,
    /// Held so the device's parent stays alive at least as long as the
    /// device. The instance is read by no other field directly but the
    /// Vulkan loader requires it to outlive every device.
    #[allow(dead_code)]
    instance: Arc<Instance>,
}

impl<'anim> Scene<'anim> {
    /// Construct a scene from configuration and a camera.
    ///
    /// On success, the Vulkan instance, device, render pass, headless
    /// render target, frame recorder, point + line pipelines, point +
    /// line instance buffers, and output sink are all live.
    /// `Output::Mp4` spawns a child `ffmpeg` process;
    /// `Output::PngSequence` creates the directory if it does not
    /// already exist.
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
        let point_pipeline = PointPipeline::new(device.clone(), &render_pass)?;
        let line_pipeline = LinePipeline::new(device.clone(), &render_pass)?;
        let point_buffer = PointInstanceBuffer::new(device.clone(), POINT_INSTANCE_CAPACITY)?;
        let line_buffer = LineInstanceBuffer::new(device.clone(), LINE_INSTANCE_CAPACITY)?;

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
            camera,
            timeline: Timeline::new(),
            sink: Some(sink),
            target,
            recorder,
            point_buffer,
            line_buffer,
            point_pipeline,
            line_pipeline,
            render_pass,
            descriptor_pool,
            pipeline_cache: PipelineCache::new(),
            device,
            instance,
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

    /// Run the frame loop with no drawables and finalise the output
    /// sink.
    ///
    /// Thin wrapper over [`Scene::render_with`] that supplies an empty
    /// record closure. Frame count is `ceil(total_run_time * fps)`;
    /// each frame clears the framebuffer to `config.background` and
    /// reads it back. The result is a monochrome video at the
    /// configured resolution and fps.
    ///
    /// Consumes `self`; the [`Drop`] impl is a no-op after this call
    /// because the sink has already been finalised.
    ///
    /// # Errors
    ///
    /// Returns [`RotorlabError::Render`] on any Vulkan failure during
    /// the loop. Returns [`RotorlabError::Encode`] if writing a frame
    /// to the sink fails or if FFmpeg exits non-zero.
    pub fn render(self) -> Result<u64, RotorlabError> {
        self.render_with(|_, _| {})
    }

    /// Run the frame loop with a per-frame record closure and finalise
    /// the output sink.
    ///
    /// Frame count is `ceil(timeline.total_run_time() * fps)`. Each
    /// frame:
    ///
    /// 1. dispatches every still-active timeline entry to its
    ///    `now = frame_index / fps`,
    /// 2. clears the per-frame point/line instance accumulators,
    /// 3. invokes `record_fn(&mut FrameContext, &Camera)` so the
    ///    caller can populate the accumulators by calling
    ///    [`Drawable::record`] on
    ///    each visible drawable,
    /// 4. uploads the accumulators to the per-scene instance buffers,
    /// 5. records the clear + line draws + point draws + readback
    ///    into a single command buffer and submits it,
    /// 6. forwards the readback bytes to the output sink.
    ///
    /// After the loop runs, a single trailing `dispatch_frame` at
    /// `total_run_time` fires any still-active animation's
    /// `interpolate(1.0) + finalize` pair without rendering a new
    /// frame.
    ///
    /// `record_fn` is `FnMut` so the caller can carry per-frame state
    /// (a frame counter, a clone of an `Arc<Mutex<...>>` driven by
    /// custom animations) across calls.
    ///
    /// # Errors
    ///
    /// Returns [`RotorlabError::Render`] on any Vulkan failure during
    /// the loop. Returns [`RotorlabError::Encode`] if writing a frame
    /// to the sink fails or if FFmpeg exits non-zero.
    pub fn render_with<F>(mut self, mut record_fn: F) -> Result<u64, RotorlabError>
    where
        F: FnMut(&mut FrameContext<'_>, &Camera),
    {
        let total_run_time = self.timeline.total_run_time();
        let total_frames = (total_run_time * self.fps as f32).ceil() as u32;

        let (width, height) = self.resolution;
        let mut host = vec![0u8; (width as usize) * (height as usize) * 4];

        // Per-frame accumulators reused across frames. Capacity 64 is
        // a small heuristic; the Vec grows on push if a busy frame
        // exceeds it.
        let mut point_instances = Vec::with_capacity(64);
        let mut line_instances = Vec::with_capacity(64);

        // Push-constant blocks rebuilt each frame. The view-proj is
        // currently constant across frames (camera does not animate in
        // Plan 3) but we recompute it per frame so future per-frame
        // camera updates land naturally.
        let view_proj = self.camera.view_proj();

        // Invariant: sink is set in `Scene::new` and only taken here.
        // `render_with` consumes self, so it cannot be called twice.
        let mut sink = self.sink.take().unwrap();

        for f in 0..total_frames {
            let now = f as f32 / self.fps as f32;
            self.timeline.dispatch_frame(now);

            point_instances.clear();
            line_instances.clear();

            // Build the FrameContext, hand it to the user record
            // closure. Drawables push into the two Vecs.
            {
                let mut ctx = FrameContext {
                    cmd_buf: self.recorder.command_buffer,
                    pipelines: &self.pipeline_cache,
                    descriptor_pool: &self.descriptor_pool,
                    point_instances: &mut point_instances,
                    line_instances: &mut line_instances,
                };
                record_fn(&mut ctx, &self.camera);
            }

            // Upload the accumulators to the per-scene instance
            // buffers. `upload` updates `len` so the draw call below
            // picks up the count.
            self.point_buffer
                .upload(&point_instances)
                .map_err(RotorlabError::Render)?;
            self.line_buffer
                .upload(&line_instances)
                .map_err(RotorlabError::Render)?;

            let point_push = PointPushConstants {
                view_proj,
                time: now,
                viewport: [width as f32, height as f32],
                _pad: 0.0,
            };
            let line_push = LinePushConstants {
                view_proj,
                viewport: [width as f32, height as f32],
                _pad: [0.0; 2],
            };

            self.recorder
                .render_drawables(
                    &self.render_pass,
                    &self.target,
                    self.background,
                    &self.point_pipeline,
                    &self.point_buffer,
                    &point_push,
                    &self.line_pipeline,
                    &self.line_buffer,
                    &line_push,
                )
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

        // Drive any still-active animation to alpha == 1.0 so its
        // `finalize` fires. The frame count is preserved (no extra
        // frame is rendered). Unconditional so instantaneous
        // animations (run_time == 0.0, total frames == 0) still
        // receive their interpolate(1.0) + finalize.
        self.timeline.dispatch_frame(total_run_time);

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

/// Marker to keep the [`Drawable`] symbol referenced even when the file
/// itself does not name a concrete drawable. The cross-references in
/// the `render_with` doc comment relied on the import; this `_` binding
/// keeps clippy's `unused_import` lint quiet without a `#[allow]`.
#[allow(dead_code)]
const _DRAWABLE_REF: Option<&dyn Drawable> = None;

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
    /// produced by the frame loop, and counts `finalize` calls so the
    /// post-loop alpha-1.0 dispatch can be observed. Mirrors the
    /// in-module `AlphaTrace` from [`crate::animation::timeline`]
    /// tests but re-declared here so the visibility surface stays
    /// clean.
    struct SceneAlphaTrace {
        calls: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
        finalize_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
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
        fn finalize(&mut self) {
            self.finalize_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
        let finalize_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let trace = SceneAlphaTrace {
            calls: calls.clone(),
            finalize_count: finalize_count.clone(),
            run_time: 1.0,
        };
        scene.play([Box::new(trace) as Box<dyn crate::animation::Animation>]);
        let n = scene.render().expect("render play(trace)");
        // small_png_config picks 30 fps, run_time=1.0 -> 30 frames.
        assert_eq!(n, 30);
        let observed = calls.lock().unwrap().clone();
        // 30 in-loop interpolate calls (frame indices 0..30 at now =
        // f/30) plus one post-loop dispatch at now = total_run_time
        // that drives alpha to 1.0 and triggers finalize.
        assert_eq!(observed.len(), 31);
        assert!((observed[0] - 0.0).abs() < 1e-6);
        let last = *observed.last().unwrap();
        assert!(
            (last - 1.0).abs() < 1e-6,
            "expected last alpha == 1.0, got {last}",
        );
        assert_eq!(
            finalize_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "expected finalize to fire exactly once",
        );
    }
}
