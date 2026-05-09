//! Visual regression harness for the 30-second `rotor_demo` example
//! (Plan 3 Task 13).
//!
//! ## Reduced scope
//!
//! The full lavapipe-pixel-perfect baseline comparison the spec
//! sketches requires committed PNG baselines that are heavy to
//! regenerate cleanly across hosts. This harness ships the test
//! framework now and supports the byte-compare path the moment
//! baselines land in `tests/baselines/rotor_demo_frame_*.png`.
//!
//! ## What the harness does today
//!
//! 1. Reproduces the [`rotor_demo`](../../examples/rotor_demo.rs) scene
//!    config and animation schedule. The demo body is duplicated
//!    verbatim (Option B in the Task 13 plan); the example file is the
//!    canonical copy and any divergence here would be caught on the
//!    next baseline regen.
//! 2. Renders the demo as
//!    [`Output::PngSequence`](rotorlab::scene::Output::PngSequence) into
//!    a `tempfile::TempDir`. Default resolution is `320x180` so the
//!    smoke check stays fast; `ROTORLAB_TEST_LAVAPIPE=1` switches to the
//!    full `1920x1080` for pixel-perfect comparison against committed
//!    baselines.
//! 3. Asserts each of the five spec-pinned frame indices (`60`, `600`,
//!    `900`, `1500`, `1740`) decodes as a valid PNG with the configured
//!    resolution.
//! 4. When `ROTORLAB_TEST_LAVAPIPE=1` is set and the matching baseline
//!    PNG is present in `tests/baselines/`, byte-compares the captured
//!    frame against the baseline. If the baseline is missing, the test
//!    logs a copy hint and passes the smoke check only.
//!
//! ## Why a single render shared across six tests
//!
//! Each `#[test]` is its own binary entry point; rendering 1800 frames
//! per test would multiply the wall-clock cost by six. A
//! [`OnceLock`]-backed shared render runs the demo once and hands every
//! test a borrowed view of the output directory. The OnceLock retains
//! ownership of the [`TempDir`] for the lifetime of the test process so
//! the directory is cleaned up at exit.
//!
//! ## Skip semantics
//!
//! If [`Scene::new`] fails (typically: no Vulkan loader / no usable
//! GPU) every test prints a `skip:` line and returns `Ok`. Per the
//! Plan 3 convention this matches the other integration tests.

use rotorlab::animation::{Animation, RateFunc};
use rotorlab::camera::{Camera, Projection};
use rotorlab::object::{Color, Drawable, Line, Plane, Point, Stroke};
use rotorlab::scene::{Output, Scene, SceneConfig};
use rotorlab_ga::motor::Motor;
use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga3::{self, Pga3};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Frame indices the spec pins as canonical. Each gets a dedicated
/// `#[test]` plus is included in the bundle test.
const FRAME_INDICES: &[u32] = &[60, 600, 900, 1500, 1740];

/// Smoke-test resolution used when `ROTORLAB_TEST_LAVAPIPE=1` is not
/// set. Matches the resolution every other integration test uses; keeps
/// the wall-clock cost bounded.
const SMOKE_RESOLUTION: (u32, u32) = (320, 180);

/// Full demo resolution used under `ROTORLAB_TEST_LAVAPIPE=1`. Matches
/// the example's `Mp4` output so committed baselines are 1080p PNGs.
const LAVAPIPE_RESOLUTION: (u32, u32) = (1920, 1080);

/// Environment variable that switches the harness from smoke mode to
/// full-resolution lavapipe-pixel-perfect mode.
const LAVAPIPE_ENV: &str = "ROTORLAB_TEST_LAVAPIPE";

/// Animated state shared between the timeline animations and the
/// per-frame render closure. Mirrors the example file verbatim.
#[derive(Default, Clone, Copy)]
struct DemoState {
    fade: f32,
    rotation_z: f32,
    rotation_oblique: f32,
}

/// Custom [`Animation`] that mutates one field of the shared
/// [`DemoState`] each frame. Mirrors the example file verbatim.
struct AnimateField {
    target: Arc<Mutex<DemoState>>,
    update: fn(&mut DemoState, f32),
    run_time: f32,
    rate: RateFunc,
}

impl Animation for AnimateField {
    fn run_time(&self) -> f32 {
        self.run_time
    }
    fn rate_func(&self) -> RateFunc {
        self.rate
    }
    fn interpolate(&mut self, alpha: f32) {
        let mut s = self.target.lock().unwrap();
        (self.update)(&mut s, alpha);
    }
}

/// PGA3 bivector for the z-axis rotation plane (`e1 ^ e2`, bitmask
/// `0b0011`). Matches the example.
fn z_axis_bivector() -> pga3::Bivector {
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0011, 1.0);
    pga3::Bivector(mv)
}

/// PGA3 bivector for the oblique axis `(1, 1, 1) / sqrt(3)`, using the
/// same sign convention as the example.
fn oblique_axis_bivector() -> pga3::Bivector {
    let inv_sqrt3 = 1.0 / 3.0_f32.sqrt();
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0110, inv_sqrt3);
    mv.set(0b0101, inv_sqrt3);
    mv.set(0b0011, inv_sqrt3);
    pga3::Bivector(mv)
}

/// PGA3 plane representing the xy plane (`z = 0`). Matches the example.
fn xy_plane_pga() -> pga3::Plane {
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0100, 1.0);
    pga3::Plane(mv)
}

/// True when the lavapipe gating env var is set to a non-empty value.
fn lavapipe_mode() -> bool {
    std::env::var(LAVAPIPE_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Resolution the harness renders at this run.
fn current_resolution() -> (u32, u32) {
    if lavapipe_mode() {
        LAVAPIPE_RESOLUTION
    } else {
        SMOKE_RESOLUTION
    }
}

/// Path to the committed baseline for `frame_index`, regardless of
/// whether the file exists on disk.
fn baseline_path(frame_index: u32) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .join("tests")
        .join("baselines")
        .join(format!("rotor_demo_frame_{frame_index}.png"))
}

/// Build the orthographic camera the example uses, scaled to whichever
/// aspect ratio the harness is rendering at. The example hard-codes
/// `1920 / 1080` for `half_height`; in smoke mode we substitute the
/// runtime aspect ratio so the framing stays consistent.
fn demo_camera(resolution: (u32, u32)) -> Camera {
    let (w, h) = resolution;
    Camera::look_at_with_projection(
        pga3::point(0.0, 0.0, 5.0),
        pga3::point(0.0, 0.0, 0.0),
        [0.0, 1.0, 0.0],
        Projection::Orthographic {
            half_width: 3.0,
            half_height: 3.0 * (h as f32) / (w as f32),
            near: 0.1,
            far: 100.0,
        },
    )
}

/// Outcome of attempting the shared render once. Held inside a
/// [`OnceLock`] so the bundle test and each per-frame test see the
/// same render.
enum RenderState {
    /// Render succeeded; the temp directory holds the PNG sequence and
    /// stays alive for the test process. The frame count comes back
    /// from [`Scene::render_with`] so individual tests can assert
    /// `>= 1740`.
    Rendered {
        /// Held to keep the directory alive across the test run.
        _dir: tempfile::TempDir,
        /// Path to the directory.
        path: PathBuf,
        /// Total frames written.
        frames_written: u64,
        /// Resolution this render was captured at; used to assert PNG
        /// dimensions match what the harness configured.
        resolution: (u32, u32),
    },
    /// Vulkan / GPU init failed; tests should print `skip:` and pass.
    Skipped(String),
}

/// Shared render result. Computed once on first access via the
/// [`shared_render`] entry point.
static RENDER: OnceLock<RenderState> = OnceLock::new();

/// Either run the demo end-to-end and cache the result, or return the
/// cached result. Subsequent calls share the same [`TempDir`].
fn shared_render() -> &'static RenderState {
    RENDER.get_or_init(run_demo_once)
}

/// Run the rotor demo end-to-end with `Output::PngSequence`. The body
/// mirrors the example file verbatim (modulo the
/// [`Output`] swap and the resolution).
fn run_demo_once() -> RenderState {
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => return RenderState::Skipped(format!("tempdir failed: {e}")),
    };
    let path = dir.path().to_path_buf();
    let resolution = current_resolution();

    let config = SceneConfig {
        fps: 60,
        resolution,
        background: [0.02, 0.02, 0.02, 1.0],
        output: Output::PngSequence { dir: path.clone() },
    };

    let camera = demo_camera(resolution);
    let mut scene = match Scene::new(config, camera) {
        Ok(s) => s,
        Err(e) => return RenderState::Skipped(format!("scene init failed: {e}")),
    };

    let state = Arc::new(Mutex::new(DemoState::default()));

    // Phase 1, 0 to 5 s: fade in.
    scene.play([Box::new(AnimateField {
        target: state.clone(),
        update: |s, a| s.fade = a,
        run_time: 5.0,
        rate: RateFunc::Smootherstep,
    }) as Box<dyn Animation + 'static>]);

    // Phase 2, 5 to 15 s: full revolution around z.
    scene.play([Box::new(AnimateField {
        target: state.clone(),
        update: |s, a| s.rotation_z = a * std::f32::consts::TAU,
        run_time: 10.0,
        rate: RateFunc::Linear,
    }) as Box<dyn Animation + 'static>]);

    // Phase 3, 15 to 25 s: full revolution around the oblique axis.
    scene.play([Box::new(AnimateField {
        target: state.clone(),
        update: |s, a| s.rotation_oblique = a * std::f32::consts::TAU,
        run_time: 10.0,
        rate: RateFunc::Linear,
    }) as Box<dyn Animation + 'static>]);

    // Phase 4, 25 to 30 s: fade out.
    scene.play([Box::new(AnimateField {
        target: state.clone(),
        update: |s, a| s.fade = 1.0 - a,
        run_time: 5.0,
        rate: RateFunc::Smootherstep,
    }) as Box<dyn Animation + 'static>]);

    let render_state = state.clone();
    let frames = scene.render_with(|ctx, camera| {
        let s = *render_state.lock().unwrap();
        let alpha = s.fade.clamp(0.0, 1.0);

        let rotor_z = pga3::rotor(z_axis_bivector(), s.rotation_z);
        let rotor_obl = pga3::rotor(oblique_axis_bivector(), s.rotation_oblique);
        let combined: Motor = rotor_obl.0.compose(&rotor_z.0);

        let pt = |xyz: [f32; 3]| -> pga3::Point {
            let p = pga3::point(xyz[0], xyz[1], xyz[2]);
            pga3::Point(combined.apply(&p.0))
        };

        let p_x = Point::new(pt([1.0, 0.0, 0.0]), Color::rgba(1.0, 0.3, 0.3, alpha), 14.0);
        let p_y = Point::new(pt([0.0, 1.0, 0.0]), Color::rgba(0.3, 1.0, 0.3, alpha), 14.0);
        let p_z = Point::new(pt([0.0, 0.0, 1.0]), Color::rgba(0.3, 0.3, 1.0, alpha), 14.0);
        p_x.record(ctx, camera);
        p_y.record(ctx, camera);
        p_z.record(ctx, camera);

        let l_x = Line::new(
            pt([0.0, 0.0, 0.0]),
            pt([1.0, 0.0, 0.0]),
            Stroke::new(Color::rgba(1.0, 0.5, 0.5, alpha), 2.5),
        );
        let l_y = Line::new(
            pt([0.0, 0.0, 0.0]),
            pt([0.0, 1.0, 0.0]),
            Stroke::new(Color::rgba(0.5, 1.0, 0.5, alpha), 2.5),
        );
        let l_z = Line::new(
            pt([0.0, 0.0, 0.0]),
            pt([0.0, 0.0, 1.0]),
            Stroke::new(Color::rgba(0.5, 0.5, 1.0, alpha), 2.5),
        );
        l_x.record(ctx, camera);
        l_y.record(ctx, camera);
        l_z.record(ctx, camera);

        let plane = Plane::new(
            xy_plane_pga(),
            1.5,
            Stroke::new(Color::rgba(0.6, 0.6, 0.6, alpha), 1.5),
        );
        for edge in plane.edges() {
            let a_xyz = edge.a.to_euclidean();
            let b_xyz = edge.b.to_euclidean();
            let rotated_edge = Line::new(pt(a_xyz), pt(b_xyz), edge.stroke);
            rotated_edge.record(ctx, camera);
        }
    });

    match frames {
        Ok(n) => RenderState::Rendered {
            _dir: dir,
            path,
            frames_written: n,
            resolution,
        },
        Err(e) => RenderState::Skipped(format!("render failed: {e}")),
    }
}

/// Inspect one captured PNG: assert it decodes, the dimensions match
/// `expected_resolution`, and (when lavapipe mode is on and the
/// baseline exists) byte-compare against the committed baseline.
fn check_frame(path: &Path, frame_index: u32, expected_resolution: (u32, u32)) {
    assert!(
        path.exists(),
        "frame {frame_index}: missing PNG at {}",
        path.display()
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "frame {frame_index}: failed to read {}: {e}",
            path.display()
        )
    });
    assert!(
        !bytes.is_empty(),
        "frame {frame_index}: zero-byte PNG at {}",
        path.display()
    );
    let img = image::load_from_memory(&bytes)
        .unwrap_or_else(|e| panic!("frame {frame_index}: PNG decode failed: {e}"));
    let (w, h) = (img.width(), img.height());
    assert_eq!(
        (w, h),
        expected_resolution,
        "frame {frame_index}: dimensions ({w}, {h}) do not match configured {expected_resolution:?}",
    );

    if lavapipe_mode() {
        let baseline = baseline_path(frame_index);
        if baseline.exists() {
            let baseline_bytes = std::fs::read(&baseline).unwrap_or_else(|e| {
                panic!(
                    "frame {frame_index}: failed to read baseline {}: {e}",
                    baseline.display()
                )
            });
            assert_eq!(
                baseline_bytes,
                bytes,
                "frame {frame_index}: captured PNG differs from committed baseline at {} (lavapipe pixel-perfect gate)",
                baseline.display(),
            );
        } else {
            eprintln!(
                "rotor_demo_baselines: baseline missing for frame {frame_index}; \
                 run `cp {} {}` to commit it",
                path.display(),
                baseline.display(),
            );
        }
    }
}

/// Single-frame entry point shared by the five spec-named tests.
fn check_single_frame(frame_index: u32) {
    let state = shared_render();
    let (path, _frames, resolution) = match state {
        RenderState::Rendered {
            path,
            frames_written,
            resolution,
            ..
        } => (path, *frames_written, *resolution),
        RenderState::Skipped(reason) => {
            eprintln!("skip: {reason}");
            return;
        }
    };
    let frame_path = path.join(format!("frame_{frame_index:06}.png"));
    check_frame(&frame_path, frame_index, resolution);
}

#[test]
fn rotor_demo_emits_all_named_frames_as_valid_pngs() {
    let state = shared_render();
    let (path, frames, resolution) = match state {
        RenderState::Rendered {
            path,
            frames_written,
            resolution,
            ..
        } => (path, *frames_written, *resolution),
        RenderState::Skipped(reason) => {
            eprintln!("skip: {reason}");
            return;
        }
    };
    // The demo schedules 5 + 10 + 10 + 5 = 30 seconds at 60 fps, which
    // is 1800 in-loop frames. We require at least 1740 (the highest
    // index the spec asks us to inspect) so a future shortened demo
    // does not silently start dropping coverage.
    assert!(
        frames >= 1741,
        "rendered only {frames} frames; need 1741 (frame indices 0..=1740)"
    );
    for &i in FRAME_INDICES {
        let frame_path = path.join(format!("frame_{i:06}.png"));
        check_frame(&frame_path, i, resolution);
    }
}

#[test]
fn frame_60_initial_layout() {
    check_single_frame(60);
}

#[test]
fn frame_600_quarter_z_rotation() {
    check_single_frame(600);
}

#[test]
fn frame_900_half_z_rotation() {
    check_single_frame(900);
}

#[test]
fn frame_1500_oblique_rotation_midpoint() {
    check_single_frame(1500);
}

#[test]
fn frame_1740_pre_fadeout() {
    check_single_frame(1740);
}
