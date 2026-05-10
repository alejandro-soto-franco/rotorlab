//! Frame-count precision tests for the Timeline + Scene::render integration.
//!
//! Per Plan 3 Task 14 spec, required precision is plus-or-minus one frame.
//! These tests pin the integration-level dispatch behavior:
//!
//! 1. At fps 60 with run_time 1.0, the in-loop dispatches plus the
//!    post-loop trailing dispatch produce 61 interpolate calls (60 in-loop
//!    at t in [0, 1) plus one trailing call at t = 1.0). The first alpha
//!    is 0.0, the last alpha is 1.0, and finalize fires exactly once.
//! 2. Two animations with run_times 1.0 and 2.0 each finalize exactly
//!    once and produce per-animation call counts that are consistent with
//!    the per-frame dispatch (61 and 121).
//! 3. wait(0.5) at fps 60 advances exactly 30 frames; wait(1.0) at fps 30
//!    advances exactly 30 frames. Both within plus-or-minus one.
//! 4. A non-Linear rate function produces a non-uniform alpha trace.

use rotorlab::animation::{Animation, RateFunc};
use rotorlab::camera::Camera;
use rotorlab::scene::{Output, Scene, SceneConfig};
use rotorlab_ga::pga3::shapes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Animation that records every alpha its `interpolate` is called with
/// into a shared vector and counts `finalize` invocations atomically.
/// The shared handles let the test read back the trace after
/// `Scene::render` consumes the boxed animation.
struct AlphaTrace {
    calls: Arc<Mutex<Vec<f32>>>,
    finalizes: Arc<AtomicUsize>,
    run_time: f32,
    rate: RateFunc,
}

impl Animation for AlphaTrace {
    fn run_time(&self) -> f32 {
        self.run_time
    }
    fn rate_func(&self) -> RateFunc {
        self.rate
    }
    fn interpolate(&mut self, alpha: f32) {
        self.calls.lock().unwrap().push(alpha);
    }
    fn finalize(&mut self) {
        self.finalizes.fetch_add(1, Ordering::SeqCst);
    }
}

/// Build a small PNG-sequence config so the tests do not depend on
/// FFmpeg and stay cheap. The temp directory is returned alongside so
/// the caller keeps it alive for the duration of the render.
fn make_config(fps: u32) -> (SceneConfig, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = SceneConfig {
        fps,
        resolution: (320, 180),
        background: [0.0, 0.0, 0.0, 1.0],
        output: Output::PngSequence {
            dir: dir.path().to_path_buf(),
        },
    };
    (cfg, dir)
}

/// Default camera reused across the timing tests. Looks down -z at the
/// origin; the actual framing does not matter because the tests inspect
/// alpha traces, not pixels.
fn camera() -> Camera {
    Camera::look_at(
        shapes::Point::new(0.0, 0.0, 5.0),
        shapes::Point::new(0.0, 0.0, 0.0),
        [0.0, 1.0, 0.0],
    )
}

/// At fps 60 with run_time 1.0, dispatch produces 60 in-loop calls plus
/// one post-loop call at alpha 1.0 (61 total), and `finalize` fires
/// exactly once. The plus-or-minus one tolerance allowed by the spec is
/// expressed here as the inclusive range `60..=61`; the actual count on
/// this engine version is 61.
#[test]
fn run_time_one_at_60_fps_emits_about_60_calls_finalize_once() {
    let (config, _dir) = make_config(60);
    let Ok(mut scene) = Scene::new(config, camera()) else {
        eprintln!("skip: vulkan unavailable");
        return;
    };

    let calls = Arc::new(Mutex::new(Vec::new()));
    let finalizes = Arc::new(AtomicUsize::new(0));
    let trace = AlphaTrace {
        calls: calls.clone(),
        finalizes: finalizes.clone(),
        run_time: 1.0,
        rate: RateFunc::Linear,
    };
    scene.play([Box::new(trace) as Box<dyn Animation + 'static>]);

    let _frames = scene.render().expect("render");

    let observed = calls.lock().unwrap();
    let n = observed.len();
    // 60 in-loop dispatches at now = f/60 for f in 0..60, plus one
    // trailing dispatch at now = total_run_time. Allow plus-or-minus one
    // per spec.
    assert!(n == 60 || n == 61, "expected 60..=61 calls, got {n}");
    assert!(
        (observed[0] - 0.0).abs() < 1e-6,
        "first alpha was {}",
        observed[0]
    );
    let last = *observed.last().unwrap();
    assert!((last - 1.0).abs() < 1e-6, "last alpha was {last}");
    assert_eq!(
        finalizes.load(Ordering::SeqCst),
        1,
        "finalize should fire exactly once",
    );
}

/// At fps 60 with run_time 1.0, the final alpha handed to `interpolate`
/// is exactly 1.0 (not 59/60 or 1.0 minus epsilon) and `finalize` runs
/// once. This pins the post-loop trailing dispatch contract that Scene
/// relies on for stock animations whose `finalize` commits state.
#[test]
fn final_alpha_is_exactly_one_and_finalize_runs_once() {
    let (config, _dir) = make_config(60);
    let Ok(mut scene) = Scene::new(config, camera()) else {
        eprintln!("skip: vulkan unavailable");
        return;
    };

    let calls = Arc::new(Mutex::new(Vec::new()));
    let finalizes = Arc::new(AtomicUsize::new(0));
    let trace = AlphaTrace {
        calls: calls.clone(),
        finalizes: finalizes.clone(),
        run_time: 1.0,
        rate: RateFunc::Linear,
    };
    scene.play([Box::new(trace) as Box<dyn Animation + 'static>]);

    let _ = scene.render().expect("render");

    let observed = calls.lock().unwrap();
    assert!(!observed.is_empty(), "expected at least one interpolate");
    let last = *observed.last().unwrap();
    assert!(
        (last - 1.0).abs() < 1e-6,
        "expected last alpha == 1.0, got {last}",
    );
    assert_eq!(finalizes.load(Ordering::SeqCst), 1);
}

/// `play(a, b)` where `a.run_time = 1.0` and `b.run_time = 2.0` makes
/// `a` finish near frame 60 and `b` near frame 120 with no drift. Each
/// animation's `finalize` fires exactly once and the per-animation
/// interpolate counts are consistent with the per-frame dispatch
/// (61 and 121 on this engine, with plus-or-minus one tolerance).
#[test]
fn two_animations_with_different_run_times_finalize_at_their_own_frame_indices() {
    let (config, _dir) = make_config(60);
    let Ok(mut scene) = Scene::new(config, camera()) else {
        eprintln!("skip: vulkan unavailable");
        return;
    };

    let calls_a = Arc::new(Mutex::new(Vec::new()));
    let finalizes_a = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::new(Mutex::new(Vec::new()));
    let finalizes_b = Arc::new(AtomicUsize::new(0));

    let a = AlphaTrace {
        calls: calls_a.clone(),
        finalizes: finalizes_a.clone(),
        run_time: 1.0,
        rate: RateFunc::Linear,
    };
    let b = AlphaTrace {
        calls: calls_b.clone(),
        finalizes: finalizes_b.clone(),
        run_time: 2.0,
        rate: RateFunc::Linear,
    };
    scene.play([
        Box::new(a) as Box<dyn Animation + 'static>,
        Box::new(b) as Box<dyn Animation + 'static>,
    ]);

    let frames = scene.render().expect("render");
    // Total run time is max(1.0, 2.0) = 2.0 seconds at 60 fps -> 120
    // frames written. Allow plus-or-minus one.
    assert!(
        (119..=121).contains(&frames),
        "expected ~120 PNG frames, got {frames}",
    );

    let na = calls_a.lock().unwrap().len();
    let nb = calls_b.lock().unwrap().len();
    // a: animation finishes at t = 1.0. The in-loop dispatch at frame
    // index 60 (now = 60/60 = 1.0) will already cross the alpha-one
    // boundary and fire a's finalize. Subsequent frames skip a. Then
    // the post-loop trailing dispatch at total_run_time = 2.0 sees a as
    // Done and skips. So a's count is the number of in-loop frames that
    // dispatched it, which is 61 (frames 0..=60).
    assert!(na == 60 || na == 61, "a calls: {na}");
    // b: animation finishes only at the post-loop dispatch at t = 2.0.
    // 120 in-loop dispatches plus 1 post-loop dispatch = 121.
    assert!(nb == 120 || nb == 121, "b calls: {nb}");

    assert_eq!(finalizes_a.load(Ordering::SeqCst), 1, "a finalize count");
    assert_eq!(finalizes_b.load(Ordering::SeqCst), 1, "b finalize count");

    // a's finalize fires the first time it sees alpha clamped to 1.0,
    // which happens at the in-loop frame whose now reaches a's
    // start_time + run_time = 0.0 + 1.0 = 1.0 (frame index 60). The
    // last alpha recorded for a must therefore be exactly 1.0.
    let last_a = *calls_a.lock().unwrap().last().unwrap();
    assert!(
        (last_a - 1.0).abs() < 1e-6,
        "a last alpha should be 1.0, got {last_a}",
    );
    let last_b = *calls_b.lock().unwrap().last().unwrap();
    assert!(
        (last_b - 1.0).abs() < 1e-6,
        "b last alpha should be 1.0, got {last_b}",
    );
}

/// `wait(0.5)` at fps 60 advances exactly 30 frames in the output
/// stream. Tolerance plus-or-minus one per spec.
#[test]
fn wait_half_second_at_60_fps_advances_30_frames() {
    let (config, _dir) = make_config(60);
    let Ok(mut scene) = Scene::new(config, camera()) else {
        eprintln!("skip: vulkan unavailable");
        return;
    };

    scene.wait(0.5);
    let frames = scene.render().expect("render");
    assert!(
        (29..=31).contains(&frames),
        "expected ~30 frames after wait(0.5) at 60 fps, got {frames}",
    );
}

/// `wait(1.0)` at fps 30 advances exactly 30 frames. Same plus-or-minus
/// one tolerance, exercised at a different fps to catch any
/// fps-coupling regression in `Scene::render_with`.
#[test]
fn wait_one_second_at_30_fps_advances_30_frames() {
    let (config, _dir) = make_config(30);
    let Ok(mut scene) = Scene::new(config, camera()) else {
        eprintln!("skip: vulkan unavailable");
        return;
    };

    scene.wait(1.0);
    let frames = scene.render().expect("render");
    assert!(
        (29..=31).contains(&frames),
        "expected ~30 frames after wait(1.0) at 30 fps, got {frames}",
    );
}

/// Sanity-check that a non-Linear rate function actually shapes the
/// alpha trace. At raw t = 0.25 the quintic smootherstep evaluates to
/// 0.103515625, which is well below 0.25; the in-loop dispatch at frame
/// index 15 (now = 15/60 = 0.25) should reflect that.
#[test]
fn smootherstep_rate_func_produces_non_uniform_alpha_trace() {
    let (config, _dir) = make_config(60);
    let Ok(mut scene) = Scene::new(config, camera()) else {
        eprintln!("skip: vulkan unavailable");
        return;
    };

    let calls = Arc::new(Mutex::new(Vec::new()));
    let finalizes = Arc::new(AtomicUsize::new(0));
    let trace = AlphaTrace {
        calls: calls.clone(),
        finalizes,
        run_time: 1.0,
        rate: RateFunc::Smootherstep,
    };
    scene.play([Box::new(trace) as Box<dyn Animation + 'static>]);
    let _ = scene.render().expect("render");

    let observed = calls.lock().unwrap();
    // Need at least 16 calls to inspect index 15. The expected post-
    // dispatch count is 61 (60 in-loop + 1 trailing); guard anyway.
    assert!(
        observed.len() >= 16,
        "expected at least 16 alpha samples, got {}",
        observed.len(),
    );
    let frame_15 = observed[15];
    assert!(
        frame_15 < 0.25,
        "expected smootherstep(0.25) < 0.25, got {frame_15}",
    );
    // Endpoints still pinned at 0 and 1 even under easing.
    assert!((observed[0] - 0.0).abs() < 1e-6);
    let last = *observed.last().unwrap();
    assert!((last - 1.0).abs() < 1e-6);
}
