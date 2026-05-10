//! 30-second orthographic rotor demo at 1080p60.
//!
//! Storyline:
//!
//! * 0 to 5 s: fade in three points on the unit x, y, z axes, plus
//!   three lines from the origin to each, plus the xy-plane wireframe.
//! * 5 to 15 s: rotate the entire scene by `2 * pi` around the z axis.
//! * 15 to 25 s: rotate the entire scene by `2 * pi` around the
//!   oblique axis `(1, 1, 1) / sqrt(3)`.
//! * 25 to 30 s: fade everything back to black.
//!
//! Run with `cargo run --release --example rotor_demo`. The output
//! `rotor_demo.mp4` lands in the current working directory. The smoke
//! check expects a non-zero mp4 in under ten seconds wall-clock when
//! Vulkan + FFmpeg are present; if either is missing, `Scene::new`
//! surfaces an error and `main` prints it without panicking.
//!
//! # Why custom AnimateField instead of the Task 11 stock set
//!
//! The Plan 3 Task 11 stock animations (`Translate`, `Rotate`,
//! `Transform`, `FadeIn`, `FadeOut`) hold `&mut dyn Translatable`,
//! `&mut dyn Rotatable`, etc. for the lifetime of the [`Scene`]. The
//! [`Timeline`] therefore borrows the drawable, which prevents the
//! per-frame render closure from also reading it: the borrow checker
//! rejects the two simultaneous live references statically.
//!
//! [`AnimateField`] sidesteps this by driving an
//! `Arc<Mutex<DemoState>>` of plain animated scalars (fade, rotation
//! angles). The custom animations hold one clone, the render closure
//! holds another; mutation is via `lock()`. Drawables are rebuilt
//! fresh from the state each frame.
//!
//! Plan 3 Task 11's stock animations remain in the library for use
//! cases where the caller holds a single drawable and animates it
//! once without also reading it from a render closure. A future plan
//! will refactor [`Animation`] to use a callback / handle model that
//! does not require static `&mut` borrows.
//!
//! [`AnimateField`]: AnimateField
//! [`Timeline`]: rotorlab::animation::Timeline

use rotorlab::animation::{Animation, RateFunc};
use rotorlab::camera::{Camera, Projection};
use rotorlab::object::{Color, Drawable, Line, Plane, Point, Stroke};
use rotorlab::scene::{H264Preset, Output, Scene, SceneConfig};
use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga3::{self, Pga3, shapes};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Animated state shared between the timeline-driven custom
/// animations and the per-frame render closure.
#[derive(Default, Clone, Copy)]
struct DemoState {
    /// Opacity multiplier in `[0, 1]`. Drives the fade in / out.
    fade: f32,
    /// Rotation angle around the world z axis in radians.
    rotation_z: f32,
    /// Rotation angle around the oblique axis `(1, 1, 1) / sqrt(3)`
    /// in radians.
    rotation_oblique: f32,
}

/// A custom [`Animation`] that mutates a single field of a shared
/// [`DemoState`].
///
/// `update` is called every frame with the eased alpha; it should
/// write into the state in whatever way the storyline requires
/// (`s.fade = a`, `s.fade = 1.0 - a`, `s.rotation_z = a *
/// std::f32::consts::TAU`, etc.).
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

fn main() {
    let config = SceneConfig {
        fps: 60,
        resolution: (1920, 1080),
        background: [0.02, 0.02, 0.02, 1.0],
        output: Output::Mp4 {
            path: PathBuf::from("rotor_demo.mp4"),
            crf: 18,
            preset: H264Preset::Slow,
        },
    };

    // Orthographic camera at z = 5 looking at the origin, +y up.
    // half_width = 3.0 keeps the unit-axis points comfortably inside
    // the frame; half_height tracks the 16:9 aspect.
    let camera = Camera::look_at_with_projection(
        shapes::Point::new(0.0, 0.0, 5.0),
        shapes::Point::new(0.0, 0.0, 0.0),
        [0.0, 1.0, 0.0],
        Projection::Orthographic {
            half_width: 3.0,
            half_height: 3.0 * 1080.0 / 1920.0,
            near: 0.1,
            far: 100.0,
        },
    );

    let mut scene = match Scene::new(config, camera) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("scene init failed: {e}");
            return;
        }
    };

    let state = Arc::new(Mutex::new(DemoState::default()));

    // Phase 1, 0 to 5 s: fade in.
    scene.play([Box::new(AnimateField {
        target: state.clone(),
        update: |s, a| s.fade = a,
        run_time: 5.0,
        rate: RateFunc::Smootherstep,
    }) as Box<dyn Animation + 'static>]);

    // Phase 2, 5 to 15 s: one full revolution around the z axis.
    scene.play([Box::new(AnimateField {
        target: state.clone(),
        update: |s, a| s.rotation_z = a * std::f32::consts::TAU,
        run_time: 10.0,
        rate: RateFunc::Linear,
    }) as Box<dyn Animation + 'static>]);

    // Phase 3, 15 to 25 s: one full revolution around the oblique
    // axis. The z rotation stays parked at 2*pi (which is
    // visually identical to 0) so the geometry transitions smoothly.
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

    // Per-frame render closure: pull the live state, rebuild the
    // rotated drawables, push them through the FrameContext.
    let render_state = state.clone();
    let frames = scene
        .render_with(|ctx, camera| {
            let s = *render_state.lock().unwrap();
            let alpha = s.fade.clamp(0.0, 1.0);

            // Combined rotor: z first, then oblique. Composition
            // order is `oblique * z` so a point first picks up the z
            // rotation, then the oblique rotation. Each rotor is
            // built on the dense surface (the standard
            // bivector-exponential factory), then converted to the
            // shape representation so we can compose via the
            // specialised [`shapes::Motor::compose`].
            let rotor_z: shapes::Motor = pga3::rotor(z_axis_bivector(), s.rotation_z).into();
            let rotor_obl: shapes::Motor =
                pga3::rotor(oblique_axis_bivector(), s.rotation_oblique).into();
            let combined: shapes::Motor = rotor_obl.compose(&rotor_z);

            let pt = |xyz: [f32; 3]| -> shapes::Point {
                let p = shapes::Point::new(xyz[0], xyz[1], xyz[2]);
                combined.apply_to_point(&p)
            };

            // Three axis points at unit positions, colored by axis.
            // Disk radius is in screen pixels; 14 reads cleanly at
            // 1080p.
            let p_x = Point::new(pt([1.0, 0.0, 0.0]), Color::rgba(1.0, 0.3, 0.3, alpha), 14.0);
            let p_y = Point::new(pt([0.0, 1.0, 0.0]), Color::rgba(0.3, 1.0, 0.3, alpha), 14.0);
            let p_z = Point::new(pt([0.0, 0.0, 1.0]), Color::rgba(0.3, 0.3, 1.0, alpha), 14.0);
            p_x.record(ctx, camera);
            p_y.record(ctx, camera);
            p_z.record(ctx, camera);

            // Three axis lines from the origin to each point. Stroke
            // is a slightly desaturated version of the matching
            // point color.
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

            // xy-plane wireframe quad. We rotate the four corners
            // explicitly via the same `combined` rotor so the
            // wireframe rotates with the rest of the scene. The
            // plane bivector itself is not rotated; only its
            // expanded corners are. Plan 5+ will rotate the plane
            // bivector directly via the motor sandwich and let the
            // wireframe expansion happen in the rotated frame.
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
        })
        .expect("render");

    println!("rendered {frames} frames to rotor_demo.mp4");
}

/// PGA3 bivector representing the z-axis rotation plane (`e1 ^ e2`,
/// bitmask `0b0011`).
fn z_axis_bivector() -> pga3::Bivector {
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0011, 1.0);
    pga3::Bivector(mv)
}

/// PGA3 bivector representing the oblique rotation axis
/// `(1, 1, 1) / sqrt(3)` in the same empirical sign convention used by
/// [`Line::from_pga`](rotorlab::object::Line::from_pga): the axis x, y,
/// z components live on `e23 = 0b0110`, `e13 = 0b0101`, `e12 = 0b0011`
/// respectively.
fn oblique_axis_bivector() -> pga3::Bivector {
    let inv_sqrt3 = 1.0 / 3.0_f32.sqrt();
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0110, inv_sqrt3); // e23 picks up the x-axis component
    mv.set(0b0101, inv_sqrt3); // e13 picks up the y-axis component
    mv.set(0b0011, inv_sqrt3); // e12 picks up the z-axis component
    pga3::Bivector(mv)
}

/// PGA3 plane representing the xy plane (`z = 0`), encoded as the
/// pure `e3` blade.
fn xy_plane_pga() -> shapes::Plane {
    shapes::Plane {
        e_0: 0.0,
        e_1: 0.0,
        e_2: 0.0,
        e_3: 1.0,
    }
}
