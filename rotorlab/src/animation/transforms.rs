//! Stock transform animations: [`Translate`], [`Rotate`], [`Transform`].
//!
//! Plan 3 Task 11 ships these three animations alongside the four
//! [capability traits](crate::object::capability) they consume.
//!
//! - [`Translate`] additively translates a [`Translatable`] target by a
//!   total `delta` over its run time. Each frame applies the
//!   incremental delta `(alpha - last_alpha) * delta`, so the running
//!   total tracks the eased alpha curve exactly without caching the
//!   starting position.
//! - [`Rotate`] applies the same incremental pattern around a PGA3
//!   axis line by a total angle.
//! - [`Transform`] runs a full motor SLERP from a `from` motor to a
//!   `to` motor and overwrites the target's motor each frame; because
//!   the operation is idempotent, no per-frame delta tracking is
//!   needed.
//!
//! All three default to the [`Smootherstep`](RateFunc::Smootherstep)
//! easing curve and expose a `with_rate` builder for callers who want
//! a different curve.

use crate::animation::{Animation, RateFunc};
use crate::object::capability::{MotorBacked, Rotatable, Translatable};
use rotorlab_ga::motor::Motor;
use rotorlab_ga::pga3::Line as PgaLine;
use rotorlab_ga::pga3::Pga3;

/// Animate a [`Translatable`] target through a total displacement
/// `delta` over `run_time` seconds.
///
/// The animation tracks cumulative alpha and emits the per-frame
/// incremental delta `(alpha - last_alpha) * delta`. Because
/// [`Translatable::translate_by`] is additive, the final position
/// after `interpolate(1.0)` is the starting position plus the full
/// `delta`, regardless of how many intermediate frames the timeline
/// dispatches.
pub struct Translate<'a> {
    /// Target drawable. Borrowed mutably for the lifetime of the
    /// animation so the timeline can drive it in place. The `Send`
    /// bound is forwarded from [`Animation`].
    obj: &'a mut (dyn Translatable + Send),
    /// Total world-space displacement to apply between `alpha = 0` and
    /// `alpha = 1`.
    delta: [f32; 3],
    /// Wall-clock duration in seconds.
    run_time: f32,
    /// Easing curve applied by the timeline before each `interpolate`
    /// call.
    rate: RateFunc,
    /// Cumulative alpha already applied to the target. Used to compute
    /// the per-frame incremental delta.
    last_alpha: f32,
}

impl<'a> Translate<'a> {
    /// Construct a `Translate` animation with the default
    /// [`Smootherstep`](RateFunc::Smootherstep) easing curve.
    pub fn new(obj: &'a mut (dyn Translatable + Send), delta: [f32; 3], run_time: f32) -> Self {
        Self {
            obj,
            delta,
            run_time,
            rate: RateFunc::Smootherstep,
            last_alpha: 0.0,
        }
    }

    /// Override the rate function. Returns `self` for builder-style
    /// chaining.
    pub fn with_rate(mut self, rate: RateFunc) -> Self {
        self.rate = rate;
        self
    }
}

impl Animation for Translate<'_> {
    fn run_time(&self) -> f32 {
        self.run_time
    }

    fn rate_func(&self) -> RateFunc {
        self.rate
    }

    fn interpolate(&mut self, alpha: f32) {
        let step = alpha - self.last_alpha;
        self.obj.translate_by([
            self.delta[0] * step,
            self.delta[1] * step,
            self.delta[2] * step,
        ]);
        self.last_alpha = alpha;
    }
}

/// Animate a [`Rotatable`] target through a total rotation of `angle`
/// radians around a PGA3 axis line over `run_time` seconds.
///
/// Same incremental pattern as [`Translate`]: per frame, rotate by
/// `(alpha - last_alpha) * angle`. Because rotor sandwich application
/// composes correctly, the final orientation after `interpolate(1.0)`
/// matches a single rotation by the full `angle`.
pub struct Rotate<'a> {
    /// Target drawable. Borrowed mutably for the lifetime of the
    /// animation so the timeline can drive it in place. The `Send`
    /// bound is forwarded from [`Animation`].
    obj: &'a mut (dyn Rotatable + Send),
    /// PGA3 axis line. Stored once at construction so successive
    /// per-frame rotations share the same numerical axis.
    axis: PgaLine,
    /// Total rotation angle in radians between `alpha = 0` and
    /// `alpha = 1`.
    angle: f32,
    /// Wall-clock duration in seconds.
    run_time: f32,
    /// Easing curve.
    rate: RateFunc,
    /// Cumulative alpha already applied.
    last_alpha: f32,
}

impl<'a> Rotate<'a> {
    /// Construct a `Rotate` animation with the default
    /// [`Smootherstep`](RateFunc::Smootherstep) easing curve.
    pub fn new(
        obj: &'a mut (dyn Rotatable + Send),
        axis: PgaLine,
        angle: f32,
        run_time: f32,
    ) -> Self {
        Self {
            obj,
            axis,
            angle,
            run_time,
            rate: RateFunc::Smootherstep,
            last_alpha: 0.0,
        }
    }

    /// Override the rate function. Returns `self` for builder-style
    /// chaining.
    pub fn with_rate(mut self, rate: RateFunc) -> Self {
        self.rate = rate;
        self
    }
}

impl Animation for Rotate<'_> {
    fn run_time(&self) -> f32 {
        self.run_time
    }

    fn rate_func(&self) -> RateFunc {
        self.rate
    }

    fn interpolate(&mut self, alpha: f32) {
        let step_angle = (alpha - self.last_alpha) * self.angle;
        self.obj.rotate_by(self.axis, step_angle);
        self.last_alpha = alpha;
    }
}

/// Animate a [`MotorBacked`] target along the SLERP geodesic from
/// `from` to `to` over `run_time` seconds.
///
/// Each `interpolate` call computes
/// [`Motor::interpolate`](rotorlab_ga::motor::Motor::interpolate) at
/// the current alpha and writes the result into the target via
/// [`MotorBacked::set_motor`]. Because the write is idempotent, no
/// cumulative-alpha tracking is required: the final motor at
/// `alpha = 1.0` equals `to` exactly.
///
/// Plan 3 ships rotor-only SLERP; if `from` or `to` carry translator
/// content, the result is a rotor-only approximation. Translator and
/// screw-motor SLERP land in a future release.
pub struct Transform<'a> {
    /// Target drawable. Borrowed mutably for the lifetime of the
    /// animation so the timeline can drive it in place. The `Send`
    /// bound is forwarded from [`Animation`].
    obj: &'a mut (dyn MotorBacked + Send),
    /// Starting motor.
    from: Motor<Pga3>,
    /// Ending motor.
    to: Motor<Pga3>,
    /// Wall-clock duration in seconds.
    run_time: f32,
    /// Easing curve.
    rate: RateFunc,
}

impl<'a> Transform<'a> {
    /// Construct a `Transform` animation with the default
    /// [`Smootherstep`](RateFunc::Smootherstep) easing curve.
    pub fn new(
        obj: &'a mut (dyn MotorBacked + Send),
        from: Motor<Pga3>,
        to: Motor<Pga3>,
        run_time: f32,
    ) -> Self {
        Self {
            obj,
            from,
            to,
            run_time,
            rate: RateFunc::Smootherstep,
        }
    }

    /// Override the rate function. Returns `self` for builder-style
    /// chaining.
    pub fn with_rate(mut self, rate: RateFunc) -> Self {
        self.rate = rate;
        self
    }
}

impl Animation for Transform<'_> {
    fn run_time(&self) -> f32 {
        self.run_time
    }

    fn rate_func(&self) -> RateFunc {
        self.rate
    }

    fn interpolate(&mut self, alpha: f32) {
        let m = self.from.interpolate(&self.to, alpha);
        self.obj.set_motor(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Point;
    use crate::object::style::Color;
    use rotorlab_ga::pga3;

    #[test]
    fn translate_alpha_one_moves_point_by_full_delta() {
        let mut p = Point::new(pga3::point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        {
            let mut anim = Translate::new(&mut p, [1.0, 0.0, 0.0], 1.0);
            anim.interpolate(1.0);
        }
        let xyz = p.position.to_euclidean();
        assert!((xyz[0] - 1.0).abs() < 1e-4, "x: {}", xyz[0]);
        assert!(xyz[1].abs() < 1e-4, "y: {}", xyz[1]);
        assert!(xyz[2].abs() < 1e-4, "z: {}", xyz[2]);
    }

    #[test]
    fn translate_incremental_steps_compose_to_full_delta() {
        // Drive through 5 alpha steps; final position must equal the
        // single-step result.
        let mut p = Point::new(pga3::point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        {
            let mut anim = Translate::new(&mut p, [2.0, 4.0, -1.0], 1.0);
            for k in 1..=5 {
                anim.interpolate((k as f32) / 5.0);
            }
        }
        let xyz = p.position.to_euclidean();
        assert!((xyz[0] - 2.0).abs() < 1e-4, "x: {}", xyz[0]);
        assert!((xyz[1] - 4.0).abs() < 1e-4, "y: {}", xyz[1]);
        assert!((xyz[2] + 1.0).abs() < 1e-4, "z: {}", xyz[2]);
    }

    #[test]
    fn translate_default_rate_is_smootherstep() {
        let mut p = Point::new(pga3::point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        let anim = Translate::new(&mut p, [1.0, 0.0, 0.0], 1.0);
        assert!(matches!(anim.rate_func(), RateFunc::Smootherstep));
    }

    #[test]
    fn translate_run_time_round_trips() {
        let mut p = Point::new(pga3::point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        let anim = Translate::new(&mut p, [1.0, 0.0, 0.0], 2.5);
        assert_eq!(anim.run_time(), 2.5);
    }

    #[test]
    fn rotate_z_axis_pi_flips_x() {
        let mut p = Point::new(pga3::point(1.0, 0.0, 0.0), Color::WHITE, 1.0);
        let z_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(0.0, 0.0, 1.0));
        {
            let mut anim = Rotate::new(&mut p, z_axis, core::f32::consts::PI, 1.0);
            anim.interpolate(1.0);
        }
        let xyz = p.position.to_euclidean();
        assert!((xyz[0] + 1.0).abs() < 1e-3, "x: {}", xyz[0]);
        assert!(xyz[1].abs() < 1e-3, "y: {}", xyz[1]);
        assert!(xyz[2].abs() < 1e-3, "z: {}", xyz[2]);
    }

    #[test]
    fn rotate_incremental_steps_compose_to_full_angle() {
        let mut p = Point::new(pga3::point(1.0, 0.0, 0.0), Color::WHITE, 1.0);
        let z_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(0.0, 0.0, 1.0));
        {
            let mut anim = Rotate::new(&mut p, z_axis, core::f32::consts::PI, 1.0);
            for k in 1..=4 {
                anim.interpolate((k as f32) / 4.0);
            }
        }
        let xyz = p.position.to_euclidean();
        assert!((xyz[0] + 1.0).abs() < 1e-3, "x: {}", xyz[0]);
        assert!(xyz[1].abs() < 1e-3, "y: {}", xyz[1]);
    }

    /// Tiny test wrapper exposing a [`MotorBacked`] target whose
    /// effect on a probe point can be observed.
    struct MotorPoint {
        m: Motor<Pga3>,
    }
    impl MotorBacked for MotorPoint {
        fn motor(&self) -> Motor<Pga3> {
            self.m
        }
        fn set_motor(&mut self, m: Motor<Pga3>) {
            self.m = m;
        }
    }

    #[test]
    fn transform_at_half_alpha_lies_on_geodesic() {
        // Identity to rotor-by-pi-around-z; midpoint should be a
        // quarter rotation, sending (1, 0, 0) to (0, 1, 0).
        let mut mv: rotorlab_ga::Multivector<rotorlab_ga::pga3::Pga3> =
            rotorlab_ga::Multivector::zero();
        mv.set(0b0011, 1.0);
        let z_bivec = pga3::Bivector(mv);
        let from = Motor::identity();
        let to = pga3::rotor(z_bivec, core::f32::consts::PI).0;

        let mut target = MotorPoint {
            m: Motor::identity(),
        };
        {
            let mut anim = Transform::new(&mut target, from, to, 1.0);
            anim.interpolate(0.5);
        }
        let probe = pga3::point(1.0, 0.0, 0.0);
        let rotated = pga3::Point(target.m.apply(&probe.0)).to_euclidean();
        assert!(rotated[0].abs() < 1e-4, "x: {}", rotated[0]);
        assert!((rotated[1] - 1.0).abs() < 1e-4, "y: {}", rotated[1]);
        assert!(rotated[2].abs() < 1e-4, "z: {}", rotated[2]);
    }

    #[test]
    fn transform_at_alpha_one_equals_target_motor() {
        let mut mv: rotorlab_ga::Multivector<rotorlab_ga::pga3::Pga3> =
            rotorlab_ga::Multivector::zero();
        mv.set(0b0011, 1.0);
        let z_bivec = pga3::Bivector(mv);
        let from = Motor::identity();
        let to = pga3::rotor(z_bivec, core::f32::consts::PI).0;

        let mut target = MotorPoint {
            m: Motor::identity(),
        };
        {
            let mut anim = Transform::new(&mut target, from, to, 1.0);
            anim.interpolate(1.0);
        }
        let probe = pga3::point(1.0, 0.0, 0.0);
        let via_target = pga3::Point(target.m.apply(&probe.0)).to_euclidean();
        let via_to = pga3::Point(to.apply(&probe.0)).to_euclidean();
        assert!((via_target[0] - via_to[0]).abs() < 1e-4);
        assert!((via_target[1] - via_to[1]).abs() < 1e-4);
        assert!((via_target[2] - via_to[2]).abs() < 1e-4);
    }
}
