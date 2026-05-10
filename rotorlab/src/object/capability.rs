//! Capability traits exposed by drawables for use by stock animations.
//!
//! Plan 3 Task 11 introduces four small capability traits that let the
//! stock animation set ([`Translate`](crate::animation::Translate),
//! [`Rotate`](crate::animation::Rotate),
//! [`Transform`](crate::animation::Transform),
//! [`FadeIn`](crate::animation::FadeIn),
//! [`FadeOut`](crate::animation::FadeOut)) operate on any drawable that
//! opts in, without those animations needing to know the concrete
//! drawable type.
//!
//! - [`Translatable`] additive translation in world space.
//! - [`Rotatable`] sandwich rotation around a PGA3 axis line.
//! - [`MotorBacked`] read/write the entire motor (used by `Transform`).
//! - [`HasOpacity`] set the alpha channel (used by fade animations).
//!
//! Plan 3 implements `Translatable + Rotatable + HasOpacity` for
//! [`Point`](crate::object::Point) and [`Line`](crate::object::Line),
//! and `HasOpacity` for [`Plane`](crate::object::Plane). Translating
//! and rotating planes requires manipulating the grade-1 vector
//! representation under the dual sandwich, which is non-trivial and
//! deferred to a later plan; the rotor demo (Task 12) only needs
//! point and line support.

use rotorlab_ga::motor::Motor;
use rotorlab_ga::pga3::Line as PgaLine;
use rotorlab_ga::pga3::Pga3;

/// A drawable whose world-space position can be additively translated.
pub trait Translatable {
    /// Translate the object by `delta` in world space (additive).
    ///
    /// Successive calls compose: calling `translate_by([1, 0, 0])`
    /// twice on a point at the origin places it at `(2, 0, 0)`. Stock
    /// animations use this property to apply per-frame deltas without
    /// caching the starting state.
    fn translate_by(&mut self, delta: [f32; 3]);
}

/// A drawable that can be rotated around a PGA3 axis line.
pub trait Rotatable {
    /// Rotate the object around the given PGA3 axis line by `angle`
    /// radians.
    ///
    /// `axis` is treated as the rotation plane's bivector; for a
    /// rotation around the world z-axis, pass
    /// `pga3::line_through(point(0,0,0), point(0,0,1))`. Successive
    /// calls compose: stock animations apply per-frame angular deltas
    /// without caching the starting state.
    fn rotate_by(&mut self, axis: PgaLine, angle: f32);
}

/// A drawable whose entire rigid motion is exposed as a single PGA3
/// motor.
pub trait MotorBacked {
    /// Read the current motor.
    fn motor(&self) -> Motor<Pga3>;
    /// Replace the motor outright. Idempotent: calling twice with the
    /// same motor leaves the drawable unchanged.
    fn set_motor(&mut self, m: Motor<Pga3>);
}

/// A drawable with a single alpha channel that fade animations can drive.
pub trait HasOpacity {
    /// Set the alpha channel to `alpha`. Values outside `[0, 1]` are
    /// not clamped at this layer; callers that want clamping should do
    /// it before invoking this method.
    fn set_opacity(&mut self, alpha: f32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::style::{Color, Stroke};
    use crate::object::{Line, Point};
    use rotorlab_ga::pga3;

    #[test]
    fn point_implements_translatable() {
        let mut p = Point::new(pga3::point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        p.translate_by([1.0, 2.0, 3.0]);
        let xyz = p.position.to_euclidean();
        assert!((xyz[0] - 1.0).abs() < 1e-5);
        assert!((xyz[1] - 2.0).abs() < 1e-5);
        assert!((xyz[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn point_implements_rotatable_z_axis() {
        let mut p = Point::new(pga3::point(1.0, 0.0, 0.0), Color::WHITE, 1.0);
        let z_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(0.0, 0.0, 1.0));
        p.rotate_by(z_axis, core::f32::consts::PI);
        let xyz = p.position.to_euclidean();
        assert!((xyz[0] + 1.0).abs() < 1e-3, "x: {}", xyz[0]);
        assert!(xyz[1].abs() < 1e-3, "y: {}", xyz[1]);
        assert!(xyz[2].abs() < 1e-3, "z: {}", xyz[2]);
    }

    #[test]
    fn point_implements_has_opacity() {
        let mut p = Point::new(pga3::point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        p.set_opacity(0.25);
        assert!((p.color.a - 0.25).abs() < 1e-6);
    }

    #[test]
    fn line_implements_translatable() {
        let mut line = Line::new(
            pga3::point(0.0, 0.0, 0.0),
            pga3::point(1.0, 0.0, 0.0),
            Stroke::default_white(),
        );
        line.translate_by([0.0, 5.0, 0.0]);
        let ae = line.a.to_euclidean();
        let be = line.b.to_euclidean();
        assert!((ae[1] - 5.0).abs() < 1e-5);
        assert!((be[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn line_implements_has_opacity() {
        let mut line = Line::new(
            pga3::point(0.0, 0.0, 0.0),
            pga3::point(1.0, 0.0, 0.0),
            Stroke::default_white(),
        );
        line.set_opacity(0.5);
        assert!((line.stroke.color.a - 0.5).abs() < 1e-6);
    }
}
