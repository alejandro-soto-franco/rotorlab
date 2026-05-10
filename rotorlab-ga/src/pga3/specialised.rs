//! Specialised hot-path methods on the named-field PGA3 [`shapes::Motor`].
//!
//! Stage 9 wires the named-field shape structs in [`super::shapes`] to
//! the dense [`Multivector<Pga3>`] sandwich / compose / SLERP routines
//! through a small set of ergonomic shape-to-shape methods on
//! [`shapes::Motor`]. Every method takes shape-struct inputs, pushes
//! them through the dense oracle, and reads the expected blades back
//! out without going via [`super::bridge::BridgeError`].
//!
//! # Why bypass [`TryFrom`]?
//!
//! The strict bridge in [`super::bridge`] rejects a multivector when any
//! blade outside the target shape's support carries a coefficient larger
//! than a scaled `f32::EPSILON`. The motor sandwich on a unit motor
//! preserves grade in exact arithmetic, but in `f32` the result of
//! `M * X * ~M` can carry sub-tolerance dust on out-of-shape blades.
//! That dust is round-off, not signal; the hot-path extraction reads the
//! support blades directly and silently drops everything else. The
//! strict [`TryFrom`] remains the validated round-trip surface (see the
//! Stage 8 round-trip tests).
//!
//! # Why "specialised" today is a delegated wrapper
//!
//! The Stage 9 deliverable is the shape-to-shape API itself: callers
//! gain a type-safe [`shapes::Motor::apply_to_point`],
//! [`shapes::Motor::apply_to_line`], etc., without having to round-trip
//! through [`Multivector<Pga3>`] manually. The bodies below delegate to
//! the dense oracle, which is correct by construction (the dense
//! [`crate::motor::Motor<Pga3>`] is the ground truth for every operation
//! in this crate). Performance equals the dense path; the win is API
//! ergonomics.
//!
//! A future optimisation pass can replace each delegated body with a
//! hand-derived closed-form expression that operates directly on the
//! shape struct's named fields, validated by the parity tests in
//! `tests/parity_pga3_specialised.rs`. The parity tests catch any drift
//! from dense-path semantics, so the swap is safe to perform method by
//! method.
//!
//! [`Multivector<Pga3>`]: crate::multivector::Multivector

use super::algebra::Pga3;
use super::shapes;
use crate::motor::Motor as DenseMotor;
use crate::multivector::Multivector;

// ---------------------------------------------------------------------
// Private extraction helpers
// ---------------------------------------------------------------------
//
// These read the support blades for each shape directly from a dense
// [`Multivector<Pga3>`] without consulting the bridge tolerance. They
// are the unchecked counterpart to [`super::bridge`]'s strict
// [`TryFrom`] impls and exist solely to absorb the sub-tolerance round-
// off that motor sandwiches deposit on out-of-shape blades. The
// bitmasks mirror the canonical assignments in [`super::shapes`]; if
// those ever change, both modules must be updated together.

fn extract_point(mv: &Multivector<Pga3>) -> shapes::Point {
    shapes::Point {
        e_023: mv.get(0b1110),
        e_013: mv.get(0b1101),
        e_012: mv.get(0b1011),
        e_123: mv.get(0b0111),
    }
}

fn extract_line(mv: &Multivector<Pga3>) -> shapes::Line {
    shapes::Line {
        e_01: mv.get(0b1001),
        e_02: mv.get(0b1010),
        e_03: mv.get(0b1100),
        e_12: mv.get(0b0011),
        e_13: mv.get(0b0101),
        e_23: mv.get(0b0110),
    }
}

fn extract_plane(mv: &Multivector<Pga3>) -> shapes::Plane {
    shapes::Plane {
        e_0: mv.get(0b1000),
        e_1: mv.get(0b0001),
        e_2: mv.get(0b0010),
        e_3: mv.get(0b0100),
    }
}

fn extract_motor(mv: &Multivector<Pga3>) -> shapes::Motor {
    shapes::Motor {
        s: mv.get(0b0000),
        e_12: mv.get(0b0011),
        e_13: mv.get(0b0101),
        e_23: mv.get(0b0110),
        e_01: mv.get(0b1001),
        e_02: mv.get(0b1010),
        e_03: mv.get(0b1100),
        e_0123: mv.get(0b1111),
    }
}

// ---------------------------------------------------------------------
// shapes::Motor specialised methods
// ---------------------------------------------------------------------

impl shapes::Motor {
    /// Apply this motor to a point via the sandwich product
    /// `M * P * ~M`, returning the transformed point.
    ///
    /// Convenience wrapper around the dense
    /// [`crate::motor::Motor::apply`] that takes and returns the
    /// named-field [`shapes::Point`] surface. Out-of-shape round-off on
    /// the dense result is dropped by reading the four point blades
    /// directly; the operation is mathematically grade-preserving on a
    /// unit motor.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga3_specialised.rs`.
    pub fn apply_to_point(&self, p: &shapes::Point) -> shapes::Point {
        let m_dense: Multivector<Pga3> = (*self).into();
        let p_dense: Multivector<Pga3> = (*p).into();
        let result = DenseMotor(m_dense).apply(&p_dense);
        extract_point(&result)
    }

    /// Apply this motor to a line via the sandwich product
    /// `M * L * ~M`, returning the transformed line.
    ///
    /// Convenience wrapper around the dense
    /// [`crate::motor::Motor::apply`] that takes and returns the
    /// named-field [`shapes::Line`] surface. Out-of-shape round-off on
    /// the dense result is dropped by reading the six grade-2 blades
    /// directly.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga3_specialised.rs`.
    pub fn apply_to_line(&self, l: &shapes::Line) -> shapes::Line {
        let m_dense: Multivector<Pga3> = (*self).into();
        let l_dense: Multivector<Pga3> = (*l).into();
        let result = DenseMotor(m_dense).apply(&l_dense);
        extract_line(&result)
    }

    /// Apply this motor to a plane via the sandwich product
    /// `M * Pi * ~M`, returning the transformed plane.
    ///
    /// Convenience wrapper around the dense
    /// [`crate::motor::Motor::apply`] that takes and returns the
    /// named-field [`shapes::Plane`] surface. Out-of-shape round-off on
    /// the dense result is dropped by reading the four grade-1 blades
    /// directly.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga3_specialised.rs`.
    pub fn apply_to_plane(&self, pl: &shapes::Plane) -> shapes::Plane {
        let m_dense: Multivector<Pga3> = (*self).into();
        let pl_dense: Multivector<Pga3> = (*pl).into();
        let result = DenseMotor(m_dense).apply(&pl_dense);
        extract_plane(&result)
    }

    /// Compose two motors: `(self . compose(other))(x) == self(other(x))`
    /// for any shape `x`.
    ///
    /// Realised as the geometric product `self * other` through the
    /// dense oracle; on a pair of even-graded motors the result is again
    /// even-graded, so the eight motor blades capture the full output
    /// and the odd-graded slots stay at round-off-only magnitudes.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga3_specialised.rs`.
    pub fn compose(&self, other: &shapes::Motor) -> shapes::Motor {
        let self_dense: Multivector<Pga3> = (*self).into();
        let other_dense: Multivector<Pga3> = (*other).into();
        let result = DenseMotor(self_dense).compose(&DenseMotor(other_dense));
        extract_motor(&result.0)
    }

    /// Interpolate between this motor and `target` along the SLERP
    /// geodesic, parameterised by `alpha` in `[0, 1]`.
    ///
    /// At `alpha == 0` the result is `self`; at `alpha == 1` the result
    /// is `target`. The implementation matches Stage 3's rotor SLERP
    /// (see [`crate::motor::Motor::interpolate`]); pure-rotor inputs
    /// produce pure-rotor outputs, and motors carrying translator
    /// content yield a rotor-only approximation until full screw-motor
    /// SLERP lands in a later release.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga3_specialised.rs`.
    pub fn interpolate(&self, target: &shapes::Motor, alpha: f32) -> shapes::Motor {
        let self_dense: Multivector<Pga3> = (*self).into();
        let target_dense: Multivector<Pga3> = (*target).into();
        let result = DenseMotor(self_dense).interpolate(&DenseMotor(target_dense), alpha);
        extract_motor(&result.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity motor as a [`shapes::Motor`] (scalar 1, every other
    /// blade 0).
    fn identity() -> shapes::Motor {
        shapes::Motor {
            s: 1.0,
            e_12: 0.0,
            e_13: 0.0,
            e_23: 0.0,
            e_01: 0.0,
            e_02: 0.0,
            e_03: 0.0,
            e_0123: 0.0,
        }
    }

    #[test]
    fn identity_apply_to_point_is_identity() {
        let m = identity();
        let p = shapes::Point {
            e_023: 1.5,
            e_013: -2.25,
            e_012: 3.75,
            e_123: 1.0,
        };
        let q = m.apply_to_point(&p);
        assert!((q.e_023 - p.e_023).abs() < 1e-6);
        assert!((q.e_013 - p.e_013).abs() < 1e-6);
        assert!((q.e_012 - p.e_012).abs() < 1e-6);
        assert!((q.e_123 - p.e_123).abs() < 1e-6);
    }

    #[test]
    fn identity_compose_is_identity() {
        let i = identity();
        let m = shapes::Motor {
            s: core::f32::consts::FRAC_1_SQRT_2,
            e_12: core::f32::consts::FRAC_1_SQRT_2,
            e_13: 0.0,
            e_23: 0.0,
            e_01: 0.0,
            e_02: 0.0,
            e_03: 0.0,
            e_0123: 0.0,
        };
        let composed = i.compose(&m);
        assert!((composed.s - m.s).abs() < 1e-6);
        assert!((composed.e_12 - m.e_12).abs() < 1e-6);
    }

    #[test]
    fn interpolate_endpoints_match_inputs() {
        let a = identity();
        let b = shapes::Motor {
            s: (core::f32::consts::FRAC_PI_4).cos(),
            e_12: (core::f32::consts::FRAC_PI_4).sin(),
            e_13: 0.0,
            e_23: 0.0,
            e_01: 0.0,
            e_02: 0.0,
            e_03: 0.0,
            e_0123: 0.0,
        };
        let at_zero = a.interpolate(&b, 0.0);
        let at_one = a.interpolate(&b, 1.0);
        // alpha == 0 returns self verbatim; alpha == 1 returns target
        // up to SLERP round-off.
        assert!((at_zero.s - a.s).abs() < 1e-5);
        assert!((at_one.s - b.s).abs() < 1e-5);
        assert!((at_one.e_12 - b.e_12).abs() < 1e-5);
    }
}
