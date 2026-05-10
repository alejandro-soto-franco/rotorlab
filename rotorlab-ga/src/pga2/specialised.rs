//! Specialised hot-path methods on the named-field PGA2 [`shapes::Motor`].
//!
//! Stage 11 backports the PGA3 Stage 9 surface to PGA2: the named-field
//! shape structs in [`super::shapes`] gain shape-to-shape methods on
//! [`shapes::Motor`] that take and return shape-struct inputs without
//! forcing callers to round-trip through [`Multivector<Pga2>`] manually.
//! Each method pushes its inputs through the dense oracle and reads the
//! expected blades back out without going via [`super::bridge::BridgeError`].
//!
//! There is no `apply_to_plane`: PGA2 has no Plane shape (the projective
//! dual of a plane in 3D collapses to a line in 2D, already covered by
//! [`shapes::Line`]).
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
//! Stage 11 round-trip tests).
//!
//! # Why "specialised" today is a delegated wrapper
//!
//! Mirrors the PGA3 Stage 9 rationale: the win is API ergonomics
//! (type-safe shape-to-shape calls). The bodies below delegate to the
//! dense oracle, which is correct by construction. A future
//! optimisation pass can replace each delegated body with a
//! hand-derived closed-form expression in the named fields, validated
//! by the parity tests in `tests/parity_pga2_specialised.rs`.
//!
//! [`Multivector<Pga2>`]: crate::multivector::Multivector

use super::algebra::Pga2;
use super::shapes;
use crate::motor::Motor as DenseMotor;
use crate::multivector::Multivector;

// ---------------------------------------------------------------------
// Private extraction helpers
// ---------------------------------------------------------------------
//
// These read the support blades for each shape directly from a dense
// [`Multivector<Pga2>`] without consulting the bridge tolerance. They
// are the unchecked counterpart to [`super::bridge`]'s strict
// [`TryFrom`] impls and exist solely to absorb the sub-tolerance round-
// off that motor sandwiches deposit on out-of-shape blades. The
// bitmasks mirror the canonical assignments in [`super::shapes`]; if
// those ever change, both modules must be updated together.

fn extract_point(mv: &Multivector<Pga2>) -> shapes::Point {
    shapes::Point {
        e_12: mv.get(0b011),
        e_02: mv.get(0b110),
        e_01: mv.get(0b101),
    }
}

fn extract_line(mv: &Multivector<Pga2>) -> shapes::Line {
    shapes::Line {
        e_1: mv.get(0b001),
        e_2: mv.get(0b010),
        e_0: mv.get(0b100),
    }
}

fn extract_motor(mv: &Multivector<Pga2>) -> shapes::Motor {
    shapes::Motor {
        s: mv.get(0b000),
        e_12: mv.get(0b011),
        e_02: mv.get(0b110),
        e_01: mv.get(0b101),
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
    /// the dense result is dropped by reading the three point blades
    /// directly; the operation is mathematically grade-preserving on a
    /// unit motor.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga2_specialised.rs`.
    pub fn apply_to_point(&self, p: &shapes::Point) -> shapes::Point {
        let m_dense: Multivector<Pga2> = (*self).into();
        let p_dense: Multivector<Pga2> = (*p).into();
        let result = DenseMotor(m_dense).apply(&p_dense);
        extract_point(&result)
    }

    /// Apply this motor to a line via the sandwich product
    /// `M * L * ~M`, returning the transformed line.
    ///
    /// Convenience wrapper around the dense
    /// [`crate::motor::Motor::apply`] that takes and returns the
    /// named-field [`shapes::Line`] surface. Out-of-shape round-off on
    /// the dense result is dropped by reading the three grade-1 blades
    /// directly.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga2_specialised.rs`.
    pub fn apply_to_line(&self, l: &shapes::Line) -> shapes::Line {
        let m_dense: Multivector<Pga2> = (*self).into();
        let l_dense: Multivector<Pga2> = (*l).into();
        let result = DenseMotor(m_dense).apply(&l_dense);
        extract_line(&result)
    }

    /// Compose two motors: `(self . compose(other))(x) == self(other(x))`
    /// for any shape `x`.
    ///
    /// Realised as the geometric product `self * other` through the
    /// dense oracle; on a pair of even-graded motors the result is again
    /// even-graded, so the four motor blades capture the full output
    /// and the odd-graded slots stay at round-off-only magnitudes.
    ///
    /// Today this method is a delegated wrapper around the dense oracle;
    /// the gain is API ergonomics (type-safe shape-to-shape calls). A
    /// future optimisation pass may replace the body with a hand-derived
    /// closed-form expression in the named fields, guarded by the parity
    /// tests in `tests/parity_pga2_specialised.rs`.
    pub fn compose(&self, other: &shapes::Motor) -> shapes::Motor {
        let self_dense: Multivector<Pga2> = (*self).into();
        let other_dense: Multivector<Pga2> = (*other).into();
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
    /// tests in `tests/parity_pga2_specialised.rs`.
    pub fn interpolate(&self, target: &shapes::Motor, alpha: f32) -> shapes::Motor {
        let self_dense: Multivector<Pga2> = (*self).into();
        let target_dense: Multivector<Pga2> = (*target).into();
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
            e_02: 0.0,
            e_01: 0.0,
        }
    }

    #[test]
    fn identity_apply_to_point_is_identity() {
        let m = identity();
        let p = shapes::Point {
            e_12: 1.0,
            e_02: 1.5,
            e_01: -2.25,
        };
        let q = m.apply_to_point(&p);
        assert!((q.e_12 - p.e_12).abs() < 1e-6);
        assert!((q.e_02 - p.e_02).abs() < 1e-6);
        assert!((q.e_01 - p.e_01).abs() < 1e-6);
    }

    #[test]
    fn identity_compose_is_identity() {
        let i = identity();
        let m = shapes::Motor {
            s: core::f32::consts::FRAC_1_SQRT_2,
            e_12: core::f32::consts::FRAC_1_SQRT_2,
            e_02: 0.0,
            e_01: 0.0,
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
            e_02: 0.0,
            e_01: 0.0,
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
