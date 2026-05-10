//! Bridge layer between [`crate::pga2::shapes`] and the dense
//! [`Multivector<Pga2>`] representation.
//!
//! Each shape struct in [`super::shapes`] stores only the blades it can
//! carry. The dense [`Multivector<Pga2>`] in [`crate::multivector`]
//! remains the reference oracle for every operation. This module wires
//! the two surfaces together with a pair of conversions per shape:
//!
//! - `From<Shape> for Multivector<Pga2>` packs a shape's named fields
//!   into the corresponding blades of a fresh zero multivector. Blades
//!   outside the shape's support stay at zero.
//! - `TryFrom<Multivector<Pga2>> for Shape` reads the shape's blades
//!   and verifies that every blade outside the support is within a
//!   scaled tolerance of zero. On violation it returns
//!   [`BridgeError::OutOfShape`].
//!
//! The tolerance is `f32::EPSILON * 64.0 * scale`, where `scale` is the
//! maximum absolute coefficient across all 8 blades (or `1.0` when the
//! input is exactly zero). This adapts the threshold to the magnitude
//! of the input so that round-tripping through compose / decompose
//! cycles does not spuriously reject results that suffered ULP-level
//! floating-point drift.
//!
//! [`BridgeError`] is re-exported from [`crate::pga3::bridge`] so the
//! engine can carry a single shared error type across PGA2 and PGA3
//! conversion surfaces. The blade indices reported in
//! [`crate::pga3::BridgeError::OutOfShape`] are PGA2 bitmasks (0..8) when
//! produced by this module.
//!
//! [`Multivector<Pga2>`]: crate::multivector::Multivector

use super::algebra::Pga2;
use super::shapes;
use crate::motor::{Motor as DenseMotor, Rotor as DenseRotor, Translator as DenseTranslator};
use crate::multivector::Multivector;

pub use crate::pga3::bridge::BridgeError;

/// Tolerance multiplier on top of `f32::EPSILON * scale` used when
/// classifying a blade as "effectively zero" during shape extraction.
const TOLERANCE_FACTOR: f32 = 64.0;

/// Compute the adaptive tolerance for a multivector: the maximum
/// absolute blade coefficient (or `1.0` when the input is the zero
/// multivector) scaled by [`TOLERANCE_FACTOR`] and `f32::EPSILON`.
fn tolerance_for(mv: &Multivector<Pga2>) -> f32 {
    let mut scale = 0.0_f32;
    for blade in 0..8 {
        let v = mv.get(blade).abs();
        if v > scale {
            scale = v;
        }
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    f32::EPSILON * TOLERANCE_FACTOR * scale
}

/// Verify that every blade in `forbidden` carries a coefficient within
/// `tolerance` of zero, returning the first offending blade as a
/// [`BridgeError::OutOfShape`].
fn check_forbidden(
    mv: &Multivector<Pga2>,
    forbidden: &[usize],
    tolerance: f32,
) -> Result<(), BridgeError> {
    for &blade in forbidden {
        let value = mv.get(blade);
        if value.abs() > tolerance {
            return Err(BridgeError::OutOfShape {
                blade,
                value,
                tolerance,
            });
        }
    }
    Ok(())
}

/// Compute the complement of `support` in the 8 PGA2 blade indices,
/// returned as a fixed-size array for stack-only checking.
const fn complement<const N: usize>(support: &[usize]) -> [usize; N] {
    let mut out = [0usize; N];
    let mut idx = 0;
    let mut blade = 0usize;
    while blade < 8 {
        let mut in_support = false;
        let mut i = 0;
        while i < support.len() {
            if support[i] == blade {
                in_support = true;
                break;
            }
            i += 1;
        }
        if !in_support {
            out[idx] = blade;
            idx += 1;
        }
        blade += 1;
    }
    out
}

// Per-shape support bitmasks. Each `*_SUPPORT` lists the blades the
// shape can carry; each `*_FORBIDDEN` is the complement against the 8
// PGA2 blades. The complement arrays are precomputed at compile time so
// the runtime extractor walks a small static slice.

const POINT_SUPPORT: &[usize] = &[0b011, 0b110, 0b101];
const POINT_FORBIDDEN: [usize; 5] = complement::<5>(POINT_SUPPORT);

const LINE_SUPPORT: &[usize] = &[0b001, 0b010, 0b100];
const LINE_FORBIDDEN: [usize; 5] = complement::<5>(LINE_SUPPORT);

const BIVECTOR_SUPPORT: &[usize] = &[0b011, 0b110, 0b101];
const BIVECTOR_FORBIDDEN: [usize; 5] = complement::<5>(BIVECTOR_SUPPORT);

const ROTOR_SUPPORT: &[usize] = &[0b000, 0b011];
const ROTOR_FORBIDDEN: [usize; 6] = complement::<6>(ROTOR_SUPPORT);

const TRANSLATOR_SUPPORT: &[usize] = &[0b000, 0b110, 0b101];
const TRANSLATOR_FORBIDDEN: [usize; 5] = complement::<5>(TRANSLATOR_SUPPORT);

const MOTOR_SUPPORT: &[usize] = &[0b000, 0b011, 0b110, 0b101];
const MOTOR_FORBIDDEN: [usize; 4] = complement::<4>(MOTOR_SUPPORT);

// ---------------------------------------------------------------------
// Point
// ---------------------------------------------------------------------

impl From<shapes::Point> for Multivector<Pga2> {
    fn from(p: shapes::Point) -> Self {
        let mut mv = Multivector::<Pga2>::zero();
        mv.set(0b011, p.e_12);
        mv.set(0b110, p.e_02);
        mv.set(0b101, p.e_01);
        mv
    }
}

impl TryFrom<Multivector<Pga2>> for shapes::Point {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga2>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &POINT_FORBIDDEN, tol)?;
        Ok(shapes::Point {
            e_12: mv.get(0b011),
            e_02: mv.get(0b110),
            e_01: mv.get(0b101),
        })
    }
}

// ---------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------

impl From<shapes::Line> for Multivector<Pga2> {
    fn from(l: shapes::Line) -> Self {
        let mut mv = Multivector::<Pga2>::zero();
        mv.set(0b001, l.e_1);
        mv.set(0b010, l.e_2);
        mv.set(0b100, l.e_0);
        mv
    }
}

impl TryFrom<Multivector<Pga2>> for shapes::Line {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga2>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &LINE_FORBIDDEN, tol)?;
        Ok(shapes::Line {
            e_1: mv.get(0b001),
            e_2: mv.get(0b010),
            e_0: mv.get(0b100),
        })
    }
}

// ---------------------------------------------------------------------
// Bivector
// ---------------------------------------------------------------------

impl From<shapes::Bivector> for Multivector<Pga2> {
    fn from(b: shapes::Bivector) -> Self {
        let mut mv = Multivector::<Pga2>::zero();
        mv.set(0b011, b.e_12);
        mv.set(0b110, b.e_02);
        mv.set(0b101, b.e_01);
        mv
    }
}

impl TryFrom<Multivector<Pga2>> for shapes::Bivector {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga2>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &BIVECTOR_FORBIDDEN, tol)?;
        Ok(shapes::Bivector {
            e_12: mv.get(0b011),
            e_02: mv.get(0b110),
            e_01: mv.get(0b101),
        })
    }
}

// ---------------------------------------------------------------------
// Rotor
// ---------------------------------------------------------------------

impl From<shapes::Rotor> for Multivector<Pga2> {
    fn from(r: shapes::Rotor) -> Self {
        let mut mv = Multivector::<Pga2>::zero();
        mv.set(0b000, r.s);
        mv.set(0b011, r.e_12);
        mv
    }
}

impl TryFrom<Multivector<Pga2>> for shapes::Rotor {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga2>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &ROTOR_FORBIDDEN, tol)?;
        Ok(shapes::Rotor {
            s: mv.get(0b000),
            e_12: mv.get(0b011),
        })
    }
}

// ---------------------------------------------------------------------
// Translator
// ---------------------------------------------------------------------

impl From<shapes::Translator> for Multivector<Pga2> {
    fn from(t: shapes::Translator) -> Self {
        let mut mv = Multivector::<Pga2>::zero();
        mv.set(0b000, t.s);
        mv.set(0b110, t.e_02);
        mv.set(0b101, t.e_01);
        mv
    }
}

impl TryFrom<Multivector<Pga2>> for shapes::Translator {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga2>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &TRANSLATOR_FORBIDDEN, tol)?;
        Ok(shapes::Translator {
            s: mv.get(0b000),
            e_02: mv.get(0b110),
            e_01: mv.get(0b101),
        })
    }
}

// ---------------------------------------------------------------------
// Motor
// ---------------------------------------------------------------------

impl From<shapes::Motor> for Multivector<Pga2> {
    fn from(m: shapes::Motor) -> Self {
        let mut mv = Multivector::<Pga2>::zero();
        mv.set(0b000, m.s);
        mv.set(0b011, m.e_12);
        mv.set(0b110, m.e_02);
        mv.set(0b101, m.e_01);
        mv
    }
}

impl TryFrom<Multivector<Pga2>> for shapes::Motor {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga2>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &MOTOR_FORBIDDEN, tol)?;
        Ok(shapes::Motor {
            s: mv.get(0b000),
            e_12: mv.get(0b011),
            e_02: mv.get(0b110),
            e_01: mv.get(0b101),
        })
    }
}

// ---------------------------------------------------------------------
// Dense Motor / Rotor / Translator -> shapes::Motor (lossy, unchecked)
// ---------------------------------------------------------------------
//
// These shortcut conversions read the 4 motor blades from the dense
// representation directly without going through the strict
// [`TryFrom`] support check. Sandwich and compose products on a unit
// motor preserve the even-grade subalgebra in exact arithmetic; in
// `f32` they can deposit sub-tolerance dust on out-of-shape blades.
// The engine drops that dust silently when crossing back into the
// shape representation. Use the strict [`TryFrom`] above when the
// caller must surface that dust as an error.

impl From<DenseMotor<Pga2>> for shapes::Motor {
    /// Read the 4 motor blades from a dense [`DenseMotor<Pga2>`] into
    /// a [`shapes::Motor`], silently dropping any sub-tolerance dust on
    /// out-of-shape blades. For the strict round-trip surface use
    /// [`TryFrom<Multivector<Pga2>> for shapes::Motor`] instead.
    fn from(m: DenseMotor<Pga2>) -> Self {
        Self {
            s: m.0.get(0b000),
            e_12: m.0.get(0b011),
            e_02: m.0.get(0b110),
            e_01: m.0.get(0b101),
        }
    }
}

impl From<DenseRotor<Pga2>> for shapes::Motor {
    /// Convert a dense [`DenseRotor<Pga2>`] into a [`shapes::Motor`]
    /// by reading the 4 motor blades from its inner motor.
    fn from(r: DenseRotor<Pga2>) -> Self {
        shapes::Motor::from(r.0)
    }
}

impl From<DenseTranslator<Pga2>> for shapes::Motor {
    /// Convert a dense [`DenseTranslator<Pga2>`] into a
    /// [`shapes::Motor`] by reading the 4 motor blades from its inner
    /// motor.
    fn from(t: DenseTranslator<Pga2>) -> Self {
        shapes::Motor::from(t.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_arrays_partition_the_8_blades() {
        // Sanity: support + forbidden must be a disjoint union covering
        // all 8 PGA2 blades, with no overlap and no gap.
        for (support, forbidden) in [
            (POINT_SUPPORT, POINT_FORBIDDEN.as_slice()),
            (LINE_SUPPORT, LINE_FORBIDDEN.as_slice()),
            (BIVECTOR_SUPPORT, BIVECTOR_FORBIDDEN.as_slice()),
            (ROTOR_SUPPORT, ROTOR_FORBIDDEN.as_slice()),
            (TRANSLATOR_SUPPORT, TRANSLATOR_FORBIDDEN.as_slice()),
            (MOTOR_SUPPORT, MOTOR_FORBIDDEN.as_slice()),
        ] {
            assert_eq!(support.len() + forbidden.len(), 8);
            for blade in 0..8usize {
                let in_s = support.contains(&blade);
                let in_f = forbidden.contains(&blade);
                assert!(in_s ^ in_f, "blade {blade} membership broken");
            }
        }
    }

    #[test]
    fn shared_bridge_error_is_reachable_via_pga2() {
        // The PGA2 bridge re-exports `pga3::BridgeError`, so engine code
        // that wants a single error type across both algebras can write
        // `Result<_, pga2::BridgeError>` and have it resolve to the
        // same underlying enum.
        let err = BridgeError::OutOfShape {
            blade: 0b001,
            value: 1.0,
            tolerance: 1e-6,
        };
        let s = format!("{err}");
        assert!(s.contains("0b"));
        let _e: &dyn std::error::Error = &err;
    }
}
