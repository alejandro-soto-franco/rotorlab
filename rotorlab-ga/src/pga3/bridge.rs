//! Bridge layer between [`crate::pga3::shapes`] and the dense
//! [`Multivector<Pga3>`] representation.
//!
//! Each shape struct in [`super::shapes`] stores only the blades it can
//! carry. The dense [`Multivector<Pga3>`] in [`crate::multivector`]
//! remains the reference oracle for every operation. This module wires
//! the two surfaces together with a pair of conversions per shape:
//!
//! - `From<Shape> for Multivector<Pga3>` packs a shape's named fields
//!   into the corresponding blades of a fresh zero multivector. Blades
//!   outside the shape's support stay at zero.
//! - `TryFrom<Multivector<Pga3>> for Shape` reads the shape's blades
//!   and verifies that every blade outside the support is within a
//!   scaled tolerance of zero. On violation it returns
//!   [`BridgeError::OutOfShape`].
//!
//! The tolerance is `f32::EPSILON * 64.0 * scale`, where `scale` is the
//! maximum absolute coefficient across all 16 blades (or `1.0` when the
//! input is exactly zero). This adapts the threshold to the magnitude
//! of the input so that round-tripping through compose / decompose
//! cycles does not spuriously reject results that suffered ULP-level
//! floating-point drift.
//!
//! [`Multivector<Pga3>`]: crate::multivector::Multivector

use core::fmt;

use super::algebra::Pga3;
use super::shapes;
use crate::motor::{Motor as DenseMotor, Rotor as DenseRotor, Translator as DenseTranslator};
use crate::multivector::Multivector;

/// Tolerance multiplier on top of `f32::EPSILON * scale` used when
/// classifying a blade as "effectively zero" during shape extraction.
const TOLERANCE_FACTOR: f32 = 64.0;

/// Errors that can occur when extracting a shape struct from a dense
/// [`Multivector<Pga3>`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BridgeError {
    /// A blade outside the target shape's expected support carried a
    /// coefficient whose magnitude exceeded the round-off tolerance.
    OutOfShape {
        /// Bitmask of the blade that violated the shape constraint.
        blade: usize,
        /// Coefficient observed at that blade.
        value: f32,
        /// Tolerance threshold the coefficient exceeded.
        tolerance: f32,
    },
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::OutOfShape {
                blade,
                value,
                tolerance,
            } => write!(
                f,
                "blade 0b{blade:04b} (index {blade}) carried {value} which exceeds the shape tolerance of {tolerance}",
            ),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Compute the adaptive tolerance for a multivector: the maximum
/// absolute blade coefficient (or `1.0` when the input is the zero
/// multivector) scaled by [`TOLERANCE_FACTOR`] and `f32::EPSILON`.
fn tolerance_for(mv: &Multivector<Pga3>) -> f32 {
    let mut scale = 0.0_f32;
    for blade in 0..16 {
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
    mv: &Multivector<Pga3>,
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

/// Compute the complement of `support` in the 16 PGA3 blade indices,
/// returned as a fixed-size array for stack-only checking.
const fn complement<const N: usize>(support: &[usize]) -> [usize; N] {
    let mut out = [0usize; N];
    let mut idx = 0;
    let mut blade = 0usize;
    while blade < 16 {
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
// shape can carry; each `*_FORBIDDEN` is the complement against the 16
// PGA3 blades. The complement arrays are precomputed at compile time so
// the runtime extractor walks a small static slice.

const POINT_SUPPORT: &[usize] = &[0b1110, 0b1101, 0b1011, 0b0111];
const POINT_FORBIDDEN: [usize; 12] = complement::<12>(POINT_SUPPORT);

const PLANE_SUPPORT: &[usize] = &[0b1000, 0b0001, 0b0010, 0b0100];
const PLANE_FORBIDDEN: [usize; 12] = complement::<12>(PLANE_SUPPORT);

const LINE_SUPPORT: &[usize] = &[0b1001, 0b1010, 0b1100, 0b0011, 0b0101, 0b0110];
const LINE_FORBIDDEN: [usize; 10] = complement::<10>(LINE_SUPPORT);

const BIVECTOR_SUPPORT: &[usize] = &[0b1001, 0b1010, 0b1100, 0b0011, 0b0101, 0b0110];
const BIVECTOR_FORBIDDEN: [usize; 10] = complement::<10>(BIVECTOR_SUPPORT);

const ROTOR_SUPPORT: &[usize] = &[0b0000, 0b0011, 0b0101, 0b0110];
const ROTOR_FORBIDDEN: [usize; 12] = complement::<12>(ROTOR_SUPPORT);

const TRANSLATOR_SUPPORT: &[usize] = &[0b0000, 0b1001, 0b1010, 0b1100];
const TRANSLATOR_FORBIDDEN: [usize; 12] = complement::<12>(TRANSLATOR_SUPPORT);

const MOTOR_SUPPORT: &[usize] = &[
    0b0000, 0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100, 0b1111,
];
const MOTOR_FORBIDDEN: [usize; 8] = complement::<8>(MOTOR_SUPPORT);

// ---------------------------------------------------------------------
// Point
// ---------------------------------------------------------------------

impl From<shapes::Point> for Multivector<Pga3> {
    fn from(p: shapes::Point) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b1110, p.e_023);
        mv.set(0b1101, p.e_013);
        mv.set(0b1011, p.e_012);
        mv.set(0b0111, p.e_123);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Point {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &POINT_FORBIDDEN, tol)?;
        Ok(shapes::Point {
            e_023: mv.get(0b1110),
            e_013: mv.get(0b1101),
            e_012: mv.get(0b1011),
            e_123: mv.get(0b0111),
        })
    }
}

// ---------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------

impl From<shapes::Plane> for Multivector<Pga3> {
    fn from(p: shapes::Plane) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b1000, p.e_0);
        mv.set(0b0001, p.e_1);
        mv.set(0b0010, p.e_2);
        mv.set(0b0100, p.e_3);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Plane {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &PLANE_FORBIDDEN, tol)?;
        Ok(shapes::Plane {
            e_0: mv.get(0b1000),
            e_1: mv.get(0b0001),
            e_2: mv.get(0b0010),
            e_3: mv.get(0b0100),
        })
    }
}

// ---------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------

impl From<shapes::Line> for Multivector<Pga3> {
    fn from(l: shapes::Line) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b1001, l.e_01);
        mv.set(0b1010, l.e_02);
        mv.set(0b1100, l.e_03);
        mv.set(0b0011, l.e_12);
        mv.set(0b0101, l.e_13);
        mv.set(0b0110, l.e_23);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Line {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &LINE_FORBIDDEN, tol)?;
        Ok(shapes::Line {
            e_01: mv.get(0b1001),
            e_02: mv.get(0b1010),
            e_03: mv.get(0b1100),
            e_12: mv.get(0b0011),
            e_13: mv.get(0b0101),
            e_23: mv.get(0b0110),
        })
    }
}

// ---------------------------------------------------------------------
// Bivector
// ---------------------------------------------------------------------

impl From<shapes::Bivector> for Multivector<Pga3> {
    fn from(b: shapes::Bivector) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b1001, b.e_01);
        mv.set(0b1010, b.e_02);
        mv.set(0b1100, b.e_03);
        mv.set(0b0011, b.e_12);
        mv.set(0b0101, b.e_13);
        mv.set(0b0110, b.e_23);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Bivector {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &BIVECTOR_FORBIDDEN, tol)?;
        Ok(shapes::Bivector {
            e_01: mv.get(0b1001),
            e_02: mv.get(0b1010),
            e_03: mv.get(0b1100),
            e_12: mv.get(0b0011),
            e_13: mv.get(0b0101),
            e_23: mv.get(0b0110),
        })
    }
}

// ---------------------------------------------------------------------
// Rotor
// ---------------------------------------------------------------------

impl From<shapes::Rotor> for Multivector<Pga3> {
    fn from(r: shapes::Rotor) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b0000, r.s);
        mv.set(0b0011, r.e_12);
        mv.set(0b0101, r.e_13);
        mv.set(0b0110, r.e_23);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Rotor {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &ROTOR_FORBIDDEN, tol)?;
        Ok(shapes::Rotor {
            s: mv.get(0b0000),
            e_12: mv.get(0b0011),
            e_13: mv.get(0b0101),
            e_23: mv.get(0b0110),
        })
    }
}

// ---------------------------------------------------------------------
// Translator
// ---------------------------------------------------------------------

impl From<shapes::Translator> for Multivector<Pga3> {
    fn from(t: shapes::Translator) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b0000, t.s);
        mv.set(0b1001, t.e_01);
        mv.set(0b1010, t.e_02);
        mv.set(0b1100, t.e_03);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Translator {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &TRANSLATOR_FORBIDDEN, tol)?;
        Ok(shapes::Translator {
            s: mv.get(0b0000),
            e_01: mv.get(0b1001),
            e_02: mv.get(0b1010),
            e_03: mv.get(0b1100),
        })
    }
}

// ---------------------------------------------------------------------
// Motor
// ---------------------------------------------------------------------

impl From<shapes::Motor> for Multivector<Pga3> {
    fn from(m: shapes::Motor) -> Self {
        let mut mv = Multivector::<Pga3>::zero();
        mv.set(0b0000, m.s);
        mv.set(0b0011, m.e_12);
        mv.set(0b0101, m.e_13);
        mv.set(0b0110, m.e_23);
        mv.set(0b1001, m.e_01);
        mv.set(0b1010, m.e_02);
        mv.set(0b1100, m.e_03);
        mv.set(0b1111, m.e_0123);
        mv
    }
}

impl TryFrom<Multivector<Pga3>> for shapes::Motor {
    type Error = BridgeError;

    fn try_from(mv: Multivector<Pga3>) -> Result<Self, Self::Error> {
        let tol = tolerance_for(&mv);
        check_forbidden(&mv, &MOTOR_FORBIDDEN, tol)?;
        Ok(shapes::Motor {
            s: mv.get(0b0000),
            e_12: mv.get(0b0011),
            e_13: mv.get(0b0101),
            e_23: mv.get(0b0110),
            e_01: mv.get(0b1001),
            e_02: mv.get(0b1010),
            e_03: mv.get(0b1100),
            e_0123: mv.get(0b1111),
        })
    }
}

// ---------------------------------------------------------------------
// Dense Motor / Rotor / Translator -> shapes::Motor (lossy, unchecked)
// ---------------------------------------------------------------------
//
// These shortcut conversions read the 8 motor blades from the dense
// representation directly without going through the strict
// [`TryFrom`] support check. Sandwich and compose products on a unit
// motor preserve the even-grade subalgebra in exact arithmetic; in
// `f32` they can deposit sub-tolerance dust on out-of-shape blades.
// The engine drops that dust silently when crossing back into the
// shape representation. Use the strict [`TryFrom`] above when the
// caller must surface that dust as an error.

impl From<DenseMotor<Pga3>> for shapes::Motor {
    /// Read the 8 motor blades from a dense [`DenseMotor<Pga3>`] into
    /// a [`shapes::Motor`], silently dropping any sub-tolerance dust on
    /// out-of-shape blades. For the strict round-trip surface use
    /// [`TryFrom<Multivector<Pga3>> for shapes::Motor`] instead.
    fn from(m: DenseMotor<Pga3>) -> Self {
        Self {
            s: m.0.get(0b0000),
            e_12: m.0.get(0b0011),
            e_13: m.0.get(0b0101),
            e_23: m.0.get(0b0110),
            e_01: m.0.get(0b1001),
            e_02: m.0.get(0b1010),
            e_03: m.0.get(0b1100),
            e_0123: m.0.get(0b1111),
        }
    }
}

impl From<DenseRotor<Pga3>> for shapes::Motor {
    /// Convert a dense [`DenseRotor<Pga3>`] into a [`shapes::Motor`]
    /// by reading the 8 motor blades from its inner motor.
    fn from(r: DenseRotor<Pga3>) -> Self {
        shapes::Motor::from(r.0)
    }
}

impl From<DenseTranslator<Pga3>> for shapes::Motor {
    /// Convert a dense [`DenseTranslator<Pga3>`] into a
    /// [`shapes::Motor`] by reading the 8 motor blades from its inner
    /// motor.
    fn from(t: DenseTranslator<Pga3>) -> Self {
        shapes::Motor::from(t.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_arrays_partition_the_16_blades() {
        // Sanity: support + forbidden must be a disjoint union covering
        // all 16 PGA3 blades, with no overlap and no gap.
        for (support, forbidden) in [
            (POINT_SUPPORT, POINT_FORBIDDEN.as_slice()),
            (PLANE_SUPPORT, PLANE_FORBIDDEN.as_slice()),
            (LINE_SUPPORT, LINE_FORBIDDEN.as_slice()),
            (BIVECTOR_SUPPORT, BIVECTOR_FORBIDDEN.as_slice()),
            (ROTOR_SUPPORT, ROTOR_FORBIDDEN.as_slice()),
            (TRANSLATOR_SUPPORT, TRANSLATOR_FORBIDDEN.as_slice()),
            (MOTOR_SUPPORT, MOTOR_FORBIDDEN.as_slice()),
        ] {
            assert_eq!(support.len() + forbidden.len(), 16);
            for blade in 0..16usize {
                let in_s = support.contains(&blade);
                let in_f = forbidden.contains(&blade);
                assert!(in_s ^ in_f, "blade {blade} membership broken");
            }
        }
    }

    #[test]
    fn bridge_error_implements_error_trait() {
        let err = BridgeError::OutOfShape {
            blade: 0b0000,
            value: 1.0,
            tolerance: 1e-6,
        };
        let s = format!("{err}");
        assert!(s.contains("0b0000"));
        let _e: &dyn std::error::Error = &err;
    }
}
