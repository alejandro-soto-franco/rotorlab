//! Dense-multivector newtype wrappers (`Point`, `Line`, `Bivector`)
//! together with the factory functions that build them.
//!
//! These are the legacy "dense" surface: each newtype wraps a full
//! [`Multivector<Pga2>`] and pays the cost of all 8 blades even when
//! only a handful are non-zero. The named-field optimisation layer
//! lives in [`crate::pga2::shapes`] (Stage 11 backport from PGA3
//! Stage 7); the bridge between the two representations lives in
//! [`crate::pga2::bridge`].

use super::algebra::Pga2;
use crate::motor::{Motor, Rotor, Translator};
use crate::multivector::Multivector;

/// Bivector (grade-2 element of PGA2).
#[derive(Copy, Clone, Default)]
pub struct Bivector(pub Multivector<Pga2>);

/// Point (grade-2 bivector in PGA2).
///
/// Note: in PGA2, points and the generic bivector subspace share the same
/// grade. The newtype distinguishes them at the type level even though the
/// underlying storage is identical.
///
/// # Bitmask layout
///
/// A normalised point at `(x, y)` is the bivector
/// `e_12 + x * e_02 + y * e_01`, with bit 0 = `e1`, bit 1 = `e2`, bit 2 = `e0`:
///
///   - weight on `e_12 = 0b011`
///   - x on `e_02 = 0b110`
///   - y on `e_01 = 0b101`
///
/// See [`point()`] for the constructor that pins this layout.
#[derive(Copy, Clone, Default)]
pub struct Point(pub Multivector<Pga2>);

/// Line (grade-1 vector in PGA2).
///
/// A 2D line `ax + by + c = 0` is encoded as the vector `a*e1 + b*e2 + c*e0`.
#[derive(Copy, Clone, Default)]
pub struct Line(pub Multivector<Pga2>);

/// Construct a Euclidean point at `(x, y)` in PGA2.
///
/// In the e0-as-null convention, a normalized point is the bivector
/// `e_12 + x * e_02 + y * e_01`. Bitmask encoding (bit 0 = e1, bit 1 = e2,
/// bit 2 = e0):
///   - `e_12 = 0b011` (bits e1, e2)
///   - `e_02 = 0b110` (bits e2, e0)
///   - `e_01 = 0b101` (bits e1, e0)
///
/// The convention mirrors PGA3's: the weight blade is the wedge of all
/// Euclidean basis vectors, and each affine coefficient sits on the
/// bivector obtained by replacing one Euclidean factor with `e0`.
pub fn point(x: f32, y: f32) -> Point {
    let mut mv: Multivector<Pga2> = Multivector::zero();
    mv.set(0b011, 1.0); // e_12
    mv.set(0b110, x); // x * e_02
    mv.set(0b101, y); // y * e_01
    Point(mv)
}

impl Point {
    /// Convert this PGA2 point to Euclidean `(x, y)` by dividing the
    /// bivector coefficients on `e_02`, `e_01` by the homogeneous weight
    /// on `e_12`.
    ///
    /// Returns `[0.0, 0.0]` for points whose weight has absolute value
    /// below `1e-12` (these represent points at projective infinity, where
    /// the affine `(x, y)` is undefined). Callers that need to distinguish
    /// a true origin from a degenerate point should inspect the underlying
    /// [`Multivector`] directly.
    pub fn to_euclidean(&self) -> [f32; 2] {
        let w = self.0.get(0b011);
        if w.abs() < 1e-12 {
            return [0.0, 0.0];
        }
        [self.0.get(0b110) / w, self.0.get(0b101) / w]
    }
}

/// Construct the line through two PGA2 points (`p ∨ q`).
///
/// Implements the regressive product `a ∨ b = J⁻¹(J(a) ∧ J(b))` using the
/// Poincaré dual `J` ([`Multivector::dual`]) and its inverse, the left
/// complement ([`Multivector::undual`]). The wedge of duals computes the
/// meet in the dual algebra, then `undual` carries the result back.
///
/// The intermediate result lives at grade 1 (a vector), where `J` and
/// `J⁻¹` differ by a sign in 3-dimensional algebras, so the final
/// `undual` is required for the correct orientation.
pub fn line_through(p: Point, q: Point) -> Line {
    let pd = p.0.dual();
    let qd = q.0.dual();
    let l_dual = pd.outer(&qd);
    Line(l_dual.undual())
}

/// Build a rotor that rotates by `angle` radians in the plane defined by
/// the unit bivector `plane`.
///
/// In 2D, the only Euclidean rotation plane is `e_12` (the `(x, y)`
/// plane); the function nevertheless accepts any bivector by analogy
/// with [`crate::pga3::rotor`].
///
/// Computes `exp(angle/2 * B) = cos(angle/2) + sin(angle/2) * B` for a
/// normalised Euclidean bivector `B` (i.e. `B^2 = -1`).
pub fn rotor(plane: Bivector, angle: f32) -> Rotor<Pga2> {
    let half = angle * 0.5;
    let mut mv: Multivector<Pga2> = Multivector::zero();
    mv.set(0, half.cos());
    let s = half.sin();
    for k in 0..8 {
        let v = plane.0.get(k);
        if v != 0.0 {
            mv.set(k, mv.get(k) + s * v);
        }
    }
    Rotor(Motor(mv))
}

/// Build a translator that translates by `distance` along the (null) direction
/// bivector `direction` (a bivector containing `e0`).
///
/// In PGA2 the only admissible direction bivectors are `e_01 = 0b101` and
/// `e_02 = 0b110`; passing `e_12 = 0b011` (the Euclidean rotation plane)
/// would degenerate the result, since `e_12` does not contain `e0`.
///
/// Computes `exp(distance/2 * T) = 1 + distance/2 * T` because `T^2 = 0`.
pub fn translator(direction: Bivector, distance: f32) -> Translator<Pga2> {
    let half = distance * 0.5;
    let mut mv: Multivector<Pga2> = Multivector::zero();
    mv.set(0, 1.0);
    for k in 0..8 {
        let v = direction.0.get(k);
        if v != 0.0 {
            mv.set(k, mv.get(k) + half * v);
        }
    }
    Translator(Motor(mv))
}
