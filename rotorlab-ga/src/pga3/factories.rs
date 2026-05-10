//! Dense-multivector newtype wrappers (`Point`, `Plane`, `Line`, `Bivector`)
//! together with the factory functions that build them.
//!
//! These are the legacy "dense" surface: each newtype wraps a full
//! [`Multivector<Pga3>`] and pays the cost of all 16 blades even when
//! only a handful are non-zero. The named-field optimisation layer
//! lives in [`crate::pga3::shapes`] (Stage 7); the bridge between the
//! two representations lands in Stage 8.

use super::algebra::Pga3;
use crate::motor::{Motor, Rotor, Translator};
use crate::multivector::Multivector;

/// Bivector (grade-2 element of PGA3).
#[derive(Copy, Clone, Default)]
pub struct Bivector(pub Multivector<Pga3>);

/// Point (grade-3 trivector in PGA3).
#[derive(Copy, Clone, Default)]
pub struct Point(pub Multivector<Pga3>);

/// Line (grade-2 bivector in PGA3).
#[derive(Copy, Clone, Default)]
pub struct Line(pub Multivector<Pga3>);

/// Plane (grade-1 vector in PGA3).
#[derive(Copy, Clone, Default)]
pub struct Plane(pub Multivector<Pga3>);

/// Construct a Euclidean point at `(x, y, z)` in PGA3.
///
/// In the e0-as-null convention, a normalized point is the trivector
/// `e_123 + x * e_023 + y * e_013 + z * e_012`. Bitmask encoding:
///   - `e_123 = 0b0111` (bits e1,e2,e3)
///   - `e_023 = 0b1110` (bits e2,e3,e0)
///   - `e_013 = 0b1101` (bits e1,e3,e0)
///   - `e_012 = 0b1011` (bits e1,e2,e0)
pub fn point(x: f32, y: f32, z: f32) -> Point {
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0111, 1.0); // e_123
    mv.set(0b1110, x); // x * e_023
    mv.set(0b1101, y); // y * e_013
    mv.set(0b1011, z); // z * e_012
    Point(mv)
}

impl Point {
    /// Convert this PGA3 point to Euclidean `(x, y, z)` by dividing the
    /// trivector coefficients on `e_023`, `e_013`, `e_012` by the
    /// homogeneous weight on `e_123`.
    ///
    /// Returns `[0.0, 0.0, 0.0]` for points whose weight has absolute
    /// value below `1e-12` (these represent points at projective
    /// infinity, where the affine `(x, y, z)` is undefined). Callers
    /// that need to distinguish a true origin from a degenerate point
    /// should inspect the underlying [`Multivector`] directly.
    pub fn to_euclidean(&self) -> [f32; 3] {
        let w = self.0.get(0b0111);
        if w.abs() < 1e-12 {
            return [0.0, 0.0, 0.0];
        }
        [
            self.0.get(0b1110) / w,
            self.0.get(0b1101) / w,
            self.0.get(0b1011) / w,
        ]
    }
}

/// Construct the line through two PGA3 points (`p ∨ q`).
///
/// Implements the regressive product `a ∨ b = J⁻¹(J(a) ∧ J(b))` using the
/// Poincaré dual `J` ([`Multivector::dual`]) and its inverse, the left
/// complement ([`Multivector::undual`]). The wedge of duals computes the
/// meet in the dual algebra, then `undual` carries the result back.
///
/// Even though the intermediate result is grade 2 (where `J` and `J⁻¹`
/// happen to coincide), we use `undual` for consistency with [`plane_through`]
/// and to make the algebraic structure explicit.
pub fn line_through(p: Point, q: Point) -> Line {
    let pd = p.0.dual();
    let qd = q.0.dual();
    let l_dual = pd.outer(&qd);
    Line(l_dual.undual())
}

/// Construct the plane through three PGA3 points (`p ∨ q ∨ r`).
///
/// Same regressive-product pattern as [`line_through`]: dualize, wedge in
/// the dual algebra, undualize. The intermediate grade is 3, on which
/// `dual` and `undual` differ by a sign, so the final `undual` is required
/// for the correct orientation.
pub fn plane_through(p: Point, q: Point, r: Point) -> Plane {
    let pd = p.0.dual();
    let qd = q.0.dual();
    let rd = r.0.dual();
    let plane_dual = pd.outer(&qd).outer(&rd);
    Plane(plane_dual.undual())
}

/// Build a rotor that rotates by `angle` radians in the plane defined by
/// the unit bivector `plane`.
///
/// Computes `exp(angle/2 * B) = cos(angle/2) + sin(angle/2) * B` for a
/// normalised Euclidean bivector `B` (i.e. `B^2 = -1`).
pub fn rotor(plane: Bivector, angle: f32) -> Rotor<Pga3> {
    let half = angle * 0.5;
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0, half.cos());
    let s = half.sin();
    for k in 0..16 {
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
/// Computes `exp(distance/2 * T) = 1 + distance/2 * T` because `T^2 = 0`.
pub fn translator(direction: Bivector, distance: f32) -> Translator<Pga3> {
    let half = distance * 0.5;
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0, 1.0);
    for k in 0..16 {
        let v = direction.0.get(k);
        if v != 0.0 {
            mv.set(k, mv.get(k) + half * v);
        }
    }
    Translator(Motor(mv))
}
