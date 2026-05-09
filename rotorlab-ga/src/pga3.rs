//! The PGA3 algebra: signature `(3, 0, 1)`, 4 ambient dimensions, 16 blades.
//!
//! Basis encoding (bitmask): bit 0 = `e1`, bit 1 = `e2`, bit 2 = `e3`, bit 3 = `e0`.
//! `e1`, `e2`, `e3` square to `+1`; `e0` (null) squares to `0`.
//!
//! Bitmask `0b1111` represents the canonical-bit-order pseudoscalar
//! `I = e1 ∧ e2 ∧ e3 ∧ e0`. This differs from `e0 ∧ e1 ∧ e2 ∧ e3` (the
//! ordering common in textbooks) by `(-1)^3 = -1` from moving `e0` past three
//! basis vectors. All sign conventions in this crate follow the bit-ascending
//! pseudoscalar consistently.

use crate::algebra::Algebra;

/// PGA3 algebra: 3D projective geometric algebra, Cl(3,0,1).
///
/// # Example
///
/// ```
/// use rotorlab_ga::{Algebra, pga3::Pga3};
/// assert_eq!(Pga3::SIGNATURE, (3, 0, 1));
/// assert_eq!(Pga3::DIM, 4);
/// assert_eq!(Pga3::N_BLADES, 16);
/// assert_eq!(Pga3::METRIC, &[1, 1, 1, 0]);
/// ```
#[derive(Copy, Clone, Default, Debug)]
pub struct Pga3;

impl Algebra for Pga3 {
    const SIGNATURE: (u32, u32, u32) = (3, 0, 1);
    const DIM: u32 = 4;
    const N_BLADES: usize = 16;
    type Storage = [f32; 16];
    type Scalar = f32;
    const METRIC: &'static [i8] = &[1, 1, 1, 0];
}

use crate::blade::{blade_product_blade, blade_product_sign};

/// The PGA3 Cayley table: `[i][j] = (sign, result_blade)` for `e_i * e_j`,
/// where `e_k` denotes the basis blade with bitmask `k`.
///
/// Materialized at compile time from the const-fn `blade_product_sign`.
pub const PGA3_CAYLEY: [[(i8, u64); 16]; 16] = pga3_cayley_table();

const fn pga3_cayley_table() -> [[(i8, u64); 16]; 16] {
    let mut table = [[(0i8, 0u64); 16]; 16];
    let mut i = 0usize;
    while i < 16 {
        let mut j = 0usize;
        while j < 16 {
            let sign = blade_product_sign(i as u64, j as u64, Pga3::METRIC);
            let blade = blade_product_blade(i as u64, j as u64);
            table[i][j] = (sign, blade);
            j += 1;
        }
        i += 1;
    }
    table
}

impl Pga3 {
    /// Reference to the compile-time-evaluated Cayley table.
    pub const fn cayley_table() -> &'static [[(i8, u64); 16]; 16] {
        &PGA3_CAYLEY
    }
}

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
    let l_dual = pd.outer_pga3(&qd);
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
    let plane_dual = pd.outer_pga3(&qd).outer_pga3(&rd);
    Plane(plane_dual.undual())
}

use crate::motor::{Motor, Rotor, Translator};

/// Build a rotor that rotates by `angle` radians in the plane defined by
/// the unit bivector `plane`.
///
/// Computes `exp(angle/2 * B) = cos(angle/2) + sin(angle/2) * B` for a
/// normalised Euclidean bivector `B` (i.e. `B^2 = -1`).
pub fn rotor(plane: Bivector, angle: f32) -> Rotor {
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
pub fn translator(direction: Bivector, distance: f32) -> Translator {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivector::Multivector;

    #[test]
    fn to_euclidean_unit_weight_round_trips() {
        let p = point(2.0, 3.0, 4.0);
        let xyz = p.to_euclidean();
        assert!((xyz[0] - 2.0).abs() < 1e-6);
        assert!((xyz[1] - 3.0).abs() < 1e-6);
        assert!((xyz[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn to_euclidean_non_unit_weight_normalizes() {
        // A point with weight 2 carrying coefficients (4, 6, 8) is the same
        // projective point as the unit-weight point (2, 3, 4).
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0b0111, 2.0);
        mv.set(0b1110, 4.0);
        mv.set(0b1101, 6.0);
        mv.set(0b1011, 8.0);
        let p = Point(mv);
        let xyz = p.to_euclidean();
        assert!((xyz[0] - 2.0).abs() < 1e-6);
        assert!((xyz[1] - 3.0).abs() < 1e-6);
        assert!((xyz[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn to_euclidean_infinity_returns_zeros() {
        // A direction (zero weight) is a point at projective infinity; we
        // collapse it to the affine origin to avoid producing NaNs.
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0b1110, 1.0);
        let p = Point(mv);
        let xyz = p.to_euclidean();
        assert_eq!(xyz, [0.0, 0.0, 0.0]);
    }
}
