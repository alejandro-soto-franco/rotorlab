//! The PGA2 algebra: signature `(2, 0, 1)`, 3 ambient dimensions, 8 blades.
//!
//! Basis encoding (bitmask): bit 0 = `e1`, bit 1 = `e2`, bit 2 = `e0`.
//! `e1`, `e2` square to `+1`; `e0` (null) squares to `0`.
//!
//! Bitmask `0b111` represents the canonical-bit-order pseudoscalar
//! `I = e1 ∧ e2 ∧ e0`. Moving `e0` past `e1, e2` gives an even-parity sign
//! change, so `e1 ∧ e2 ∧ e0` and `e0 ∧ e1 ∧ e2` agree on the pseudoscalar's
//! sign in PGA2. The bit-ascending convention is preserved here for
//! consistency with PGA3.

use crate::algebra::Algebra;

/// PGA2 algebra: 2D projective geometric algebra, Cl(2,0,1).
///
/// # Example
///
/// ```
/// use rotorlab_ga::{Algebra, pga2::Pga2};
/// assert_eq!(Pga2::SIGNATURE, (2, 0, 1));
/// assert_eq!(Pga2::DIM, 3);
/// assert_eq!(Pga2::N_BLADES, 8);
/// assert_eq!(Pga2::METRIC, &[1, 1, 0]);
/// // Bit 2 (e0) is the only null basis vector.
/// assert_eq!(Pga2::NULL_MASK, 0b100);
/// // Flat Cayley table is row-major over 8 x 8 blades.
/// assert_eq!(Pga2::CAYLEY.len(), 8 * 8);
/// ```
#[derive(Copy, Clone, Default, Debug)]
pub struct Pga2;

impl Algebra for Pga2 {
    const SIGNATURE: (u32, u32, u32) = (2, 0, 1);
    const DIM: u32 = 3;
    const N_BLADES: usize = 8;
    type Storage = [f32; 8];
    type Scalar = f32;
    const METRIC: &'static [i8] = &[1, 1, 0];
    const CAYLEY: &'static [(i8, u64)] = &PGA2_CAYLEY_FLAT;
    /// Bit 2 corresponds to `e0`, the sole null basis vector of PGA2.
    const NULL_MASK: u64 = 0b100;
}

use crate::blade::{blade_product_blade, blade_product_sign};

/// The PGA2 Cayley table: `[i][j] = (sign, result_blade)` for `e_i * e_j`,
/// where `e_k` denotes the basis blade with bitmask `k`.
///
/// Materialized at compile time from the const-fn `blade_product_sign`.
pub const PGA2_CAYLEY: [[(i8, u64); 8]; 8] = pga2_cayley_table();

/// Flat row-major version of [`PGA2_CAYLEY`], length `8 * 8 = 64`.
///
/// Entry `(i, j)` lives at index `i * 8 + j`. Built at compile time by
/// copying [`PGA2_CAYLEY`] entry by entry, so the two are guaranteed to
/// agree element for element. This is the storage backing
/// [`Algebra::CAYLEY`] for [`Pga2`].
pub const PGA2_CAYLEY_FLAT: [(i8, u64); 64] = pga2_cayley_flat();

const fn pga2_cayley_table() -> [[(i8, u64); 8]; 8] {
    let mut table = [[(0i8, 0u64); 8]; 8];
    let mut i = 0usize;
    while i < 8 {
        let mut j = 0usize;
        while j < 8 {
            let sign = blade_product_sign(i as u64, j as u64, Pga2::METRIC);
            let blade = blade_product_blade(i as u64, j as u64);
            table[i][j] = (sign, blade);
            j += 1;
        }
        i += 1;
    }
    table
}

const fn pga2_cayley_flat() -> [(i8, u64); 64] {
    let mut flat = [(0i8, 0u64); 64];
    let mut i = 0usize;
    while i < 8 {
        let mut j = 0usize;
        while j < 8 {
            flat[i * 8 + j] = PGA2_CAYLEY[i][j];
            j += 1;
        }
        i += 1;
    }
    flat
}

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

use crate::motor::{Motor, Rotor, Translator};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::Motor;
    use crate::multivector::Multivector;

    #[test]
    fn pga2_signature() {
        // Stage-5 smoke: trait constants must match the PGA2 spec.
        assert_eq!(Pga2::SIGNATURE, (2, 0, 1));
        assert_eq!(Pga2::DIM, 3);
        assert_eq!(Pga2::N_BLADES, 8);
        assert_eq!(Pga2::NULL_MASK, 0b100);
        assert_eq!(Pga2::METRIC, &[1, 1, 0]);
        assert_eq!(Pga2::CAYLEY.len(), 64);
    }

    #[test]
    fn multivector_default_constructs() {
        // Smoke: `Multivector::<Pga2>::default()` must compile and produce
        // an all-zero multivector of the right storage size.
        let mv: Multivector<Pga2> = Multivector::default();
        for k in 0..8 {
            assert_eq!(mv.get(k), 0.0);
        }
    }

    #[test]
    fn multivector_size_matches_storage() {
        // 8 blades * f32 = 32 bytes, no padding.
        assert_eq!(core::mem::size_of::<Multivector<Pga2>>(), 8 * 4);
        assert_eq!(
            core::mem::align_of::<Multivector<Pga2>>(),
            core::mem::align_of::<f32>()
        );
    }

    #[test]
    fn null_mask_is_e0_only() {
        // e0 is bit 2 in the PGA2 bitmask encoding and is the only null
        // basis vector (METRIC[2] == 0). Bits 0, 1 are e1, e2 (both +1).
        assert_eq!(Pga2::NULL_MASK, 0b100);
        for i in 0..3usize {
            let bit = 1u64 << i;
            let in_null = (Pga2::NULL_MASK & bit) != 0;
            let metric_zero = Pga2::METRIC[i] == 0;
            assert_eq!(
                in_null, metric_zero,
                "NULL_MASK / METRIC mismatch on basis vector e_{i}",
            );
        }
    }

    #[test]
    fn cayley_flat_matches_nested_table() {
        // Mirror the PGA3 acceptance: flat table agrees with the nested one.
        assert_eq!(Pga2::CAYLEY.len(), 64);
        for (i, row) in PGA2_CAYLEY.iter().enumerate() {
            for (j, &from_nested) in row.iter().enumerate() {
                let flat = Pga2::CAYLEY[i * 8 + j];
                assert_eq!(
                    flat, from_nested,
                    "mismatch at (i={i}, j={j}): flat={flat:?} nested={from_nested:?}",
                );
            }
        }
    }

    #[test]
    fn geometric_product_e1_times_e1() {
        // e1 * e1 = +1
        let mut e1: Multivector<Pga2> = Multivector::zero();
        e1.set(0b001, 1.0);
        let result = e1.geometric(&e1);
        assert_eq!(result.get(0), 1.0);
        for k in 1..8 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero");
        }
    }

    #[test]
    fn geometric_product_e1_times_e2() {
        // e1 * e2 = e_12 (bitmask 0b011)
        let mut e1: Multivector<Pga2> = Multivector::zero();
        e1.set(0b001, 1.0);
        let mut e2: Multivector<Pga2> = Multivector::zero();
        e2.set(0b010, 1.0);
        let result = e1.geometric(&e2);
        assert_eq!(result.get(0b011), 1.0);
        for k in 0..8 {
            if k != 0b011 {
                assert_eq!(result.get(k), 0.0, "blade {k} should be zero");
            }
        }
    }

    #[test]
    fn geometric_product_e0_times_e0_is_zero() {
        // e0 is null in PGA2.
        let mut e0: Multivector<Pga2> = Multivector::zero();
        e0.set(0b100, 1.0);
        let result = e0.geometric(&e0);
        for k in 0..8 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero (e0 is null)");
        }
    }

    #[test]
    fn dual_of_scalar_is_pseudoscalar() {
        // J(1) = +I where I is the bitmask 0b111 pseudoscalar.
        let mut s: Multivector<Pga2> = Multivector::zero();
        s.set(0, 1.0);
        let d = s.dual();
        assert_eq!(d.get(0b111), 1.0);
        for k in 0..8 {
            if k != 0b111 {
                assert_eq!(d.get(k), 0.0, "blade {k} should be zero");
            }
        }
    }

    #[test]
    fn to_euclidean_unit_weight_round_trips() {
        let p = point(2.0, 3.0);
        let xy = p.to_euclidean();
        assert!((xy[0] - 2.0).abs() < 1e-6);
        assert!((xy[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn to_euclidean_non_unit_weight_normalizes() {
        // A point with weight 2 carrying coefficients (4, 6) is the same
        // projective point as the unit-weight point (2, 3).
        let mut mv: Multivector<Pga2> = Multivector::zero();
        mv.set(0b011, 2.0);
        mv.set(0b110, 4.0);
        mv.set(0b101, 6.0);
        let p = Point(mv);
        let xy = p.to_euclidean();
        assert!((xy[0] - 2.0).abs() < 1e-6);
        assert!((xy[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn to_euclidean_infinity_returns_zeros() {
        // A direction (zero weight) is a point at projective infinity; we
        // collapse it to the affine origin to avoid producing NaNs.
        let mut mv: Multivector<Pga2> = Multivector::zero();
        mv.set(0b110, 1.0);
        let p = Point(mv);
        let xy = p.to_euclidean();
        assert_eq!(xy, [0.0, 0.0]);
    }

    #[test]
    fn point_pins_bitmask_convention() {
        // Convention pin at the bitmask level: weight on e_12 = 0b011,
        // x on e_02 = 0b110, y on e_01 = 0b101. Round-trip behaviour is
        // already covered by `to_euclidean_unit_weight_round_trips`.
        let p = point(1.0, 0.0);
        assert_eq!(p.0.get(0b011), 1.0, "weight blade is e_12 = 0b011");
        assert_eq!(p.0.get(0b110), 1.0, "x blade is e_02 = 0b110");
        assert_eq!(p.0.get(0b101), 0.0, "y blade is e_01 = 0b101");
        let q = point(0.0, 1.0);
        assert_eq!(q.0.get(0b011), 1.0);
        assert_eq!(q.0.get(0b110), 0.0);
        assert_eq!(q.0.get(0b101), 1.0);
    }

    /// Build the unit Euclidean bivector representing the world rotation
    /// plane (`e1 ∧ e2`, bitmask `0b011`). This is the *only* Euclidean
    /// rotation plane in 2D PGA, and the `e_12` blade IS the xy-plane
    /// bivector.
    fn xy_plane_bivector() -> Bivector {
        let mut mv: Multivector<Pga2> = Multivector::zero();
        mv.set(0b011, 1.0);
        Bivector(mv)
    }

    #[test]
    fn rotor_is_unit_norm() {
        // exp(theta/2 * e_12) = cos(theta/2) + sin(theta/2) * e_12.
        // Squared norm: cos^2 + sin^2 = 1.
        let r = rotor(xy_plane_bivector(), 1.234);
        assert!((r.0.0.norm_sq() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn line_through_three_collinear_points_meets_third() {
        // Construct three collinear points along the line y = 0:
        // p = (0, 0), q = (1, 0), r = (2, 0).
        // The line through p and q must wedge to zero against r.
        let p = point(0.0, 0.0);
        let q = point(1.0, 0.0);
        let r = point(2.0, 0.0);
        let line = line_through(p, q);

        // In PGA2, a point lies on a line iff their wedge is the zero
        // pseudoscalar. Wedging `line.0` (grade 1) with `r.0` (grade 2)
        // produces a grade-3 element (the pseudoscalar slot 0b111).
        let meet = line.0.outer(&r.0);
        for k in 0..8 {
            assert!(
                meet.get(k).abs() < 1e-6,
                "collinearity wedge nonzero at blade {k:03b}: {}",
                meet.get(k),
            );
        }
    }

    #[test]
    fn interpolate_xy_plane_rotation_at_half_is_quarter_rotation() {
        // Identity to rotor-by-pi-around-e_12, midpoint = rotor-by-pi/2.
        let m_a: Motor<Pga2> = Motor::identity();
        let m_b: Motor<Pga2> = rotor(xy_plane_bivector(), core::f32::consts::PI).0;
        let m = m_a.interpolate(&m_b, 0.5);
        let test_point = point(1.0, 0.0);
        let rotated = Point(m.apply(&test_point.0));
        let xy = rotated.to_euclidean();
        // Rotation by pi/2 in the (x, y) plane sends (1, 0) to (0, 1).
        assert!(xy[0].abs() < 1e-4, "x: {}", xy[0]);
        assert!((xy[1] - 1.0).abs() < 1e-4, "y: {}", xy[1]);
    }
}
