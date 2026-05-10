//! The PGA4 algebra: signature `(4, 0, 1)`, 5 ambient dimensions, 32 blades.
//!
//! Basis encoding (bitmask): bit 0 = `e1`, bit 1 = `e2`, bit 2 = `e3`,
//! bit 3 = `e4`, bit 4 = `e0`. `e1`, `e2`, `e3`, `e4` square to `+1`;
//! `e0` (null) squares to `0`.
//!
//! Bitmask `0b11111` represents the canonical-bit-order pseudoscalar
//! `I = e1 ∧ e2 ∧ e3 ∧ e4 ∧ e0`. Moving `e0` past four basis vectors
//! is an even permutation, so this agrees in sign with the
//! `e0 ∧ e1 ∧ e2 ∧ e3 ∧ e4` ordering common in textbooks. The
//! bit-ascending convention is preserved here for consistency with PGA2
//! and PGA3.
//!
//! PGA4 represents 4D Euclidean geometry. Stage 6 supplies only the
//! algebra implementation, the compile-time Cayley tables, and enough
//! convenience to build [`Rotor<Pga4>`] / [`Translator<Pga4>`] for SLERP
//! tests. Geometric primitive newtypes (Point / Line / Plane / Volume)
//! are deferred to Stage 11; users of PGA4 work directly with
//! [`Multivector<Pga4>`].

use crate::algebra::Algebra;
use crate::blade::{blade_product_blade, blade_product_sign};

/// PGA4 algebra: 4D projective geometric algebra, Cl(4,0,1).
///
/// # Example
///
/// ```
/// use rotorlab_ga::{Algebra, pga4::Pga4};
/// assert_eq!(Pga4::SIGNATURE, (4, 0, 1));
/// assert_eq!(Pga4::DIM, 5);
/// assert_eq!(Pga4::N_BLADES, 32);
/// assert_eq!(Pga4::METRIC, &[1, 1, 1, 1, 0]);
/// // Bit 4 (e0) is the only null basis vector.
/// assert_eq!(Pga4::NULL_MASK, 0b10000);
/// // Flat Cayley table is row-major over 32 x 32 blades.
/// assert_eq!(Pga4::CAYLEY.len(), 32 * 32);
/// ```
#[derive(Copy, Clone, Default, Debug)]
pub struct Pga4;

impl Algebra for Pga4 {
    const SIGNATURE: (u32, u32, u32) = (4, 0, 1);
    const DIM: u32 = 5;
    const N_BLADES: usize = 32;
    type Storage = [f32; 32];
    type Scalar = f32;
    const METRIC: &'static [i8] = &[1, 1, 1, 1, 0];
    const CAYLEY: &'static [(i8, u64)] = &PGA4_CAYLEY_FLAT;
    /// Bit 4 corresponds to `e0`, the sole null basis vector of PGA4.
    const NULL_MASK: u64 = 0b10000;
}

/// The PGA4 Cayley table: `[i][j] = (sign, result_blade)` for `e_i * e_j`,
/// where `e_k` denotes the basis blade with bitmask `k`.
///
/// Materialized at compile time from the const-fn `blade_product_sign`.
pub const PGA4_CAYLEY: [[(i8, u64); 32]; 32] = pga4_cayley_table();

/// Flat row-major version of [`PGA4_CAYLEY`], length `32 * 32 = 1024`.
///
/// Entry `(i, j)` lives at index `i * 32 + j`. Built at compile time by
/// copying [`PGA4_CAYLEY`] entry by entry, so the two are guaranteed to
/// agree element for element. This is the storage backing
/// [`Algebra::CAYLEY`] for [`Pga4`].
pub const PGA4_CAYLEY_FLAT: [(i8, u64); 1024] = pga4_cayley_flat();

const fn pga4_cayley_table() -> [[(i8, u64); 32]; 32] {
    let mut table = [[(0i8, 0u64); 32]; 32];
    let mut i = 0usize;
    while i < 32 {
        let mut j = 0usize;
        while j < 32 {
            let sign = blade_product_sign(i as u64, j as u64, Pga4::METRIC);
            let blade = blade_product_blade(i as u64, j as u64);
            table[i][j] = (sign, blade);
            j += 1;
        }
        i += 1;
    }
    table
}

const fn pga4_cayley_flat() -> [(i8, u64); 1024] {
    let mut flat = [(0i8, 0u64); 1024];
    let mut i = 0usize;
    while i < 32 {
        let mut j = 0usize;
        while j < 32 {
            flat[i * 32 + j] = PGA4_CAYLEY[i][j];
            j += 1;
        }
        i += 1;
    }
    flat
}

use crate::motor::{Motor, Rotor, Translator};
use crate::multivector::Multivector;

/// Bivector (grade-2 element of PGA4).
///
/// PGA4 has `C(5, 2) = 10` grade-2 blades: six purely Euclidean
/// (combinations of `e1, e2, e3, e4`) and four containing `e0`. The
/// Euclidean bivectors are the rotation planes; the `e0`-bearing
/// bivectors are the translation directions. This newtype carries no
/// invariant beyond "the underlying multivector is a grade-2 element";
/// callers select the geometric meaning by which blade(s) they populate.
#[derive(Copy, Clone, Default)]
pub struct Bivector(pub Multivector<Pga4>);

/// Build a rotor that rotates by `angle` radians in the plane defined by
/// the unit Euclidean bivector `plane`.
///
/// In 4D there are six independent Euclidean rotation planes
/// (`e_12, e_13, e_14, e_23, e_24, e_34`); the function accepts any
/// bivector and lets the caller select the plane by populating the
/// corresponding blade.
///
/// Computes `exp(angle/2 * B) = cos(angle/2) + sin(angle/2) * B` for a
/// normalised Euclidean bivector `B` (i.e. `B^2 = -1`).
pub fn rotor(plane: Bivector, angle: f32) -> Rotor<Pga4> {
    let half = angle * 0.5;
    let mut mv: Multivector<Pga4> = Multivector::zero();
    mv.set(0, half.cos());
    let s = half.sin();
    for k in 0..32 {
        let v = plane.0.get(k);
        if v != 0.0 {
            mv.set(k, mv.get(k) + s * v);
        }
    }
    Rotor(Motor(mv))
}

/// Build a translator that translates by `distance` along the (null)
/// direction bivector `direction` (a bivector containing `e0`).
///
/// Admissible direction bivectors in PGA4 are the four `e0`-bearing
/// blades `e_01 = 0b10001`, `e_02 = 0b10010`, `e_03 = 0b10100`, and
/// `e_04 = 0b11000`. Passing a purely Euclidean bivector (no `e0` factor)
/// would degenerate the result, since the exponential closure
/// `T^2 = 0` only holds for null bivectors.
///
/// Computes `exp(distance/2 * T) = 1 + distance/2 * T` because `T^2 = 0`.
pub fn translator(direction: Bivector, distance: f32) -> Translator<Pga4> {
    let half = distance * 0.5;
    let mut mv: Multivector<Pga4> = Multivector::zero();
    mv.set(0, 1.0);
    for k in 0..32 {
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
    fn pga4_signature() {
        // Stage-6 smoke: trait constants must match the PGA4 spec.
        assert_eq!(Pga4::SIGNATURE, (4, 0, 1));
        assert_eq!(Pga4::DIM, 5);
        assert_eq!(Pga4::N_BLADES, 32);
        assert_eq!(Pga4::NULL_MASK, 0b10000);
        assert_eq!(Pga4::METRIC, &[1, 1, 1, 1, 0]);
        assert_eq!(Pga4::CAYLEY.len(), 1024);
    }

    #[test]
    fn multivector_default_constructs() {
        // Smoke: `Multivector::<Pga4>::default()` must compile and
        // produce an all-zero multivector of the right storage size.
        let mv: Multivector<Pga4> = Multivector::default();
        for k in 0..32 {
            assert_eq!(mv.get(k), 0.0);
        }
    }

    #[test]
    fn multivector_size_matches_storage() {
        // 32 blades * f32 = 128 bytes, no padding.
        assert_eq!(core::mem::size_of::<Multivector<Pga4>>(), 32 * 4);
        assert_eq!(
            core::mem::align_of::<Multivector<Pga4>>(),
            core::mem::align_of::<f32>()
        );
    }

    #[test]
    fn null_mask_is_e0_only() {
        // e0 is bit 4 in the PGA4 bitmask encoding and is the only null
        // basis vector (METRIC[4] == 0). Bits 0..=3 are e1..=e4 (all +1).
        assert_eq!(Pga4::NULL_MASK, 0b10000);
        for i in 0..5usize {
            let bit = 1u64 << i;
            let in_null = (Pga4::NULL_MASK & bit) != 0;
            let metric_zero = Pga4::METRIC[i] == 0;
            assert_eq!(
                in_null, metric_zero,
                "NULL_MASK / METRIC mismatch on basis vector e_{i}",
            );
        }
    }

    #[test]
    fn cayley_flat_matches_nested_table() {
        // Mirror the PGA3 / PGA2 acceptance: flat table agrees with
        // the nested one on every entry.
        assert_eq!(Pga4::CAYLEY.len(), 1024);
        for (i, row) in PGA4_CAYLEY.iter().enumerate() {
            for (j, &from_nested) in row.iter().enumerate() {
                let flat = Pga4::CAYLEY[i * 32 + j];
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
        let mut e1: Multivector<Pga4> = Multivector::zero();
        e1.set(0b00001, 1.0);
        let result = e1.geometric(&e1);
        assert_eq!(result.get(0), 1.0);
        for k in 1..32 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero");
        }
    }

    #[test]
    fn geometric_product_e4_times_e4() {
        // e4 * e4 = +1 (e4 is Euclidean in PGA4).
        let mut e4: Multivector<Pga4> = Multivector::zero();
        e4.set(0b01000, 1.0);
        let result = e4.geometric(&e4);
        assert_eq!(result.get(0), 1.0);
        for k in 1..32 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero");
        }
    }

    #[test]
    fn geometric_product_e1_times_e2() {
        // e1 * e2 = e_12 (bitmask 0b00011)
        let mut e1: Multivector<Pga4> = Multivector::zero();
        e1.set(0b00001, 1.0);
        let mut e2: Multivector<Pga4> = Multivector::zero();
        e2.set(0b00010, 1.0);
        let result = e1.geometric(&e2);
        assert_eq!(result.get(0b00011), 1.0);
        for k in 0..32 {
            if k != 0b00011 {
                assert_eq!(result.get(k), 0.0, "blade {k} should be zero");
            }
        }
    }

    #[test]
    fn geometric_product_e0_times_e0_is_zero() {
        // e0 is null in PGA4.
        let mut e0: Multivector<Pga4> = Multivector::zero();
        e0.set(0b10000, 1.0);
        let result = e0.geometric(&e0);
        for k in 0..32 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero (e0 is null)");
        }
    }

    #[test]
    fn dual_of_scalar_is_pseudoscalar() {
        // J(1) = +I where I is the bitmask 0b11111 pseudoscalar.
        let mut s: Multivector<Pga4> = Multivector::zero();
        s.set(0, 1.0);
        let d = s.dual();
        assert_eq!(d.get(0b11111), 1.0);
        for k in 0..32 {
            if k != 0b11111 {
                assert_eq!(d.get(k), 0.0, "blade {k} should be zero");
            }
        }
    }

    #[test]
    fn dual_undual_round_trip_on_basis_blades() {
        // For every basis blade, undual(dual(b)) == b. This is the
        // fundamental compatibility check for the dual map and exercises
        // the metric-independent right-complement infrastructure on a
        // 5-dimensional algebra.
        for k in 0..32 {
            let mut mv: Multivector<Pga4> = Multivector::zero();
            mv.set(k, 1.0);
            let round = mv.dual().undual();
            for j in 0..32 {
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (round.get(j) - expected).abs() < 1e-6,
                    "round-trip mismatch on blade {k:05b} at {j:05b}: got {}",
                    round.get(j),
                );
            }
        }
    }

    /// Build the unit Euclidean bivector representing the `e1 ∧ e2` plane
    /// (bitmask `0b00011`). PGA4 has six independent Euclidean rotation
    /// planes; we exercise this one since it lifts the PGA3 `z`-axis
    /// rotation pattern verbatim.
    fn e12_bivector() -> Bivector {
        let mut mv: Multivector<Pga4> = Multivector::zero();
        mv.set(0b00011, 1.0);
        Bivector(mv)
    }

    #[test]
    fn rotor_is_unit_norm() {
        // exp(theta/2 * e_12) = cos(theta/2) + sin(theta/2) * e_12.
        // Squared norm: cos^2 + sin^2 = 1.
        let r = rotor(e12_bivector(), 1.234);
        assert!((r.0.0.norm_sq() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn translator_identity_at_zero_distance() {
        // exp(0 * T) = 1, regardless of which null direction T points along.
        let mut dir: Multivector<Pga4> = Multivector::zero();
        dir.set(0b10001, 1.0); // e_01
        let t = translator(Bivector(dir), 0.0);
        assert_eq!(t.0.0.get(0), 1.0);
        for k in 1..32 {
            assert_eq!(t.0.0.get(k), 0.0, "blade {k} should be zero");
        }
    }

    #[test]
    fn interpolate_e12_rotation_at_half_is_quarter_rotation() {
        // Identity to rotor-by-pi-around-e_12, midpoint = rotor-by-pi/2.
        // For a unit vector v the sandwich product
        //   R * v * ~R  with  R = cos(theta/2) + sin(theta/2) * e_12
        // computes a rotation of v *by 2 * (theta/2) = theta* in the
        // (e1, e2) plane. The convention in this crate maps a positive
        // `e_12` bivector to the rotation that sends e1 -> -e2 -> -e1
        // -> e2 -> e1 (cf. the PGA3 SLERP test, which uses a Point
        // trivector that rotates with the opposite handedness because
        // it picks up the conjugation parity of grade 3).
        //
        // For theta = pi/2, the midpoint rotor takes e1 to -e2.
        let m_a: Motor<Pga4> = Motor::identity();
        let m_b: Motor<Pga4> = rotor(e12_bivector(), core::f32::consts::PI).0;
        let m = m_a.interpolate(&m_b, 0.5);

        let mut e1: Multivector<Pga4> = Multivector::zero();
        e1.set(0b00001, 1.0);
        let rotated = m.apply(&e1);

        // Quarter rotation in the (e1, e2) plane sends e1 to -e2.
        assert!(
            rotated.get(0b00001).abs() < 1e-4,
            "e1: {}",
            rotated.get(0b00001)
        );
        assert!(
            (rotated.get(0b00010) + 1.0).abs() < 1e-4,
            "e2: {}",
            rotated.get(0b00010),
        );
        // Other blades should be untouched (zero).
        for k in 0..32 {
            if k == 0b00001 || k == 0b00010 {
                continue;
            }
            assert!(
                rotated.get(k).abs() < 1e-4,
                "blade {k:05b} leaked: {}",
                rotated.get(k),
            );
        }
    }

    #[test]
    fn interpolate_at_zero_returns_self() {
        // Endpoint behaviour: alpha = 0 reproduces the source motor.
        let m_a: Motor<Pga4> = rotor(e12_bivector(), 0.5).0;
        let m_b: Motor<Pga4> = rotor(e12_bivector(), 1.5).0;
        let m = m_a.interpolate(&m_b, 0.0);
        let mut e1: Multivector<Pga4> = Multivector::zero();
        e1.set(0b00001, 1.0);
        let p_via_a = m_a.apply(&e1);
        let p_via_m = m.apply(&e1);
        for k in 0..32 {
            assert!(
                (p_via_a.get(k) - p_via_m.get(k)).abs() < 1e-5,
                "blade {k:05b}: a={} m={}",
                p_via_a.get(k),
                p_via_m.get(k),
            );
        }
    }

    #[test]
    fn interpolate_at_one_returns_target() {
        // Endpoint behaviour: alpha = 1 reproduces the target motor.
        let m_a: Motor<Pga4> = rotor(e12_bivector(), 0.5).0;
        let m_b: Motor<Pga4> = rotor(e12_bivector(), 1.5).0;
        let m = m_a.interpolate(&m_b, 1.0);
        let mut e1: Multivector<Pga4> = Multivector::zero();
        e1.set(0b00001, 1.0);
        let p_via_b = m_b.apply(&e1);
        let p_via_m = m.apply(&e1);
        for k in 0..32 {
            assert!(
                (p_via_b.get(k) - p_via_m.get(k)).abs() < 1e-5,
                "blade {k:05b}: b={} m={}",
                p_via_b.get(k),
                p_via_m.get(k),
            );
        }
    }

    #[test]
    fn euclidean_grade2_blade_count_fits_motor_scratch() {
        // Motor::interpolate sizes its bivector scratch by the private
        // constant A_MAX_GRADE_2 = 16 in motor.rs. PGA4 has C(4, 2) = 6
        // Euclidean grade-2 blades; this test pins the count so a future
        // change to either the algebra or the scratch is forced to
        // revisit the assumption together.
        let mut count = 0usize;
        for blade in 0..Pga4::N_BLADES {
            let b = blade as u64;
            if b.count_ones() == 2 && (b & Pga4::NULL_MASK) == 0 {
                count += 1;
            }
        }
        assert_eq!(count, 6, "PGA4 should have C(4,2) = 6 Euclidean bivectors");
        assert!(count <= 16, "scratch in Motor::interpolate must fit count");
    }
}
