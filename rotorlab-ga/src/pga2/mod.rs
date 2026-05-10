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
//!
//! # Module layout
//!
//! - [`algebra`]: the [`Pga2`] type, its [`crate::algebra::Algebra`]
//!   impl, and the compile-time Cayley tables [`PGA2_CAYLEY`] and
//!   [`PGA2_CAYLEY_FLAT`].
//! - [`factories`]: the dense-multivector newtypes ([`Point`], [`Line`],
//!   [`Bivector`]) and the factory functions that build them ([`point`],
//!   [`line_through`], [`rotor`], [`translator`]).
//! - [`shapes`]: named-field shape structs that store only the
//!   non-zero blades for each geometric primitive. The dense
//!   newtypes in [`factories`] remain the reference oracle; the
//!   shape structs are an optimisation layer (Stage 11 backport
//!   from PGA3 Stage 7).
//! - [`bridge`]: round-trip conversions between the named-field
//!   shape structs in [`shapes`] and the dense
//!   [`crate::multivector::Multivector<Pga2>`]. Re-exports
//!   [`crate::pga3::BridgeError`] as [`BridgeError`] so the engine can
//!   share a single error type across PGA2 and PGA3 surfaces.
//! - [`specialised`]: shape-to-shape methods on
//!   [`shapes::Motor`] mirroring the PGA3 Stage 9 surface, scaled to
//!   the 2D primitive set (no plane in 2D).

pub mod algebra;
pub mod bridge;
pub mod factories;
pub mod shapes;
pub mod specialised;

pub use algebra::{PGA2_CAYLEY, PGA2_CAYLEY_FLAT, Pga2};
pub use bridge::BridgeError;
pub use factories::{Bivector, Line, Point, line_through, point, rotor, translator};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Algebra;
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
    /// plane (`e1 ∧ e2`, bitmask `0b011`). This is the only Euclidean
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
