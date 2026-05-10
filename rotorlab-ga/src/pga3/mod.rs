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
//!
//! # Module layout
//!
//! - [`algebra`]: the [`Pga3`] type, its [`crate::algebra::Algebra`]
//!   impl, and the compile-time Cayley tables [`PGA3_CAYLEY`] and
//!   [`PGA3_CAYLEY_FLAT`].
//! - [`factories`]: the dense-multivector newtypes ([`Point`],
//!   [`Plane`], [`Line`], [`Bivector`]) and the factory functions
//!   that build them ([`point`], [`line_through`], [`plane_through`],
//!   [`rotor`], [`translator`]).
//! - [`shapes`]: named-field shape structs that store only the
//!   non-zero blades for each geometric primitive. The dense
//!   newtypes in [`factories`] remain the reference oracle; the
//!   shape structs are an optimisation layer (Stage 7).
//! - [`bridge`]: round-trip conversions between the named-field
//!   shape structs in [`shapes`] and the dense
//!   [`crate::multivector::Multivector<Pga3>`] (Stage 8). Defines
//!   [`BridgeError`] for out-of-shape extraction failures.

pub mod algebra;
pub mod bridge;
pub mod factories;
pub mod shapes;

pub use algebra::{PGA3_CAYLEY, PGA3_CAYLEY_FLAT, Pga3};
pub use bridge::BridgeError;
pub use factories::{
    Bivector, Line, Plane, Point, line_through, plane_through, point, rotor, translator,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Algebra;
    use crate::multivector::Multivector;

    #[test]
    #[allow(deprecated)]
    fn cayley_flat_matches_nested_table() {
        // Stage-1 acceptance: the new flat row-major slice must agree with the
        // legacy nested-array accessor on every one of the 256 entries.
        let nested = Pga3::cayley_table();
        assert_eq!(Pga3::CAYLEY.len(), 256);
        for (i, row) in nested.iter().enumerate() {
            for (j, &from_nested) in row.iter().enumerate() {
                let flat = Pga3::CAYLEY[i * 16 + j];
                assert_eq!(
                    flat, from_nested,
                    "mismatch at (i={i}, j={j}): flat={flat:?} nested={from_nested:?}",
                );
            }
        }
    }

    #[test]
    fn null_mask_is_e0_only() {
        // e0 is bit 3 in the PGA3 bitmask encoding and is the only null
        // basis vector (METRIC[3] == 0). Bits 0, 1, 2 are e1, e2, e3 (all +1).
        assert_eq!(Pga3::NULL_MASK, 0b1000);
        for i in 0..4usize {
            let bit = 1u64 << i;
            let in_null = (Pga3::NULL_MASK & bit) != 0;
            let metric_zero = Pga3::METRIC[i] == 0;
            assert_eq!(
                in_null, metric_zero,
                "NULL_MASK / METRIC mismatch on basis vector e_{i}",
            );
        }
    }

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
