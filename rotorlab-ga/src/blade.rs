//! Const-fn primitives for blade arithmetic.
//!
//! A *blade* is a basis multivector encoded as a `u64` bitmask: bit `i` set
//! means basis vector `e_i` is part of the wedge product. For example, in PGA3
//! (4D, basis `e1 e2 e3 e0`), blade `0b1011` is `e1 ∧ e2 ∧ e0`.
//!
//! The two functions in this module compute the sign and the resulting blade
//! of a Clifford product `blade1 * blade2`. They are `const fn` so the full
//! Cayley table for an algebra can be a `pub const`, evaluated at compile time.

/// Compute the sign of `blade1 * blade2` under the given metric.
///
/// Returns `+1`, `-1`, or `0` (the latter when the product is degenerate
/// because some null basis vector squares to zero in the inner reduction).
///
/// `metric` is a slice of `dim` entries, each `+1`, `-1`, or `0`, where the
/// `i`-th entry is `e_i^2`.
///
/// # Example
///
/// ```
/// use rotorlab_ga::blade::blade_product_sign;
/// // PGA3 metric: e1, e2, e3 are positive, e0 is null.
/// let pga3 = &[1, 1, 1, 0];
/// assert_eq!(blade_product_sign(0b0001, 0b0001, pga3), 1);  // e1*e1 = +1
/// assert_eq!(blade_product_sign(0b1000, 0b1000, pga3), 0);  // e0*e0 = 0
/// assert_eq!(blade_product_sign(0b0001, 0b0010, pga3), 1);  // e1*e2 = +e12
/// assert_eq!(blade_product_sign(0b0010, 0b0001, pga3), -1); // e2*e1 = -e12
/// ```
pub const fn blade_product_sign(blade1: u64, blade2: u64, metric: &[i8]) -> i8 {
    let mut sign = false;
    let mut mask: u64 = 0;
    let mut b1 = blade1;
    let mut b2 = blade2;
    while b1 != 0 {
        let power1 = 1u64 << (63 - b1.leading_zeros());
        b2 &= power1.wrapping_mul(2).wrapping_sub(1);
        let b2_floor = if b2 == 0 {
            0
        } else {
            1u64 << (63 - b2.leading_zeros())
        };
        if power1 == b2_floor {
            let i = power1.trailing_zeros() as usize;
            if metric[i] == -1 {
                sign = !sign;
            } else if metric[i] == 0 {
                return 0;
            }
        }
        b1 ^= power1;
        mask ^= power1.wrapping_sub(1);
    }
    let parity = (mask & blade2).count_ones() % 2 == 1;
    sign ^= parity;
    if sign { -1 } else { 1 }
}

/// Compute the blade resulting from `blade1 * blade2` (regardless of sign).
///
/// In a Clifford algebra over orthogonal basis vectors, the result blade
/// is just the symmetric difference of the two inputs.
///
/// # Example
///
/// ```
/// use rotorlab_ga::blade::blade_product_blade;
/// assert_eq!(blade_product_blade(0b0001, 0b0010), 0b0011); // e1 * e2 = e12
/// assert_eq!(blade_product_blade(0b0011, 0b0010), 0b0001); // e12 * e2 = e1
/// ```
pub const fn blade_product_blade(blade1: u64, blade2: u64) -> u64 {
    blade1 ^ blade2
}

/// Sign of the right complement (Poincaré dual / Lengyel `J`-map) of a basis
/// blade in a `dim`-dimensional algebra.
///
/// The right complement `J(b)` is the unique scalar multiple of the bit-
/// complement blade for which `b ∧ J(b) = +I` where `I` is the canonical-bit-
/// order pseudoscalar (bitmask `(1 << dim) - 1`). Returns the sign `ε ∈ {±1}`
/// such that `J(b) = ε · complement(b)`.
///
/// # Why this is metric-independent
///
/// `b` and `complement(b)` share no basis vectors, so the wedge `b ∧ comp` has
/// no contractions; the sign is purely the bit-permutation parity needed to
/// sort the concatenated bit list into canonical (ascending) order. This is
/// what makes the right complement well-defined in PGA, where `e0² = 0` makes
/// the metric pseudoscalar product `b · I` degenerate.
///
/// # Example
///
/// ```
/// use rotorlab_ga::blade::right_complement_sign;
/// // J(1) = +I (no swaps needed).
/// assert_eq!(right_complement_sign(0b0000, 4), 1);
/// // J(I) = +1 (no factors to commute).
/// assert_eq!(right_complement_sign(0b1111, 4), 1);
/// // J(e1) = +e2∧e3∧e0 (parity 0).
/// assert_eq!(right_complement_sign(0b0001, 4), 1);
/// // J(e0) = -e1∧e2∧e3 (e0 must hop past three vectors to reach the front).
/// assert_eq!(right_complement_sign(0b1000, 4), -1);
/// ```
pub const fn right_complement_sign(blade: u64, dim: u32) -> i8 {
    let full: u64 = if dim >= 64 {
        u64::MAX
    } else {
        (1u64 << dim) - 1
    };
    let comp = (!blade) & full;
    // Parity of the swaps required to interleave the bits of `blade` (in
    // ascending bit order) with the bits of `comp` (in ascending bit order)
    // to produce the fully-sorted bit list. For each bit p in `blade`, count
    // bits of `comp` strictly below p (each is one transposition).
    let mut sign = false;
    let mut mask: u64 = 0;
    let mut b1 = blade;
    while b1 != 0 {
        let power1 = 1u64 << (63 - b1.leading_zeros());
        b1 ^= power1;
        mask ^= power1.wrapping_sub(1);
    }
    let parity = (mask & comp).count_ones() % 2 == 1;
    sign ^= parity;
    if sign { -1 } else { 1 }
}

/// Sign of the left complement (`L`-map, the inverse of the right complement)
/// of a basis blade.
///
/// Defined dually by `L(b) ∧ b = +I`. The relation to the right complement is
/// `L = (-1)^(k(n-k)) · J` on grade-`k` blades, so `L(J(b)) = J(L(b)) = b`.
///
/// # Example
///
/// ```
/// use rotorlab_ga::blade::{left_complement_sign, right_complement_sign};
/// // On even grades (0, 2, 4) of n=4, L = J.
/// assert_eq!(left_complement_sign(0b0000, 4), right_complement_sign(0b0000, 4));
/// assert_eq!(left_complement_sign(0b0011, 4), right_complement_sign(0b0011, 4));
/// assert_eq!(left_complement_sign(0b1111, 4), right_complement_sign(0b1111, 4));
/// // On odd grades (1, 3) of n=4, L = -J.
/// assert_eq!(left_complement_sign(0b0001, 4), -right_complement_sign(0b0001, 4));
/// assert_eq!(left_complement_sign(0b0111, 4), -right_complement_sign(0b0111, 4));
/// ```
pub const fn left_complement_sign(blade: u64, dim: u32) -> i8 {
    let full: u64 = if dim >= 64 {
        u64::MAX
    } else {
        (1u64 << dim) - 1
    };
    let comp = (!blade) & full;
    let mut sign = false;
    let mut mask: u64 = 0;
    let mut b1 = comp;
    while b1 != 0 {
        let power1 = 1u64 << (63 - b1.leading_zeros());
        b1 ^= power1;
        mask ^= power1.wrapping_sub(1);
    }
    let parity = (mask & blade).count_ones() % 2 == 1;
    sign ^= parity;
    if sign { -1 } else { 1 }
}
