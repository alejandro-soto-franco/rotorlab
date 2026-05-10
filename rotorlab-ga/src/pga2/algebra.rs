//! The PGA2 algebra struct, its [`Algebra`] impl, and the compile-time
//! Cayley tables backing it.
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
use crate::blade::{blade_product_blade, blade_product_sign};

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
