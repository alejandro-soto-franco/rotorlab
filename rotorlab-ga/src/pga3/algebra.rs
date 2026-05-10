//! The PGA3 algebra struct, its [`Algebra`] impl, and the compile-time
//! Cayley tables backing it.
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
use crate::blade::{blade_product_blade, blade_product_sign};

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
/// // Bit 3 (e0) is the only null basis vector.
/// assert_eq!(Pga3::NULL_MASK, 0b1000);
/// // Flat Cayley table is row-major over 16 x 16 blades.
/// assert_eq!(Pga3::CAYLEY.len(), 16 * 16);
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
    const CAYLEY: &'static [(i8, u64)] = &PGA3_CAYLEY_FLAT;
    /// Bit 3 corresponds to `e0`, the sole null basis vector of PGA3.
    const NULL_MASK: u64 = 0b1000;
}

/// The PGA3 Cayley table: `[i][j] = (sign, result_blade)` for `e_i * e_j`,
/// where `e_k` denotes the basis blade with bitmask `k`.
///
/// Materialized at compile time from the const-fn `blade_product_sign`.
pub const PGA3_CAYLEY: [[(i8, u64); 16]; 16] = pga3_cayley_table();

/// Flat row-major version of [`PGA3_CAYLEY`], length `16 * 16 = 256`.
///
/// Entry `(i, j)` lives at index `i * 16 + j`. Built at compile time by
/// copying [`PGA3_CAYLEY`] entry by entry, so the two are guaranteed to
/// agree element for element. This is the storage backing
/// [`Algebra::CAYLEY`] for [`Pga3`].
pub const PGA3_CAYLEY_FLAT: [(i8, u64); 256] = pga3_cayley_flat();

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

const fn pga3_cayley_flat() -> [(i8, u64); 256] {
    let mut flat = [(0i8, 0u64); 256];
    let mut i = 0usize;
    while i < 16 {
        let mut j = 0usize;
        while j < 16 {
            flat[i * 16 + j] = PGA3_CAYLEY[i][j];
            j += 1;
        }
        i += 1;
    }
    flat
}

impl Pga3 {
    /// Reference to the compile-time-evaluated Cayley table.
    #[deprecated(note = "use Pga3::CAYLEY (flat row-major) instead; will be removed in 0.2.0")]
    pub const fn cayley_table() -> &'static [[(i8, u64); 16]; 16] {
        &PGA3_CAYLEY
    }
}
