//! The `Algebra` trait: defines a Clifford / geometric algebra by signature.

use crate::scalar::Scalar;

/// A geometric algebra over signature `(P, Q, R)`.
///
/// `P` basis vectors square to `+1`, `Q` to `-1`, `R` to `0` (null/degenerate).
/// Total ambient dimension is `P + Q + R`; total blade count is `2^(P+Q+R)`.
///
/// Implementations are unit structs: one per algebra. v0.0.1 ships [`crate::pga3::Pga3`]
/// (signature `(3, 0, 1)`). See that type for a worked example.
pub trait Algebra: Copy + Default + 'static {
    /// `(P, Q, R)`: counts of positive, negative, null basis vectors.
    const SIGNATURE: (u32, u32, u32);
    /// Total ambient dimension `P + Q + R`.
    const DIM: u32;
    /// Number of basis blades `2^DIM`.
    const N_BLADES: usize;
    /// Storage for one multivector. Must be `[Self::Scalar; N_BLADES]`.
    type Storage: AsRef<[Self::Scalar]> + AsMut<[Self::Scalar]> + Copy + Default + bytemuck::Pod;
    /// The scalar field. Always `f32` or `f64` in v0.0.1.
    type Scalar: Scalar;
    /// Metric vector: a slice of `DIM` entries, each `+1`, `-1`, or `0`.
    /// Index `i` gives the square of basis vector `e_i`.
    const METRIC: &'static [i8];

    /// Flat row-major Cayley table, length `N_BLADES * N_BLADES`.
    ///
    /// Entry `(i, j)` of the geometric product table is at index
    /// `i * N_BLADES + j` and decomposes as `(sign, result_blade)`, where
    /// `sign` is `+1`, `-1`, or `0` and `result_blade` is the bitmask of
    /// the basis blade produced by `e_i * e_j`. The flat layout lets
    /// downstream code iterate the table generically without specialising
    /// on a fixed nested-array shape.
    const CAYLEY: &'static [(i8, u64)];

    /// Bitmask of basis vectors that square to `0`.
    ///
    /// Bit `i` is set iff `METRIC[i] == 0`. Lets downstream code recognise
    /// blades that contain a null factor in `O(1)` via a single mask test.
    /// For purely Euclidean algebras this is `0`.
    const NULL_MASK: u64;
}
