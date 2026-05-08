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
}
