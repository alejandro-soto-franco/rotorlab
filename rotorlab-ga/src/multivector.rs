//! The universal `Multivector<A: Algebra>` type.

use crate::algebra::Algebra;
use core::marker::PhantomData;

/// A multivector in algebra `A`.
///
/// Stored as a flat `[Scalar; N_BLADES]` indexed by blade bitmask. The
/// `PhantomData<A>` carries the algebra type at compile time without
/// affecting runtime size or alignment.
///
/// # Example
///
/// ```
/// use rotorlab_ga::{Multivector, pga3::Pga3};
/// let mv: Multivector<Pga3> = Multivector::default();
/// assert_eq!(mv.coeffs[0], 0.0);
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Multivector<A: Algebra> {
    /// Per-blade coefficients, indexed by blade bitmask.
    pub coeffs: A::Storage,
    _marker: PhantomData<A>,
}

impl<A: Algebra> Default for Multivector<A> {
    fn default() -> Self {
        Self {
            coeffs: A::Storage::default(),
            _marker: PhantomData,
        }
    }
}

// Safety:
// - `#[repr(C)]` guarantees field order and no struct-level padding.
// - The only data field is `A::Storage`, which is `bytemuck::Pod` by trait bound.
// - `PhantomData<A>` is a ZST and contributes no bytes, alignment, or padding.
// - All bit patterns of `[f32; N]` (the only `Storage` shape in v0.0.1) are
//   valid f32 values (NaN inclusive), satisfying Pod's "all bit patterns valid"
//   requirement.
unsafe impl<A: Algebra> bytemuck::Pod for Multivector<A> {}
// Safety: An all-zero `Multivector` is a valid value (every coefficient is 0.0).
unsafe impl<A: Algebra> bytemuck::Zeroable for Multivector<A> {}

impl<A: Algebra> Multivector<A> {
    /// Construct the additive-zero multivector.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Construct a pure-scalar multivector with grade-0 coefficient `s`.
    pub fn from_scalar(s: A::Scalar) -> Self {
        let mut mv = Self::default();
        mv.coeffs.as_mut()[0] = s;
        mv
    }

    /// Read the coefficient at blade index `blade`.
    pub fn get(&self, blade: usize) -> A::Scalar {
        self.coeffs.as_ref()[blade]
    }

    /// Write the coefficient at blade index `blade`.
    pub fn set(&mut self, blade: usize, value: A::Scalar) {
        self.coeffs.as_mut()[blade] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pga3::Pga3;

    #[test]
    fn size_matches_storage() {
        assert_eq!(core::mem::size_of::<Multivector<Pga3>>(), 16 * 4);
        assert_eq!(
            core::mem::align_of::<Multivector<Pga3>>(),
            core::mem::align_of::<f32>()
        );
    }

    #[test]
    fn pod_round_trip() {
        let mut mv: Multivector<Pga3> = Multivector::default();
        mv.set(0, 1.0);
        mv.set(3, 2.5);
        let bytes: &[u8] = bytemuck::bytes_of(&mv);
        assert_eq!(bytes.len(), 16 * 4);
        let recovered: &Multivector<Pga3> = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.get(0), 1.0);
        assert_eq!(recovered.get(3), 2.5);
    }

    #[test]
    fn slice_to_gpu_bytes() {
        let mvs: [Multivector<Pga3>; 4] = [Multivector::default(); 4];
        let bytes: &[u8] = bytemuck::cast_slice(&mvs);
        assert_eq!(bytes.len(), 4 * 16 * 4);
    }
}
