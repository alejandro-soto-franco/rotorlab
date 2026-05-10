//! The `Scalar` trait: what can be a multivector coefficient.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// A real-valued scalar that can be a multivector coefficient.
///
/// Implemented for `f32` and `f64`. The trait is sealed; downstream crates
/// cannot add new scalar types.
///
/// # Example
///
/// ```
/// use rotorlab_ga::Scalar;
/// fn add_one<S: Scalar>(x: S) -> S { x + S::ONE }
/// assert_eq!(add_one(2.0_f32), 3.0_f32);
/// ```
pub trait Scalar:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + bytemuck::Pod
    + private::Sealed
    + 'static
{
    /// The additive identity (`0.0`).
    const ZERO: Self;
    /// The multiplicative identity (`1.0`).
    const ONE: Self;
    /// `-1.0`.
    const NEG_ONE: Self;
    /// Returns `self.sqrt()`.
    fn sqrt(self) -> Self;
    /// Returns `self.abs()`.
    fn abs(self) -> Self;
    /// Returns `self.sin()`.
    fn sin(self) -> Self;
    /// Returns `self.cos()`.
    fn cos(self) -> Self;

    /// Convert a Cayley-table sign (`-1`, `0`, `+1`) into the scalar field.
    ///
    /// Used by generic geometric / outer / inner product loops to multiply
    /// the integer sign drawn from [`crate::algebra::Algebra::CAYLEY`] by
    /// scalar coefficients without resorting to a per-`Scalar` numeric cast.
    ///
    /// Any input outside `{-1, 0, 1}` is treated as `0`; downstream call
    /// sites only ever feed values produced by the Cayley-table constructor,
    /// which itself emits exactly those three values.
    fn from_sign(s: i8) -> Self {
        match s {
            1 => Self::ONE,
            -1 => Self::NEG_ONE,
            _ => Self::ZERO,
        }
    }
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn sin(self) -> Self {
        f32::sin(self)
    }
    fn cos(self) -> Self {
        f32::cos(self)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
}
