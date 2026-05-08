//! The PGA3 algebra: signature `(3, 0, 1)`, 4 ambient dimensions, 16 blades.
//!
//! Basis encoding (bitmask): bit 0 = `e1`, bit 1 = `e2`, bit 2 = `e3`, bit 3 = `e0`.
//! `e1`, `e2`, `e3` square to `+1`; `e0` (null) squares to `0`.
//! The pseudoscalar `I = e0 ∧ e1 ∧ e2 ∧ e3` has bitmask `0b1111`.

use crate::algebra::Algebra;

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
}
