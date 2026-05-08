//! The universal `Multivector<A: Algebra>` type.

use crate::algebra::Algebra;
use crate::pga3::Pga3;
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

    /// Reverse (`~self` in many GA notations): negates blades of grade 2 (mod 4)
    /// and grade 3 (mod 4); keeps grades 0, 1, 4, 5, 8, ... unchanged.
    pub fn reverse(&self) -> Self {
        let mut out = *self;
        for blade in 0..A::N_BLADES {
            let g = (blade as u64).count_ones();
            // Reverse sign is (-1)^(g(g-1)/2).
            let neg = matches!(g % 4, 2 | 3);
            if neg {
                let v = out.coeffs.as_mut();
                v[blade] = -v[blade];
            }
        }
        out
    }

    /// Grade-`k` projection: zero out all coefficients whose blade has grade ≠ k.
    pub fn grade(&self, k: u32) -> Self {
        let mut out = Self::zero();
        for blade in 0..A::N_BLADES {
            if (blade as u64).count_ones() == k {
                let v = self.coeffs.as_ref()[blade];
                out.coeffs.as_mut()[blade] = v;
            }
        }
        out
    }
}

impl Multivector<Pga3> {
    /// Geometric product `self * rhs` in PGA3.
    ///
    /// Reads from the compile-time `PGA3_CAYLEY` table: for each blade pair
    /// `(i, j)` with non-zero coefficients, looks up `(sign, out_blade)` and
    /// accumulates `sign * a_i * b_j` into the output blade.
    pub fn geometric_pga3(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        let table = Pga3::cayley_table();
        for (i, blade_i) in table.iter().enumerate() {
            let a = self.get(i);
            if a == 0.0 {
                continue;
            }
            for (j, &(sign, out_blade)) in blade_i.iter().enumerate() {
                let b = rhs.get(j);
                if b == 0.0 {
                    continue;
                }
                if sign == 0 {
                    continue;
                }
                let cur = out.get(out_blade as usize);
                let contrib = (sign as f32) * a * b;
                out.set(out_blade as usize, cur + contrib);
            }
        }
        out
    }

    /// Outer (wedge) product `self ∧ rhs` in PGA3.
    ///
    /// Same as the geometric product but only keeps terms where the input
    /// blades share no common basis vectors — i.e. `i & j == 0`. This means
    /// the result blade's grade equals the sum of input grades.
    pub fn outer_pga3(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        let table = Pga3::cayley_table();
        for (i, blade_i) in table.iter().enumerate() {
            let a = self.get(i);
            if a == 0.0 {
                continue;
            }
            for (j, &(sign, out_blade)) in blade_i.iter().enumerate() {
                if (i & j) != 0 {
                    continue; // shared basis vector → no wedge contribution
                }
                let b = rhs.get(j);
                if b == 0.0 {
                    continue;
                }
                if sign == 0 {
                    continue;
                }
                let cur = out.get(out_blade as usize);
                let contrib = (sign as f32) * a * b;
                out.set(out_blade as usize, cur + contrib);
            }
        }
        out
    }

    /// Inner product (left contraction) `self · rhs` in PGA3.
    ///
    /// Defined as the grade-`|r - s|` part of the geometric product when
    /// `self` is grade `r` and `rhs` is grade `s`. Implemented blade-wise:
    /// for each pair `(i, j)`, keep the contribution iff
    /// `popcount(out_blade) == |popcount(i) - popcount(j)|`.
    pub fn inner_pga3(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        let table = Pga3::cayley_table();
        for (i, blade_i) in table.iter().enumerate() {
            let a = self.get(i);
            if a == 0.0 {
                continue;
            }
            let gi = i.count_ones() as i32;
            for (j, &(sign, out_blade)) in blade_i.iter().enumerate() {
                let b = rhs.get(j);
                if b == 0.0 {
                    continue;
                }
                let gj = j.count_ones() as i32;
                let target_grade = (gi - gj).unsigned_abs();
                if sign == 0 {
                    continue;
                }
                if out_blade.count_ones() != target_grade {
                    continue;
                }
                let cur = out.get(out_blade as usize);
                let contrib = (sign as f32) * a * b;
                out.set(out_blade as usize, cur + contrib);
            }
        }
        out
    }

    /// Dual: maps a blade `b` to the complementary blade `~b & ((1 << DIM) - 1)`,
    /// with a sign determined by the dimension and the original blade's grade.
    pub fn dual(&self) -> Self {
        let mut out = Self::zero();
        let mask = (1u64 << Pga3::DIM) - 1; // 0b1111 for PGA3
        let table = Pga3::cayley_table();
        for blade in 0..16u64 {
            let v = self.get(blade as usize);
            if v == 0.0 {
                continue;
            }
            let dual_blade = (!blade) & mask;
            // Sign convention: dual(b) gets the sign from b * I.
            let (sign, _result) = table[blade as usize][mask as usize];
            let s = sign as f32;
            let cur = out.get(dual_blade as usize);
            out.set(dual_blade as usize, cur + s * v);
        }
        out
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

    #[test]
    fn geometric_product_e1_times_e1() {
        // e1 * e1 = +1 (scalar)
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 1.0);
        let result = e1.geometric_pga3(&e1);
        assert_eq!(result.get(0), 1.0);
        for k in 1..16 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero");
        }
    }

    #[test]
    fn geometric_product_e1_times_e2() {
        // e1 * e2 = e12 (bitmask 0b0011)
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 1.0);
        let mut e2: Multivector<Pga3> = Multivector::zero();
        e2.set(0b0010, 1.0);
        let result = e1.geometric_pga3(&e2);
        assert_eq!(result.get(0b0011), 1.0);
    }

    #[test]
    fn geometric_product_e0_times_e0_is_zero() {
        // e0 is null in PGA3
        let mut e0: Multivector<Pga3> = Multivector::zero();
        e0.set(0b1000, 1.0);
        let result = e0.geometric_pga3(&e0);
        for k in 0..16 {
            assert_eq!(result.get(k), 0.0, "blade {k} should be zero (e0 is null)");
        }
    }

    #[test]
    fn outer_product_e1_wedge_e2() {
        // e1 ∧ e2 = e12 (grade 1 ∧ grade 1 = grade 2)
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 1.0);
        let mut e2: Multivector<Pga3> = Multivector::zero();
        e2.set(0b0010, 1.0);
        let result = e1.outer_pga3(&e2);
        assert_eq!(result.get(0b0011), 1.0);
        // No grade-0 contribution
        assert_eq!(result.get(0), 0.0);
    }

    #[test]
    fn outer_product_e1_wedge_e1_is_zero() {
        // a ∧ a = 0 always
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 1.0);
        let result = e1.outer_pga3(&e1);
        for k in 0..16 {
            assert_eq!(result.get(k), 0.0);
        }
    }

    #[test]
    fn inner_product_e1_dot_e1() {
        // e1 · e1 = 1 (grade 1 · grade 1 → grade 0)
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 1.0);
        let result = e1.inner_pga3(&e1);
        assert_eq!(result.get(0), 1.0);
    }

    #[test]
    fn inner_product_e1_dot_e2_is_zero() {
        // e1 · e2 = 0 (orthogonal)
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 1.0);
        let mut e2: Multivector<Pga3> = Multivector::zero();
        e2.set(0b0010, 1.0);
        let result = e1.inner_pga3(&e2);
        for k in 0..16 {
            assert_eq!(result.get(k), 0.0);
        }
    }

    #[test]
    fn reverse_grade_2_negates() {
        // reverse(e12) = -e12 (grade 2)
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0b0011, 1.0);
        let r = mv.reverse();
        assert_eq!(r.get(0b0011), -1.0);
    }

    #[test]
    fn reverse_grade_1_unchanged() {
        // reverse(e1) = e1 (grade 1)
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0b0001, 1.0);
        let r = mv.reverse();
        assert_eq!(r.get(0b0001), 1.0);
    }

    #[test]
    fn grade_projection_isolates_grade() {
        // mv = 1 + e1 + e12; project grade 1 → only e1 remains
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0, 1.0);
        mv.set(0b0001, 2.0);
        mv.set(0b0011, 3.0);
        let g1 = mv.grade(1);
        assert_eq!(g1.get(0), 0.0);
        assert_eq!(g1.get(0b0001), 2.0);
        assert_eq!(g1.get(0b0011), 0.0);
    }

    #[test]
    fn dual_swaps_grade_with_complement() {
        // dual(1) = pseudoscalar I (or ±I depending on convention)
        let mut s: Multivector<Pga3> = Multivector::zero();
        s.set(0, 1.0);
        let d = s.dual();
        // The dual is at the pseudoscalar blade 0b1111.
        // Sign depends on convention — assert magnitude only.
        assert_eq!(d.get(0b1111).abs(), 1.0);
    }
}
