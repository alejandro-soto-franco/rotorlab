//! The universal `Multivector<A: Algebra>` type.

use crate::algebra::Algebra;
use crate::pga3::{PGA3_CAYLEY, Pga3};
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
#[derive(Copy, Clone, Debug)]
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

    /// Element-wise addition.
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        for ((dst, &a), &b) in out
            .coeffs
            .as_mut()
            .iter_mut()
            .zip(self.coeffs.as_ref().iter())
            .zip(rhs.coeffs.as_ref().iter())
        {
            *dst = a + b;
        }
        out
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        for ((dst, &a), &b) in out
            .coeffs
            .as_mut()
            .iter_mut()
            .zip(self.coeffs.as_ref().iter())
            .zip(rhs.coeffs.as_ref().iter())
        {
            *dst = a - b;
        }
        out
    }

    /// Multiply every coefficient by a scalar.
    pub fn scale(&self, s: A::Scalar) -> Self {
        let mut out = Self::zero();
        for (dst, &a) in out
            .coeffs
            .as_mut()
            .iter_mut()
            .zip(self.coeffs.as_ref().iter())
        {
            *dst = a * s;
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
        let table = &PGA3_CAYLEY;
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
    /// blades share no common basis vectors (i.e. `i & j == 0`). This means
    /// the result blade's grade equals the sum of input grades.
    pub fn outer_pga3(&self, rhs: &Self) -> Self {
        let mut out = Self::zero();
        let table = &PGA3_CAYLEY;
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
        let table = &PGA3_CAYLEY;
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

    /// Right complement (Poincaré dual / Lengyel `J`-map).
    ///
    /// For each basis blade `b`, sends the coefficient at `b` to the
    /// complementary blade `~b & 0b1111` with the unique sign making
    /// `b ∧ J(b) = +I`.
    ///
    /// # Why this is *not* `b · I` in PGA
    ///
    /// In PGA3 the metric is degenerate (`e0² = 0`), so `I² = 0` and the
    /// metric-Hodge dual `b · I⁻¹` is undefined. The geometric-product
    /// approach `b * I` collapses every blade containing `e0` to zero, which
    /// would annihilate three of the four components of any point. The right
    /// complement is purely combinatorial — sign comes from bit-permutation
    /// parity alone — and is well-defined regardless of metric signature.
    ///
    /// # Properties
    ///
    /// * Maps grade `k` to grade `4 - k`.
    /// * `dual(dual(x))` equals `(-1)^(k(4-k)) · x` on grade-`k` components.
    ///   Specifically `+x` on grades 0/2/4 and `-x` on grades 1/3, so it is
    ///   *not* a strict involution. Use [`undual`](Self::undual) for the
    ///   inverse.
    pub fn dual(&self) -> Self {
        let mut out = Self::zero();
        let full = (1u64 << Pga3::DIM) - 1; // 0b1111 for PGA3
        for blade in 0..Pga3::N_BLADES {
            let v = self.get(blade);
            if v == 0.0 {
                continue;
            }
            let comp = ((!(blade as u64)) & full) as usize;
            let s = crate::blade::right_complement_sign(blade as u64, Pga3::DIM) as f32;
            let cur = out.get(comp);
            out.set(comp, cur + s * v);
        }
        out
    }

    /// Left complement, the formal inverse of [`dual`](Self::dual).
    ///
    /// Defined by `L(b) ∧ b = +I`. Satisfies `dual().undual() == self` and
    /// `undual().dual() == self` exactly. On grade `k` it equals
    /// `(-1)^(k(4-k)) · dual()`, so it agrees with `dual` on grades 0/2/4
    /// and flips sign on grades 1/3.
    pub fn undual(&self) -> Self {
        let mut out = Self::zero();
        let full = (1u64 << Pga3::DIM) - 1;
        for blade in 0..Pga3::N_BLADES {
            let v = self.get(blade);
            if v == 0.0 {
                continue;
            }
            let comp = ((!(blade as u64)) & full) as usize;
            let s = crate::blade::left_complement_sign(blade as u64, Pga3::DIM) as f32;
            let cur = out.get(comp);
            out.set(comp, cur + s * v);
        }
        out
    }

    /// Squared norm: the grade-0 part of `self * ~self`.
    pub fn norm_sq(&self) -> f32 {
        let r = self.reverse();
        let p = self.geometric_pga3(&r);
        p.get(0)
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
    fn dual_of_scalar_is_pseudoscalar() {
        // J(1) = +I exactly: no swaps needed because the scalar has no factors.
        let mut s: Multivector<Pga3> = Multivector::zero();
        s.set(0, 1.0);
        let d = s.dual();
        assert_eq!(d.get(0b1111), 1.0);
        for k in 0..16 {
            if k != 0b1111 {
                assert_eq!(d.get(k), 0.0, "blade {k} should be zero");
            }
        }
    }

    #[test]
    fn dual_of_pseudoscalar_is_scalar() {
        // J(I) = +1.
        let mut p: Multivector<Pga3> = Multivector::zero();
        p.set(0b1111, 1.0);
        let d = p.dual();
        assert_eq!(d.get(0), 1.0);
        for k in 1..16 {
            assert_eq!(d.get(k), 0.0);
        }
    }

    #[test]
    fn dual_of_e0_is_nonzero_grade_3() {
        // The old metric-based dual returned zero here (regression guard).
        // The right complement gives J(e0) = -e1∧e2∧e3 (bitmask 0b0111).
        let mut e0: Multivector<Pga3> = Multivector::zero();
        e0.set(0b1000, 1.0);
        let d = e0.dual();
        assert_eq!(d.get(0b0111), -1.0, "J(e0) must be -e1∧e2∧e3");
        for k in 0..16 {
            if k != 0b0111 {
                assert_eq!(d.get(k), 0.0);
            }
        }
    }

    #[test]
    fn dual_of_every_basis_blade_is_a_basis_blade_with_unit_coefficient() {
        // Regression guard: confirms no basis blade is silently zeroed.
        for blade in 0..16 {
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let d = mv.dual();
            let comp = (!blade) & 0b1111;
            let coeff = d.get(comp);
            assert!(
                coeff == 1.0 || coeff == -1.0,
                "dual of basis blade {blade:04b} produced coeff {coeff} at {comp:04b}",
            );
            for k in 0..16 {
                if k != comp {
                    assert_eq!(d.get(k), 0.0, "dual of {blade:04b} leaked into {k:04b}");
                }
            }
        }
    }

    #[test]
    fn dual_swaps_grade_to_complement_grade() {
        // For each input grade k, dual sends to grade (4 - k).
        for blade in 0..16 {
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let d = mv.dual();
            let in_grade = (blade as u64).count_ones();
            let expected_out_grade = 4 - in_grade;
            for k in 0..16 {
                if d.get(k) != 0.0 {
                    assert_eq!(
                        (k as u64).count_ones(),
                        expected_out_grade,
                        "dual of grade-{in_grade} blade leaked to grade {}",
                        (k as u64).count_ones()
                    );
                }
            }
        }
    }

    #[test]
    fn undual_inverts_dual_on_every_basis_blade() {
        // The strict round-trip: undual ∘ dual = identity, no sign flip.
        for blade in 0..16 {
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let round = mv.dual().undual();
            for k in 0..16 {
                let got = round.get(k);
                let want = if k == blade { 1.0 } else { 0.0 };
                assert_eq!(
                    got, want,
                    "undual(dual(blade {blade:04b})) wrong at {k:04b}: got {got}, want {want}",
                );
            }
        }
    }

    #[test]
    fn dual_inverts_undual_on_every_basis_blade() {
        for blade in 0..16 {
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let round = mv.undual().dual();
            for k in 0..16 {
                let got = round.get(k);
                let want = if k == blade { 1.0 } else { 0.0 };
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn dual_dual_signs_match_grade_formula() {
        // J²|_grade k = (-1)^(k(n-k)) for n = 4: +1 on grades 0/2/4, −1 on 1/3.
        for blade in 0..16 {
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let twice = mv.dual().dual();
            let k = (blade as u64).count_ones();
            let expected_sign = if (k * (4 - k)) % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(twice.get(blade), expected_sign);
        }
    }

    #[test]
    fn dual_undual_agree_on_even_grades() {
        // On grades 0, 2, 4 the right and left complements coincide.
        for blade in 0..16 {
            let k = (blade as u64).count_ones();
            if k % 2 != 0 {
                continue;
            }
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let d = mv.dual();
            let u = mv.undual();
            for k_out in 0..16 {
                assert_eq!(
                    d.get(k_out),
                    u.get(k_out),
                    "blade {blade:04b}, out {k_out:04b}"
                );
            }
        }
    }

    #[test]
    fn dual_undual_differ_by_sign_on_odd_grades() {
        // On grades 1 and 3, undual = −dual.
        for blade in 0..16 {
            let k = (blade as u64).count_ones();
            if k % 2 == 0 {
                continue;
            }
            let mut mv: Multivector<Pga3> = Multivector::zero();
            mv.set(blade, 1.0);
            let d = mv.dual();
            let u = mv.undual();
            for k_out in 0..16 {
                assert_eq!(
                    d.get(k_out),
                    -u.get(k_out),
                    "blade {blade:04b}, out {k_out:04b}"
                );
            }
        }
    }

    #[test]
    fn dual_is_linear() {
        // dual(αa + βb) = α dual(a) + β dual(b).
        let mut a: Multivector<Pga3> = Multivector::zero();
        a.set(0b0001, 1.0);
        a.set(0b1000, 2.0);
        let mut b: Multivector<Pga3> = Multivector::zero();
        b.set(0b0010, -1.0);
        b.set(0b1001, 3.0);
        let combined = a.scale(2.0).add(&b.scale(5.0));
        let direct = combined.dual();
        let composed = a.dual().scale(2.0).add(&b.dual().scale(5.0));
        for k in 0..16 {
            assert_eq!(direct.get(k), composed.get(k));
        }
    }

    #[test]
    fn dual_is_metric_independent_for_full_multivector() {
        // A multivector with every blade populated round-trips exactly under
        // undual ∘ dual. This was impossible with the old `b * I` definition
        // because eight of the sixteen blades contain `e0`.
        let mut mv: Multivector<Pga3> = Multivector::zero();
        for k in 0..16 {
            mv.set(k, (k as f32) + 1.0);
        }
        let round = mv.dual().undual();
        for k in 0..16 {
            assert_eq!(round.get(k), (k as f32) + 1.0, "blade {k:04b}");
        }
    }

    #[test]
    fn add_two_multivectors() {
        let mut a: Multivector<Pga3> = Multivector::zero();
        a.set(0b0001, 1.0);
        let mut b: Multivector<Pga3> = Multivector::zero();
        b.set(0b0001, 2.0);
        let c = a.add(&b);
        assert_eq!(c.get(0b0001), 3.0);
    }

    #[test]
    fn scale_multivector() {
        let mut a: Multivector<Pga3> = Multivector::zero();
        a.set(0b0001, 1.0);
        a.set(0b0010, 2.0);
        let s = a.scale(3.0);
        assert_eq!(s.get(0b0001), 3.0);
        assert_eq!(s.get(0b0010), 6.0);
    }

    #[test]
    fn norm_sq_scalar() {
        let mut e1: Multivector<Pga3> = Multivector::zero();
        e1.set(0b0001, 3.0);
        // norm_sq(e1) = e1 · e1 = 9.0 (since e1^2 = 1 and it's grade-1)
        assert_eq!(e1.norm_sq(), 9.0);
    }
}
