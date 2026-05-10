//! proptest harness: random multivectors must satisfy GA identities.

use proptest::prelude::*;
use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga3::Pga3;

fn arb_pga3_mv() -> impl Strategy<Value = Multivector<Pga3>> {
    proptest::collection::vec(-10.0f32..10.0, 16).prop_map(|coeffs| {
        let mut mv: Multivector<Pga3> = Multivector::zero();
        for (i, c) in coeffs.iter().enumerate() {
            mv.set(i, *c);
        }
        mv
    })
}

fn approx_eq(a: &Multivector<Pga3>, b: &Multivector<Pga3>, tol: f32) -> bool {
    (0..16)
        .zip(std::iter::repeat_with(|| ()))
        .all(|(k, _)| (a.get(k) - b.get(k)).abs() <= tol)
}

proptest! {
    /// Geometric product is bilinear: (a + b) * c == a*c + b*c.
    #[test]
    fn geometric_product_is_bilinear(
        a in arb_pga3_mv(),
        b in arb_pga3_mv(),
        c in arb_pga3_mv(),
    ) {
        let lhs = a.add(&b).geometric(&c);
        let rhs = a.geometric(&c).add(&b.geometric(&c));
        prop_assert!(approx_eq(&lhs, &rhs, 1e-3));
    }

    /// Outer product is antisymmetric for vectors: u ∧ v = -(v ∧ u).
    #[test]
    fn outer_product_is_antisymmetric_on_grade1(
        u_coeffs in proptest::array::uniform4(-10.0f32..10.0),
        v_coeffs in proptest::array::uniform4(-10.0f32..10.0),
    ) {
        let mut u: Multivector<Pga3> = Multivector::zero();
        let mut v: Multivector<Pga3> = Multivector::zero();
        for (b, c) in [0b0001u64, 0b0010, 0b0100, 0b1000].iter().zip(u_coeffs.iter()) {
            u.set(*b as usize, *c);
        }
        for (b, c) in [0b0001u64, 0b0010, 0b0100, 0b1000].iter().zip(v_coeffs.iter()) {
            v.set(*b as usize, *c);
        }
        let uv = u.outer(&v);
        let vu = v.outer(&u);
        let neg_vu = vu.scale(-1.0);
        prop_assert!(approx_eq(&uv, &neg_vu, 1e-3));
    }

    /// Reverse is an involution: ~(~a) = a.
    #[test]
    fn reverse_is_involution(a in arb_pga3_mv()) {
        let r2 = a.reverse().reverse();
        prop_assert!(approx_eq(&a, &r2, 1e-6));
    }
}
