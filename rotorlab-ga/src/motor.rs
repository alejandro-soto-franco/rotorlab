//! Motors, rotors, and translators over an arbitrary [`Algebra`].
//!
//! A *motor* is an even-grade element of the algebra that acts on geometry
//! by the sandwich product `M * X * ~M`. In PGA3 a motor encodes a rigid
//! motion (rotation + translation); the same algebraic shape lifts to any
//! Clifford algebra exposed via the [`Algebra`] trait. Rotors and
//! translators are tagged subsets: a [`Rotor`] is a motor whose translator
//! component is zero, a [`Translator`] is a motor whose rotor component is
//! the identity scalar.
//!
//! The Stage 3 generalisation parameterises [`Motor`], [`Rotor`], and
//! [`Translator`] over `A: Algebra` so that the same SLERP / sandwich /
//! composition logic serves PGA3, future PGA2 / PGA4 modules, and synthetic
//! algebras (see the `interpolate_generic` integration test). Hot-path
//! per-algebra specialisations remain a separate stage.

use crate::algebra::Algebra;
use crate::multivector::Multivector;
use crate::scalar::Scalar;

/// A motor in algebra `A`: an even-grade element representing a rigid
/// motion via the sandwich product.
///
/// Defined as a transparent newtype around [`Multivector<A>`]. The wrapper
/// type carries no extra data; it exists only to mark the multivector as a
/// motor in the type system so callers cannot accidentally pass an arbitrary
/// multivector where a unit motor is required.
#[derive(Copy, Clone)]
pub struct Motor<A: Algebra>(pub Multivector<A>);

/// A pure rotation: a motor whose translator component vanishes.
///
/// Stored as the underlying [`Motor<A>`] so all the motor methods apply
/// transparently. Constructors that produce rotors live alongside their
/// algebra (for PGA3 see [`crate::pga3::rotor`]).
#[derive(Copy, Clone)]
pub struct Rotor<A: Algebra>(pub Motor<A>);

/// A pure translation: a motor whose rotor component is the identity.
///
/// Stored as the underlying [`Motor<A>`] so all the motor methods apply
/// transparently. Constructors that produce translators live alongside
/// their algebra (for PGA3 see [`crate::pga3::translator`]).
#[derive(Copy, Clone)]
pub struct Translator<A: Algebra>(pub Motor<A>);

impl<A: Algebra> Default for Motor<A> {
    fn default() -> Self {
        Self(Multivector::default())
    }
}

impl<A: Algebra> Default for Rotor<A> {
    fn default() -> Self {
        Self(Motor::default())
    }
}

impl<A: Algebra> Default for Translator<A> {
    fn default() -> Self {
        Self(Motor::default())
    }
}

impl<A: Algebra> Motor<A> {
    /// The identity motor: scalar coefficient `1`, every other blade `0`.
    ///
    /// Independent of algebra signature; the grade-0 blade is always at
    /// bit index `0`.
    pub fn identity() -> Self {
        let mut mv: Multivector<A> = Multivector::zero();
        mv.set(0, A::Scalar::ONE);
        Motor(mv)
    }

    /// Apply this motor to a multivector via the sandwich product
    /// `M * X * ~M`.
    ///
    /// Composes via two geometric products against the motor and its
    /// reverse. Both products dispatch through [`Multivector::geometric`],
    /// which reads its sign data from `A::CAYLEY` and so works for any
    /// algebra.
    pub fn apply(&self, x: &Multivector<A>) -> Multivector<A> {
        let m_rev = self.0.reverse();
        let mx = self.0.geometric(x);
        mx.geometric(&m_rev)
    }

    /// Compose two motors: `(self ∘ other)(x) = self(other(x))`.
    ///
    /// Realised as the geometric product `self.0 * other.0`; verifying
    /// that the result is again a unit motor (rather than picking up a
    /// non-unit norm under round-off) is the caller's responsibility.
    pub fn compose(&self, other: &Motor<A>) -> Motor<A> {
        Motor(self.0.geometric(&other.0))
    }

    /// Interpolate between this motor and `target` along the geodesic.
    ///
    /// Stage 3 ships rotor SLERP only: both motors are assumed to be pure
    /// rotors (no translator content on null-bearing blades). For pure
    /// rotors this is the standard quaternion-style spherical linear
    /// interpolation, packaged as `result = exp(alpha * log(target * ~self)) * self`.
    ///
    /// # Limits
    ///
    /// Supported up to `A::DIM == 6`. The implementation enumerates the
    /// Euclidean grade-2 blades into a fixed-length scratch array sized
    /// by the private constant `A_MAX_GRADE_2 = 16`, which covers
    /// `C(6, 2) = 15` blades with one slot of headroom. An algebra with
    /// `A::DIM >= 7` would have `C(7, 2) = 21` grade-2 blades and
    /// overflow the scratch; in debug builds this trips a
    /// `debug_assert!`, in release builds it would silently corrupt the
    /// computation. Adding such an algebra requires either raising
    /// `A_MAX_GRADE_2` (and updating this paragraph) or splitting the
    /// scratch sizing per-algebra (e.g. via an associated const on
    /// [`Algebra`]).
    ///
    /// For a unit rotor `R = w + B` with scalar part `w = cos(theta/2)`
    /// and Euclidean bivector part `B` of norm `|B| = sin(theta/2)`, the
    /// log is `(theta/2) * (B / |B|)`. Scaling by `alpha` and taking the
    /// exponential gives `cos(alpha * theta/2) + sin(alpha * theta/2) * (B / |B|)`.
    ///
    /// # Generic bivector enumeration
    ///
    /// The Euclidean-bivector subspace of an arbitrary algebra is the set
    /// of grade-2 blades whose bitmask shares no factor with
    /// [`Algebra::NULL_MASK`]. Enumerating that subspace by iterating
    /// `0..A::N_BLADES` and gating on
    /// `count_ones() == 2 && (b & NULL_MASK) == 0` lets the same loop
    /// serve PGA3 (one null factor `e0`, three Euclidean planes) and
    /// purely Euclidean algebras such as Cl(3, 0, 0).
    ///
    /// # Endpoint and degeneracy behaviour
    ///
    /// * `interpolate(target, 0.0)` returns `self`.
    /// * `interpolate(target, 1.0)` returns `target`.
    /// * When `self` and `target` represent the same rotation (relative
    ///   bivector norm below `1e-6`), the result is `self` to avoid the
    ///   `0/0` in the `sin(alpha*theta_half)/|bivec|` factor.
    /// * Rotors `R` and `-R` represent the same `SO(3)` element. When
    ///   the relative rotor's scalar part is below `-1e-6` (encoding a
    ///   rotation greater than `pi` in the long-way representative), the
    ///   implementation negates the relevant coefficients en masse so
    ///   SLERP traverses the short geodesic. The threshold buffers
    ///   floating-point noise around `w == 0` where both representatives
    ///   are geodesic-equivalent.
    ///
    /// Translator and screw-motor SLERP are deferred to a later release;
    /// passing motors with null-bearing blades produces a rotor-only
    /// approximation.
    pub fn interpolate(&self, target: &Motor<A>, alpha: A::Scalar) -> Motor<A> {
        // Relative rotor R = target * reverse(self). For pure rotors this
        // is again a pure rotor, and log(R) lies in the Euclidean bivector
        // subspace defined by the NULL_MASK gate.
        let self_rev = self.0.reverse();
        let r = target.0.geometric(&self_rev);

        let w = r.get(0);

        // Collect Euclidean bivector coefficients (grade 2, no null factor).
        // We materialise the indices once and reuse them for the negate /
        // norm / writeback passes so the three loops stay in sync without
        // duplicating the gate.
        let mut bivec_indices: [usize; A_MAX_GRADE_2] = [0; A_MAX_GRADE_2];
        let mut bivec_count = 0usize;
        for blade in 0..A::N_BLADES {
            let b = blade as u64;
            if b.count_ones() != 2 {
                continue;
            }
            if (b & A::NULL_MASK) != 0 {
                continue;
            }
            debug_assert!(
                bivec_count < A_MAX_GRADE_2,
                "Motor::interpolate scratch overflow: A::DIM = {} produces more than {} grade-2 \
                 Euclidean blades. Raise A_MAX_GRADE_2 in motor.rs (or split the scratch sizing \
                 per algebra) and update the Limits paragraph on Motor::interpolate.",
                A::DIM,
                A_MAX_GRADE_2,
            );
            bivec_indices[bivec_count] = blade;
            bivec_count += 1;
        }

        // Double-cover shortest-path fix: pick the representative with
        // w >= 0. The threshold (-1e-6) keeps round-off near w == 0 from
        // arbitrarily flipping direction, where both halves of the cover
        // are geodesic-equivalent.
        let eps = <A::Scalar as Scalar>::from_f32(1e-6);
        let neg_threshold = -eps;
        let needs_flip = w < neg_threshold;
        let w = if needs_flip { -w } else { w };

        // Compute |bivec| from the post-flip coefficients.
        let mut bivec_norm_sq = A::Scalar::ZERO;
        for &idx in bivec_indices[..bivec_count].iter() {
            let raw = r.get(idx);
            let coef = if needs_flip { -raw } else { raw };
            bivec_norm_sq = bivec_norm_sq + coef * coef;
        }
        let bivec_norm = bivec_norm_sq.sqrt();

        // If the relative rotor is (numerically) the identity, log is
        // zero and any scaling collapses to the identity rotor; the SLERP
        // result is therefore self.
        if bivec_norm < eps {
            return *self;
        }

        // R = cos(theta/2) + sin(theta/2) * B_unit, so atan2 recovers
        // theta/2 directly. We then scale by alpha and rebuild
        // exp(alpha * log(R)).
        let theta_half = bivec_norm.atan2(w);
        let alpha_theta_half = alpha * theta_half;
        let cos_a = alpha_theta_half.cos();
        let sin_a = alpha_theta_half.sin();
        let scale = sin_a / bivec_norm;

        let mut exp_part: Multivector<A> = Multivector::zero();
        exp_part.set(0, cos_a);
        for &idx in bivec_indices[..bivec_count].iter() {
            let raw = r.get(idx);
            let coef = if needs_flip { -raw } else { raw };
            exp_part.set(idx, coef * scale);
        }

        // Final SLERP'd motor: exp(alpha * log(R)) * self.
        Motor(exp_part.geometric(&self.0))
    }
}

/// Maximum number of grade-2 blades any algebra we ship will need to
/// enumerate inside `Motor::interpolate`. PGA4 (Cl(4, 0, 1), 5 ambient
/// vectors) tops out at C(5, 2) = 10 grade-2 blades, so 16 leaves
/// headroom for hypothetical six-vector algebras (C(6, 2) = 15) without
/// allocating. The constant is private so callers can not depend on it.
const A_MAX_GRADE_2: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pga3;
    use crate::pga3::Pga3;

    /// Build the unit Euclidean bivector representing the world z-axis
    /// rotation plane (`e1 ∧ e2`, bitmask `0b0011`).
    fn z_axis_bivector() -> pga3::Bivector {
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0b0011, 1.0);
        pga3::Bivector(mv)
    }

    fn approx_eq_mv(a: &Multivector<Pga3>, b: &Multivector<Pga3>, eps: f32) -> bool {
        (0..16).all(|k| (a.get(k) - b.get(k)).abs() < eps)
    }

    #[test]
    fn interpolate_at_zero_returns_self() {
        let m_a: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 0.5).0;
        let m_b: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 1.5).0;
        let m = m_a.interpolate(&m_b, 0.0);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let p_via_a = m_a.apply(&test_point.0);
        let p_via_m = m.apply(&test_point.0);
        assert!(approx_eq_mv(&p_via_a, &p_via_m, 1e-5));
    }

    #[test]
    fn interpolate_at_one_returns_target() {
        let m_a: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 0.5).0;
        let m_b: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 1.5).0;
        let m = m_a.interpolate(&m_b, 1.0);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let p_via_b = m_b.apply(&test_point.0);
        let p_via_m = m.apply(&test_point.0);
        assert!(approx_eq_mv(&p_via_b, &p_via_m, 1e-5));
    }

    #[test]
    fn interpolate_z_rotation_at_half_is_quarter_rotation() {
        // Identity to rotor-by-pi-around-z, midpoint = rotor-by-pi/2.
        let m_a: Motor<Pga3> = Motor::identity();
        let m_b: Motor<Pga3> = pga3::rotor(z_axis_bivector(), core::f32::consts::PI).0;
        let m = m_a.interpolate(&m_b, 0.5);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let rotated = pga3::Point(m.apply(&test_point.0));
        let xyz = rotated.to_euclidean();
        // Rotation by pi/2 around z sends (1, 0, 0) to (0, 1, 0).
        assert!(xyz[0].abs() < 1e-4, "x: {}", xyz[0]);
        assert!((xyz[1] - 1.0).abs() < 1e-4, "y: {}", xyz[1]);
        assert!(xyz[2].abs() < 1e-4, "z: {}", xyz[2]);
    }

    #[test]
    fn interpolate_takes_shortest_path_around_double_cover() {
        // A rotor by 1.5*pi around z has raw scalar cos(0.75*pi) < 0, the long
        // way around. The shortest-path fix negates the rotor representative
        // before SLERP, so the midpoint should rotate (1,0,0) by -pi/4 (the
        // short way), landing at (cos(-pi/4), sin(-pi/4), 0). Without the fix
        // the midpoint would land at the long-way (-cos(pi/4), sin(pi/4), 0).
        let m_a: Motor<Pga3> = Motor::identity();
        let m_b: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 1.5 * core::f32::consts::PI).0;
        let m = m_a.interpolate(&m_b, 0.5);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let rotated = pga3::Point(m.apply(&test_point.0));
        let xyz = rotated.to_euclidean();
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!(
            (xyz[0] - inv_sqrt2).abs() < 1e-4,
            "x: {} (long-way bug if negative)",
            xyz[0]
        );
        assert!(
            (xyz[1] + inv_sqrt2).abs() < 1e-4,
            "y: {} (long-way bug if positive)",
            xyz[1]
        );
        assert!(xyz[2].abs() < 1e-4);
    }

    #[test]
    fn interpolate_equal_rotors_is_self() {
        let m_a: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 0.7).0;
        let m_b: Motor<Pga3> = pga3::rotor(z_axis_bivector(), 0.7).0;
        let m = m_a.interpolate(&m_b, 0.3);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let p_via_a = m_a.apply(&test_point.0);
        let p_via_m = m.apply(&test_point.0);
        assert!(approx_eq_mv(&p_via_a, &p_via_m, 1e-5));
    }
}
