//! Motors, rotors, and translators in PGA3.

use crate::multivector::Multivector;
use crate::pga3::Pga3;

/// A PGA3 motor: an even-grade element representing a rigid motion
/// (rotation + translation) via the sandwich product.
#[derive(Copy, Clone, Default)]
pub struct Motor(pub Multivector<Pga3>);

/// A pure rotation (motor with zero translation component).
#[derive(Copy, Clone, Default)]
pub struct Rotor(pub Motor);

/// A pure translation (motor with zero rotation component).
#[derive(Copy, Clone, Default)]
pub struct Translator(pub Motor);

impl Motor {
    /// The identity motor.
    pub fn identity() -> Self {
        let mut mv: Multivector<Pga3> = Multivector::zero();
        mv.set(0, 1.0);
        Motor(mv)
    }

    /// Apply this motor to a multivector via the sandwich product `M * X * ~M`.
    pub fn apply(&self, x: &Multivector<Pga3>) -> Multivector<Pga3> {
        let m_rev = self.0.reverse();
        let mx = self.0.geometric_pga3(x);
        mx.geometric_pga3(&m_rev)
    }

    /// Compose two motors: `(self ∘ other)(x) = self(other(x))`.
    pub fn compose(&self, other: &Motor) -> Motor {
        Motor(self.0.geometric_pga3(&other.0))
    }

    /// Interpolate between this motor and `target` along the geodesic.
    ///
    /// Plan 3 ships rotor SLERP only: both motors are assumed to be pure
    /// rotors (no translator content on the `e0`-bearing blades). For pure
    /// rotors this is the standard quaternion-style spherical linear
    /// interpolation, packaged as `result = exp(alpha * log(target * ~self)) * self`.
    ///
    /// For a unit rotor `R = w + (b12 e12 + b13 e13 + b23 e23)` with
    /// `w = cos(theta/2)` and `|bivec| = sin(theta/2)`, the log is
    /// `(theta/2) * (bivec / |bivec|)`. Scaling by `alpha` and taking the
    /// exponential gives `cos(alpha * theta/2) + sin(alpha * theta/2) * (bivec / |bivec|)`,
    /// which is exactly the SLERP at parameter `alpha`.
    ///
    /// Endpoint behaviour: `interpolate(target, 0.0)` returns `self`,
    /// `interpolate(target, 1.0)` returns `target`. When `self` and
    /// `target` represent the same rotation (relative bivector norm
    /// below `1e-6`), the result is `self` to avoid a `0/0` in the
    /// `sin/|bivec|` factor.
    ///
    /// Shortest-path: rotors `r` and `-r` represent the same SO(3)
    /// element (the double cover). When the relative rotor's scalar
    /// part is negative (encoding a > pi rotation), the implementation
    /// negates the representative so SLERP traverses the short geodesic
    /// rather than the long one.
    ///
    /// Translator and screw-motor SLERP are deferred to a later release;
    /// passing motors with non-zero `e0`-bearing blades will produce a
    /// rotor-only approximation.
    pub fn interpolate(&self, target: &Motor, alpha: f32) -> Motor {
        // Relative rotor R = target * reverse(self). For pure rotors this is
        // again a pure rotor, and log(R) lies in the Euclidean bivector subspace.
        let self_rev = self.0.reverse();
        let r = target.0.geometric_pga3(&self_rev);

        let w = r.get(0);
        let b12 = r.get(0b0011);
        let b13 = r.get(0b0101);
        let b23 = r.get(0b0110);
        // Double-cover shortest-path fix: rotors r and -r encode the same
        // SO(3) element, but `atan2(|bivec|, w)` returns a half-angle in
        // [0, pi]. Picking the representative with w >= 0 forces SLERP to
        // take the short way around. Without this, a relative rotor whose
        // raw scalar is clearly negative (encoding a > pi rotation)
        // traverses the long way. The threshold (-1e-6) keeps floating
        // point noise around w == 0 from arbitrarily flipping the
        // direction at the half-turn boundary, where both representatives
        // are geodesic-equivalent.
        let (w, b12, b13, b23) = if w < -1e-6 {
            (-w, -b12, -b13, -b23)
        } else {
            (w, b12, b13, b23)
        };
        let bivec_norm = (b12 * b12 + b13 * b13 + b23 * b23).sqrt();

        // If the relative rotor is (numerically) the identity, log is zero
        // and any scaling collapses to the identity rotor; the SLERP result
        // is therefore self.
        if bivec_norm < 1e-6 {
            return *self;
        }

        // R = cos(theta/2) + sin(theta/2) * B_unit, so atan2 recovers
        // theta/2 directly. We then scale that half-angle by alpha and
        // rebuild exp(alpha * log(R)).
        let theta_half = bivec_norm.atan2(w);
        let alpha_theta_half = alpha * theta_half;
        let cos_a = alpha_theta_half.cos();
        let sin_a = alpha_theta_half.sin();
        let scale = sin_a / bivec_norm;

        let mut exp_part: Multivector<Pga3> = Multivector::zero();
        exp_part.set(0, cos_a);
        exp_part.set(0b0011, b12 * scale);
        exp_part.set(0b0101, b13 * scale);
        exp_part.set(0b0110, b23 * scale);

        // Final SLERP'd motor: exp(alpha * log(R)) * self.
        Motor(exp_part.geometric_pga3(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pga3;

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
        let m_a = pga3::rotor(z_axis_bivector(), 0.5).0;
        let m_b = pga3::rotor(z_axis_bivector(), 1.5).0;
        let m = m_a.interpolate(&m_b, 0.0);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let p_via_a = m_a.apply(&test_point.0);
        let p_via_m = m.apply(&test_point.0);
        assert!(approx_eq_mv(&p_via_a, &p_via_m, 1e-5));
    }

    #[test]
    fn interpolate_at_one_returns_target() {
        let m_a = pga3::rotor(z_axis_bivector(), 0.5).0;
        let m_b = pga3::rotor(z_axis_bivector(), 1.5).0;
        let m = m_a.interpolate(&m_b, 1.0);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let p_via_b = m_b.apply(&test_point.0);
        let p_via_m = m.apply(&test_point.0);
        assert!(approx_eq_mv(&p_via_b, &p_via_m, 1e-5));
    }

    #[test]
    fn interpolate_z_rotation_at_half_is_quarter_rotation() {
        // Identity to rotor-by-pi-around-z, midpoint = rotor-by-pi/2.
        let m_a = Motor::identity();
        let m_b = pga3::rotor(z_axis_bivector(), core::f32::consts::PI).0;
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
        let m_a = Motor::identity();
        let m_b = pga3::rotor(z_axis_bivector(), 1.5 * core::f32::consts::PI).0;
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
        let m_a = pga3::rotor(z_axis_bivector(), 0.7).0;
        let m_b = pga3::rotor(z_axis_bivector(), 0.7).0;
        let m = m_a.interpolate(&m_b, 0.3);
        let test_point = pga3::point(1.0, 0.0, 0.0);
        let p_via_a = m_a.apply(&test_point.0);
        let p_via_m = m.apply(&test_point.0);
        assert!(approx_eq_mv(&p_via_a, &p_via_m, 1e-5));
    }
}
