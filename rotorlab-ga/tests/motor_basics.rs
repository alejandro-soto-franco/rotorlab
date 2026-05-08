//! Integration tests for Motor identity and sandwich.

use rotorlab_ga::motor::Motor;
use rotorlab_ga::pga3;

#[test]
fn identity_motor_preserves_point() {
    let m = Motor::identity();
    let p = pga3::point(1.0, 2.0, 3.0);
    let p_moved = m.apply(&p.0);
    for k in 0..16 {
        assert!(
            (p_moved.get(k) - p.0.get(k)).abs() < 1e-6,
            "blade {k}: got {}, expected {}",
            p_moved.get(k),
            p.0.get(k)
        );
    }
}

#[test]
fn identity_motor_compose_is_identity() {
    let i = Motor::identity();
    let i2 = i.compose(&i);
    let p = pga3::point(1.0, 2.0, 3.0);
    let p_moved = i2.apply(&p.0);
    for k in 0..16 {
        assert!((p_moved.get(k) - p.0.get(k)).abs() < 1e-6);
    }
}

#[test]
fn rotor_zero_angle_is_identity() {
    // Bivector e12 (rotation in xy-plane).
    let mut b: rotorlab_ga::multivector::Multivector<rotorlab_ga::pga3::Pga3> =
        rotorlab_ga::multivector::Multivector::zero();
    b.set(0b0011, 1.0);
    let plane = rotorlab_ga::pga3::Bivector(b);
    let r = rotorlab_ga::pga3::rotor(plane, 0.0);
    let i = Motor::identity();
    for k in 0..16 {
        let diff = (r.0.0.get(k) - i.0.get(k)).abs();
        assert!(
            diff < 1e-6,
            "blade {k}: rotor at θ=0 differs from identity by {diff}"
        );
    }
}

#[test]
fn rotor_pi_in_xy_squares_to_identity() {
    // Rotor(e12, π) should square to identity (rotor(e12, 2π) = identity).
    let mut b: rotorlab_ga::multivector::Multivector<rotorlab_ga::pga3::Pga3> =
        rotorlab_ga::multivector::Multivector::zero();
    b.set(0b0011, 1.0);
    let plane = rotorlab_ga::pga3::Bivector(b);
    let r = rotorlab_ga::pga3::rotor(plane, std::f32::consts::PI);
    let r_squared = r.0.compose(&r.0);
    // r_squared rotates by 2π = identity.
    let p = rotorlab_ga::pga3::point(1.0, 0.0, 0.0);
    let p_moved = r_squared.apply(&p.0);
    for k in 0..16 {
        let diff = (p_moved.get(k) - p.0.get(k)).abs();
        assert!(diff < 1e-5, "blade {k} diff = {diff}");
    }
}
