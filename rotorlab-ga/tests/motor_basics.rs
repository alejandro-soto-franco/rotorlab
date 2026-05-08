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
