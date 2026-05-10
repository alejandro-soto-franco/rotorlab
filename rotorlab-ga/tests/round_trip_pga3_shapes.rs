//! Stage 8 integration tests for the [`rotorlab_ga::pga3::bridge`]
//! layer.
//!
//! Three properties are checked for each of the seven PGA3 shape
//! structs (`Point`, `Plane`, `Line`, `Bivector`, `Rotor`, `Translator`,
//! `Motor`):
//!
//! 1. Round-trip: `Shape -> Multivector -> Shape` reconstructs the
//!    original on every field, with exact `f32` equality (the bridge
//!    is a pure shuffle of blade slots; no arithmetic enters the
//!    round-trip path).
//! 2. Out-of-shape rejection: poking a blade outside the shape's
//!    support and round-tripping yields `Err(BridgeError::OutOfShape
//!    { blade, .. })` naming the perturbed blade.
//! 3. Zero round-trip: `Shape::default() -> Multivector -> Shape`
//!    equals `Shape::default()`.
//!
//! Random instances are produced by a local linear-congruential
//! generator with a fixed seed. This keeps the tests deterministic
//! without adding `rand` as a dev-dependency.

use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga3::Pga3;
use rotorlab_ga::pga3::bridge::BridgeError;
use rotorlab_ga::pga3::shapes;

/// Numbers of random samples to draw per round-trip test.
const SAMPLES: usize = 64;

/// A deterministic LCG suitable for generating reproducible `f32`
/// samples in tests. Numerical recipes constants; period 2^64.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Sample an `f32` in `[-bound, bound)`.
    fn next_f32(&mut self, bound: f32) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 high-quality bits.
        let unit = (bits as f32) / ((1u32 << 24) as f32); // [0, 1)
        (unit * 2.0 - 1.0) * bound
    }
}

// ---------------------------------------------------------------------
// Point
// ---------------------------------------------------------------------

#[test]
fn point_round_trip_random() {
    let mut rng = Lcg::new(0xC0FFEE);
    for _ in 0..SAMPLES {
        let original = shapes::Point {
            e_023: rng.next_f32(10.0),
            e_013: rng.next_f32(10.0),
            e_012: rng.next_f32(10.0),
            e_123: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Point = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_023, original.e_023);
        assert_eq!(back.e_013, original.e_013);
        assert_eq!(back.e_012, original.e_012);
        assert_eq!(back.e_123, original.e_123);
    }
}

#[test]
fn point_rejects_scalar_perturbation() {
    let p = shapes::Point {
        e_023: 1.0,
        e_013: 2.0,
        e_012: 3.0,
        e_123: 4.0,
    };
    let mut mv: Multivector<Pga3> = p.into();
    // Scalar blade 0b0000 is outside Point's support.
    mv.set(0b0000, 0.5);
    let result: Result<shapes::Point, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b0000),
        other => panic!("expected OutOfShape on scalar blade, got {other:?}"),
    }
}

#[test]
fn point_zero_round_trips() {
    let zero = shapes::Point::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Point = mv.try_into().expect("zero point round trip");
    assert_eq!(back.e_023, 0.0);
    assert_eq!(back.e_013, 0.0);
    assert_eq!(back.e_012, 0.0);
    assert_eq!(back.e_123, 0.0);
}

// ---------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------

#[test]
fn plane_round_trip_random() {
    let mut rng = Lcg::new(0xDECAF);
    for _ in 0..SAMPLES {
        let original = shapes::Plane {
            e_0: rng.next_f32(10.0),
            e_1: rng.next_f32(10.0),
            e_2: rng.next_f32(10.0),
            e_3: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Plane = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_0, original.e_0);
        assert_eq!(back.e_1, original.e_1);
        assert_eq!(back.e_2, original.e_2);
        assert_eq!(back.e_3, original.e_3);
    }
}

#[test]
fn plane_rejects_pseudoscalar_perturbation() {
    let p = shapes::Plane {
        e_0: 1.0,
        e_1: 2.0,
        e_2: 3.0,
        e_3: 4.0,
    };
    let mut mv: Multivector<Pga3> = p.into();
    // Pseudoscalar blade 0b1111 is outside Plane's support.
    mv.set(0b1111, 0.5);
    let result: Result<shapes::Plane, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b1111),
        other => panic!("expected OutOfShape on pseudoscalar blade, got {other:?}"),
    }
}

#[test]
fn plane_zero_round_trips() {
    let zero = shapes::Plane::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Plane = mv.try_into().expect("zero plane round trip");
    assert_eq!(back.e_0, 0.0);
    assert_eq!(back.e_1, 0.0);
    assert_eq!(back.e_2, 0.0);
    assert_eq!(back.e_3, 0.0);
}

// ---------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------

#[test]
fn line_round_trip_random() {
    let mut rng = Lcg::new(0xFACADE);
    for _ in 0..SAMPLES {
        let original = shapes::Line {
            e_01: rng.next_f32(10.0),
            e_02: rng.next_f32(10.0),
            e_03: rng.next_f32(10.0),
            e_12: rng.next_f32(10.0),
            e_13: rng.next_f32(10.0),
            e_23: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Line = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_01, original.e_01);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_03, original.e_03);
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_13, original.e_13);
        assert_eq!(back.e_23, original.e_23);
    }
}

#[test]
fn line_rejects_scalar_perturbation() {
    let l = shapes::Line {
        e_01: 1.0,
        e_02: 2.0,
        e_03: 3.0,
        e_12: 4.0,
        e_13: 5.0,
        e_23: 6.0,
    };
    let mut mv: Multivector<Pga3> = l.into();
    // Scalar blade 0b0000 is outside Line's support.
    mv.set(0b0000, 0.5);
    let result: Result<shapes::Line, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b0000),
        other => panic!("expected OutOfShape on scalar blade, got {other:?}"),
    }
}

#[test]
fn line_zero_round_trips() {
    let zero = shapes::Line::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Line = mv.try_into().expect("zero line round trip");
    assert_eq!(back.e_01, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_03, 0.0);
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_13, 0.0);
    assert_eq!(back.e_23, 0.0);
}

// ---------------------------------------------------------------------
// Bivector
// ---------------------------------------------------------------------

#[test]
fn bivector_round_trip_random() {
    let mut rng = Lcg::new(0xBADBED);
    for _ in 0..SAMPLES {
        let original = shapes::Bivector {
            e_01: rng.next_f32(10.0),
            e_02: rng.next_f32(10.0),
            e_03: rng.next_f32(10.0),
            e_12: rng.next_f32(10.0),
            e_13: rng.next_f32(10.0),
            e_23: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Bivector = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_01, original.e_01);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_03, original.e_03);
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_13, original.e_13);
        assert_eq!(back.e_23, original.e_23);
    }
}

#[test]
fn bivector_rejects_pseudoscalar_perturbation() {
    let b = shapes::Bivector {
        e_01: 1.0,
        e_02: 2.0,
        e_03: 3.0,
        e_12: 4.0,
        e_13: 5.0,
        e_23: 6.0,
    };
    let mut mv: Multivector<Pga3> = b.into();
    // Pseudoscalar blade 0b1111 is outside Bivector's support.
    mv.set(0b1111, 0.5);
    let result: Result<shapes::Bivector, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b1111),
        other => panic!("expected OutOfShape on pseudoscalar blade, got {other:?}"),
    }
}

#[test]
fn bivector_zero_round_trips() {
    let zero = shapes::Bivector::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Bivector = mv.try_into().expect("zero bivector round trip");
    assert_eq!(back.e_01, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_03, 0.0);
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_13, 0.0);
    assert_eq!(back.e_23, 0.0);
}

// ---------------------------------------------------------------------
// Rotor
// ---------------------------------------------------------------------

#[test]
fn rotor_round_trip_random() {
    let mut rng = Lcg::new(0xABCDEF);
    for _ in 0..SAMPLES {
        let original = shapes::Rotor {
            s: rng.next_f32(10.0),
            e_12: rng.next_f32(10.0),
            e_13: rng.next_f32(10.0),
            e_23: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Rotor = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.s, original.s);
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_13, original.e_13);
        assert_eq!(back.e_23, original.e_23);
    }
}

#[test]
fn rotor_rejects_null_bivector_perturbation() {
    let r = shapes::Rotor {
        s: 1.0,
        e_12: 0.5,
        e_13: 0.25,
        e_23: 0.125,
    };
    let mut mv: Multivector<Pga3> = r.into();
    // Null bivector blade 0b1001 (e_01) is outside Rotor's support.
    mv.set(0b1001, 0.5);
    let result: Result<shapes::Rotor, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b1001),
        other => panic!("expected OutOfShape on e_01 blade, got {other:?}"),
    }
}

#[test]
fn rotor_zero_round_trips() {
    let zero = shapes::Rotor::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Rotor = mv.try_into().expect("zero rotor round trip");
    assert_eq!(back.s, 0.0);
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_13, 0.0);
    assert_eq!(back.e_23, 0.0);
}

// ---------------------------------------------------------------------
// Translator
// ---------------------------------------------------------------------

#[test]
fn translator_round_trip_random() {
    let mut rng = Lcg::new(0x123456);
    for _ in 0..SAMPLES {
        let original = shapes::Translator {
            s: rng.next_f32(10.0),
            e_01: rng.next_f32(10.0),
            e_02: rng.next_f32(10.0),
            e_03: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Translator = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.s, original.s);
        assert_eq!(back.e_01, original.e_01);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_03, original.e_03);
    }
}

#[test]
fn translator_rejects_euclidean_bivector_perturbation() {
    let t = shapes::Translator {
        s: 1.0,
        e_01: 0.5,
        e_02: 0.25,
        e_03: 0.125,
    };
    let mut mv: Multivector<Pga3> = t.into();
    // Euclidean bivector blade 0b0011 (e_12) is outside Translator's support.
    mv.set(0b0011, 0.5);
    let result: Result<shapes::Translator, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b0011),
        other => panic!("expected OutOfShape on e_12 blade, got {other:?}"),
    }
}

#[test]
fn translator_zero_round_trips() {
    let zero = shapes::Translator::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Translator = mv.try_into().expect("zero translator round trip");
    assert_eq!(back.s, 0.0);
    assert_eq!(back.e_01, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_03, 0.0);
}

// ---------------------------------------------------------------------
// Motor
// ---------------------------------------------------------------------

#[test]
fn motor_round_trip_random() {
    let mut rng = Lcg::new(0x789ABC);
    for _ in 0..SAMPLES {
        let original = shapes::Motor {
            s: rng.next_f32(10.0),
            e_12: rng.next_f32(10.0),
            e_13: rng.next_f32(10.0),
            e_23: rng.next_f32(10.0),
            e_01: rng.next_f32(10.0),
            e_02: rng.next_f32(10.0),
            e_03: rng.next_f32(10.0),
            e_0123: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga3> = original.into();
        let back: shapes::Motor = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.s, original.s);
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_13, original.e_13);
        assert_eq!(back.e_23, original.e_23);
        assert_eq!(back.e_01, original.e_01);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_03, original.e_03);
        assert_eq!(back.e_0123, original.e_0123);
    }
}

#[test]
fn motor_rejects_vector_perturbation() {
    let m = shapes::Motor {
        s: 1.0,
        e_12: 0.5,
        e_13: 0.25,
        e_23: 0.125,
        e_01: 0.0625,
        e_02: 0.03125,
        e_03: 0.015625,
        e_0123: 0.0078125,
    };
    let mut mv: Multivector<Pga3> = m.into();
    // Grade-1 vector blade 0b0001 (e_1) is outside Motor's support
    // (Motor is even-graded).
    mv.set(0b0001, 0.5);
    let result: Result<shapes::Motor, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b0001),
        other => panic!("expected OutOfShape on e_1 blade, got {other:?}"),
    }
}

#[test]
fn motor_zero_round_trips() {
    let zero = shapes::Motor::default();
    let mv: Multivector<Pga3> = zero.into();
    let back: shapes::Motor = mv.try_into().expect("zero motor round trip");
    assert_eq!(back.s, 0.0);
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_13, 0.0);
    assert_eq!(back.e_23, 0.0);
    assert_eq!(back.e_01, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_03, 0.0);
    assert_eq!(back.e_0123, 0.0);
}
