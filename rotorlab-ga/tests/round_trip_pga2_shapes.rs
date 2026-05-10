//! Stage 11 integration tests for the [`rotorlab_ga::pga2::bridge`]
//! layer.
//!
//! Three properties are checked for each of the six PGA2 shape structs
//! (`Point`, `Line`, `Bivector`, `Rotor`, `Translator`, `Motor`):
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
//! There is no Plane test row: PGA2 has no Plane shape struct.
//!
//! Random instances are produced by a local linear-congruential
//! generator with a fixed seed. This keeps the tests deterministic
//! without adding `rand` as a dev-dependency.

use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga2::Pga2;
use rotorlab_ga::pga2::bridge::BridgeError;
use rotorlab_ga::pga2::shapes;

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
            e_12: rng.next_f32(10.0),
            e_02: rng.next_f32(10.0),
            e_01: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga2> = original.into();
        let back: shapes::Point = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_01, original.e_01);
    }
}

#[test]
fn point_rejects_scalar_perturbation() {
    let p = shapes::Point {
        e_12: 1.0,
        e_02: 2.0,
        e_01: 3.0,
    };
    let mut mv: Multivector<Pga2> = p.into();
    // Scalar blade 0b000 is outside Point's support.
    mv.set(0b000, 0.5);
    let result: Result<shapes::Point, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b000),
        other => panic!("expected OutOfShape on scalar blade, got {other:?}"),
    }
}

#[test]
fn point_zero_round_trips() {
    let zero = shapes::Point::default();
    let mv: Multivector<Pga2> = zero.into();
    let back: shapes::Point = mv.try_into().expect("zero point round trip");
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_01, 0.0);
}

// ---------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------

#[test]
fn line_round_trip_random() {
    let mut rng = Lcg::new(0xFACADE);
    for _ in 0..SAMPLES {
        let original = shapes::Line {
            e_1: rng.next_f32(10.0),
            e_2: rng.next_f32(10.0),
            e_0: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga2> = original.into();
        let back: shapes::Line = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_1, original.e_1);
        assert_eq!(back.e_2, original.e_2);
        assert_eq!(back.e_0, original.e_0);
    }
}

#[test]
fn line_rejects_pseudoscalar_perturbation() {
    let l = shapes::Line {
        e_1: 1.0,
        e_2: 2.0,
        e_0: 3.0,
    };
    let mut mv: Multivector<Pga2> = l.into();
    // Pseudoscalar blade 0b111 is outside Line's support.
    mv.set(0b111, 0.5);
    let result: Result<shapes::Line, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b111),
        other => panic!("expected OutOfShape on pseudoscalar blade, got {other:?}"),
    }
}

#[test]
fn line_zero_round_trips() {
    let zero = shapes::Line::default();
    let mv: Multivector<Pga2> = zero.into();
    let back: shapes::Line = mv.try_into().expect("zero line round trip");
    assert_eq!(back.e_1, 0.0);
    assert_eq!(back.e_2, 0.0);
    assert_eq!(back.e_0, 0.0);
}

// ---------------------------------------------------------------------
// Bivector
// ---------------------------------------------------------------------

#[test]
fn bivector_round_trip_random() {
    let mut rng = Lcg::new(0xBADBED);
    for _ in 0..SAMPLES {
        let original = shapes::Bivector {
            e_12: rng.next_f32(10.0),
            e_02: rng.next_f32(10.0),
            e_01: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga2> = original.into();
        let back: shapes::Bivector = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_01, original.e_01);
    }
}

#[test]
fn bivector_rejects_pseudoscalar_perturbation() {
    let b = shapes::Bivector {
        e_12: 1.0,
        e_02: 2.0,
        e_01: 3.0,
    };
    let mut mv: Multivector<Pga2> = b.into();
    // Pseudoscalar blade 0b111 is outside Bivector's support.
    mv.set(0b111, 0.5);
    let result: Result<shapes::Bivector, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b111),
        other => panic!("expected OutOfShape on pseudoscalar blade, got {other:?}"),
    }
}

#[test]
fn bivector_zero_round_trips() {
    let zero = shapes::Bivector::default();
    let mv: Multivector<Pga2> = zero.into();
    let back: shapes::Bivector = mv.try_into().expect("zero bivector round trip");
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_01, 0.0);
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
        };
        let mv: Multivector<Pga2> = original.into();
        let back: shapes::Rotor = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.s, original.s);
        assert_eq!(back.e_12, original.e_12);
    }
}

#[test]
fn rotor_rejects_null_bivector_perturbation() {
    let r = shapes::Rotor { s: 1.0, e_12: 0.5 };
    let mut mv: Multivector<Pga2> = r.into();
    // Null bivector blade 0b110 (e_02) is outside Rotor's support.
    mv.set(0b110, 0.5);
    let result: Result<shapes::Rotor, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b110),
        other => panic!("expected OutOfShape on e_02 blade, got {other:?}"),
    }
}

#[test]
fn rotor_zero_round_trips() {
    let zero = shapes::Rotor::default();
    let mv: Multivector<Pga2> = zero.into();
    let back: shapes::Rotor = mv.try_into().expect("zero rotor round trip");
    assert_eq!(back.s, 0.0);
    assert_eq!(back.e_12, 0.0);
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
            e_02: rng.next_f32(10.0),
            e_01: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga2> = original.into();
        let back: shapes::Translator = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.s, original.s);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_01, original.e_01);
    }
}

#[test]
fn translator_rejects_euclidean_bivector_perturbation() {
    let t = shapes::Translator {
        s: 1.0,
        e_02: 0.5,
        e_01: 0.25,
    };
    let mut mv: Multivector<Pga2> = t.into();
    // Euclidean bivector blade 0b011 (e_12) is outside Translator's support.
    mv.set(0b011, 0.5);
    let result: Result<shapes::Translator, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b011),
        other => panic!("expected OutOfShape on e_12 blade, got {other:?}"),
    }
}

#[test]
fn translator_zero_round_trips() {
    let zero = shapes::Translator::default();
    let mv: Multivector<Pga2> = zero.into();
    let back: shapes::Translator = mv.try_into().expect("zero translator round trip");
    assert_eq!(back.s, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_01, 0.0);
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
            e_02: rng.next_f32(10.0),
            e_01: rng.next_f32(10.0),
        };
        let mv: Multivector<Pga2> = original.into();
        let back: shapes::Motor = mv.try_into().expect("round trip should succeed");
        assert_eq!(back.s, original.s);
        assert_eq!(back.e_12, original.e_12);
        assert_eq!(back.e_02, original.e_02);
        assert_eq!(back.e_01, original.e_01);
    }
}

#[test]
fn motor_rejects_vector_perturbation() {
    let m = shapes::Motor {
        s: 1.0,
        e_12: 0.5,
        e_02: 0.25,
        e_01: 0.125,
    };
    let mut mv: Multivector<Pga2> = m.into();
    // Grade-1 vector blade 0b001 (e_1) is outside Motor's support
    // (Motor is even-graded).
    mv.set(0b001, 0.5);
    let result: Result<shapes::Motor, BridgeError> = mv.try_into();
    match result {
        Err(BridgeError::OutOfShape { blade, .. }) => assert_eq!(blade, 0b001),
        other => panic!("expected OutOfShape on e_1 blade, got {other:?}"),
    }
}

#[test]
fn motor_zero_round_trips() {
    let zero = shapes::Motor::default();
    let mv: Multivector<Pga2> = zero.into();
    let back: shapes::Motor = mv.try_into().expect("zero motor round trip");
    assert_eq!(back.s, 0.0);
    assert_eq!(back.e_12, 0.0);
    assert_eq!(back.e_02, 0.0);
    assert_eq!(back.e_01, 0.0);
}
