//! Stage 9 parity tests for [`rotorlab_ga::pga3::specialised`].
//!
//! Each specialised method on the [`shapes::Motor`] surface is paired
//! with the dense oracle (the [`crate::motor::Motor<Pga3>`] sandwich /
//! compose / interpolate path) and asserted to agree component-wise on
//! random inputs.
//!
//! The shape-struct API is currently a thin delegated wrapper, so on
//! today's code the parity is exact modulo the float roundoff that
//! creeps in when the dense result carries sub-tolerance values on
//! out-of-shape blades. A future optimisation pass that replaces each
//! delegated body with a hand-derived closed-form expression will
//! continue to be guarded by these parity assertions.
//!
//! Random instances are produced by a local linear-congruential
//! generator with a fixed seed so the suite stays deterministic without
//! adding `rand` as a dev-dependency.

use rotorlab_ga::motor::Motor as DenseMotor;
use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga3::Pga3;
use rotorlab_ga::pga3::shapes;

/// Number of random samples to draw per parity test.
const SAMPLES: usize = 32;

/// Absolute tolerance for sandwich and compose parity checks.
const PARITY_TOL: f32 = 1e-5;

/// Absolute tolerance for the interpolate parity check; SLERP introduces
/// transcendental round-off that the simple sandwich does not.
const INTERPOLATE_TOL: f32 = 1e-4;

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
        let bits = (self.next_u64() >> 40) as u32;
        let unit = (bits as f32) / ((1u32 << 24) as f32);
        (unit * 2.0 - 1.0) * bound
    }
}

/// Build a normalised PGA3 motor from random angle / axis / translation
/// samples drawn from `rng`. The result is a unit motor (rotor composed
/// with translator) so the sandwich product preserves projective shape
/// classes within the bridge tolerance.
fn random_motor(rng: &mut Lcg) -> shapes::Motor {
    // Random unit Euclidean bivector axis for the rotor part.
    let mut bx = rng.next_f32(1.0);
    let mut by = rng.next_f32(1.0);
    let mut bz = rng.next_f32(1.0);
    let bnorm = (bx * bx + by * by + bz * bz).sqrt().max(1e-6);
    bx /= bnorm;
    by /= bnorm;
    bz /= bnorm;

    let angle = rng.next_f32(core::f32::consts::PI);
    let half = angle * 0.5;
    let c = half.cos();
    let s = half.sin();

    // rotor: cos(half) + sin(half) * (bx e_12 + by e_13 + bz e_23)
    let rotor = shapes::Motor {
        s: c,
        e_12: s * bx,
        e_13: s * by,
        e_23: s * bz,
        e_01: 0.0,
        e_02: 0.0,
        e_03: 0.0,
        e_0123: 0.0,
    };

    // translator: 1 + (tx/2) e_01 + (ty/2) e_02 + (tz/2) e_03
    let tx = rng.next_f32(3.0);
    let ty = rng.next_f32(3.0);
    let tz = rng.next_f32(3.0);
    let translator = shapes::Motor {
        s: 1.0,
        e_12: 0.0,
        e_13: 0.0,
        e_23: 0.0,
        e_01: 0.5 * tx,
        e_02: 0.5 * ty,
        e_03: 0.5 * tz,
        e_0123: 0.0,
    };

    // Compose translator * rotor through the dense oracle.
    let t_dense: Multivector<Pga3> = translator.into();
    let r_dense: Multivector<Pga3> = rotor.into();
    let composed = DenseMotor(t_dense).compose(&DenseMotor(r_dense));
    shapes::Motor {
        s: composed.0.get(0b0000),
        e_12: composed.0.get(0b0011),
        e_13: composed.0.get(0b0101),
        e_23: composed.0.get(0b0110),
        e_01: composed.0.get(0b1001),
        e_02: composed.0.get(0b1010),
        e_03: composed.0.get(0b1100),
        e_0123: composed.0.get(0b1111),
    }
}

fn random_point(rng: &mut Lcg) -> shapes::Point {
    shapes::Point {
        e_023: rng.next_f32(5.0),
        e_013: rng.next_f32(5.0),
        e_012: rng.next_f32(5.0),
        e_123: 1.0,
    }
}

fn random_line(rng: &mut Lcg) -> shapes::Line {
    shapes::Line {
        e_01: rng.next_f32(2.0),
        e_02: rng.next_f32(2.0),
        e_03: rng.next_f32(2.0),
        e_12: rng.next_f32(2.0),
        e_13: rng.next_f32(2.0),
        e_23: rng.next_f32(2.0),
    }
}

fn random_plane(rng: &mut Lcg) -> shapes::Plane {
    shapes::Plane {
        e_0: rng.next_f32(2.0),
        e_1: rng.next_f32(2.0),
        e_2: rng.next_f32(2.0),
        e_3: rng.next_f32(2.0),
    }
}

fn dense_apply_to_point(m: &shapes::Motor, p: &shapes::Point) -> shapes::Point {
    let m_dense: Multivector<Pga3> = (*m).into();
    let p_dense: Multivector<Pga3> = (*p).into();
    let result = DenseMotor(m_dense).apply(&p_dense);
    shapes::Point {
        e_023: result.get(0b1110),
        e_013: result.get(0b1101),
        e_012: result.get(0b1011),
        e_123: result.get(0b0111),
    }
}

fn dense_apply_to_line(m: &shapes::Motor, l: &shapes::Line) -> shapes::Line {
    let m_dense: Multivector<Pga3> = (*m).into();
    let l_dense: Multivector<Pga3> = (*l).into();
    let result = DenseMotor(m_dense).apply(&l_dense);
    shapes::Line {
        e_01: result.get(0b1001),
        e_02: result.get(0b1010),
        e_03: result.get(0b1100),
        e_12: result.get(0b0011),
        e_13: result.get(0b0101),
        e_23: result.get(0b0110),
    }
}

fn dense_apply_to_plane(m: &shapes::Motor, pl: &shapes::Plane) -> shapes::Plane {
    let m_dense: Multivector<Pga3> = (*m).into();
    let pl_dense: Multivector<Pga3> = (*pl).into();
    let result = DenseMotor(m_dense).apply(&pl_dense);
    shapes::Plane {
        e_0: result.get(0b1000),
        e_1: result.get(0b0001),
        e_2: result.get(0b0010),
        e_3: result.get(0b0100),
    }
}

fn dense_compose(a: &shapes::Motor, b: &shapes::Motor) -> shapes::Motor {
    let a_dense: Multivector<Pga3> = (*a).into();
    let b_dense: Multivector<Pga3> = (*b).into();
    let result = DenseMotor(a_dense).compose(&DenseMotor(b_dense));
    shapes::Motor {
        s: result.0.get(0b0000),
        e_12: result.0.get(0b0011),
        e_13: result.0.get(0b0101),
        e_23: result.0.get(0b0110),
        e_01: result.0.get(0b1001),
        e_02: result.0.get(0b1010),
        e_03: result.0.get(0b1100),
        e_0123: result.0.get(0b1111),
    }
}

fn dense_interpolate(a: &shapes::Motor, b: &shapes::Motor, alpha: f32) -> shapes::Motor {
    let a_dense: Multivector<Pga3> = (*a).into();
    let b_dense: Multivector<Pga3> = (*b).into();
    let result = DenseMotor(a_dense).interpolate(&DenseMotor(b_dense), alpha);
    shapes::Motor {
        s: result.0.get(0b0000),
        e_12: result.0.get(0b0011),
        e_13: result.0.get(0b0101),
        e_23: result.0.get(0b0110),
        e_01: result.0.get(0b1001),
        e_02: result.0.get(0b1010),
        e_03: result.0.get(0b1100),
        e_0123: result.0.get(0b1111),
    }
}

#[test]
fn apply_to_point_matches_dense() {
    let mut rng = Lcg::new(0xA9_5E_C1_A1_15_ED_00_01);
    for sample in 0..SAMPLES {
        let m = random_motor(&mut rng);
        let p = random_point(&mut rng);
        let specialised = m.apply_to_point(&p);
        let dense = dense_apply_to_point(&m, &p);
        let fields = [
            ("e_023", specialised.e_023, dense.e_023),
            ("e_013", specialised.e_013, dense.e_013),
            ("e_012", specialised.e_012, dense.e_012),
            ("e_123", specialised.e_123, dense.e_123),
        ];
        for (name, sp, dn) in fields {
            assert!(
                (sp - dn).abs() < PARITY_TOL,
                "sample {sample} field {name}: specialised {sp} vs dense {dn} \
                 (diff {}); motor = {m:?}; point = {p:?}",
                (sp - dn).abs(),
            );
        }
    }
}

#[test]
fn apply_to_line_matches_dense() {
    let mut rng = Lcg::new(0xA9_5E_C1_A1_15_ED_00_02);
    for sample in 0..SAMPLES {
        let m = random_motor(&mut rng);
        let l = random_line(&mut rng);
        let specialised = m.apply_to_line(&l);
        let dense = dense_apply_to_line(&m, &l);
        let fields = [
            ("e_01", specialised.e_01, dense.e_01),
            ("e_02", specialised.e_02, dense.e_02),
            ("e_03", specialised.e_03, dense.e_03),
            ("e_12", specialised.e_12, dense.e_12),
            ("e_13", specialised.e_13, dense.e_13),
            ("e_23", specialised.e_23, dense.e_23),
        ];
        for (name, sp, dn) in fields {
            assert!(
                (sp - dn).abs() < PARITY_TOL,
                "sample {sample} field {name}: specialised {sp} vs dense {dn} \
                 (diff {}); motor = {m:?}; line = {l:?}",
                (sp - dn).abs(),
            );
        }
    }
}

#[test]
fn apply_to_plane_matches_dense() {
    let mut rng = Lcg::new(0xA9_5E_C1_A1_15_ED_00_03);
    for sample in 0..SAMPLES {
        let m = random_motor(&mut rng);
        let pl = random_plane(&mut rng);
        let specialised = m.apply_to_plane(&pl);
        let dense = dense_apply_to_plane(&m, &pl);
        let fields = [
            ("e_0", specialised.e_0, dense.e_0),
            ("e_1", specialised.e_1, dense.e_1),
            ("e_2", specialised.e_2, dense.e_2),
            ("e_3", specialised.e_3, dense.e_3),
        ];
        for (name, sp, dn) in fields {
            assert!(
                (sp - dn).abs() < PARITY_TOL,
                "sample {sample} field {name}: specialised {sp} vs dense {dn} \
                 (diff {}); motor = {m:?}; plane = {pl:?}",
                (sp - dn).abs(),
            );
        }
    }
}

#[test]
fn compose_matches_dense() {
    let mut rng = Lcg::new(0xA9_5E_C1_A1_15_ED_00_04);
    for sample in 0..SAMPLES {
        let a = random_motor(&mut rng);
        let b = random_motor(&mut rng);
        let specialised = a.compose(&b);
        let dense = dense_compose(&a, &b);
        let fields = [
            ("s", specialised.s, dense.s),
            ("e_12", specialised.e_12, dense.e_12),
            ("e_13", specialised.e_13, dense.e_13),
            ("e_23", specialised.e_23, dense.e_23),
            ("e_01", specialised.e_01, dense.e_01),
            ("e_02", specialised.e_02, dense.e_02),
            ("e_03", specialised.e_03, dense.e_03),
            ("e_0123", specialised.e_0123, dense.e_0123),
        ];
        for (name, sp, dn) in fields {
            assert!(
                (sp - dn).abs() < PARITY_TOL,
                "sample {sample} field {name}: specialised {sp} vs dense {dn} \
                 (diff {}); a = {a:?}; b = {b:?}",
                (sp - dn).abs(),
            );
        }
    }
}

#[test]
fn interpolate_matches_dense() {
    let mut rng = Lcg::new(0xA9_5E_C1_A1_15_ED_00_05);
    for sample in 0..SAMPLES {
        let a = random_motor(&mut rng);
        let b = random_motor(&mut rng);
        let alpha_bits = (rng.next_u64() >> 40) as u32;
        let alpha = (alpha_bits as f32) / ((1u32 << 24) as f32);
        let specialised = a.interpolate(&b, alpha);
        let dense = dense_interpolate(&a, &b, alpha);
        let fields = [
            ("s", specialised.s, dense.s),
            ("e_12", specialised.e_12, dense.e_12),
            ("e_13", specialised.e_13, dense.e_13),
            ("e_23", specialised.e_23, dense.e_23),
            ("e_01", specialised.e_01, dense.e_01),
            ("e_02", specialised.e_02, dense.e_02),
            ("e_03", specialised.e_03, dense.e_03),
            ("e_0123", specialised.e_0123, dense.e_0123),
        ];
        for (name, sp, dn) in fields {
            assert!(
                (sp - dn).abs() < INTERPOLATE_TOL,
                "sample {sample} field {name}: specialised {sp} vs dense {dn} \
                 (diff {}); a = {a:?}; b = {b:?}; alpha = {alpha}",
                (sp - dn).abs(),
            );
        }
    }
}
