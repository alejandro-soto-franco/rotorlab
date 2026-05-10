//! Stage 3 acceptance: `Motor::interpolate` enumerates Euclidean bivector
//! blades using `Algebra::NULL_MASK` rather than the hard-coded PGA3
//! bitmasks. This file pins that behaviour against a synthetic Cl(3, 0, 0)
//! algebra (no null vector, three Euclidean bivector planes).
//!
//! The stub algebra `Cl3` is defined inline so the test does not bleed
//! non-PGA3 surface into the public crate root. It is sufficient for SLERP
//! because rotor SLERP only touches the scalar and grade-2 blades.

use rotorlab_ga::algebra::Algebra;
use rotorlab_ga::blade::{blade_product_blade, blade_product_sign};
use rotorlab_ga::motor::Motor;
use rotorlab_ga::multivector::Multivector;

/// Cl(3, 0, 0): three Euclidean basis vectors `e1`, `e2`, `e3`, all
/// squaring to `+1`. Eight blades, no null component.
#[derive(Copy, Clone, Default, Debug)]
struct Cl3;

/// Compile-time row-major Cayley table for `Cl3`. Built the same way as
/// the PGA3 table but over a flat 8 x 8 grid.
const CL3_METRIC: [i8; 3] = [1, 1, 1];

const fn cl3_cayley_flat() -> [(i8, u64); 64] {
    let mut flat = [(0i8, 0u64); 64];
    let mut i = 0usize;
    while i < 8 {
        let mut j = 0usize;
        while j < 8 {
            let sign = blade_product_sign(i as u64, j as u64, &CL3_METRIC);
            let blade = blade_product_blade(i as u64, j as u64);
            flat[i * 8 + j] = (sign, blade);
            j += 1;
        }
        i += 1;
    }
    flat
}

const CL3_CAYLEY_FLAT: [(i8, u64); 64] = cl3_cayley_flat();

impl Algebra for Cl3 {
    const SIGNATURE: (u32, u32, u32) = (3, 0, 0);
    const DIM: u32 = 3;
    const N_BLADES: usize = 8;
    type Storage = [f32; 8];
    type Scalar = f32;
    const METRIC: &'static [i8] = &CL3_METRIC;
    const CAYLEY: &'static [(i8, u64)] = &CL3_CAYLEY_FLAT;
    /// No basis vector is null in Cl(3, 0, 0).
    const NULL_MASK: u64 = 0;
}

/// Blade indices for the three Euclidean bivector planes in Cl3.
const E12: usize = 0b011;
const E13: usize = 0b101;
const E23: usize = 0b110;

/// Build a unit rotor in Cl3 that rotates by `angle` radians in plane
/// `e_i ∧ e_j` (passed as a blade bitmask). Mirrors the PGA3 `rotor`
/// constructor but specialised to Cl3 grade-2 blades.
fn cl3_rotor(plane: usize, angle: f32) -> Motor<Cl3> {
    let half = angle * 0.5;
    let mut mv: Multivector<Cl3> = Multivector::zero();
    mv.set(0, half.cos());
    mv.set(plane, half.sin());
    Motor(mv)
}

/// Apply a unit Cl3 rotor to a grade-1 vector via the sandwich product
/// and read out the three Euclidean components `(x, y, z)`.
fn rotate_vec(m: &Motor<Cl3>, x: f32, y: f32, z: f32) -> [f32; 3] {
    let mut v: Multivector<Cl3> = Multivector::zero();
    v.set(0b001, x);
    v.set(0b010, y);
    v.set(0b100, z);
    let out = m.apply(&v);
    [out.get(0b001), out.get(0b010), out.get(0b100)]
}

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn cl3_identity_rotor_is_grade_zero_unit() {
    let id = Motor::<Cl3>::identity();
    assert!(approx_eq(id.0.get(0), 1.0, 1e-6));
    for k in 1..8 {
        assert!(approx_eq(id.0.get(k), 0.0, 1e-6), "blade {k}: nonzero");
    }
}

#[test]
fn cl3_interpolate_xy_plane_half_is_quarter_rotation() {
    // SLERP from identity to a pi-rotation in e12 at alpha=0.5 must be a
    // quarter rotation in the e12 plane. The rotation sends (1, 0, 0) to
    // a unit vector in the xy-plane with vanishing x and z components.
    //
    // Sign locked in 2026-05-10 against Cl3 Cayley with `[+1, +1, +1]`
    // metric and PGA3-parity bit ordering: the y-component lands at
    // exactly `-1`. A regression on rotation direction in
    // `Motor::interpolate` (e.g. swapping `target * ~self` for
    // `~self * target` in the relative-rotor build, or flipping the
    // double-cover branch) would change this sign and trip the assert.
    let from = Motor::<Cl3>::identity();
    let to = cl3_rotor(E12, core::f32::consts::PI);
    let mid = from.interpolate(&to, 0.5);
    let xyz = rotate_vec(&mid, 1.0, 0.0, 0.0);
    assert!(approx_eq(xyz[0], 0.0, 1e-4), "x: {}", xyz[0]);
    assert!(
        approx_eq(xyz[1], -1.0, 1e-4),
        "y: {} (rotation-direction regression flips this sign)",
        xyz[1],
    );
    assert!(approx_eq(xyz[2], 0.0, 1e-4), "z: {}", xyz[2]);
}

#[test]
fn cl3_interpolate_xz_plane_half_is_quarter_rotation() {
    // Acceptance for the generic enumeration: the e13 blade (bitmask
    // 0b101) is a Euclidean bivector in Cl3 the same way it is in PGA3,
    // but in PGA3 it sat alongside two other grade-2 Euclidean blades.
    // This test fails to take the e13 contribution unless `interpolate`
    // enumerates blades by `count_ones() == 2 && (b & NULL_MASK) == 0`
    // rather than hard-coding the three PGA3 indices.
    //
    // Sign locked in 2026-05-10 against Cl3 Cayley with `[+1, +1, +1]`
    // metric and PGA3-parity bit ordering: the z-component lands at
    // exactly `-1`. A regression on rotation direction in
    // `Motor::interpolate` would change this sign and trip the assert.
    let from = Motor::<Cl3>::identity();
    let to = cl3_rotor(E13, core::f32::consts::PI);
    let mid = from.interpolate(&to, 0.5);
    let xyz = rotate_vec(&mid, 1.0, 0.0, 0.0);
    assert!(approx_eq(xyz[0], 0.0, 1e-4), "x: {}", xyz[0]);
    assert!(approx_eq(xyz[1], 0.0, 1e-4), "y: {}", xyz[1]);
    assert!(
        approx_eq(xyz[2], -1.0, 1e-4),
        "z: {} (rotation-direction regression flips this sign)",
        xyz[2],
    );
}

#[test]
fn cl3_interpolate_yz_plane_half_is_quarter_rotation() {
    // Quarter rotation in e23 sends (0, 1, 0) to a unit vector with no
    // x-component and with y near zero.
    //
    // Sign locked in 2026-05-10 against Cl3 Cayley with `[+1, +1, +1]`
    // metric and PGA3-parity bit ordering: the z-component lands at
    // exactly `-1`. A regression on rotation direction in
    // `Motor::interpolate` would change this sign and trip the assert.
    let from = Motor::<Cl3>::identity();
    let to = cl3_rotor(E23, core::f32::consts::PI);
    let mid = from.interpolate(&to, 0.5);
    let xyz = rotate_vec(&mid, 0.0, 1.0, 0.0);
    assert!(approx_eq(xyz[0], 0.0, 1e-4), "x: {}", xyz[0]);
    assert!(approx_eq(xyz[1], 0.0, 1e-4), "y: {}", xyz[1]);
    assert!(
        approx_eq(xyz[2], -1.0, 1e-4),
        "z: {} (rotation-direction regression flips this sign)",
        xyz[2],
    );
}

#[test]
fn cl3_interpolate_at_zero_returns_self() {
    let from = cl3_rotor(E12, 0.5);
    let to = cl3_rotor(E12, 1.5);
    let r = from.interpolate(&to, 0.0);
    let xyz_from = rotate_vec(&from, 1.0, 0.0, 0.0);
    let xyz_r = rotate_vec(&r, 1.0, 0.0, 0.0);
    for k in 0..3 {
        assert!(
            approx_eq(xyz_from[k], xyz_r[k], 1e-5),
            "component {k}: from={}, r={}",
            xyz_from[k],
            xyz_r[k],
        );
    }
}

#[test]
fn cl3_interpolate_equal_rotors_is_self() {
    // Numerical-identity guard: when from == to the relative-rotor
    // bivector vanishes and the early-out path must fire, returning a
    // copy of `from` rather than producing NaNs from the `0/0` divide.
    let from = cl3_rotor(E23, 0.7);
    let r = from.interpolate(&from, 0.4);
    let xyz_from = rotate_vec(&from, 0.0, 1.0, 0.0);
    let xyz_r = rotate_vec(&r, 0.0, 1.0, 0.0);
    for k in 0..3 {
        assert!(
            approx_eq(xyz_from[k], xyz_r[k], 1e-5),
            "component {k}: from={}, r={}",
            xyz_from[k],
            xyz_r[k],
        );
    }
}
