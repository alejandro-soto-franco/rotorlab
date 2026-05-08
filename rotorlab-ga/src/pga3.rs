//! The PGA3 algebra: signature `(3, 0, 1)`, 4 ambient dimensions, 16 blades.
//!
//! Basis encoding (bitmask): bit 0 = `e1`, bit 1 = `e2`, bit 2 = `e3`, bit 3 = `e0`.
//! `e1`, `e2`, `e3` square to `+1`; `e0` (null) squares to `0`.
//! The pseudoscalar `I = e0 ∧ e1 ∧ e2 ∧ e3` has bitmask `0b1111`.

use crate::algebra::Algebra;

/// PGA3 algebra: 3D projective geometric algebra, Cl(3,0,1).
///
/// # Example
///
/// ```
/// use rotorlab_ga::{Algebra, pga3::Pga3};
/// assert_eq!(Pga3::SIGNATURE, (3, 0, 1));
/// assert_eq!(Pga3::DIM, 4);
/// assert_eq!(Pga3::N_BLADES, 16);
/// assert_eq!(Pga3::METRIC, &[1, 1, 1, 0]);
/// ```
#[derive(Copy, Clone, Default, Debug)]
pub struct Pga3;

impl Algebra for Pga3 {
    const SIGNATURE: (u32, u32, u32) = (3, 0, 1);
    const DIM: u32 = 4;
    const N_BLADES: usize = 16;
    type Storage = [f32; 16];
    type Scalar = f32;
    const METRIC: &'static [i8] = &[1, 1, 1, 0];
}

use crate::blade::{blade_product_blade, blade_product_sign};

/// The PGA3 Cayley table: `[i][j] = (sign, result_blade)` for `e_i * e_j`,
/// where `e_k` denotes the basis blade with bitmask `k`.
///
/// Materialized at compile time from the const-fn `blade_product_sign`.
pub const PGA3_CAYLEY: [[(i8, u64); 16]; 16] = pga3_cayley_table();

const fn pga3_cayley_table() -> [[(i8, u64); 16]; 16] {
    let mut table = [[(0i8, 0u64); 16]; 16];
    let mut i = 0usize;
    while i < 16 {
        let mut j = 0usize;
        while j < 16 {
            let sign = blade_product_sign(i as u64, j as u64, Pga3::METRIC);
            let blade = blade_product_blade(i as u64, j as u64);
            table[i][j] = (sign, blade);
            j += 1;
        }
        i += 1;
    }
    table
}

impl Pga3 {
    /// Reference to the compile-time-evaluated Cayley table.
    pub const fn cayley_table() -> &'static [[(i8, u64); 16]; 16] {
        &PGA3_CAYLEY
    }
}

use crate::multivector::Multivector;

/// Bivector (grade-2 element of PGA3).
#[derive(Copy, Clone, Default)]
pub struct Bivector(pub Multivector<Pga3>);

/// Point (grade-3 trivector in PGA3).
#[derive(Copy, Clone, Default)]
pub struct Point(pub Multivector<Pga3>);

/// Line (grade-2 bivector in PGA3).
#[derive(Copy, Clone, Default)]
pub struct Line(pub Multivector<Pga3>);

/// Plane (grade-1 vector in PGA3).
#[derive(Copy, Clone, Default)]
pub struct Plane(pub Multivector<Pga3>);

/// Construct a Euclidean point at `(x, y, z)` in PGA3.
///
/// In the e0-as-null convention, a normalized point is the trivector
/// `e_123 + x * e_023 + y * e_013 + z * e_012`. Bitmask encoding:
///   - `e_123 = 0b0111` (bits e1,e2,e3)
///   - `e_023 = 0b1110` (bits e2,e3,e0)
///   - `e_013 = 0b1101` (bits e1,e3,e0)
///   - `e_012 = 0b1011` (bits e1,e2,e0)
pub fn point(x: f32, y: f32, z: f32) -> Point {
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0b0111, 1.0); // e_123
    mv.set(0b1110, x); // x * e_023
    mv.set(0b1101, y); // y * e_013
    mv.set(0b1011, z); // z * e_012
    Point(mv)
}

/// Construct a line through two PGA3 points (`p ∨ q` — the meet, dual of join).
///
/// For v0.0.1 we use the wedge of duals: dualise p and q to grade-1 vectors,
/// take their wedge (a bivector), and dualise back.
pub fn line_through(p: Point, q: Point) -> Line {
    let pd = p.0.dual();
    let qd = q.0.dual();
    let l_dual = pd.outer_pga3(&qd);
    Line(l_dual.dual())
}

/// Construct a plane through three PGA3 points (`p ∨ q ∨ r`).
pub fn plane_through(p: Point, q: Point, r: Point) -> Plane {
    let pd = p.0.dual();
    let qd = q.0.dual();
    let rd = r.0.dual();
    let plane_dual = pd.outer_pga3(&qd).outer_pga3(&rd);
    Plane(plane_dual.dual())
}
