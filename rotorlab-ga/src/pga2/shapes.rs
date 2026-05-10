//! Named-field shape structs for PGA2 geometric primitives.
//!
//! Each struct in this module stores **only** the non-zero blades for a
//! particular geometric type, packed as a `#[repr(C)]` array of `f32`s
//! with one named field per blade. The dense [`Multivector<Pga2>`]
//! representation in [`crate::pga2::factories`] remains the reference
//! oracle; these shape structs are an optimisation layer that exists
//! to:
//!
//! 1. Cut storage from the 8-blade dense form (32 bytes) to the
//!    minimal subset required by each primitive (12 to 16 bytes).
//! 2. Keep cache-resident data dense by packing only the live blades
//!    contiguously, with no `0.0` padding for blades the type cannot
//!    carry.
//! 3. Expose semantically named fields (`e_12`, `e_02`, ...) so call
//!    sites can manipulate blades by name rather than by bitmask
//!    indexing.
//!
//! Round-tripping with [`Multivector<Pga2>`] is the responsibility of
//! the bridge layer in [`crate::pga2::bridge`]; this module ships the
//! data layout only and deliberately exposes no `From` impls between
//! shape structs and the dense newtype surface.
//!
//! Every struct here is `#[repr(C)] + Copy + Clone + Default + Debug`
//! and implements [`bytemuck::Pod`] + [`bytemuck::Zeroable`] so that
//! shape buffers can be uploaded to the GPU or memcpyed to disk
//! without further conversion.
//!
//! There is no `Plane` shape struct in PGA2: the projective dual of a
//! plane in 3D collapses to a line in 2D, and [`Line`] already covers
//! the grade-1 vector slice.
//!
//! [`Multivector<Pga2>`]: crate::multivector::Multivector

/// Named-field PGA2 point: a grade-2 bivector that occupies exactly the
/// three blades whose bitmasks select two of `{e1, e2, e0}`.
///
/// Stores 3 of the 8 PGA2 blades. The dense oracle is
/// [`crate::pga2::Point`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga2::Pga2>`] lives in
/// [`crate::pga2::bridge`].
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Point {
    /// Coefficient on `e_12` (bitmask `0b011`); homogeneous weight
    /// (always `1.0` for a normalised point).
    pub e_12: f32,
    /// Coefficient on `e_02` (bitmask `0b110`); affine x-coordinate
    /// after weight normalisation.
    pub e_02: f32,
    /// Coefficient on `e_01` (bitmask `0b101`); affine y-coordinate
    /// after weight normalisation.
    pub e_01: f32,
}

impl Point {
    /// Construct a unit-weight Euclidean point at `(x, y)`.
    ///
    /// Mirrors [`crate::pga2::point`] but produces the named-field
    /// shape struct directly (`e_12 = 1.0`, `e_02 = x`, `e_01 = y`).
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            e_12: 1.0,
            e_02: x,
            e_01: y,
        }
    }

    /// Convert this PGA2 point to Euclidean `(x, y)` by dividing the
    /// bivector coefficients by the homogeneous weight `e_12`.
    ///
    /// Returns `[0.0, 0.0]` for points whose weight has absolute value
    /// below `1e-12` (these represent points at projective infinity).
    pub fn to_euclidean(&self) -> [f32; 2] {
        let w = self.e_12;
        if w.abs() < 1e-12 {
            return [0.0, 0.0];
        }
        [self.e_02 / w, self.e_01 / w]
    }
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Point {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Point {}

/// Named-field PGA2 line: a grade-1 vector encoding `a*x + b*y + c = 0`
/// as `a*e1 + b*e2 + c*e0`.
///
/// Stores 3 of the 8 PGA2 blades. The dense oracle is
/// [`crate::pga2::Line`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga2::Pga2>`] lives in
/// [`crate::pga2::bridge`].
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Line {
    /// Coefficient on `e_1` (bitmask `0b001`); the normal x-component
    /// `a`.
    pub e_1: f32,
    /// Coefficient on `e_2` (bitmask `0b010`); the normal y-component
    /// `b`.
    pub e_2: f32,
    /// Coefficient on `e_0` (bitmask `0b100`); the affine constant
    /// `c` in `a*x + b*y + c = 0`.
    pub e_0: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Line {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Line {}

/// Named-field PGA2 bivector: a generic grade-2 element with the same
/// three-blade layout as the bivector slice of [`Motor`].
///
/// Stores 3 of the 8 PGA2 blades. The dense oracle is
/// [`crate::pga2::Bivector`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga2::Pga2>`] lives in
/// [`crate::pga2::bridge`].
///
/// Bivectors arise as rotor / translator generators and as arbitrary
/// grade-2 elements. The field layout matches [`Point`]'s by
/// construction (PGA2 points are themselves grade-2 bivectors); the
/// newtype distinction is semantic, not structural.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Bivector {
    /// Coefficient on `e_12` (bitmask `0b011`); Euclidean bivector
    /// component (xy-plane), the only Euclidean rotation plane in 2D.
    pub e_12: f32,
    /// Coefficient on `e_02` (bitmask `0b110`); null bivector
    /// component (translation generator along x).
    pub e_02: f32,
    /// Coefficient on `e_01` (bitmask `0b101`); null bivector
    /// component (translation generator along y).
    pub e_01: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Bivector {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Bivector {}

/// Named-field PGA2 rotor: the scalar plus Euclidean-bivector slice of
/// an even multivector that represents a pure rotation in the
/// `xy`-plane.
///
/// Stores 2 of the 8 PGA2 blades. The dense oracle is
/// [`crate::motor::Rotor<crate::pga2::Pga2>`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga2::Pga2>`] lives in
/// [`crate::pga2::bridge`].
///
/// A unit rotor satisfies `s^2 + e_12^2 = 1` and acts on
/// points/lines via the sandwich product `R x R~`.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Rotor {
    /// Coefficient on the scalar blade (bitmask `0b000`); equals
    /// `cos(angle / 2)` for a rotation by `angle`.
    pub s: f32,
    /// Coefficient on `e_12` (bitmask `0b011`); xy-plane bivector
    /// component scaled by `sin(angle / 2)`.
    pub e_12: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Rotor {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Rotor {}

/// Named-field PGA2 translator: the scalar plus null-bivector slice of
/// an even multivector that represents a pure translation.
///
/// Stores 3 of the 8 PGA2 blades. The dense oracle is
/// [`crate::motor::Translator<crate::pga2::Pga2>`]; the bridge to and
/// from [`crate::multivector::Multivector<crate::pga2::Pga2>`] lives
/// in [`crate::pga2::bridge`].
///
/// Because the null-bivector generators square to zero, the
/// exponential collapses to `exp(t * T) = 1 + t * T` and the scalar
/// part `s` is always `1.0` for a unit translator.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Translator {
    /// Coefficient on the scalar blade (bitmask `0b000`); always
    /// `1.0` for a unit translator (the null bivector generators
    /// square to zero, so `exp` truncates after the linear term).
    pub s: f32,
    /// Coefficient on `e_02` (bitmask `0b110`); null bivector
    /// component carrying half the x-translation.
    pub e_02: f32,
    /// Coefficient on `e_01` (bitmask `0b101`); null bivector
    /// component carrying half the y-translation.
    pub e_01: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Translator {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Translator {}

/// Named-field PGA2 motor: the full even-grade subalgebra slice
/// (scalar plus all three grade-2 bivectors).
///
/// Stores 4 of the 8 PGA2 blades. The dense oracle is
/// [`crate::motor::Motor<crate::pga2::Pga2>`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga2::Pga2>`] lives in
/// [`crate::pga2::bridge`].
///
/// A general motor is the composition of a [`Rotor`] and a
/// [`Translator`] (`M = T R`). Unlike PGA3, the PGA2 even subalgebra
/// has no pseudoscalar component (the pseudoscalar `e_120` lives at
/// grade 3, which is odd in PGA2), so the motor surface is exactly
/// the four blades listed below.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Motor {
    /// Coefficient on the scalar blade (bitmask `0b000`).
    pub s: f32,
    /// Coefficient on `e_12` (bitmask `0b011`); xy-plane Euclidean
    /// bivector component (rotation generator).
    pub e_12: f32,
    /// Coefficient on `e_02` (bitmask `0b110`); null bivector
    /// component (translation generator along x).
    pub e_02: f32,
    /// Coefficient on `e_01` (bitmask `0b101`); null bivector
    /// component (translation generator along y).
    pub e_01: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Motor {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Motor {}

impl Motor {
    /// The identity motor: scalar coefficient `1`, every other blade
    /// `0`.
    ///
    /// Mirrors [`crate::motor::Motor::identity`] in the named-field
    /// shape representation. Applying this to any shape leaves it
    /// unchanged (up to round-off).
    pub const fn identity() -> Self {
        Self {
            s: 1.0,
            e_12: 0.0,
            e_02: 0.0,
            e_01: 0.0,
        }
    }
}

impl From<Rotor> for Motor {
    /// Embed a [`Rotor`] into a [`Motor`] by copying the two rotor
    /// blades into the matching motor slots and zeroing the translator
    /// part.
    fn from(r: Rotor) -> Self {
        Self {
            s: r.s,
            e_12: r.e_12,
            e_02: 0.0,
            e_01: 0.0,
        }
    }
}

impl From<Translator> for Motor {
    /// Embed a [`Translator`] into a [`Motor`] by copying the scalar
    /// and two null-bivector blades into the matching motor slots and
    /// zeroing the rotor part.
    fn from(t: Translator) -> Self {
        Self {
            s: t.s,
            e_12: 0.0,
            e_02: t.e_02,
            e_01: t.e_01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn point_size_matches_field_count() {
        assert_eq!(mem::size_of::<Point>(), 3 * mem::size_of::<f32>());
    }

    #[test]
    fn point_default_is_all_zeros() {
        let p = Point::default();
        let bytes: &[u8] = bytemuck::bytes_of(&p);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn line_size_matches_field_count() {
        assert_eq!(mem::size_of::<Line>(), 3 * mem::size_of::<f32>());
    }

    #[test]
    fn line_default_is_all_zeros() {
        let l = Line::default();
        let bytes: &[u8] = bytemuck::bytes_of(&l);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn bivector_size_matches_field_count() {
        assert_eq!(mem::size_of::<Bivector>(), 3 * mem::size_of::<f32>());
    }

    #[test]
    fn bivector_default_is_all_zeros() {
        let b = Bivector::default();
        let bytes: &[u8] = bytemuck::bytes_of(&b);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn rotor_size_matches_field_count() {
        assert_eq!(mem::size_of::<Rotor>(), 2 * mem::size_of::<f32>());
    }

    #[test]
    fn rotor_default_is_all_zeros() {
        let r = Rotor::default();
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn translator_size_matches_field_count() {
        assert_eq!(mem::size_of::<Translator>(), 3 * mem::size_of::<f32>());
    }

    #[test]
    fn translator_default_is_all_zeros() {
        let t = Translator::default();
        let bytes: &[u8] = bytemuck::bytes_of(&t);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn motor_size_matches_field_count() {
        assert_eq!(mem::size_of::<Motor>(), 4 * mem::size_of::<f32>());
    }

    #[test]
    fn motor_default_is_all_zeros() {
        let m = Motor::default();
        let bytes: &[u8] = bytemuck::bytes_of(&m);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
