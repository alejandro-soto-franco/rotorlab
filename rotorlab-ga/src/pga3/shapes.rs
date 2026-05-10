//! Named-field shape structs for PGA3 geometric primitives.
//!
//! Each struct in this module stores **only** the non-zero blades for a
//! particular geometric type, packed as a `#[repr(C)]` array of `f32`s
//! with one named field per blade. The dense [`Multivector<Pga3>`]
//! representation in [`crate::pga3::factories`] remains the reference
//! oracle; these shape structs are an optimisation layer that exists
//! to:
//!
//! 1. Cut storage from the 16-blade dense form (64 bytes) to the
//!    minimal subset required by each primitive (16 to 32 bytes).
//! 2. Keep cache-resident data dense by packing only the live blades
//!    contiguously, with no `0.0` padding for blades the type cannot
//!    carry.
//! 3. Expose semantically named fields (`e_023`, `e_12`, ...) so call
//!    sites can manipulate blades by name rather than by bitmask
//!    indexing.
//!
//! Round-tripping with [`Multivector<Pga3>`] is the responsibility of
//! the bridge layer landing in **Stage 8**; this module ships the
//! data layout only and deliberately exposes no `From` impls between
//! shape structs and the dense newtype surface.
//!
//! Every struct here is `#[repr(C)] + Copy + Clone + Default + Debug`
//! and implements [`bytemuck::Pod`] + [`bytemuck::Zeroable`] so that
//! shape buffers can be uploaded to the GPU or memcpyed to disk
//! without further conversion.
//!
//! [`Multivector<Pga3>`]: crate::multivector::Multivector

/// Named-field PGA3 point: a grade-3 trivector that occupies exactly
/// the four blades whose bitmasks omit one of `e1`, `e2`, `e3`, `e0`.
///
/// Stores 4 of the 16 PGA3 blades. The dense oracle is
/// [`crate::pga3::Point`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands in
/// Stage 8.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Point {
    /// Coefficient on `e_023` (bitmask `0b1110`); affine x-coordinate
    /// after weight normalisation.
    pub e_023: f32,
    /// Coefficient on `e_013` (bitmask `0b1101`); affine y-coordinate
    /// after weight normalisation.
    pub e_013: f32,
    /// Coefficient on `e_012` (bitmask `0b1011`); affine z-coordinate
    /// after weight normalisation.
    pub e_012: f32,
    /// Coefficient on `e_123` (bitmask `0b0111`); homogeneous weight
    /// (always `1.0` for a normalised point).
    pub e_123: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Point {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Point {}

/// Named-field PGA3 plane: a grade-1 vector encoding
/// `a*x + b*y + c*z = d` as `a*e1 + b*e2 + c*e3 + d*e0`.
///
/// Stores 4 of the 16 PGA3 blades. The dense oracle is
/// [`crate::pga3::Plane`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands in
/// Stage 8.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Plane {
    /// Coefficient on `e_0` (bitmask `0b1000`); the plane offset `d`
    /// in `a*x + b*y + c*z = d`.
    pub e_0: f32,
    /// Coefficient on `e_1` (bitmask `0b0001`); the normal x-component
    /// `a`.
    pub e_1: f32,
    /// Coefficient on `e_2` (bitmask `0b0010`); the normal y-component
    /// `b`.
    pub e_2: f32,
    /// Coefficient on `e_3` (bitmask `0b0100`); the normal z-component
    /// `c`.
    pub e_3: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Plane {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Plane {}

/// Named-field PGA3 line: a grade-2 element carrying a Plücker-style
/// pair (Euclidean direction part `e_12, e_13, e_23` and null moment
/// part `e_01, e_02, e_03`).
///
/// Stores 6 of the 16 PGA3 blades. The dense oracle is
/// [`crate::pga3::Line`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands in
/// Stage 8.
///
/// `Line` is the **geometric** newtype: it represents the join of two
/// points (or the meet of two planes) as a projective 3-space line.
/// The bit-identical [`Bivector`] type is the **algebraic** newtype
/// for any grade-2 element of PGA3, regardless of whether it has a
/// geometric interpretation as a line.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Line {
    /// Coefficient on `e_01` (bitmask `0b1001`); null moment component.
    pub e_01: f32,
    /// Coefficient on `e_02` (bitmask `0b1010`); null moment component.
    pub e_02: f32,
    /// Coefficient on `e_03` (bitmask `0b1100`); null moment component.
    pub e_03: f32,
    /// Coefficient on `e_12` (bitmask `0b0011`); Euclidean direction
    /// component (xy-plane).
    pub e_12: f32,
    /// Coefficient on `e_13` (bitmask `0b0101`); Euclidean direction
    /// component (xz-plane).
    pub e_13: f32,
    /// Coefficient on `e_23` (bitmask `0b0110`); Euclidean direction
    /// component (yz-plane).
    pub e_23: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Line {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Line {}

/// Named-field PGA3 bivector: a generic grade-2 element with the same
/// six-blade layout as [`Line`].
///
/// Stores 6 of the 16 PGA3 blades. The dense oracle is
/// [`crate::pga3::Bivector`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands in
/// Stage 8.
///
/// `Bivector` is the **algebraic** newtype, kept distinct from
/// [`Line`] (the geometric newtype) even though the field layout is
/// identical. Bivectors arise as rotor / translator generators and as
/// arbitrary grade-2 elements that may not satisfy the Plücker
/// constraint of a true line.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Bivector {
    /// Coefficient on `e_01` (bitmask `0b1001`); null bivector component.
    pub e_01: f32,
    /// Coefficient on `e_02` (bitmask `0b1010`); null bivector component.
    pub e_02: f32,
    /// Coefficient on `e_03` (bitmask `0b1100`); null bivector component.
    pub e_03: f32,
    /// Coefficient on `e_12` (bitmask `0b0011`); Euclidean bivector
    /// component (xy-plane).
    pub e_12: f32,
    /// Coefficient on `e_13` (bitmask `0b0101`); Euclidean bivector
    /// component (xz-plane).
    pub e_13: f32,
    /// Coefficient on `e_23` (bitmask `0b0110`); Euclidean bivector
    /// component (yz-plane).
    pub e_23: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Bivector {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Bivector {}

/// Named-field PGA3 rotor: the scalar plus Euclidean-bivector slice of
/// an even multivector that represents a pure rotation.
///
/// Stores 4 of the 16 PGA3 blades. The dense oracle is
/// [`crate::motor::Rotor<crate::pga3::Pga3>`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands in
/// Stage 8.
///
/// A unit rotor satisfies `s^2 + e_12^2 + e_13^2 + e_23^2 = 1` and
/// acts on points/lines/planes via the sandwich product `R x R~`.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Rotor {
    /// Coefficient on the scalar blade (bitmask `0b0000`); equals
    /// `cos(angle / 2)` for a rotation by `angle`.
    pub s: f32,
    /// Coefficient on `e_12` (bitmask `0b0011`); xy-plane bivector
    /// component scaled by `sin(angle / 2)`.
    pub e_12: f32,
    /// Coefficient on `e_13` (bitmask `0b0101`); xz-plane bivector
    /// component scaled by `sin(angle / 2)`.
    pub e_13: f32,
    /// Coefficient on `e_23` (bitmask `0b0110`); yz-plane bivector
    /// component scaled by `sin(angle / 2)`.
    pub e_23: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Rotor {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Rotor {}

/// Named-field PGA3 translator: the scalar plus null-bivector slice of
/// an even multivector that represents a pure translation.
///
/// Stores 4 of the 16 PGA3 blades. The dense oracle is
/// [`crate::motor::Translator<crate::pga3::Pga3>`]; the bridge to and
/// from [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands
/// in Stage 8.
///
/// Because the null-bivector generators square to zero, the
/// exponential collapses to `exp(t * T) = 1 + t * T` and the scalar
/// part `s` is always `1.0` for a unit translator.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Translator {
    /// Coefficient on the scalar blade (bitmask `0b0000`); always
    /// `1.0` for a unit translator (the null bivector generators
    /// square to zero, so `exp` truncates after the linear term).
    pub s: f32,
    /// Coefficient on `e_01` (bitmask `0b1001`); null bivector
    /// component carrying half the x-translation.
    pub e_01: f32,
    /// Coefficient on `e_02` (bitmask `0b1010`); null bivector
    /// component carrying half the y-translation.
    pub e_02: f32,
    /// Coefficient on `e_03` (bitmask `0b1100`); null bivector
    /// component carrying half the z-translation.
    pub e_03: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Translator {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Translator {}

/// Named-field PGA3 motor: the full even-grade subalgebra slice
/// (scalar, all six bivectors, and the pseudoscalar).
///
/// Stores 8 of the 16 PGA3 blades. The dense oracle is
/// [`crate::motor::Motor<crate::pga3::Pga3>`]; the bridge to and from
/// [`crate::multivector::Multivector<crate::pga3::Pga3>`] lands in
/// Stage 8.
///
/// A general motor is the composition of a [`Rotor`] and a
/// [`Translator`] (`M = T R`); its even-grade structure includes the
/// pseudoscalar component `e_0123` that arises from such a product.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Motor {
    /// Coefficient on the scalar blade (bitmask `0b0000`).
    pub s: f32,
    /// Coefficient on `e_12` (bitmask `0b0011`); xy-plane Euclidean
    /// bivector component.
    pub e_12: f32,
    /// Coefficient on `e_13` (bitmask `0b0101`); xz-plane Euclidean
    /// bivector component.
    pub e_13: f32,
    /// Coefficient on `e_23` (bitmask `0b0110`); yz-plane Euclidean
    /// bivector component.
    pub e_23: f32,
    /// Coefficient on `e_01` (bitmask `0b1001`); null bivector
    /// component (translation generator).
    pub e_01: f32,
    /// Coefficient on `e_02` (bitmask `0b1010`); null bivector
    /// component (translation generator).
    pub e_02: f32,
    /// Coefficient on `e_03` (bitmask `0b1100`); null bivector
    /// component (translation generator).
    pub e_03: f32,
    /// Coefficient on the pseudoscalar `e_0123` (bitmask `0b1111`);
    /// arises from products of rotors and translators.
    pub e_0123: f32,
}

// SAFETY: `#[repr(C)]` with only `f32` fields. The all-zero bit
// pattern is valid for every `f32`, so `Zeroable` applies.
unsafe impl bytemuck::Zeroable for Motor {}
// SAFETY: `#[repr(C)]` with only `f32` fields. Every bit pattern is a
// valid `f32`, the struct contains no padding, and it derives `Copy`,
// so `Pod` applies.
unsafe impl bytemuck::Pod for Motor {}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn point_size_matches_field_count() {
        assert_eq!(mem::size_of::<Point>(), 4 * mem::size_of::<f32>());
    }

    #[test]
    fn point_default_is_all_zeros() {
        let p = Point::default();
        let bytes: &[u8] = bytemuck::bytes_of(&p);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn plane_size_matches_field_count() {
        assert_eq!(mem::size_of::<Plane>(), 4 * mem::size_of::<f32>());
    }

    #[test]
    fn plane_default_is_all_zeros() {
        let p = Plane::default();
        let bytes: &[u8] = bytemuck::bytes_of(&p);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn line_size_matches_field_count() {
        assert_eq!(mem::size_of::<Line>(), 6 * mem::size_of::<f32>());
    }

    #[test]
    fn line_default_is_all_zeros() {
        let l = Line::default();
        let bytes: &[u8] = bytemuck::bytes_of(&l);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn bivector_size_matches_field_count() {
        assert_eq!(mem::size_of::<Bivector>(), 6 * mem::size_of::<f32>());
    }

    #[test]
    fn bivector_default_is_all_zeros() {
        let b = Bivector::default();
        let bytes: &[u8] = bytemuck::bytes_of(&b);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn rotor_size_matches_field_count() {
        assert_eq!(mem::size_of::<Rotor>(), 4 * mem::size_of::<f32>());
    }

    #[test]
    fn rotor_default_is_all_zeros() {
        let r = Rotor::default();
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn translator_size_matches_field_count() {
        assert_eq!(mem::size_of::<Translator>(), 4 * mem::size_of::<f32>());
    }

    #[test]
    fn translator_default_is_all_zeros() {
        let t = Translator::default();
        let bytes: &[u8] = bytemuck::bytes_of(&t);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn motor_size_matches_field_count() {
        assert_eq!(mem::size_of::<Motor>(), 8 * mem::size_of::<f32>());
    }

    #[test]
    fn motor_default_is_all_zeros() {
        let m = Motor::default();
        let bytes: &[u8] = bytemuck::bytes_of(&m);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
