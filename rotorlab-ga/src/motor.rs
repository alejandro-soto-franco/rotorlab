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
}
