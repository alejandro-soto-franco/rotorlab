# rotorlab-ga

Const-generic geometric algebra (GA) core for the [rotorlab](https://github.com/alejandro-soto-franco/rotorlab) animation engine.

[![crates.io](https://img.shields.io/crates/v/rotorlab-ga.svg)](https://crates.io/crates/rotorlab-ga)
[![docs.rs](https://img.shields.io/docsrs/rotorlab-ga)](https://docs.rs/rotorlab-ga)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../LICENSE-APACHE)

## What ships in v0.0.1

- `Algebra` trait (const-generic over signature `(P, Q, R)`)
- `Multivector<A: Algebra>` universal type
- PGA3 (`Pga3`, signature `(3, 0, 1)`) fully implemented: 16 blades, geometric / outer / inner products, reverse, dual, grade projection
- `pga3::Motor`, `pga3::Rotor`, `pga3::Translator` newtypes; closed-form `exp` and `log`
- `pga3::point`, `pga3::line_through`, `pga3::plane_through` constructors

## Soundness

`Multivector<A>` derives `bytemuck::Pod` via two manual `unsafe impl` blocks, with inline safety arguments. The crate has no other unsafe code. Verified by `cargo miri test` in CI.

## Quick example

```rust
use rotorlab_ga::pga3::{self, Pga3};
use rotorlab_ga::Algebra;

// Construct two PGA3 points and the line through them.
let p = pga3::point(0.0, 0.0, 0.0);
let q = pga3::point(1.0, 0.0, 0.0);
let _line = pga3::line_through(p, q);

// Verify PGA3's signature.
assert_eq!(Pga3::SIGNATURE, (3, 0, 1));
assert_eq!(Pga3::N_BLADES, 16);
```

## License

Apache-2.0.
