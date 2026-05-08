# Miri verification log

| Date | Toolchain | Test target | Result |
|------|-----------|-------------|--------|
| 2026-05-08 | nightly-x86_64-unknown-linux-gnu (rustc 1.96.0-nightly) | `cargo +nightly miri test -p rotorlab-ga --lib` | clean (17 lib tests pass under Miri) |

This confirms the two `unsafe impl bytemuck::Pod` / `bytemuck::Zeroable` blocks
in `src/multivector.rs` are sound under Miri's stacked-borrows / tree-borrows
analysis. No UB detected.
