//! Compiles GLSL shaders in `shaders/` to SPIR-V at build time.
//! Task 7 fills this in; for now it's a no-op so the crate compiles.

fn main() {
    println!("cargo:rerun-if-changed=shaders");
}
