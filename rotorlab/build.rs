//! Compiles GLSL shaders in `shaders/` to SPIR-V at build time.
//!
//! Outputs `OUT_DIR/<name>.spv` for each `<name>.vert` / `<name>.frag` /
//! `<name>.comp` found in `shaders/`. Source code can include them with
//! `include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"))`.

use shaderc::{Compiler, ShaderKind};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let shaders_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("shaders");

    println!("cargo:rerun-if-changed=shaders");
    if !shaders_dir.exists() {
        return;
    }

    let compiler = Compiler::new().expect("shaderc compiler");

    for entry in fs::read_dir(&shaders_dir).expect("read shaders dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        let kind = match path.extension().and_then(|s| s.to_str()) {
            Some("vert") => ShaderKind::Vertex,
            Some("frag") => ShaderKind::Fragment,
            Some("comp") => ShaderKind::Compute,
            _ => continue,
        };
        let source = fs::read_to_string(&path).expect("read shader source");
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let out_path = out_dir.join(format!("{file_name}.spv"));
        let artifact = compiler
            .compile_into_spirv(&source, kind, file_name, "main", None)
            .unwrap_or_else(|e| panic!("compile {file_name}: {e}"));
        fs::write(&out_path, artifact.as_binary_u8()).expect("write spv");
        println!("cargo:rerun-if-changed=shaders/{file_name}");
    }
}
