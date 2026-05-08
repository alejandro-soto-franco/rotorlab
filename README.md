# rotorlab

Math-animation engine for explainer videos, written in Rust, built on a const-generic geometric-algebra core, rendered via raw Vulkan.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

**Status:** v0.0.1, `rotorlab-ga` (geometric-algebra core) only. The animation engine ships in v0.1.

## Crates

| Crate | Purpose |
|---|---|
| [`rotorlab-ga`](rotorlab-ga/) | Pure const-generic GA: PGA3 multivectors, motors, geometric/outer/inner products. No I/O, no GPU. |
| `rotorlab` (v0.1) | Animation engine. Vulkan render path, scene graph, FFmpeg encode, LaTeX text. |

## License

Apache-2.0. See [LICENSE-APACHE](LICENSE-APACHE) and [NOTICE](NOTICE).
