//! rotorlab: GA-first Rust math-animation engine.
//!
//! See the [crate README](../README.md) for an overview.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod animation;
pub mod camera;
pub mod encode;
pub mod error;
pub mod object;
pub mod render;
pub mod scene;

pub use error::{EncodeError, RenderError, RotorlabError};
