//! Animation primitives: timeline, rate functions, stock animations.
//!
//! Currently houses only the [`Timeline`] stub for Task 2. Plan 3 Tasks
//! 9 through 11 grow this module with the `Animation` trait, `RateFunc`,
//! and stock animations (Translate, Rotate, Transform, FadeIn, FadeOut).

pub mod timeline;

pub use timeline::Timeline;
