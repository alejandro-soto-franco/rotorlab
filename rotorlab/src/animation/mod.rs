//! Animation primitives: timeline, rate functions, stock animations.
//!
//! Plan 3 Task 9 introduces the [`Animation`] trait and the [`RateFunc`]
//! easing-curve enum. Task 10 grows [`Timeline`] from its current stub into
//! a real scheduler, and Task 11 ships the stock animation set
//! ([`Translate`], [`Rotate`], [`Transform`], [`FadeIn`], [`FadeOut`]).

#[allow(clippy::module_inception)]
pub mod animation;
pub mod fade;
pub mod rate_func;
pub mod timeline;
pub mod transforms;

pub use animation::Animation;
pub use fade::{FadeIn, FadeOut};
pub use rate_func::RateFunc;
pub use timeline::Timeline;
pub use transforms::{Rotate, Transform, Translate};
