//! Drawable objects rendered by a [`Scene`](crate::scene::Scene).
//!
//! Plan 3 Task 3 introduces the [`Drawable`] trait and the [`Aabb`]
//! bounds type. Plan 3 Task 5 adds the first concrete drawable
//! ([`Point`]) and the [`Color`] visual-style primitive. Line and
//! plane drawables land in Plan 3 Tasks 7 and 8.

pub mod drawable;
pub mod point;
pub mod style;

pub use drawable::{Aabb, Drawable};
pub use point::Point;
pub use style::Color;
