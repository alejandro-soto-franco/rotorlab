//! Drawable objects rendered by a [`Scene`](crate::scene::Scene).
//!
//! Plan 3 Task 3 introduces the [`Drawable`] trait and the [`Aabb`]
//! bounds type. Concrete drawables (point, line, plane) land in
//! Plan 3 Tasks 5, 7, and 8 respectively.

pub mod drawable;

pub use drawable::{Aabb, Drawable};
