//! Drawable objects rendered by a [`Scene`](crate::scene::Scene).
//!
//! Plan 3 Task 3 introduces the [`Drawable`] trait and the [`Aabb`]
//! bounds type. Plan 3 Task 5 adds the first concrete drawable
//! ([`Point`]) and the [`Color`] visual-style primitive. Plan 3
//! Task 7 adds the [`Line`] drawable and the [`Stroke`] styling
//! primitive. Plan 3 Task 8 adds the [`Plane`] wireframe drawable.

pub mod drawable;
pub mod line;
pub mod plane;
pub mod point;
pub mod style;

pub use drawable::{Aabb, Drawable};
pub use line::Line;
pub use plane::Plane;
pub use point::Point;
pub use style::{Color, Stroke};
