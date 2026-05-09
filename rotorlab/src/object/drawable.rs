//! [`Drawable`] trait and supporting [`Aabb`] type.
//!
//! Every object that the scene can render implements [`Drawable`].
//! Recording happens once per frame inside
//! [`Scene::render`](crate::scene::Scene) by calling
//! [`Drawable::record`] with a [`FrameContext`] (the command buffer +
//! pipeline cache + descriptor pool) and the active
//! [`Camera`]. Plan 3 ships only the trait and
//! its bounds type; concrete drawables (point, line, plane) follow in
//! Plan 3 Tasks 5, 7, and 8.

use crate::camera::Camera;
use crate::scene::FrameContext;

/// Axis-aligned bounding box in world space.
///
/// Used by [`Drawable::bounds`] to feed culling and auto-framing.
/// Both endpoints are inclusive; an empty box is constructed via
/// [`Aabb::empty`] (with `min = +inf`, `max = -inf`) so that
/// [`Aabb::union`] over a sequence of points yields the tight bound.
#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    /// Minimum corner of the box, component-wise.
    pub min: [f32; 3],
    /// Maximum corner of the box, component-wise.
    pub max: [f32; 3],
}

impl Aabb {
    /// An empty box: `min = +inf` and `max = -inf` on every axis.
    ///
    /// Acts as the identity for [`Aabb::union`]: unioning any other
    /// box (or the [`Aabb::from_point`] of any finite point) with
    /// `Aabb::empty()` yields that other box.
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    /// A degenerate box that contains exactly the supplied point.
    pub fn from_point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    /// The component-wise union of two boxes.
    ///
    /// `min` of the result is the per-axis minimum of the two `min`s;
    /// `max` is the per-axis maximum of the two `max`s. Unioning with
    /// [`Aabb::empty`] returns the other operand.
    pub fn union(self, other: Aabb) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
}

/// Anything the scene can record into a command buffer.
///
/// Implementations must be [`Send`] so the scene can plausibly
/// migrate recording off the main thread later. They do *not* need
/// to be [`Sync`]: each drawable is recorded by exactly one thread
/// per frame.
///
/// The `&Camera` argument is read-only; [`Camera::view_proj`]
/// supplies the matrix that almost every drawable needs as a push
/// constant or uniform.
pub trait Drawable: Send {
    /// Record this drawable into the per-frame command buffer
    /// supplied via `ctx`.
    fn record(&self, ctx: &mut FrameContext<'_>, camera: &Camera);

    /// World-space bounds of this drawable.
    ///
    /// Used for culling and auto-framing. Drawables that have no
    /// well-defined world bounds (such as a screen-space overlay)
    /// may return [`Aabb::empty`]; the scene treats that as
    /// "contributes nothing to the union".
    fn bounds(&self) -> Aabb;

    /// Sort key used by the scene to pick a stable draw order
    /// across overlapping translucent drawables.
    ///
    /// Defaults to `0.0`. Drawables that need to be drawn behind or
    /// in front of others can override this; the scene sorts in
    /// ascending order (smaller `z_order` is drawn first, larger
    /// last).
    fn z_order(&self) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial drawable used to prove the trait surface compiles.
    /// The body of `record` is empty because Plan 3 Task 3 has no
    /// real Vulkan-backed `FrameContext` to drive; the test is that
    /// the call signature compiles end to end.
    struct DummyDrawable {
        half_extent: f32,
    }
    impl Drawable for DummyDrawable {
        fn record(&self, _ctx: &mut FrameContext<'_>, _camera: &Camera) {}
        fn bounds(&self) -> Aabb {
            Aabb {
                min: [-self.half_extent, -self.half_extent, -self.half_extent],
                max: [self.half_extent, self.half_extent, self.half_extent],
            }
        }
    }

    #[test]
    fn drawable_default_z_order_is_zero() {
        let d = DummyDrawable { half_extent: 1.0 };
        assert_eq!(d.z_order(), 0.0);
    }

    #[test]
    fn drawable_bounds_round_trip() {
        let d = DummyDrawable { half_extent: 2.0 };
        let b = d.bounds();
        assert_eq!(b.min, [-2.0, -2.0, -2.0]);
        assert_eq!(b.max, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn aabb_empty_union_with_point_returns_point_bounds() {
        let p = Aabb::from_point([1.0, 2.0, 3.0]);
        let u = Aabb::empty().union(p);
        assert_eq!(u.min, [1.0, 2.0, 3.0]);
        assert_eq!(u.max, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn aabb_union_takes_extremes() {
        let a = Aabb {
            min: [-1.0, -2.0, -3.0],
            max: [1.0, 2.0, 3.0],
        };
        let b = Aabb {
            min: [0.0, -5.0, 0.0],
            max: [4.0, 0.0, 0.5],
        };
        let u = a.union(b);
        assert_eq!(u.min, [-1.0, -5.0, -3.0]);
        assert_eq!(u.max, [4.0, 2.0, 3.0]);
    }
}
