//! The [`Point`] drawable: a single GA point rendered as a billboard
//! disk via the [`PointPipeline`](crate::render::pipelines::PointPipeline).
//!
//! Each [`Point`] carries a PGA3 position, a [`Color`], and a
//! screen-pixel radius. [`Drawable::record`] converts the point into a
//! [`PointInstance`](crate::render::pipelines::PointInstance) ready for
//! upload to the pipeline's instance buffer. The actual command-buffer
//! recording (`cmd_bind_pipeline` + `cmd_draw`) is wired in Plan 3
//! Tasks 11-12 once [`FrameContext`] exposes an instance-buffer staging
//! API; Task 5 ships the geometry-to-GPU conversion only.
//!
//! The `radius` field lives in screen-space pixels (constant size on
//! screen regardless of camera distance), matching the Manim-style
//! convention used by the upstream [`PointPipeline`].

use crate::camera::Camera;
use crate::object::capability::{HasOpacity, Rotatable, Translatable};
use crate::object::drawable::{Aabb, Drawable};
use crate::object::style::Color;
use crate::render::pipelines::PointInstance;
use crate::scene::FrameContext;
use rotorlab_ga::pga3::{self, Bivector, Line as PgaLine, Point as PgaPoint};

/// A drawable point: a GA position, a linear-sRGB color, and a
/// screen-pixel disk radius.
///
/// The Aabb returned by [`Drawable::bounds`] approximates the
/// screen-pixel radius as a world-unit half-extent on each axis. This
/// is technically wrong (mixing world and pixel units), but Plan 3
/// does not use `bounds()` for culling, only auto-framing, so the
/// approximation is fine until Plan 5 introduces proper screen-space
/// bounds.
#[derive(Copy, Clone)]
pub struct Point {
    /// World-space position, stored as a PGA3 trivector.
    pub position: PgaPoint,
    /// Disk color (linear sRGB). Alpha is multiplied by the disk's
    /// edge-antialiasing factor in the fragment shader.
    pub color: Color,
    /// Disk radius in screen-space pixels.
    pub radius: f32,
}

impl Point {
    /// Construct a point with the supplied position, color, and
    /// screen-pixel radius.
    pub fn new(position: PgaPoint, color: Color, radius: f32) -> Self {
        Self {
            position,
            color,
            radius,
        }
    }

    /// Convert this drawable into the GPU-ready [`PointInstance`]
    /// expected by [`PointPipeline`](crate::render::pipelines::PointPipeline).
    ///
    /// Divides the PGA3 trivector by its `e_123` weight via
    /// [`PgaPoint::to_euclidean`] to recover affine `(x, y, z)`, then
    /// packs into a 48-byte instance with `world_pos.w = 1.0` (the
    /// homogeneous-finite-point marker).
    pub fn to_instance(&self) -> PointInstance {
        let xyz = self.position.to_euclidean();
        PointInstance {
            world_pos: [xyz[0], xyz[1], xyz[2], 1.0],
            color: self.color.to_array(),
            radius: self.radius,
            _pad: [0.0; 3],
        }
    }
}

impl Drawable for Point {
    fn record(&self, ctx: &mut FrameContext<'_>, _camera: &Camera) {
        // Plan 3 Task 12: push one PointInstance into the per-frame
        // accumulator on `FrameContext`. The Scene::render_with frame
        // loop drains the accumulator each frame, uploads it to the
        // per-scene PointInstanceBuffer, and issues the bind + draw.
        ctx.point_instances.push(self.to_instance());
    }

    fn bounds(&self) -> Aabb {
        let xyz = self.position.to_euclidean();
        Aabb {
            min: [
                xyz[0] - self.radius,
                xyz[1] - self.radius,
                xyz[2] - self.radius,
            ],
            max: [
                xyz[0] + self.radius,
                xyz[1] + self.radius,
                xyz[2] + self.radius,
            ],
        }
    }
}

impl Translatable for Point {
    /// Translate the point by adding `delta` to its Euclidean
    /// coordinates and rebuilding a unit-weight PGA3 point.
    fn translate_by(&mut self, delta: [f32; 3]) {
        let xyz = self.position.to_euclidean();
        self.position = pga3::point(xyz[0] + delta[0], xyz[1] + delta[1], xyz[2] + delta[2]);
    }
}

impl Rotatable for Point {
    /// Rotate the point around the supplied PGA3 axis line by `angle`
    /// radians, via the rotor sandwich product.
    ///
    /// The axis bivector is taken straight from the line's
    /// [`Multivector`](rotorlab_ga::Multivector) storage so that
    /// rotations composed by the caller (axis built once, multiple
    /// rotations of the same drawable) remain numerically consistent.
    fn rotate_by(&mut self, axis: PgaLine, angle: f32) {
        let bivector = Bivector(axis.0);
        let rotor = pga3::rotor(bivector, angle);
        self.position = PgaPoint(rotor.0.apply(&self.position.0));
    }
}

impl HasOpacity for Point {
    /// Drive the disk's straight alpha channel.
    fn set_opacity(&mut self, alpha: f32) {
        self.color.a = alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rotorlab_ga::pga3::point as pga_point;

    #[test]
    fn to_instance_unit_weight() {
        let p = Point::new(pga_point(1.0, 2.0, 3.0), Color::RED, 4.0);
        let inst = p.to_instance();
        assert_eq!(inst.world_pos, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(inst.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(inst.radius, 4.0);
    }

    #[test]
    fn point_bounds_are_centered_at_position() {
        let p = Point::new(pga_point(5.0, 0.0, 0.0), Color::WHITE, 2.0);
        let b = p.bounds();
        assert_eq!(b.min, [3.0, -2.0, -2.0]);
        assert_eq!(b.max, [7.0, 2.0, 2.0]);
    }

    #[test]
    fn record_emits_one_instance() {
        // We cannot construct a real `FrameContext` without Vulkan, so
        // we exercise `to_instance` directly and assert on its shape.
        // Tasks 11-12 will exercise the full `record` path with a live
        // command buffer.
        let p = Point::new(pga_point(0.0, 0.0, 0.0), Color::BLUE, 1.0);
        let inst = p.to_instance();
        assert_eq!(inst.world_pos[3], 1.0);
        assert_eq!(inst.color[2], 1.0);
        assert_eq!(inst.radius, 1.0);
    }

    #[test]
    fn point_default_z_order_is_zero() {
        let p = Point::new(pga_point(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        assert_eq!(p.z_order(), 0.0);
    }
}
