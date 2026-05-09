//! The [`Line`] drawable: a line segment between two PGA3 points,
//! rendered as a screen-space-thickness rectangle via the
//! [`LinePipeline`](crate::render::pipelines::LinePipeline).
//!
//! Each [`Line`] carries two PGA3 endpoints and a [`Stroke`].
//! [`Drawable::record`] converts the line into a
//! [`LineInstance`](crate::render::pipelines::LineInstance) ready for
//! upload to the pipeline's instance buffer. The actual command-buffer
//! recording is wired in Plan 3 Tasks 11-12 once
//! [`FrameContext`](crate::scene::FrameContext) exposes an
//! instance-buffer staging API; Task 7 ships the geometry-to-GPU
//! conversion only.
//!
//! Plan 3 ships two constructors:
//! - [`Line::new`] takes two explicit PGA3 endpoints (the primary
//!   path).
//! - [`Line::from_pga`] takes a PGA3 line bivector plus a `view_extent`
//!   and returns the segment of length `2 * view_extent` centered at
//!   the line's closest point to the origin. Plan 3 simplifies the
//!   general case to lines through the origin (moment vanishes);
//!   Plan 5+ extends to general-position lines and a true bounding
//!   sphere intersection.

use crate::camera::Camera;
use crate::object::drawable::{Aabb, Drawable};
use crate::object::style::Stroke;
use crate::render::pipelines::LineInstance;
use crate::scene::FrameContext;
use rotorlab_ga::pga3::{self, Point as PgaPoint};

/// A drawable line segment: two PGA3 endpoints and a [`Stroke`].
///
/// The Aabb returned by [`Drawable::bounds`] approximates the
/// screen-pixel stroke thickness as a world-unit half-extent on each
/// axis. This is technically wrong (mixing world and pixel units), but
/// Plan 3 does not use `bounds()` for culling, only auto-framing, so
/// the approximation is fine until Plan 5 introduces proper
/// screen-space bounds.
#[derive(Copy, Clone)]
pub struct Line {
    /// First endpoint in world space (PGA3 trivector).
    pub a: PgaPoint,
    /// Second endpoint in world space (PGA3 trivector).
    pub b: PgaPoint,
    /// Stroke styling (color + screen-pixel thickness).
    pub stroke: Stroke,
}

impl Line {
    /// Construct a line segment from two explicit PGA3 endpoints and a
    /// [`Stroke`].
    pub fn new(a: PgaPoint, b: PgaPoint, stroke: Stroke) -> Self {
        Self { a, b, stroke }
    }

    /// Extract a finite segment from a PGA3 line bivector by walking
    /// `view_extent` units in each direction along the line's
    /// (normalized) Euclidean direction, starting from the closest
    /// point to the origin.
    ///
    /// The direction is read from the bivector's `e23`, `e13`, `e12`
    /// coefficients. The empirical sign convention (verified against
    /// `pga3::line_through(origin, e_k)` for `k = x, y, z`) is
    /// `d = (+get(e23), +get(e13), +get(e12))`, with bitmasks
    /// `0b0110`, `0b0101`, `0b0011`.
    ///
    /// **Plan 3 simplification.** Plan 3 only renders lines through
    /// the origin, where the Plücker moment `m = (e01, e02, e03)`
    /// vanishes and the closest point to the origin is the origin
    /// itself. The implementation therefore ignores the moment part
    /// and returns the segment `[-d̂ * view_extent, +d̂ * view_extent]`.
    /// **Plan 5+ follow-up:** extract the closest point via
    /// `(m × d) / |d|^2` (using the moment bitmasks `e01 = 0b1001`,
    /// `e02 = 0b1010`, `e03 = 0b1100`) and intersect with the camera
    /// frustum's bounding sphere instead of walking a fixed
    /// `view_extent`.
    ///
    /// Returns a stroke styled with [`Stroke::default_white`]; callers
    /// who need a different stroke construct via [`Line::new`] or
    /// mutate `stroke` after this returns.
    ///
    /// Degenerate input (a bivector with zero Euclidean direction part)
    /// falls back to the x-axis direction so that the returned segment
    /// is still well-defined; Plan 5 will surface this case as a
    /// `Result` once line construction can plausibly fail.
    pub fn from_pga(line: pga3::Line, view_extent: f32) -> Self {
        let dx = line.0.get(0b0110);
        let dy = line.0.get(0b0101);
        let dz = line.0.get(0b0011);
        let len_sq = dx * dx + dy * dy + dz * dz;
        let (nx, ny, nz) = if len_sq > 1e-12 {
            let inv = 1.0 / len_sq.sqrt();
            (dx * inv, dy * inv, dz * inv)
        } else {
            (1.0, 0.0, 0.0)
        };
        let a = pga3::point(-nx * view_extent, -ny * view_extent, -nz * view_extent);
        let b = pga3::point(nx * view_extent, ny * view_extent, nz * view_extent);
        Line::new(a, b, Stroke::default_white())
    }

    /// Convert this drawable into the GPU-ready [`LineInstance`]
    /// expected by [`LinePipeline`](crate::render::pipelines::LinePipeline).
    ///
    /// Each endpoint is divided by its homogeneous weight via
    /// [`PgaPoint::to_euclidean`] before being packed with `w = 1.0`
    /// (the homogeneous-finite-point marker).
    pub fn to_instance(&self) -> LineInstance {
        let ae = self.a.to_euclidean();
        let be = self.b.to_euclidean();
        LineInstance {
            a: [ae[0], ae[1], ae[2], 1.0],
            b: [be[0], be[1], be[2], 1.0],
            color: self.stroke.color.to_array(),
            thickness: self.stroke.thickness,
            _pad: [0.0; 3],
        }
    }
}

impl Drawable for Line {
    fn record(&self, ctx: &mut FrameContext<'_>, _camera: &Camera) {
        // Plan 3 Task 7 stops at the conversion; Tasks 11-12 wire the
        // resulting `LineInstance` through an instance-buffer staging
        // API on `FrameContext` that does not exist yet. Until then we
        // call `to_instance()` for its conversion side effect (and to
        // keep this method honest about doing the geometric work) and
        // discard the result.
        let _ = ctx;
        let _ = self.to_instance();
    }

    fn bounds(&self) -> Aabb {
        let ae = self.a.to_euclidean();
        let be = self.b.to_euclidean();
        let t = self.stroke.thickness;
        Aabb {
            min: [
                ae[0].min(be[0]) - t,
                ae[1].min(be[1]) - t,
                ae[2].min(be[2]) - t,
            ],
            max: [
                ae[0].max(be[0]) + t,
                ae[1].max(be[1]) + t,
                ae[2].max(be[2]) + t,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::style::{Color, Stroke};
    use rotorlab_ga::pga3;

    #[test]
    fn from_pga_x_axis_extent_five_returns_axis_aligned_segment() {
        let x_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(1.0, 0.0, 0.0));
        let line = Line::from_pga(x_axis, 5.0);
        let ae = line.a.to_euclidean();
        let be = line.b.to_euclidean();
        let (lo, hi) = if ae[0] < be[0] { (ae, be) } else { (be, ae) };
        assert!((lo[0] + 5.0).abs() < 1e-4, "lo.x: {}", lo[0]);
        assert!((hi[0] - 5.0).abs() < 1e-4, "hi.x: {}", hi[0]);
        assert!(lo[1].abs() < 1e-4 && lo[2].abs() < 1e-4);
        assert!(hi[1].abs() < 1e-4 && hi[2].abs() < 1e-4);
    }

    #[test]
    fn from_pga_y_axis_extent_three_axis_aligned() {
        let y_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(0.0, 1.0, 0.0));
        let line = Line::from_pga(y_axis, 3.0);
        let ae = line.a.to_euclidean();
        let be = line.b.to_euclidean();
        let (lo, hi) = if ae[1] < be[1] { (ae, be) } else { (be, ae) };
        assert!((lo[1] + 3.0).abs() < 1e-4, "lo.y: {}", lo[1]);
        assert!((hi[1] - 3.0).abs() < 1e-4, "hi.y: {}", hi[1]);
        assert!(lo[0].abs() < 1e-4 && lo[2].abs() < 1e-4);
        assert!(hi[0].abs() < 1e-4 && hi[2].abs() < 1e-4);
    }

    #[test]
    fn from_pga_z_axis_extent_two_axis_aligned() {
        let z_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(0.0, 0.0, 1.0));
        let line = Line::from_pga(z_axis, 2.0);
        let ae = line.a.to_euclidean();
        let be = line.b.to_euclidean();
        let (lo, hi) = if ae[2] < be[2] { (ae, be) } else { (be, ae) };
        assert!((lo[2] + 2.0).abs() < 1e-4, "lo.z: {}", lo[2]);
        assert!((hi[2] - 2.0).abs() < 1e-4, "hi.z: {}", hi[2]);
        assert!(lo[0].abs() < 1e-4 && lo[1].abs() < 1e-4);
        assert!(hi[0].abs() < 1e-4 && hi[1].abs() < 1e-4);
    }

    #[test]
    fn from_pga_default_stroke_is_white() {
        let x_axis = pga3::line_through(pga3::point(0.0, 0.0, 0.0), pga3::point(1.0, 0.0, 0.0));
        let line = Line::from_pga(x_axis, 1.0);
        assert_eq!(line.stroke.color, Color::WHITE);
        assert_eq!(line.stroke.thickness, 1.5);
    }

    #[test]
    fn to_instance_packs_endpoints() {
        let a = pga3::point(0.0, 0.0, 0.0);
        let b = pga3::point(3.0, 4.0, 0.0);
        let line = Line::new(a, b, Stroke::new(Color::GREEN, 2.0));
        let inst = line.to_instance();
        assert_eq!(inst.a, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(inst.b, [3.0, 4.0, 0.0, 1.0]);
        assert_eq!(inst.color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(inst.thickness, 2.0);
    }

    #[test]
    fn line_bounds_include_endpoints_and_thickness() {
        let line = Line::new(
            pga3::point(0.0, 0.0, 0.0),
            pga3::point(2.0, 4.0, 0.0),
            Stroke::new(Color::WHITE, 1.0),
        );
        let b = line.bounds();
        assert_eq!(b.min, [-1.0, -1.0, -1.0]);
        assert_eq!(b.max, [3.0, 5.0, 1.0]);
    }

    #[test]
    fn line_default_z_order_is_zero() {
        let line = Line::new(
            pga3::point(0.0, 0.0, 0.0),
            pga3::point(1.0, 0.0, 0.0),
            Stroke::default_white(),
        );
        assert_eq!(line.z_order(), 0.0);
    }
}
