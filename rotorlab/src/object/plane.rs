//! The [`Plane`] drawable: a PGA3 plane rendered as a four-edge
//! wireframe quad via the
//! [`LinePipeline`](crate::render::pipelines::LinePipeline).
//!
//! Plan 3 Task 8 ships the wireframe-only path: the plane is clipped to
//! a square of side `2 * view_extent` centered at the plane's closest
//! point to the world origin, and rendered as four [`Line`] edges
//! sharing a common [`Stroke`]. The four corners are computed by
//! sweeping `view_extent` units along an in-plane orthonormal basis
//! `(u, v)` derived from the plane normal.
//!
//! Plan 3 simplification. The clipping geometry is a fixed square in
//! the plane, not a real intersection with the camera frustum. The
//! orthographic-bounding-box and perspective-near-plane intersection
//! paths land in Plan 5+ once the camera carries enough state to drive
//! them. Filled translucent rendering (the typical "plane" look in
//! Manim-style scenes) is deferred to Plan 6's `FillPipeline`.
//!
//! Bitmask convention. PGA3 planes are grade-1 elements with basis
//! `e1 = 0b0001`, `e2 = 0b0010`, `e3 = 0b0100`, `e0 = 0b1000`. The
//! plane equation `a*x + b*y + c*z = d` is encoded as
//! `a*e1 + b*e2 + c*e3 + d*e0`; the xy-plane (`z = 0`) is therefore the
//! pure `e3` blade.

use crate::camera::Camera;
use crate::object::drawable::{Aabb, Drawable};
use crate::object::line::Line;
use crate::object::style::Stroke;
use crate::scene::FrameContext;
use rotorlab_ga::pga3::{self, Plane as PgaPlane};

/// A drawable plane rendered as a four-edge wireframe quad.
///
/// The quad is the square of side `2 * view_extent` centered at the
/// plane's closest point to the world origin. Plan 3 ships the
/// wireframe path only; the filled translucent variant is deferred to
/// Plan 6's `FillPipeline`.
#[derive(Copy, Clone)]
pub struct Plane {
    /// The PGA3 plane (grade-1 vector with components on `e1`, `e2`,
    /// `e3`, `e0` encoding `a*x + b*y + c*z = d`).
    pub plane: PgaPlane,
    /// Half-side of the rendered square, in world units.
    pub view_extent: f32,
    /// Edge stroke styling shared by all four wireframe edges.
    pub stroke: Stroke,
}

impl Plane {
    /// Construct a plane drawable from a PGA3 plane, a view extent, and
    /// an edge stroke.
    pub fn new(plane: PgaPlane, view_extent: f32, stroke: Stroke) -> Self {
        Self {
            plane,
            view_extent,
            stroke,
        }
    }

    /// Compute the four wireframe corners of the visible portion of the
    /// plane by sweeping `view_extent` units along each in-plane basis
    /// axis from the plane's closest point to the world origin.
    ///
    /// The in-plane basis `(u, v)` is built from the plane normal
    /// `n = (a, b, c)` by Gram-Schmidt against a world-space reference
    /// vector chosen to avoid colinearity with `n`: world-x when
    /// `|n.x| < 0.9` (i.e. the normal is not nearly x-aligned), world-y
    /// otherwise. The corners are returned in the cyclic order
    /// `(-u, -v)`, `(+u, -v)`, `(+u, +v)`, `(-u, +v)` so that
    /// connecting consecutive corners yields a closed quad.
    ///
    /// Plan 3 ships orthographic-only with a fixed view extent; the
    /// orthographic bounding-box and perspective near-frustum-plane
    /// paths land in Plan 5+.
    ///
    /// Degenerate input (a plane with zero Euclidean normal) falls
    /// back to the xy-plane at the origin so that the returned corners
    /// remain well defined; Plan 5 will surface this case as a
    /// `Result` once plane construction can plausibly fail.
    pub fn corners(&self) -> [[f32; 3]; 4] {
        let a = self.plane.0.get(0b0001);
        let b = self.plane.0.get(0b0010);
        let c = self.plane.0.get(0b0100);
        let d = self.plane.0.get(0b1000);

        let len_sq = a * a + b * b + c * c;
        let (nx, ny, nz, inv_len) = if len_sq > 1e-12 {
            let inv = 1.0 / len_sq.sqrt();
            (a * inv, b * inv, c * inv, inv)
        } else {
            (0.0, 0.0, 1.0, 1.0)
        };

        // Closest point on the plane to the world origin.
        let signed_dist = d * inv_len;
        let p0 = [nx * signed_dist, ny * signed_dist, nz * signed_dist];

        // In-plane orthonormal basis (u, v) by Gram-Schmidt against a
        // world reference vector chosen to avoid colinearity with n.
        let (rx, ry, rz) = if nx.abs() < 0.9 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };
        let dot_rn = rx * nx + ry * ny + rz * nz;
        let mut ux = rx - dot_rn * nx;
        let mut uy = ry - dot_rn * ny;
        let mut uz = rz - dot_rn * nz;
        let u_len_sq = ux * ux + uy * uy + uz * uz;
        if u_len_sq > 1e-12 {
            let inv_u = 1.0 / u_len_sq.sqrt();
            ux *= inv_u;
            uy *= inv_u;
            uz *= inv_u;
        }
        // v = n x u.
        let vx = ny * uz - nz * uy;
        let vy = nz * ux - nx * uz;
        let vz = nx * uy - ny * ux;

        let e = self.view_extent;
        let corner = |su: f32, sv: f32| {
            [
                p0[0] + su * e * ux + sv * e * vx,
                p0[1] + su * e * uy + sv * e * vy,
                p0[2] + su * e * uz + sv * e * vz,
            ]
        };
        [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ]
    }

    /// Generate the four edge [`Line`]s that form the wireframe quad,
    /// each carrying this plane's [`Stroke`].
    ///
    /// Edges are produced in the same cyclic order as [`Plane::corners`],
    /// so the segments form a closed loop: `c0->c1`, `c1->c2`, `c2->c3`,
    /// `c3->c0`.
    pub fn edges(&self) -> [Line; 4] {
        let c = self.corners();
        let p = |xyz: [f32; 3]| pga3::point(xyz[0], xyz[1], xyz[2]);
        [
            Line::new(p(c[0]), p(c[1]), self.stroke),
            Line::new(p(c[1]), p(c[2]), self.stroke),
            Line::new(p(c[2]), p(c[3]), self.stroke),
            Line::new(p(c[3]), p(c[0]), self.stroke),
        ]
    }
}

impl Drawable for Plane {
    fn record(&self, ctx: &mut FrameContext<'_>, camera: &Camera) {
        for edge in self.edges() {
            edge.record(ctx, camera);
        }
    }

    fn bounds(&self) -> Aabb {
        let c = self.corners();
        let mut min = c[0];
        let mut max = c[0];
        for corner in &c[1..] {
            for axis in 0..3 {
                min[axis] = min[axis].min(corner[axis]);
                max[axis] = max[axis].max(corner[axis]);
            }
        }
        // Inflate by the screen-pixel stroke thickness, treated as a
        // world-unit half-extent. Same approximation as `Point` and
        // `Line`; Plan 5+ replaces this with proper screen-space
        // bounds.
        let t = self.stroke.thickness;
        Aabb {
            min: [min[0] - t, min[1] - t, min[2] - t],
            max: [max[0] + t, max[1] + t, max[2] + t],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::style::{Color, Stroke};
    use rotorlab_ga::multivector::Multivector;
    use rotorlab_ga::pga3;

    fn xy_plane() -> PgaPlane {
        // Construct the plane z = 0 directly: only the e3 coefficient
        // is +1.
        let mut mv: Multivector<pga3::Pga3> = Multivector::zero();
        mv.set(0b0100, 1.0);
        PgaPlane(mv)
    }

    #[test]
    fn xy_plane_corners_are_axis_aligned_at_extent() {
        let p = Plane::new(xy_plane(), 5.0, Stroke::default_white());
        let c = p.corners();
        let xs: Vec<f32> = c.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = c.iter().map(|p| p[1]).collect();
        let zs: Vec<f32> = c.iter().map(|p| p[2]).collect();
        for z in &zs {
            assert!(z.abs() < 1e-4, "z: {}", z);
        }
        let count_pos_x = xs.iter().filter(|x| (**x - 5.0).abs() < 1e-4).count();
        let count_neg_x = xs.iter().filter(|x| (**x + 5.0).abs() < 1e-4).count();
        let count_pos_y = ys.iter().filter(|y| (**y - 5.0).abs() < 1e-4).count();
        let count_neg_y = ys.iter().filter(|y| (**y + 5.0).abs() < 1e-4).count();
        assert_eq!(count_pos_x, 2, "expected 2 corners with x=+5");
        assert_eq!(count_neg_x, 2, "expected 2 corners with x=-5");
        assert_eq!(count_pos_y, 2, "expected 2 corners with y=+5");
        assert_eq!(count_neg_y, 2, "expected 2 corners with y=-5");
    }

    #[test]
    fn xy_plane_edges_form_closed_loop() {
        let p = Plane::new(xy_plane(), 5.0, Stroke::default_white());
        let edges = p.edges();
        assert_eq!(edges.len(), 4);
        for i in 0..4 {
            let cur_b = edges[i].b.to_euclidean();
            let next_a = edges[(i + 1) % 4].a.to_euclidean();
            for axis in 0..3 {
                assert!(
                    (cur_b[axis] - next_a[axis]).abs() < 1e-4,
                    "edge {} b != edge {} a at axis {}",
                    i,
                    (i + 1) % 4,
                    axis
                );
            }
        }
    }

    #[test]
    fn plane_bounds_inflate_by_stroke_thickness() {
        let p = Plane::new(xy_plane(), 5.0, Stroke::new(Color::WHITE, 0.5));
        let b = p.bounds();
        assert!((b.min[0] + 5.5).abs() < 1e-4);
        assert!((b.max[0] - 5.5).abs() < 1e-4);
        assert!((b.min[1] + 5.5).abs() < 1e-4);
        assert!((b.max[1] - 5.5).abs() < 1e-4);
        assert!((b.min[2] + 0.5).abs() < 1e-4);
        assert!((b.max[2] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn plane_default_z_order_is_zero() {
        let p = Plane::new(xy_plane(), 5.0, Stroke::default_white());
        assert_eq!(p.z_order(), 0.0);
    }
}
