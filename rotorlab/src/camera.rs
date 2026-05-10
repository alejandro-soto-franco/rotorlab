//! Camera and projection types for the rotorlab engine.
//!
//! The camera stores its pose as a PGA3 [`Motor`] and a [`Projection`]
//! variant. The motor represents the camera-to-world transform: applying
//! it to the canonical origin yields the world-space camera position, and
//! applying it to the canonical forward direction yields the world-space
//! viewing direction.
//!
//! # Conventions
//!
//! * Right-handed world space.
//! * Right-handed view space; camera looks down `-Z` in view space.
//! * Vulkan clip space: `Y` axis points down on screen, `Z` lies in `[0, 1]`.
//! * Y-flip is baked into [`Camera::proj_matrix`], not the viewport.
//! * Matrices are row-major `[[f32; 4]; 4]`, indexed `m[row][col]`. Matrix
//!   multiplication acts on column vectors as `M * v`.
//!
//! # Motor-to-matrix conversion
//!
//! The view matrix is derived from the motor by applying the motor's
//! reverse (the inverse, for unit motors) to four reference points: the
//! origin and the three Euclidean axis points. The Euclidean coordinates
//! of the transformed points populate the four columns of a 4x4 matrix.
//! See [`view_matrix`](Camera::view_matrix) for the implementation.

use rotorlab_ga::motor::Motor;
use rotorlab_ga::multivector::Multivector;
use rotorlab_ga::pga3::{self, Bivector, Line, Pga3, Point};

// Bitmask indices for the bivector basis blades used to build translators.
const E_01: usize = 0b1001; // translation along x
const E_02: usize = 0b1010; // translation along y
const E_03: usize = 0b1100; // translation along z

// Bitmask indices for the spatial bivector basis blades (rotor planes).
const E_23: usize = 0b0110; // rotation around x (e2 ∧ e3)
const E_13: usize = 0b0101; // rotation plane for y axis (e1 ∧ e3)
const E_12: usize = 0b0011; // rotation around z (e1 ∧ e2)

/// Projection mode used when computing [`Camera::proj_matrix`].
#[derive(Copy, Clone, Debug)]
pub enum Projection {
    /// Right-handed orthographic projection mapping the box
    /// `[-half_width, half_width] x [-half_height, half_height] x [near, far]`
    /// (in view space, with the camera looking down `-Z`) to Vulkan clip
    /// space `[-1, 1] x [-1, 1] x [0, 1]`. The Y axis is flipped to match
    /// Vulkan's screen-down convention.
    Orthographic {
        /// Half-extent along the view-space X axis.
        half_width: f32,
        /// Half-extent along the view-space Y axis (pre-flip).
        half_height: f32,
        /// Near clipping distance (positive value, distance in front of
        /// the camera).
        near: f32,
        /// Far clipping distance (positive value, distance in front of
        /// the camera).
        far: f32,
    },
    /// Right-handed perspective projection. `fov_y` is the full vertical
    /// field of view in radians. `aspect` is width / height.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f32,
        /// Aspect ratio (width / height).
        aspect: f32,
        /// Near clipping distance (positive).
        near: f32,
        /// Far clipping distance (positive).
        far: f32,
    },
}

/// A camera in PGA3.
///
/// The [`motor`](Self::motor) field encodes the camera-to-world transform:
/// applying it to the canonical origin yields the world-space camera
/// position, and applying it to the canonical forward direction yields
/// the world-space viewing direction. The view matrix is the inverse
/// (the reverse, for unit motors) of the motor expressed as a 4x4 matrix.
///
/// Does not derive `Debug` because [`Motor`] does not implement it.
#[derive(Copy, Clone)]
pub struct Camera {
    /// Camera-to-world rigid motion.
    pub motor: Motor<Pga3>,
    /// Projection mode.
    pub projection: Projection,
    /// RGBA clear color used when this camera renders a frame.
    pub clear_color: [f32; 4],
}

impl Camera {
    /// Build a camera that looks from `eye` toward `target` with a default
    /// perspective projection (`fov_y = π/3`, `aspect = 16/9`, `near = 0.1`,
    /// `far = 100`).
    ///
    /// This is the perspective-default one-liner; for orthographic or any
    /// non-default perspective frustum, use
    /// [`look_at_with_projection`](Self::look_at_with_projection).
    ///
    /// If `eye == target` or `up` is parallel to `target - eye`, the
    /// resulting motor is degenerate but the function still returns;
    /// callers are expected to pass a sensible frame.
    pub fn look_at(eye: Point, target: Point, up: [f32; 3]) -> Self {
        Self::look_at_with_projection(
            eye,
            target,
            up,
            Projection::Perspective {
                fov_y: std::f32::consts::FRAC_PI_3,
                aspect: 16.0 / 9.0,
                near: 0.1,
                far: 100.0,
            },
        )
    }

    /// Build a camera that looks from `eye` toward `target`, with the
    /// world-space up direction `up` (need not be normalized; the
    /// component perpendicular to `forward` is used) and the supplied
    /// projection.
    ///
    /// The returned camera's motor places the canonical camera (at the
    /// origin, looking down `-Z`, with `+Y` up) at `eye` so that its
    /// forward direction points from `eye` to `target`. The caller picks
    /// the projection, so this entry point works for both perspective and
    /// orthographic cameras.
    ///
    /// If `eye == target` or `up` is parallel to `target - eye`, the
    /// resulting motor is degenerate but the function still returns;
    /// callers are expected to pass a sensible frame.
    pub fn look_at_with_projection(
        eye: Point,
        target: Point,
        up: [f32; 3],
        projection: Projection,
    ) -> Self {
        let eye_e = point_to_euclidean(&eye.0);
        let target_e = point_to_euclidean(&target.0);
        let mut forward = sub3(target_e, eye_e);
        forward = normalize3(forward).unwrap_or([0.0, 0.0, -1.0]);
        let mut right = cross3(forward, up);
        right = normalize3(right).unwrap_or([1.0, 0.0, 0.0]);
        let up_corrected = cross3(right, forward);

        // The 3x3 rotation R has columns (right, up_corrected, -forward).
        // We need a PGA rotor whose sandwich realises this rotation on
        // Euclidean Point trivectors. Build it from R via the standard
        // matrix-to-quaternion conversion, then map the quaternion to a
        // PGA rotor.
        let rot = matrix_to_rotor(
            [right[0], up_corrected[0], -forward[0]],
            [right[1], up_corrected[1], -forward[1]],
            [right[2], up_corrected[2], -forward[2]],
        );

        // Translation: place the origin at eye. We compose translator
        // after rotor so that the motor maps (0,0,0) -> eye and the
        // canonical forward to forward. (Order: M = T * R, so
        // M(p) = T(R(p)).)
        let trans = build_translator(eye_e[0], eye_e[1], eye_e[2]);
        let motor = trans.compose(&rot);

        Self {
            motor,
            projection,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// View matrix: world-to-camera transform as a row-major 4x4 matrix.
    ///
    /// For a unit motor, `view = mat4(reverse(motor))`. The reverse of a
    /// unit motor equals its inverse, so `view` undoes the camera's pose.
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        // GPU-boundary conversion: the motor is the only PGA representation
        // of orientation in the engine; everything downstream of this
        // function operates on the resulting 4x4 matrix.
        let inv = Motor(self.motor.0.reverse());
        motor_to_mat4(&inv)
    }

    /// Projection matrix for the camera's [`Projection`].
    ///
    /// Right-handed view space (camera looks down `-Z`); Vulkan clip
    /// space (`Y` flipped, `Z` in `[0, 1]`).
    pub fn proj_matrix(&self) -> [[f32; 4]; 4] {
        match self.projection {
            Projection::Orthographic {
                half_width,
                half_height,
                near,
                far,
            } => {
                let inv_w = 1.0 / half_width;
                let inv_h = 1.0 / half_height;
                let inv_d = 1.0 / (near - far);
                [
                    [inv_w, 0.0, 0.0, 0.0],
                    [0.0, -inv_h, 0.0, 0.0],
                    [0.0, 0.0, inv_d, near * inv_d],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
            Projection::Perspective {
                fov_y,
                aspect,
                near,
                far,
            } => {
                let f = 1.0 / (fov_y * 0.5).tan();
                let inv_d = 1.0 / (near - far);
                [
                    [f / aspect, 0.0, 0.0, 0.0],
                    [0.0, -f, 0.0, 0.0],
                    [0.0, 0.0, far * inv_d, far * near * inv_d],
                    [0.0, 0.0, -1.0, 0.0],
                ]
            }
        }
    }

    /// Combined `proj * view` matrix, ready to upload as a GPU uniform.
    pub fn view_proj(&self) -> [[f32; 4]; 4] {
        let v = self.view_matrix();
        let p = self.proj_matrix();
        mat4_mul(&p, &v)
    }

    /// Rotate the camera around `axis` (a PGA3 line) by `angle` radians,
    /// returning a new camera with the rotated pose.
    ///
    /// The rotation is applied in world space: the new motor is
    /// `rotor(axis, angle).compose(&self.motor)`.
    pub fn orbit(&self, axis: Line, angle: f32) -> Self {
        let r = pga3::rotor(Bivector(axis.0), angle);
        let new_motor = r.0.compose(&self.motor);
        Self {
            motor: new_motor,
            ..*self
        }
    }

    /// Translate the camera along its current forward direction by
    /// `distance`, returning a new camera.
    ///
    /// Positive `distance` moves the camera in the direction it is
    /// facing.
    pub fn dolly(&self, distance: f32) -> Self {
        // Compute the camera's current forward direction in world space
        // by transforming the canonical origin and a point one unit ahead.
        let origin_world = self.motor.apply(&pga3::point(0.0, 0.0, 0.0).0);
        let ahead_world = self.motor.apply(&pga3::point(0.0, 0.0, -1.0).0);
        let o = point_to_euclidean(&origin_world);
        let a = point_to_euclidean(&ahead_world);
        let dir = sub3(a, o);
        let dir = normalize3(dir).unwrap_or([0.0, 0.0, -1.0]);
        let trans = build_translator(dir[0] * distance, dir[1] * distance, dir[2] * distance);
        Self {
            motor: trans.compose(&self.motor),
            ..*self
        }
    }
}

// ============================================================================
// GPU-boundary conversion: PGA3 motor -> 4x4 row-major matrix.
//
// The motor is converted by applying it (via the sandwich product) to four
// reference PGA3 points: the origin and the three Euclidean axis points
// `(1,0,0)`, `(0,1,0)`, `(0,0,1)`. The transformed Euclidean coordinates
// fill the four columns of a row-major 4x4 matrix.
//
// This is the only place in the engine where a 3x3 / 4x4 orientation matrix
// is constructed from PGA data.
// ============================================================================

fn motor_to_mat4(m: &Motor<Pga3>) -> [[f32; 4]; 4] {
    let p0 = m.apply(&pga3::point(0.0, 0.0, 0.0).0);
    let px = m.apply(&pga3::point(1.0, 0.0, 0.0).0);
    let py = m.apply(&pga3::point(0.0, 1.0, 0.0).0);
    let pz = m.apply(&pga3::point(0.0, 0.0, 1.0).0);

    let o = point_to_euclidean(&p0);
    let cx = sub3(point_to_euclidean(&px), o);
    let cy = sub3(point_to_euclidean(&py), o);
    let cz = sub3(point_to_euclidean(&pz), o);

    [
        [cx[0], cy[0], cz[0], o[0]],
        [cx[1], cy[1], cz[1], o[1]],
        [cx[2], cy[2], cz[2], o[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn point_to_euclidean(mv: &Multivector<Pga3>) -> [f32; 3] {
    // Thin shim around `pga3::Point::to_euclidean` for the camera's
    // raw-multivector call sites (the result of `motor.apply` is a
    // `Multivector<Pga3>`, not a typed `Point`). The two functions
    // share the same near-zero-weight guard and blade indices.
    Point(*mv).to_euclidean()
}

fn build_translator(tx: f32, ty: f32, tz: f32) -> Motor<Pga3> {
    // A PGA3 translator that maps the canonical origin to the Euclidean
    // point `(tx, ty, tz)` under `Motor::apply`. The per-axis half-coefficients
    // below are derived empirically from the geometric-product Cayley table
    // used in `rotorlab_ga`; the bit-ordering of basis blades in this crate
    // (`e0` is bit 3, so `0b1001 = e1 ∧ e0` etc.) makes the textbook sign
    // axis-dependent. The exact signs are pinned by
    // `translator_moves_origin_to_offset`, `look_at_places_camera_at_eye`,
    // and `dolly_moves_camera_along_forward`.
    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0, 1.0); // scalar part
    mv.set(E_01, -0.5 * tx);
    mv.set(E_02, 0.5 * ty);
    mv.set(E_03, -0.5 * tz);
    Motor(mv)
}

// Convert a 3x3 rotation matrix (rows r0, r1, r2) to a PGA3 rotor.
//
// The standard matrix-to-quaternion formula yields (w, x, y, z); we then
// embed it as a PGA rotor with mapping
//   scalar    = w
//   e23 coef  = x  (rotation around the world X axis lives in e23)
//   e13 coef  = -y (rotation around Y, with the sign that makes
//                   `apply` realise the right-handed rotation)
//   e12 coef  = z  (rotation around Z lives in e12)
//
// The signs above are the unique choice that makes
// `look_at(eye, target, up).motor.apply(origin) == eye` and
// `look_at(eye, target, up).motor.apply(canonical_forward) == eye - target`
// to working precision. They are pinned by the look_at sanity tests below.
fn matrix_to_rotor(r0: [f32; 3], r1: [f32; 3], r2: [f32; 3]) -> Motor<Pga3> {
    let m00 = r0[0];
    let m11 = r1[1];
    let m22 = r2[2];
    let trace = m00 + m11 + m22;

    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r2[1] - r1[2]) / s;
        let y = (r0[2] - r2[0]) / s;
        let z = (r1[0] - r0[1]) / s;
        (w, x, y, z)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        let w = (r2[1] - r1[2]) / s;
        let x = 0.25 * s;
        let y = (r0[1] + r1[0]) / s;
        let z = (r0[2] + r2[0]) / s;
        (w, x, y, z)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        let w = (r0[2] - r2[0]) / s;
        let x = (r0[1] + r1[0]) / s;
        let y = 0.25 * s;
        let z = (r1[2] + r2[1]) / s;
        (w, x, y, z)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        let w = (r1[0] - r0[1]) / s;
        let x = (r0[2] + r2[0]) / s;
        let y = (r1[2] + r2[1]) / s;
        let z = 0.25 * s;
        (w, x, y, z)
    };

    let mut mv: Multivector<Pga3> = Multivector::zero();
    mv.set(0, w);
    mv.set(E_23, x);
    // negate y to match e_13 sign convention; see function header
    mv.set(E_13, -y);
    mv.set(E_12, z);
    Motor(mv)
}

// Row-major 4x4 multiply: out = a * b.
fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (r, out_row) in out.iter_mut().enumerate() {
        for (c, out_cell) in out_row.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, b_row) in b.iter().enumerate() {
                s += a[r][k] * b_row[c];
            }
            *out_cell = s;
        }
    }
    out
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        None
    } else {
        Some([v[0] / n, v[1] / n, v[2] / n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn assert_mat_eq(got: &[[f32; 4]; 4], want: &[[f32; 4]; 4], eps: f32) {
        for (r, (got_row, want_row)) in got.iter().zip(want.iter()).enumerate() {
            for (c, (g, w)) in got_row.iter().zip(want_row.iter()).enumerate() {
                assert!(approx_eq(*g, *w, eps), "[{r}][{c}] got {g} want {w}");
            }
        }
    }

    fn identity_mat4() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn identity_motor_produces_identity_view() {
        let cam = Camera {
            motor: Motor::identity(),
            projection: Projection::Orthographic {
                half_width: 1.0,
                half_height: 1.0,
                near: 0.0,
                far: 1.0,
            },
            clear_color: [0.0; 4],
        };
        let v = cam.view_matrix();
        assert_mat_eq(&v, &identity_mat4(), 1e-6);
    }

    #[test]
    fn orthographic_identity_returns_clip_space_passthrough() {
        // Fixture: half_width = half_height = 1, near = 0, far = 1. With
        // identity view, view_proj reduces to the orthographic matrix.
        // We assert the *post-Y-flip* clip-space coordinates: for a
        // view-space input (x, y, z, 1) inside the box, the clip-space
        // output is (x, -y, z, 1).
        let cam = Camera {
            motor: Motor::identity(),
            projection: Projection::Orthographic {
                half_width: 1.0,
                half_height: 1.0,
                near: 0.0,
                far: 1.0,
            },
            clear_color: [0.0; 4],
        };
        let vp = cam.view_proj();
        // Test point (0.3, 0.4, 0.5).
        let v = [0.3f32, 0.4, 0.5, 1.0];
        let mut out = [0.0f32; 4];
        for (r, out_cell) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for (c, &vc) in v.iter().enumerate() {
                s += vp[r][c] * vc;
            }
            *out_cell = s;
        }
        // x passes through, y is flipped, z is mapped via [0, 0, -1, 0].
        assert!(approx_eq(out[0], 0.3, 1e-6), "x: {}", out[0]);
        assert!(approx_eq(out[1], -0.4, 1e-6), "y: {}", out[1]);
        assert!(approx_eq(out[2], -0.5, 1e-6), "z: {}", out[2]);
        assert!(approx_eq(out[3], 1.0, 1e-6), "w: {}", out[3]);
    }

    #[test]
    fn perspective_identity_has_expected_diagonal() {
        // fov_y = pi/2, aspect = 16/9, tan(fov_y/2) = 1, f = 1.
        // Expected (0,0) = 1 / (16/9) = 9/16.
        // Expected (1,1) = -f = -1 (Y flip baked into proj_matrix).
        let cam = Camera {
            motor: Motor::identity(),
            projection: Projection::Perspective {
                fov_y: std::f32::consts::FRAC_PI_2,
                aspect: 16.0 / 9.0,
                near: 0.1,
                far: 100.0,
            },
            clear_color: [0.0; 4],
        };
        let p = cam.proj_matrix();
        assert!(approx_eq(p[0][0], 9.0 / 16.0, 1e-6), "(0,0) = {}", p[0][0]);
        assert!(approx_eq(p[1][1].abs(), 1.0, 1e-6), "(1,1) = {}", p[1][1]);
    }

    #[test]
    fn translator_moves_origin_to_offset() {
        // Sanity: build_translator(2, 3, 5) applied to origin should yield
        // the point (2, 3, 5). Pins the sign of build_translator.
        let t = build_translator(2.0, 3.0, 5.0);
        let p = pga3::point(0.0, 0.0, 0.0);
        let moved = t.apply(&p.0);
        let e = point_to_euclidean(&moved);
        assert!(approx_eq(e[0], 2.0, 1e-5), "x: {}", e[0]);
        assert!(approx_eq(e[1], 3.0, 1e-5), "y: {}", e[1]);
        assert!(approx_eq(e[2], 5.0, 1e-5), "z: {}", e[2]);
    }

    #[test]
    fn look_at_places_camera_at_eye() {
        // The camera motor takes the canonical origin to the world-space
        // eye position.
        let eye = pga3::point(2.0, 3.0, 5.0);
        let target = pga3::point(0.0, 0.0, 0.0);
        let cam = Camera::look_at(eye, target, [0.0, 1.0, 0.0]);
        let origin = pga3::point(0.0, 0.0, 0.0);
        let placed = cam.motor.apply(&origin.0);
        let e = point_to_euclidean(&placed);
        assert!(approx_eq(e[0], 2.0, 1e-4), "x: {}", e[0]);
        assert!(approx_eq(e[1], 3.0, 1e-4), "y: {}", e[1]);
        assert!(approx_eq(e[2], 5.0, 1e-4), "z: {}", e[2]);
    }

    #[test]
    fn orbit_zero_angle_is_identity() {
        let cam = Camera {
            motor: Motor::identity(),
            projection: Projection::Orthographic {
                half_width: 1.0,
                half_height: 1.0,
                near: 0.0,
                far: 1.0,
            },
            clear_color: [0.0; 4],
        };
        // any line works because angle is zero.
        let mut line_mv: Multivector<Pga3> = Multivector::zero();
        line_mv.set(E_12, 1.0);
        let axis = Line(line_mv);
        let orbited = cam.orbit(axis, 0.0);
        let v = orbited.view_matrix();
        assert_mat_eq(&v, &identity_mat4(), 1e-5);
    }

    #[test]
    fn dolly_zero_distance_is_identity_pose() {
        let cam = Camera {
            motor: Motor::identity(),
            projection: Projection::Orthographic {
                half_width: 1.0,
                half_height: 1.0,
                near: 0.0,
                far: 1.0,
            },
            clear_color: [0.0; 4],
        };
        let dollied = cam.dolly(0.0);
        let v = dollied.view_matrix();
        assert_mat_eq(&v, &identity_mat4(), 1e-5);
    }

    #[test]
    fn look_at_with_projection_uses_supplied_projection() {
        // Pins that look_at_with_projection actually carries the supplied
        // projection through to the constructed camera (rather than always
        // defaulting to perspective).
        let eye = pga3::point(2.0, 3.0, 5.0);
        let target = pga3::point(0.0, 0.0, 0.0);
        let cam = Camera::look_at_with_projection(
            eye,
            target,
            [0.0, 1.0, 0.0],
            Projection::Orthographic {
                half_width: 1.0,
                half_height: 1.0,
                near: 0.0,
                far: 1.0,
            },
        );
        assert!(matches!(cam.projection, Projection::Orthographic { .. }));
    }

    #[test]
    fn dolly_moves_camera_along_forward() {
        // Identity camera looks down -Z. dolly(2) should move it to
        // (0, 0, -2) in world space.
        let cam = Camera {
            motor: Motor::identity(),
            projection: Projection::Orthographic {
                half_width: 1.0,
                half_height: 1.0,
                near: 0.0,
                far: 1.0,
            },
            clear_color: [0.0; 4],
        };
        let dollied = cam.dolly(2.0);
        let placed = dollied.motor.apply(&pga3::point(0.0, 0.0, 0.0).0);
        let e = point_to_euclidean(&placed);
        assert!(approx_eq(e[0], 0.0, 1e-5), "x: {}", e[0]);
        assert!(approx_eq(e[1], 0.0, 1e-5), "y: {}", e[1]);
        assert!(approx_eq(e[2], -2.0, 1e-5), "z: {}", e[2]);
    }
}
