//! Visual-style primitives shared by drawables.
//!
//! Plan 3 Task 5 introduces [`Color`], a linear-space RGBA value used by
//! the [`Point`](crate::object::Point) drawable and (Task 7) the
//! [`Line`](crate::object::Line) drawable. Plan 3 Task 7 adds
//! [`Stroke`], the line-styling primitive carried by `Line` (and, in
//! Task 8, by the wireframe edges of `Plane`).

/// Linear-space RGBA color with `f32` components.
///
/// Components live in `[0, 1]` for in-gamut colors but the type does
/// not enforce that range; HDR / wide-gamut workflows can store
/// out-of-range values and rely on the pipeline to tone-map. Alpha is
/// straight (not premultiplied); the point pipeline multiplies the
/// alpha by an antialiasing edge factor in the fragment shader.
///
/// Plan 4 / Plan 6 may add gamma helpers (`from_srgb_u8`, `to_srgb`);
/// for now everything in the engine is linear-sRGB f32.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    /// Red channel, linear sRGB.
    pub r: f32,
    /// Green channel, linear sRGB.
    pub g: f32,
    /// Blue channel, linear sRGB.
    pub b: f32,
    /// Straight alpha (not premultiplied).
    pub a: f32,
}

impl Color {
    /// Opaque white `(1, 1, 1, 1)`.
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// Opaque black `(0, 0, 0, 1)`.
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Opaque pure red `(1, 0, 0, 1)`.
    pub const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Opaque pure green `(0, 1, 0, 1)`.
    pub const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };

    /// Opaque pure blue `(0, 0, 1, 1)`.
    pub const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };

    /// Build a color from explicit RGBA components.
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Color { r, g, b, a }
    }

    /// Build an opaque color from explicit RGB components (alpha = 1).
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Color { r, g, b, a: 1.0 }
    }

    /// Pack the color into a `[r, g, b, a]` array suitable for
    /// uploading to a GPU instance buffer.
    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// Stroke styling for lines, plane wireframes, and any future
/// path-stroked drawable.
///
/// `thickness` is in screen-space pixels (matches the `radius`
/// semantic on [`Point`](crate::object::Point)): a stroke with
/// `thickness = 2.0` covers a 2-pixel-wide band on screen,
/// regardless of camera distance or projection.
#[derive(Copy, Clone, Debug)]
pub struct Stroke {
    /// Stroke color (linear sRGB).
    pub color: Color,
    /// Stroke thickness in screen-space pixels.
    pub thickness: f32,
}

impl Stroke {
    /// Construct a stroke from an explicit color and screen-pixel
    /// thickness.
    pub const fn new(color: Color, thickness: f32) -> Self {
        Self { color, thickness }
    }

    /// Convenience default: opaque white at 1.5 screen-pixel
    /// thickness. Adjust per drawable as needed.
    pub fn default_white() -> Self {
        Self {
            color: Color::WHITE,
            thickness: 1.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_round_trip() {
        let c = Color::rgba(0.1, 0.2, 0.3, 0.4);
        assert_eq!(c.to_array(), [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn rgb_sets_alpha_to_one() {
        let c = Color::rgb(0.5, 0.5, 0.5);
        assert_eq!(c.a, 1.0);
    }

    #[test]
    fn const_palette_alpha_is_one() {
        for c in [
            Color::WHITE,
            Color::BLACK,
            Color::RED,
            Color::GREEN,
            Color::BLUE,
        ] {
            assert_eq!(c.a, 1.0);
        }
    }

    #[test]
    fn stroke_new_round_trips_color_and_thickness() {
        let s = Stroke::new(Color::RED, 3.0);
        assert_eq!(s.color, Color::RED);
        assert_eq!(s.thickness, 3.0);
    }

    #[test]
    fn stroke_default_white_is_white_and_one_point_five_pixels() {
        let s = Stroke::default_white();
        assert_eq!(s.color, Color::WHITE);
        assert_eq!(s.thickness, 1.5);
    }
}
