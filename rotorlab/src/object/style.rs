//! Visual-style primitives shared by drawables.
//!
//! Plan 3 Task 5 introduces [`Color`], a linear-space RGBA value used by
//! the [`Point`](crate::object::Point) drawable and (Task 7) the
//! upcoming `Line` drawable. Stroke and fill abstractions arrive in
//! Plan 3 Task 7.

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
}
