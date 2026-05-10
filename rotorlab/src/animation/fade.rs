//! Stock opacity animations: [`FadeIn`] and [`FadeOut`].
//!
//! Plan 3 Task 11 ships the simplest possible opacity animations:
//! both drive a [`HasOpacity`] target by writing the eased alpha into
//! the target's opacity each frame.
//!
//! - [`FadeIn`] writes `alpha` directly. After `interpolate(1.0)` and
//!   the timeline's `finalize` call, the target's opacity is `1.0`.
//! - [`FadeOut`] writes `1 - alpha`. After `interpolate(1.0)` and
//!   `finalize`, the target's opacity is `0.0`.
//!
//! Both default to the [`Smootherstep`](RateFunc::Smootherstep)
//! easing curve and expose a `with_rate` builder for callers who
//! want a different curve. `finalize` pins the final opacity to
//! defend against floating-point drift in the eased alpha.

use crate::animation::{Animation, RateFunc};
use crate::object::capability::HasOpacity;

/// Animate a [`HasOpacity`] target from opacity `0.0` to `1.0` over
/// `run_time` seconds.
pub struct FadeIn<'a> {
    /// Target drawable. Borrowed mutably for the lifetime of the
    /// animation so the timeline can drive it in place.
    obj: &'a mut (dyn HasOpacity + Send),
    /// Wall-clock duration in seconds.
    run_time: f32,
    /// Easing curve.
    rate: RateFunc,
}

impl<'a> FadeIn<'a> {
    /// Construct a `FadeIn` animation with the default
    /// [`Smootherstep`](RateFunc::Smootherstep) easing curve.
    pub fn new(obj: &'a mut (dyn HasOpacity + Send), run_time: f32) -> Self {
        Self {
            obj,
            run_time,
            rate: RateFunc::Smootherstep,
        }
    }

    /// Override the rate function. Returns `self` for builder-style
    /// chaining.
    pub fn with_rate(mut self, rate: RateFunc) -> Self {
        self.rate = rate;
        self
    }
}

impl Animation for FadeIn<'_> {
    fn run_time(&self) -> f32 {
        self.run_time
    }

    fn rate_func(&self) -> RateFunc {
        self.rate
    }

    fn interpolate(&mut self, alpha: f32) {
        self.obj.set_opacity(alpha);
    }

    fn finalize(&mut self) {
        self.obj.set_opacity(1.0);
    }
}

/// Animate a [`HasOpacity`] target from opacity `1.0` to `0.0` over
/// `run_time` seconds.
pub struct FadeOut<'a> {
    /// Target drawable. Borrowed mutably for the lifetime of the
    /// animation so the timeline can drive it in place.
    obj: &'a mut (dyn HasOpacity + Send),
    /// Wall-clock duration in seconds.
    run_time: f32,
    /// Easing curve.
    rate: RateFunc,
}

impl<'a> FadeOut<'a> {
    /// Construct a `FadeOut` animation with the default
    /// [`Smootherstep`](RateFunc::Smootherstep) easing curve.
    pub fn new(obj: &'a mut (dyn HasOpacity + Send), run_time: f32) -> Self {
        Self {
            obj,
            run_time,
            rate: RateFunc::Smootherstep,
        }
    }

    /// Override the rate function. Returns `self` for builder-style
    /// chaining.
    pub fn with_rate(mut self, rate: RateFunc) -> Self {
        self.rate = rate;
        self
    }
}

impl Animation for FadeOut<'_> {
    fn run_time(&self) -> f32 {
        self.run_time
    }

    fn rate_func(&self) -> RateFunc {
        self.rate
    }

    fn interpolate(&mut self, alpha: f32) {
        self.obj.set_opacity(1.0 - alpha);
    }

    fn finalize(&mut self) {
        self.obj.set_opacity(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Point;
    use crate::object::style::Color;
    use rotorlab_ga::pga3::shapes;

    #[test]
    fn fade_in_full_alpha_sets_opacity_one() {
        let mut p = Point::new(
            shapes::Point::new(0.0, 0.0, 0.0),
            Color::rgba(1.0, 0.0, 0.0, 0.0),
            1.0,
        );
        {
            let mut anim = FadeIn::new(&mut p, 1.0);
            anim.interpolate(1.0);
        }
        assert!((p.color.a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fade_in_half_alpha_sets_opacity_half() {
        let mut p = Point::new(
            shapes::Point::new(0.0, 0.0, 0.0),
            Color::rgba(1.0, 0.0, 0.0, 0.0),
            1.0,
        );
        {
            let mut anim = FadeIn::new(&mut p, 1.0);
            anim.interpolate(0.5);
        }
        assert!((p.color.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fade_in_finalize_pins_opacity_to_one() {
        let mut p = Point::new(
            shapes::Point::new(0.0, 0.0, 0.0),
            Color::rgba(1.0, 0.0, 0.0, 0.0),
            1.0,
        );
        {
            let mut anim = FadeIn::new(&mut p, 1.0);
            // Drift a touch off 1.0 to confirm finalize snaps it.
            anim.interpolate(0.9999);
            anim.finalize();
        }
        assert_eq!(p.color.a, 1.0);
    }

    #[test]
    fn fade_out_full_alpha_sets_opacity_zero() {
        let mut p = Point::new(shapes::Point::new(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        {
            let mut anim = FadeOut::new(&mut p, 1.0);
            anim.interpolate(1.0);
        }
        assert!(p.color.a.abs() < 1e-6);
    }

    #[test]
    fn fade_out_half_alpha_sets_opacity_half() {
        let mut p = Point::new(shapes::Point::new(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        {
            let mut anim = FadeOut::new(&mut p, 1.0);
            anim.interpolate(0.5);
        }
        assert!((p.color.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fade_out_finalize_pins_opacity_to_zero() {
        let mut p = Point::new(shapes::Point::new(0.0, 0.0, 0.0), Color::WHITE, 1.0);
        {
            let mut anim = FadeOut::new(&mut p, 1.0);
            anim.interpolate(0.0001);
            anim.finalize();
        }
        assert_eq!(p.color.a, 0.0);
    }
}
