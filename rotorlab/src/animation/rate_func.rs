//! Easing curves used to reshape a normalised animation parameter.
//!
//! A [`RateFunc`] is a pure function `f: [0, 1] -> R` that an
//! [`Animation`](crate::animation::Animation) applies to its raw alpha before
//! handing the result to `interpolate`. The default curve, [`Smootherstep`],
//! is the quintic `6 t^5 - 15 t^4 + 10 t^3` whose first and second derivatives
//! both vanish at the endpoints, giving visually smooth starts and stops.
//!
//! The catalogue and naming follow ManimGL's `rate_funcs.py` so that prior
//! art on Manim curves transfers directly. Formulas are inlined alongside
//! each variant so the file is self-documenting.
//!
//! [`Smootherstep`]: RateFunc::Smootherstep

/// Easing curve applied to a normalised animation parameter.
///
/// Every variant maps the closed interval `[0, 1]` onto values that an
/// animation uses to drive its visual state. Monotone variants
/// ([`Linear`](Self::Linear), [`Smoothstep`](Self::Smoothstep),
/// [`Smootherstep`](Self::Smootherstep), [`Smoothererstep`](Self::Smoothererstep),
/// [`RushInto`](Self::RushInto), [`RushFrom`](Self::RushFrom)) satisfy
/// `f(0) = 0` and `f(1) = 1`. The "go-and-return" variants
/// ([`ThereAndBack`](Self::ThereAndBack),
/// [`ThereAndBackWithPause`](Self::ThereAndBackWithPause)) instead satisfy
/// `f(0) = f(1) = 0` and peak at `1.0` near the middle of the interval.
#[derive(Clone, Copy)]
pub enum RateFunc {
    /// Identity curve. `f(t) = t`. Monotone, `f(0) = 0`, `f(1) = 1`.
    Linear,
    /// Cubic smoothstep. `f(t) = 3 t^2 - 2 t^3`. Monotone with
    /// `f'(0) = f'(1) = 0`.
    Smoothstep,
    /// Quintic smootherstep, the default rate function.
    /// `f(t) = 6 t^5 - 15 t^4 + 10 t^3`. Monotone with
    /// `f'(0) = f'(1) = f''(0) = f''(1) = 0`.
    Smootherstep,
    /// Septic smoothing curve.
    /// `f(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7`. Monotone with
    /// `f^(k)(0) = f^(k)(1) = 0` for `k = 1, 2, 3` (the Hermite-style
    /// interpolant of order three at each endpoint).
    Smoothererstep,
    /// Symmetric "go and come back" curve. Returns `0` at both endpoints
    /// and `1.0` at `t = 0.5`. Built from
    /// [`Smootherstep`](Self::Smootherstep) on a folded parameter.
    ThereAndBack,
    /// Eased ease-out: race in, settle smoothly. `f(t) = 2 * smootherstep(t / 2)`.
    /// Monotone, `f(0) = 0`, `f(1) = 1`. Acceleration at the start.
    RushInto,
    /// Eased ease-in: settle, then race out. `f(t) = 1 - 2 * smootherstep((1 - t) / 2)`.
    /// Monotone, `f(0) = 0`, `f(1) = 1`. Acceleration at the end.
    RushFrom,
    /// Symmetric "go, hold, come back" curve. Returns `0` at both endpoints,
    /// rises to `1.0` by `t = 1/3`, holds the plateau through `t = 2/3`, then
    /// falls back to `0`. Useful for momentary highlight animations.
    ThereAndBackWithPause,
    /// User-supplied easing function. Stored as a function pointer
    /// (`fn(f32) -> f32`), not a closure with captures, so [`RateFunc`]
    /// stays `Copy` and `Send`.
    Custom(fn(f32) -> f32),
}

impl RateFunc {
    /// Apply this rate function to a normalised parameter `t`.
    ///
    /// Callers must clamp `t` to `[0, 1]` before invoking `apply`; the
    /// `Timeline` driver (Plan 3 Task 10) is responsible for that
    /// clamping. This routine performs no bounds checking and the formulas
    /// are not guaranteed to behave outside `[0, 1]`.
    pub fn apply(self, t: f32) -> f32 {
        match self {
            Self::Linear => t,
            Self::Smoothstep => 3.0 * t * t - 2.0 * t * t * t,
            Self::Smootherstep => smootherstep_raw(t),
            Self::Smoothererstep => {
                let t4 = t * t * t * t;
                let t5 = t4 * t;
                let t6 = t5 * t;
                let t7 = t6 * t;
                35.0 * t4 - 84.0 * t5 + 70.0 * t6 - 20.0 * t7
            }
            Self::ThereAndBack => {
                let u = if t < 0.5 { 2.0 * t } else { 2.0 * (1.0 - t) };
                smootherstep_raw(u)
            }
            Self::RushInto => 2.0 * smootherstep_raw(t / 2.0),
            Self::RushFrom => 1.0 - 2.0 * smootherstep_raw((1.0 - t) / 2.0),
            Self::ThereAndBackWithPause => {
                let third = 1.0 / 3.0;
                if t < third {
                    smootherstep_raw(3.0 * t)
                } else if t > 2.0 / 3.0 {
                    smootherstep_raw(3.0 * (1.0 - t))
                } else {
                    1.0
                }
            }
            Self::Custom(f) => f(t),
        }
    }
}

/// Quintic smootherstep helper shared by several [`RateFunc`] variants.
///
/// Computes `6 t^5 - 15 t^4 + 10 t^3` directly. Inputs are assumed to lie
/// in `[0, 1]`; values outside that range are returned as-is by the
/// polynomial without clamping.
fn smootherstep_raw(t: f32) -> f32 {
    let t3 = t * t * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    6.0 * t5 - 15.0 * t4 + 10.0 * t3
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn quintic(t: f32) -> f32 {
        let t3 = t * t * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        6.0 * t5 - 15.0 * t4 + 10.0 * t3
    }

    #[test]
    fn linear_endpoints_match_identity() {
        assert!((RateFunc::Linear.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::Linear.apply(1.0) - 1.0).abs() < EPS);
        assert!((RateFunc::Linear.apply(0.5) - 0.5).abs() < EPS);
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!((RateFunc::Smoothstep.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::Smoothstep.apply(1.0) - 1.0).abs() < EPS);
        assert!((RateFunc::Smoothstep.apply(0.5) - 0.5).abs() < EPS);
    }

    #[test]
    fn smootherstep_matches_quintic_polynomial() {
        let samples = [0.0_f32, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9, 1.0];
        for &t in &samples {
            let actual = RateFunc::Smootherstep.apply(t);
            let expected = quintic(t);
            assert!(
                (actual - expected).abs() < EPS,
                "smootherstep mismatch at t = {t}: actual = {actual}, expected = {expected}",
            );
        }
    }

    #[test]
    fn smoothererstep_endpoints() {
        assert!((RateFunc::Smoothererstep.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::Smoothererstep.apply(1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn there_and_back_peaks_at_midpoint() {
        assert!((RateFunc::ThereAndBack.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::ThereAndBack.apply(1.0) - 0.0).abs() < EPS);
        assert!((RateFunc::ThereAndBack.apply(0.5) - 1.0).abs() < EPS);
    }

    #[test]
    fn rush_into_saturates_at_endpoint() {
        assert!((RateFunc::RushInto.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::RushInto.apply(1.0) - 1.0).abs() < EPS);
        let expected_mid = 2.0 * quintic(0.25);
        assert!(
            (RateFunc::RushInto.apply(0.5) - expected_mid).abs() < EPS,
            "RushInto mid expected {}",
            expected_mid,
        );
    }

    #[test]
    fn rush_from_endpoints() {
        assert!((RateFunc::RushFrom.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::RushFrom.apply(1.0) - 1.0).abs() < EPS);
        let expected_mid = 1.0 - 2.0 * quintic(0.25);
        assert!((RateFunc::RushFrom.apply(0.5) - expected_mid).abs() < EPS);
    }

    #[test]
    fn there_and_back_with_pause_holds_in_middle_third() {
        assert!((RateFunc::ThereAndBackWithPause.apply(0.0) - 0.0).abs() < EPS);
        assert!((RateFunc::ThereAndBackWithPause.apply(1.0) - 0.0).abs() < EPS);
        assert!((RateFunc::ThereAndBackWithPause.apply(0.5) - 1.0).abs() < EPS);
        assert!((RateFunc::ThereAndBackWithPause.apply(0.4) - 1.0).abs() < EPS);
        assert!((RateFunc::ThereAndBackWithPause.apply(0.6) - 1.0).abs() < EPS);
    }

    #[test]
    fn custom_invokes_supplied_function() {
        fn double(t: f32) -> f32 {
            t * 2.0
        }
        assert!((RateFunc::Custom(double).apply(0.25) - 0.5).abs() < EPS);
    }

    #[test]
    fn default_rate_func_is_smootherstep() {
        use crate::animation::Animation;

        struct StubAnim;
        impl Animation for StubAnim {
            fn run_time(&self) -> f32 {
                1.0
            }
            fn interpolate(&mut self, _alpha: f32) {}
        }

        let stub = StubAnim;
        assert!(matches!(stub.rate_func(), RateFunc::Smootherstep));
    }
}
