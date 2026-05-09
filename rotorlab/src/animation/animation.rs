//! The [`Animation`] trait shared by every stock and user-defined animation.
//!
//! An animation is a stateful object that mutates a target between two
//! visual states over a fixed wall-clock duration. It exposes:
//!
//! * `run_time()`: the duration in seconds the [`Timeline`](crate::animation::Timeline)
//!   should allocate to this animation.
//! * `rate_func()`: the easing curve applied to the raw `[0, 1]` parameter
//!   before it reaches `interpolate`. Defaults to [`RateFunc::Smootherstep`].
//! * `lag_ratio()`: fraction of the run time staggered onto child animations
//!   when this object is itself a composite. Defaults to `0.0`, meaning all
//!   children animate in lock-step.
//! * `interpolate(&mut self, alpha)`: the workhorse, called every frame with
//!   `alpha` already passed through the rate function and clamped to
//!   `[0, 1]`. Animations are stateful (`&mut self`) so they may cache the
//!   start state on the first call.
//! * `finalize(&mut self)`: invoked once after the final `interpolate(1.0)`
//!   call. Default no-op; stock animations override it to release caches or
//!   commit state changes that should outlive the animation.

use super::rate_func::RateFunc;

/// Mutates a target between two visual states over a fixed wall-clock duration.
///
/// Implementors are stateful so they may capture starting state on the first
/// `interpolate` call. The default rate function is [`RateFunc::Smootherstep`]
/// and the default lag ratio is `0.0` (all child animations run in lock-step).
///
/// `Send` is required so animations can move between threads inside the
/// [`Timeline`](crate::animation::Timeline) scheduler.
pub trait Animation: Send {
    /// Wall-clock duration this animation should run for, in seconds.
    fn run_time(&self) -> f32;

    /// Easing curve applied to the raw `[0, 1]` alpha before
    /// [`interpolate`](Self::interpolate). Defaults to
    /// [`RateFunc::Smootherstep`].
    fn rate_func(&self) -> RateFunc {
        RateFunc::Smootherstep
    }

    /// Fraction of `run_time` staggered onto child animations in a composite.
    ///
    /// `0.0` (the default) means all children animate together; values in
    /// `(0.0, 1.0]` cascade them. Single-target animations may ignore this.
    fn lag_ratio(&self) -> f32 {
        0.0
    }

    /// Drive the animation to the visual state at parameter `alpha`.
    ///
    /// `alpha` is already eased through [`rate_func`](Self::rate_func) and
    /// clamped to `[0, 1]` by the [`Timeline`](crate::animation::Timeline)
    /// before reaching this method.
    fn interpolate(&mut self, alpha: f32);

    /// Called once after the final [`interpolate(1.0)`](Self::interpolate)
    /// invocation. Default no-op; override to release caches or commit
    /// state changes that should outlive the animation.
    fn finalize(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubAnim;
    impl Animation for StubAnim {
        fn run_time(&self) -> f32 {
            1.0
        }
        fn interpolate(&mut self, _alpha: f32) {}
    }

    #[test]
    fn stub_anim_uses_default_lag_ratio() {
        let stub = StubAnim;
        assert_eq!(stub.lag_ratio(), 0.0);
    }

    #[test]
    fn finalize_default_is_no_op() {
        let mut stub = StubAnim;
        stub.finalize();
    }

    #[test]
    fn stub_anim_default_rate_func_is_smootherstep() {
        let stub = StubAnim;
        assert!(matches!(stub.rate_func(), RateFunc::Smootherstep));
    }
}
