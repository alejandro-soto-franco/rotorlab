//! Timeline of animations driving a [`Scene`](crate::scene::Scene).
//!
//! Stub for Task 10: see plan. The full timeline (scheduled animations,
//! `play`, `wait`, durations, rate functions) lands in Plan 3 Task 10. For
//! Task 2 the [`Scene`] only needs to name a `Timeline` field and to ask it
//! for a total run time, which the empty default reports as `0.0` seconds.

/// Ordered collection of scheduled animations.
///
/// Stub for Task 10: see plan. In Plan 3 Task 10 this becomes a real
/// container with `play(&mut self, anim, duration)` and `wait(&mut self,
/// duration)` entry points; for Task 2 it stores nothing and reports a
/// total run time of zero.
#[derive(Default)]
pub struct Timeline {
    /// Reserved private state for Task 10. The empty struct keeps the
    /// public surface small and lets us evolve the representation freely
    /// when scheduled animations land.
    _private: (),
}

impl Timeline {
    /// Construct an empty timeline.
    ///
    /// Equivalent to `Timeline::default()`. Provided so call sites can
    /// stay readable without importing the `Default` trait.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Total run time of the timeline in seconds.
    ///
    /// Stub for Task 10: see plan. Always returns `0.0` until Task 10
    /// wires up scheduled animations. This is the value the [`Scene`]
    /// frame-loop multiplies by `fps` to compute the total frame count;
    /// returning zero here is what makes a freshly-constructed scene
    /// emit zero frames.
    pub fn total_run_time(&self) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_default_agree() {
        let a = Timeline::new();
        let b = Timeline::default();
        assert_eq!(a.total_run_time(), b.total_run_time());
        assert_eq!(a.total_run_time(), 0.0);
    }
}
