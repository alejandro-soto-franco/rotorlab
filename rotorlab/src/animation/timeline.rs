//! Timeline of animations driving a [`Scene`](crate::scene::Scene).
//!
//! The [`Timeline`] is the per-[`Scene`](crate::scene::Scene) scheduler that owns
//! [`Animation`] trait objects, advances a
//! global cursor through `play` and `wait`, and dispatches per-frame
//! `interpolate` and `finalize` calls during
//! [`Scene::render`](crate::scene::Scene::render).
//!
//! # Lifetime parameter
//!
//! [`Timeline`] is generic over `'anim`, the lifetime of any data the
//! boxed animations may borrow (typically a drawable owned by the
//! caller). For the common case where every animation owns its target
//! state outright, use the unconstrained form [`Timeline<'static>`],
//! which is what callers get from [`Timeline::new`] when type inference
//! has nothing to bind against.
//!
//! # `lag_ratio` v0.1
//!
//! When [`Timeline::play`] is called with a non-empty iterable, the
//! `lag_ratio` of the first animation is read once and used to stagger
//! every subsequent animation in the iterable by
//! `lag_ratio * first.run_time()` seconds. Composite animations'
//! internal lag ratios are not recursed into; this matches the design
//! doc 4.2 "single level honoured" rule for v0.1.

use super::animation::Animation;

/// Per-entry bookkeeping for the dispatch loop.
///
/// `Pending` and `Active` differ only in whether the entry has received
/// at least one [`Animation::interpolate`] call. Stock animations may
/// use the first-call edge to capture starting state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryState {
    /// `now < start_time`: not yet scheduled to fire.
    Pending,
    /// `start_time <= now < start_time + run_time`: at least one
    /// `interpolate` call has been issued, no `finalize` yet.
    Active,
    /// `finalize` has been called; the entry is inert from now on.
    Done,
}

/// A scheduled [`Animation`] together with its global start time and
/// dispatch state.
struct TimelineEntry<'anim> {
    /// The boxed animation. `'anim` is the borrow lifetime any captured
    /// references must outlive.
    anim: Box<dyn Animation + 'anim>,
    /// Wall-clock seconds (in the [`Timeline`] cursor frame) at which
    /// this animation's first `interpolate(0.0)` should fire.
    start_time: f32,
    /// Lifecycle marker for the dispatch loop.
    state: EntryState,
}

/// Ordered collection of scheduled animations.
///
/// See the module-level docs for the lifetime parameter and the v0.1
/// `lag_ratio` rule. The cursor advances monotonically through
/// [`play`](Self::play) and [`wait`](Self::wait); there is no rewinding.
pub struct Timeline<'anim> {
    /// Every animation ever scheduled, in insertion order. `Done`
    /// entries are kept in place so the dispatch loop can iterate
    /// without re-indexing; their per-frame work is skipped.
    entries: Vec<TimelineEntry<'anim>>,
    /// Current "now" in seconds. Advances by `play` (by
    /// `max(run_time)`, plus any tail lag) and `wait` (by the supplied
    /// duration). Reported externally via
    /// [`total_run_time`](Self::total_run_time).
    cursor: f32,
}

impl Default for Timeline<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'anim> Timeline<'anim> {
    /// Construct an empty timeline with cursor at `0.0` seconds.
    ///
    /// Equivalent to [`Timeline::default`]. Provided so call sites can
    /// stay readable without importing the `Default` trait.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            cursor: 0.0,
        }
    }

    /// Schedule an iterable of animations to start at the current cursor.
    ///
    /// Reads the first animation's [`Animation::lag_ratio`] once; each
    /// subsequent animation in `anims` starts `lag_ratio * first_run_time`
    /// seconds later than its predecessor. The cursor is advanced by the
    /// schedule's full footprint, i.e. the maximum
    /// `start_offset + run_time` across the iterable. Composite
    /// animations' internal lag ratios are not recursed into for v0.1.
    ///
    /// Calling `play` on an empty iterable is a no-op.
    pub fn play<I>(&mut self, anims: I)
    where
        I: IntoIterator<Item = Box<dyn Animation + 'anim>>,
    {
        let mut iter = anims.into_iter();
        let Some(first) = iter.next() else {
            return;
        };
        let base = self.cursor;
        let lag_ratio = first.lag_ratio().clamp(0.0, 1.0);
        let first_run_time = first.run_time();
        let stagger = lag_ratio * first_run_time;

        let mut footprint = first_run_time;
        let mut index = 0_u32;

        self.entries.push(TimelineEntry {
            anim: first,
            start_time: base,
            state: EntryState::Pending,
        });
        index += 1;

        for anim in iter {
            let offset = stagger * index as f32;
            let run_time = anim.run_time();
            let local_end = offset + run_time;
            if local_end > footprint {
                footprint = local_end;
            }
            self.entries.push(TimelineEntry {
                anim,
                start_time: base + offset,
                state: EntryState::Pending,
            });
            index += 1;
        }

        self.cursor = base + footprint;
    }

    /// Advance the cursor by `seconds` without scheduling anything.
    ///
    /// Negative inputs are clamped to `0.0`; this keeps the cursor
    /// monotone. Callers that pass `0.0` see no change.
    pub fn wait(&mut self, seconds: f32) {
        if seconds > 0.0 {
            self.cursor += seconds;
        }
    }

    /// Total wall-clock time covered by the schedule, in seconds.
    ///
    /// This is what [`Scene::render`](crate::scene::Scene::render)
    /// multiplies by the configured FPS to compute the total frame
    /// count.
    pub fn total_run_time(&self) -> f32 {
        self.cursor
    }

    /// Drive every still-active entry to time `now` (seconds in the
    /// timeline's cursor frame).
    ///
    /// For each entry not already `Done`:
    ///
    /// * if `now < start_time`: skip (still `Pending`);
    /// * if `start_time <= now < start_time + run_time`: clamp the raw
    ///   parameter to `[0, 1)`, ease through
    ///   [`Animation::rate_func`], and call [`Animation::interpolate`].
    ///   The entry transitions to `Active`.
    /// * if `now >= start_time + run_time`: clamp to `1.0`, ease, call
    ///   `interpolate(eased_at_1)`, then [`Animation::finalize`], in
    ///   that order. The entry transitions to `Done` and never fires
    ///   again.
    ///
    /// `run_time <= 0.0` collapses to an immediate `interpolate(1.0) +
    /// finalize` pair on the frame the entry fires.
    pub(crate) fn dispatch_frame(&mut self, now: f32) {
        for entry in &mut self.entries {
            if entry.state == EntryState::Done {
                continue;
            }
            if now < entry.start_time {
                continue;
            }
            let run_time = entry.anim.run_time();
            let raw = if run_time > 0.0 {
                (now - entry.start_time) / run_time
            } else {
                1.0
            };
            if raw < 1.0 {
                let clamped = raw.max(0.0);
                let eased = entry.anim.rate_func().apply(clamped);
                entry.anim.interpolate(eased);
                entry.state = EntryState::Active;
            } else {
                let eased = entry.anim.rate_func().apply(1.0);
                entry.anim.interpolate(eased);
                entry.anim.finalize();
                entry.state = EntryState::Done;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::RateFunc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Animation whose `interpolate` records every alpha it was given
    /// into a shared vector. Lets tests observe the rate-eased trace
    /// produced by [`Timeline::dispatch_frame`].
    #[derive(Clone)]
    struct AlphaTrace {
        calls: Arc<Mutex<Vec<f32>>>,
        finalize_count: Arc<AtomicUsize>,
        run_time: f32,
        rate: RateFunc,
    }

    impl AlphaTrace {
        fn new(run_time: f32, rate: RateFunc) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                finalize_count: Arc::new(AtomicUsize::new(0)),
                run_time,
                rate,
            }
        }
    }

    impl Animation for AlphaTrace {
        fn run_time(&self) -> f32 {
            self.run_time
        }
        fn rate_func(&self) -> RateFunc {
            self.rate
        }
        fn interpolate(&mut self, alpha: f32) {
            self.calls.lock().unwrap().push(alpha);
        }
        fn finalize(&mut self) {
            self.finalize_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn quintic(t: f32) -> f32 {
        let t3 = t * t * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        6.0 * t5 - 15.0 * t4 + 10.0 * t3
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn empty_timeline_total_run_time_is_zero() {
        let tl: Timeline<'static> = Timeline::new();
        assert_eq!(tl.total_run_time(), 0.0);
    }

    #[test]
    fn new_and_default_agree() {
        let a: Timeline<'static> = Timeline::new();
        let b: Timeline<'static> = Timeline::default();
        assert_eq!(a.total_run_time(), b.total_run_time());
    }

    #[test]
    fn wait_advances_cursor_by_seconds() {
        let mut tl: Timeline<'static> = Timeline::new();
        tl.wait(0.5);
        assert!((tl.total_run_time() - 0.5).abs() < EPS);
        tl.wait(0.25);
        assert!((tl.total_run_time() - 0.75).abs() < EPS);
    }

    #[test]
    fn wait_with_non_positive_is_no_op() {
        let mut tl: Timeline<'static> = Timeline::new();
        tl.wait(0.0);
        tl.wait(-1.0);
        assert!((tl.total_run_time() - 0.0).abs() < EPS);
    }

    #[test]
    fn play_advances_cursor_by_max_run_time() {
        let a = AlphaTrace::new(1.0, RateFunc::Linear);
        let b = AlphaTrace::new(2.0, RateFunc::Linear);
        let mut tl: Timeline<'static> = Timeline::new();
        tl.play([
            Box::new(a) as Box<dyn Animation>,
            Box::new(b) as Box<dyn Animation>,
        ]);
        assert!((tl.total_run_time() - 2.0).abs() < EPS);
    }

    #[test]
    fn play_then_wait_then_play_total_is_sequential() {
        let mut tl: Timeline<'static> = Timeline::new();
        tl.play([Box::new(AlphaTrace::new(1.0, RateFunc::Linear)) as Box<dyn Animation>]);
        tl.wait(0.5);
        tl.play([Box::new(AlphaTrace::new(0.75, RateFunc::Linear)) as Box<dyn Animation>]);
        assert!((tl.total_run_time() - 2.25).abs() < EPS);
    }

    #[test]
    fn dispatch_calls_interpolate_with_eased_alpha() {
        let trace = AlphaTrace::new(1.0, RateFunc::Linear);
        let calls = trace.calls.clone();
        let mut tl: Timeline<'static> = Timeline::new();
        tl.play([Box::new(trace) as Box<dyn Animation>]);

        tl.dispatch_frame(0.0);
        tl.dispatch_frame(0.5);
        tl.dispatch_frame(1.0);

        let observed = calls.lock().unwrap().clone();
        assert_eq!(observed.len(), 3);
        assert!((observed[0] - 0.0).abs() < EPS);
        assert!((observed[1] - 0.5).abs() < EPS);
        assert!((observed[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn finalize_runs_exactly_once_at_alpha_one() {
        let trace = AlphaTrace::new(1.0, RateFunc::Linear);
        let finalize_count = trace.finalize_count.clone();
        let mut tl: Timeline<'static> = Timeline::new();
        tl.play([Box::new(trace) as Box<dyn Animation>]);

        tl.dispatch_frame(0.5);
        assert_eq!(finalize_count.load(Ordering::SeqCst), 0);
        tl.dispatch_frame(1.0);
        assert_eq!(finalize_count.load(Ordering::SeqCst), 1);
        tl.dispatch_frame(2.0);
        assert_eq!(finalize_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pending_animation_skipped_before_start_time() {
        let trace = AlphaTrace::new(1.0, RateFunc::Linear);
        let calls = trace.calls.clone();
        let mut tl: Timeline<'static> = Timeline::new();
        tl.wait(0.5);
        tl.play([Box::new(trace) as Box<dyn Animation>]);

        tl.dispatch_frame(0.0);
        tl.dispatch_frame(0.25);
        assert!(calls.lock().unwrap().is_empty());

        tl.dispatch_frame(0.5);
        let observed = calls.lock().unwrap().clone();
        assert_eq!(observed.len(), 1);
        assert!((observed[0] - 0.0).abs() < EPS);
    }

    #[test]
    fn two_animations_complete_in_max_run_time() {
        let a = AlphaTrace::new(1.0, RateFunc::Linear);
        let b = AlphaTrace::new(2.0, RateFunc::Linear);
        let fa = a.finalize_count.clone();
        let fb = b.finalize_count.clone();
        let mut tl: Timeline<'static> = Timeline::new();
        tl.play([
            Box::new(a) as Box<dyn Animation>,
            Box::new(b) as Box<dyn Animation>,
        ]);

        // Walk the dispatch through every "frame" at 60 fps so both
        // animations cross alpha == 1 cleanly.
        for f in 0..=120 {
            let now = f as f32 / 60.0;
            tl.dispatch_frame(now);
        }

        assert_eq!(fa.load(Ordering::SeqCst), 1);
        assert_eq!(fb.load(Ordering::SeqCst), 1);
        assert!((tl.total_run_time() - 2.0).abs() < EPS);
    }

    #[test]
    fn wait_produces_no_animation_calls() {
        let mut tl: Timeline<'static> = Timeline::new();
        tl.wait(1.0);
        // No animations scheduled, so dispatch_frame is a no-op. The
        // Scene-level "60 frames" half of this rule lives in
        // src/scene/scene.rs.
        tl.dispatch_frame(0.0);
        tl.dispatch_frame(0.5);
        tl.dispatch_frame(1.0);
        assert!((tl.total_run_time() - 1.0).abs() < EPS);
    }

    #[test]
    fn non_default_rate_func_observed_in_alpha_trace() {
        let linear = AlphaTrace::new(1.0, RateFunc::Linear);
        let smoother = AlphaTrace::new(1.0, RateFunc::Smootherstep);
        let lin_calls = linear.calls.clone();
        let smo_calls = smoother.calls.clone();

        let mut tl_lin: Timeline<'static> = Timeline::new();
        tl_lin.play([Box::new(linear) as Box<dyn Animation>]);
        let mut tl_smo: Timeline<'static> = Timeline::new();
        tl_smo.play([Box::new(smoother) as Box<dyn Animation>]);

        for &t in &[0.0_f32, 0.25, 0.5, 0.75] {
            tl_lin.dispatch_frame(t);
            tl_smo.dispatch_frame(t);
        }

        let lin_obs = lin_calls.lock().unwrap().clone();
        let smo_obs = smo_calls.lock().unwrap().clone();
        assert_eq!(lin_obs.len(), 4);
        assert_eq!(smo_obs.len(), 4);

        for (i, &t) in [0.0_f32, 0.25, 0.5, 0.75].iter().enumerate() {
            assert!(
                (lin_obs[i] - t).abs() < EPS,
                "linear alpha mismatch at t = {t}: {}",
                lin_obs[i]
            );
            let expected = quintic(t);
            assert!(
                (smo_obs[i] - expected).abs() < EPS,
                "smootherstep alpha mismatch at t = {t}: {} vs {}",
                smo_obs[i],
                expected,
            );
        }
    }
}
